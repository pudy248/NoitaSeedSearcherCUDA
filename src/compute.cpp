#include "../platforms/platform_implementation.h"

#include "../include/compute.h"
#include "../include/misc_funcs.h"
#include "../include/pngutils.h"
#include "../include/primitives.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

_compute SpanRet PLATFORM_API::EvaluateSpan(
	SearchConfig config, SpanParams span, void* threadMemBlock, void* outputPtr) {
	if (span.seedCount == 0)
		return {0, 0, false, 0};

	int seedsFound = 0;
	MemoryArena arena = {(uint8_t*)threadMemBlock, 0};
	for (int currentSeed = span.seedStart; currentSeed < span.seedStart + span.seedCount; currentSeed++) {
		arena.offset = 0;
		bool seedPassed = true;
		MemSpan output = ArenaAlloc(arena, config.memSizes.outputSize, 4);

		seedPassed &= PrecheckSeed(currentSeed, config.precheckCfg);
		if (!seedPassed)
			continue;

		if (config.precheckCfg.precheckUpwarps) {
			bool tmp = config.spawnableCfg.biomeChests;
			config.spawnableCfg.biomeChests = true;
			MemSpan upwarps = ArenaAlloc(arena, config.memSizes.spawnableMemSize, 8);
			MemSpan miscMem = ArenaAlloc(arena, 20 * TOTAL_FILTER_COUNT + 128, 8);
			int offset = 0;
			int _ = 0;
			spawnChest(315, 17, {currentSeed, {}, config.spawnableCfg, upwarps, offset, _});
			MemSpan ptr1 = {upwarps.ptr + offset, upwarps.sz - offset};
			spawnChest(75, 117, {currentSeed, {}, config.spawnableCfg, upwarps, offset, _});
			Spawnable* spawnables[] = {(Spawnable*)upwarps.ptr, (Spawnable*)ptr1.ptr};
			SpawnableBlock b = {currentSeed, 2, spawnables};
			config.spawnableCfg.biomeChests = tmp;

			seedPassed &= SpawnablesPassed(b, config.filterCfg, output, miscMem, true, true);
			ArenaSetOffset(arena, upwarps.ptr);
			if (!seedPassed)
				continue;

			if (!config.outputCfg.countPassesOnly && !config.spawnableCfg.biomeChests) {
				memcpy(output.ptr, &currentSeed, 4);
				memcpy((uint8_t*)outputPtr, output.ptr, config.memSizes.outputSize);

				int extraSeeds = span.seedStart + span.seedCount - currentSeed - 1;
				return {span.seedStart, span.seedCount, true, extraSeeds};
			}
		}

		int spawnableCount = 0;
		int spawnableOffset = 0;
		MemSpan mapMem = ArenaAlloc(arena, config.memSizes.mapDataSize, 8);
		MemSpan visited = ArenaAlloc(arena, config.memSizes.visitedMemSize);
		MemSpan spawnableDat = ArenaAlloc(arena, config.memSizes.spawnableMemSize, 4);
		MemSpan spawnables = ArenaAlloc(arena, config.memSizes.spawnableMemSize / 4, 4);
		MemSpan miscMem = ArenaAlloc(arena, max(config.memSizes.miscMemSize, 20 * TOTAL_FILTER_COUNT + 128), 8);

#ifdef DO_WORLDGEN
		for (int biomeNum = 1; biomeNum < config.biomeCount; biomeNum++) {
			GeneratedBiome b =
				GenerateMap(currentSeed, *config.biomeScopes[biomeNum], output, mapMem, visited, miscMem);
			threadSync();
#ifndef SEEDS_AS_TRIES
			SpawnParams p = {currentSeed, *config.biomeScopes[biomeNum], config.spawnableCfg, spawnableDat,
				spawnableOffset, spawnableCount};
			CheckSpawnables(b, p);
			threadSync();
#endif
		}
#endif
#ifndef SEEDS_AS_TRIES
		SpawnParams p = {currentSeed, *config.biomeScopes[0], config.spawnableCfg, spawnableDat, spawnableOffset,
			spawnableCount};
		CheckMountains(p);
		CheckEyeRooms(p);
		CheckNightmareSpawnWands(p);
		threadSync();

		SpawnableBlock result =
			ParseSpawnableBlock(spawnableDat.ptr, spawnables, config.spawnableCfg, currentSeed, spawnableCount);
		threadSync();
		seedPassed &=
			SpawnablesPassed(result, config.filterCfg, output, miscMem, true, config.precheckCfg.precheckUpwarps);

		if (!seedPassed)
			continue;
#endif
		if (config.outputCfg.countPassesOnly)
			seedsFound++;
		else {
			memcpy(output.ptr, &currentSeed, 4);
			memcpy((uint8_t*)outputPtr, output.ptr, config.memSizes.outputSize);

			int extraSeeds = span.seedStart + span.seedCount - currentSeed - 1;
			return {span.seedStart, span.seedCount, true, extraSeeds};
		}
	}
	if (config.outputCfg.countPassesOnly)
		return {span.seedStart, span.seedCount, false, seedsFound};
	else
		return {span.seedStart, span.seedCount, false, 0};
}

using namespace API_INTERNAL;
Vec2i OutputLoop(FILE* outputFile, time_t startTime, OutputProgressData& progress, void (*appendOutput)(char*, char*)) {
	uint32_t displayIntervals = 0;
	uint64_t recountIntervals = 2;
	uint64_t recountTimestamp = 0;

	uint32_t lastDiff = 0;
	uint32_t lastSeed = 0;

	uint32_t checkedSeeds = 0;
	uint32_t passedSeeds = 0;

	int dbg_seed_loop_ctr = 1;
	int dbg_seed_loop_max = 1;

	uint32_t currentSeed = config.generalCfg.seedStart;
	int index = 0;
	int stoppedBlocks = 0;

	double returnedBlocksThisIter = 0;

	std::vector<Worker*> workers;
	SpanParams* params = (SpanParams*)malloc(WorkerAppetite * sizeof(SpanParams));
	bool* stopped = (bool*)malloc(NumWorkers);
	memset(stopped, false, NumWorkers);
	uint8_t* hOutput = (uint8_t*)malloc(NumWorkers * WorkerAppetite * config.memSizes.outputSize);

	//initial dispatch
	for (int i = 0; i < NumWorkers; i++) {
		workers.emplace_back(CreateWorker());
	}
	for (int i = 0; i < NumWorkers; i++) {
		if (currentSeed <= config.generalCfg.seedEnd) {
			for (int j = 0; j < WorkerAppetite; j++) {
				if (currentSeed > config.generalCfg.seedEnd) {
					params[j] = {0, 0};
					continue;
				}
				uint32_t nextSeed = currentSeed;
				if (!config.generalCfg.seedStart)
					nextSeed = seed_list[currentSeed];
#ifdef REALTIME_SEEDS
				uint8_t* output = hOutput + (i * WorkerAppetite + j) * config.memSizes.outputSize;
				int _ = 0;
				writeInt({output, 4}, _, currentSeed);
				nextSeed = pick_world_seed(startTime + currentSeed);
#endif
				uint32_t length =
					std::min(config.generalCfg.seedBlockSize, config.generalCfg.seedEnd - currentSeed + 1);
				params[j] = {(int)nextSeed, (int)length};
				currentSeed += length;
			}
			DispatchJob(*workers[i], params);
		} else {
			stoppedBlocks++;
			stopped[i] = true;
			//printf("Sleeping worker %i.\n", index);
		}
	}

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	//loop
	while (stoppedBlocks < NumWorkers) {
		if (progress.abort) {
			for (int i = 0; i < NumWorkers; i++)
				AbortJob(*workers[i]);
			break;
		}
		if (currentSeed > config.generalCfg.seedEnd && dbg_seed_loop_ctr < dbg_seed_loop_max) {
			dbg_seed_loop_ctr++;
			currentSeed = config.generalCfg.seedStart;
		}

		if (index == 0) {
			std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
			std::chrono::nanoseconds duration = time2 - time1;
			uint64_t milliseconds = (uint64_t)(duration.count() / 1000000);
			progress.elapsedMillis = milliseconds;
			progress.searchedSeeds = checkedSeeds;
			progress.validSeeds = passedSeeds;
			uint64_t timescale1 = recountIntervals * recountIntervals * recountIntervals * recountIntervals;
			if (timescale1 < milliseconds) {
				recountIntervals++;
				uint64_t elapsed = milliseconds - recountTimestamp;
				recountTimestamp = milliseconds;
				if (returnedBlocksThisIter == 0)
					goto recount_end;
				{
					double expected = (DispatchRate * elapsed) / 1000.;
					double predicted_optimal = config.generalCfg.seedBlockSize * returnedBlocksThisIter / expected;
					if (DEBUG_FLAGS & DEBUG::LOG_SEED_BLOCKS)
						fprintf(stderr,
							"%llims: Recalculating dispatch block size. Current size %i, predicted optimal size %i.\n",
							milliseconds, config.generalCfg.seedBlockSize, (uint32_t)predicted_optimal);
					returnedBlocksThisIter = 0;
#ifndef REALTIME_SEEDS
					if (!config.generalCfg.seedBlockOverride)
						// Weighted geometric mean
						config.generalCfg.seedBlockSize = std::max(
							1u, (uint32_t)std::exp(
									(std::log(config.generalCfg.seedBlockSize) + 2 * std::log(predicted_optimal)) / 3));
#endif
				}
recount_end:
			}
			if (displayIntervals * config.outputCfg.printInterval * 1000 < milliseconds) {
				lastDiff = checkedSeeds - lastSeed;
				lastSeed = checkedSeeds;
				displayIntervals++;
				float percentComplete =
					((float)(checkedSeeds) / (config.generalCfg.seedEnd - config.generalCfg.seedStart + 1));
				progress.progressPercent = percentComplete;
				int seconds = (displayIntervals - 1) * config.outputCfg.printInterval;
				int minutes = seconds / 60;
				int hours = minutes / 60;
				if (config.outputCfg.printProgressLog) {
					printf(
						"[%02ih %02im %02is]: %2.3f%% complete. Searched %i seeds (+%i this interval), found %i valid seeds.\n",
						hours, minutes % 60, seconds % 60, percentComplete * 100, checkedSeeds, lastDiff, passedSeeds);
					fflush(stdout);
				}
			}
		}

		if (!stopped[index] && QueryWorker(*workers[index])) {
			SpanRet* returns = SubmitJob(*workers[index]);
			memset(params, 0, sizeof(SpanParams) * WorkerAppetite);
			int* times = (int*)malloc(4 * WorkerAppetite);
			bool* hasOutput = (bool*)malloc(WorkerAppetite);
			for (int i = 0; i < WorkerAppetite; i++)
				hasOutput[i] = false;
			int inputIdx = 0;

			if (config.outputCfg.countPassesOnly) {
				returnedBlocksThisIter++;
				for (int i = 0; i < WorkerAppetite; i++) {
					checkedSeeds += returns[i].seedCount;
					passedSeeds += returns[i].leftoverSeeds;
				}
			} else {
				for (int i = 0; i < WorkerAppetite; i++) {
					returnedBlocksThisIter += (returns[i].seedCount - returns[i].leftoverSeeds) /
											  (double)(WorkerAppetite * config.generalCfg.seedBlockSize);
					checkedSeeds += returns[i].seedCount - returns[i].leftoverSeeds;
					if (!returns[i].seedFound)
						continue;
					passedSeeds++;

					uint8_t* uOutput = (uint8_t*)returns[i].outputPtr;
					uint8_t* output = hOutput + (index * WorkerAppetite + i) * config.memSizes.outputSize;
					int _ = 0;
					times[i] = readInt(output, _);
					memcpy(output, uOutput, config.memSizes.outputSize);
					hasOutput[i] = true;

					if (returns[i].leftoverSeeds > 0) {
						params[inputIdx++] = {returns[i].seedStart + returns[i].seedCount - returns[i].leftoverSeeds,
							returns[i].leftoverSeeds};
					};
				}
			}
			if (inputIdx > 0 || currentSeed <= config.generalCfg.seedEnd) {
				for (int i = 0; i < WorkerAppetite; i++) {
					if (currentSeed > config.generalCfg.seedEnd || returns[i].seedFound)
						continue;
					uint32_t nextSeed = currentSeed;
					if (!config.generalCfg.seedStart)
						nextSeed = seed_list[currentSeed];
#ifdef REALTIME_SEEDS
					uint8_t* output = hOutput + (index * WorkerAppetite + i) * config.memSizes.outputSize;
					int _ = 0;
					writeInt({output, 4}, _, currentSeed);
					nextSeed = pick_world_seed(startTime + currentSeed);
#endif
					uint32_t length =
						std::min(config.generalCfg.seedBlockSize, config.generalCfg.seedEnd - currentSeed + 1);
					params[inputIdx++] = {(int)nextSeed, (int)length};
					currentSeed += length;
				}
				DispatchJob(*workers[index], params);
			} else {
				stoppedBlocks++;
				stopped[index] = true;
				//printf("Sleeping worker %i.\n", index);
			}

			for (int i = 0; i < WorkerAppetite; i++) {
				if (!hasOutput[i])
					continue;
				uint8_t* output = hOutput + (index * WorkerAppetite + i) * config.memSizes.outputSize;

				int time[2] = {times[i], (int)startTime};
				PrintOutputBlock(output, time, outputFile, config.outputCfg, appendOutput);
			}
			free(times);
			free(hasOutput);
		}
		index = (index + 1) % NumWorkers;
	}
	for (int i = 0; i < NumWorkers; i++) {
		DestroyWorker(*workers[i]);
		delete workers[i];
	}
	workers.clear();
	free(params);
	free(stopped);
	free(hOutput);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	std::chrono::nanoseconds duration = time2 - time1;
	uint64_t milliseconds = (uint64_t)(duration.count() / 1000000);
	progress.elapsedMillis = milliseconds;
	progress.searchedSeeds = checkedSeeds;
	progress.validSeeds = passedSeeds;

	return {(int)checkedSeeds, (int)passedSeeds};
}

void InstantiateBiomes(
	BiomeWangScope** ss, int& biomeCount, int& maxMapArea, BiomeMapChunks map, std::vector<Biome>& b) {
	void* dPtr = UploadToDevice(map.map, map.w * map.h);
	BiomeWangScope nullScope = {};
	nullScope.map = {map.w, map.h, (Biome*)dPtr};
	BiomeWangScope* dnScope = (BiomeWangScope*)UploadToDevice(&nullScope, sizeof(BiomeWangScope));
	ss[biomeCount++] = dnScope;

	for (auto& c : map.chunks) {
		if (std::ranges::find(b, c.b) == b.end())
			continue;
		BiomeSector sector = {c.b, c.x, c.y, c.w, c.h};
		const std::string& wang_tiles = get_wak_file(HTables::wang_paths[c.b]);
		Vec2i tileDims = GetBufferImageDimensions((uint8_t*)wang_tiles.c_str());

		sector.tiles_w = tileDims.x;
		sector.tiles_h = tileDims.y;
		sector.map_w = GetWidthFromPix(sector.worldX, sector.worldX + sector.worldW);
		sector.map_h = GetWidthFromPix(sector.worldY, sector.worldY + sector.worldH);

		uint8_t* hTileData = (uint8_t*)malloc(3 * tileDims.x * tileDims.y);
		ReadBufferImage((uint8_t*)wang_tiles.c_str(), hTileData, false);

		BiomeWangScope scope;
		scope.map = {map.w, map.h, (Biome*)dPtr};
		if (DEBUG_FLAGS & DEBUG::LOG_SPAWN_PIXELS)
			fprintf(stderr, "%s\n", HTables::wang_paths[c.b]);
		scope.ts = stbhw_build_tileset_from_image(hTileData, sector.b, 3 * tileDims.x, tileDims.x, tileDims.y);
		sector.wang_w = (sector.map_w + scope.ts.short_side_len - 1) / scope.ts.short_side_len;
		sector.wang_h = (sector.map_h + scope.ts.short_side_len + 3) / scope.ts.short_side_len;
		maxMapArea = max(maxMapArea, (int)(sector.map_w * sector.map_h));

		scope.ts.tileData = (uint8_t*)UploadToDevice(scope.ts.tileData, 3 * tileDims.x * tileDims.y);
		scope.bSec = sector;
		free(hTileData);
		BiomeWangScope* dScope = (BiomeWangScope*)UploadToDevice(&scope, sizeof(BiomeWangScope));
		ss[biomeCount++] = dScope;
	}
	free(map.map);
}

void SearchMain(OutputProgressData& progress, void (*appendOutput)(char*, char*)) {
	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	UploadBiomeData();

	FILE* f = config.outputCfg.printOutputToFile ? fopen(config.outputCfg.outputFile, "wb") : NULL;

	time_t startTime = time(NULL) - 100000;
	Vec2i seedCounts = OutputLoop(f, startTime, progress, appendOutput);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	std::chrono::nanoseconds duration = time2 - time1;

	if (!QUIET)
		printf("Search finished in %ims. Checked %i seeds, found %i valid seeds (%.3f%%).\n",
			(int)(duration.count() / 1000000), seedCounts.x, seedCounts.y, (double)seedCounts.y / seedCounts.x * 100);

	if (f)
		fclose(f);
}
