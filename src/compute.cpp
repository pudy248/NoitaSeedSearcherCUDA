#include "../platforms/platform_implementation.h"

#include "../include/compute.h"
#include "../include/misc_funcs.h"
#include "../include/pngutils.h"
#include "../include/primitives.h"

#include <chrono>
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
			MemSpan miscMem2 = ArenaAlloc(arena, 16 * TOTAL_FILTER_COUNT + 128, 8);
			int offset = 0;
			int _ = 0;
			spawnChest(315, 17, {currentSeed, {}, config.spawnableCfg, upwarps, offset, _});
			MemSpan ptr1 = {upwarps.ptr + offset, upwarps.sz - offset};
			spawnChest(75, 117, {currentSeed, {}, config.spawnableCfg, upwarps, offset, _});
			Spawnable* spawnables[] = {(Spawnable*)upwarps.ptr, (Spawnable*)ptr1.ptr};
			SpawnableBlock b = {currentSeed, 2, spawnables};
			config.spawnableCfg.biomeChests = tmp;

			seedPassed &= SpawnablesPassed(b, config.filterCfg, output, miscMem2, true, true);
			ArenaSetOffset(arena, upwarps.ptr);
			if (!seedPassed)
				continue;

			if (!config.spawnableCfg.biomeChests) {
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
		MemSpan miscMem = ArenaAlloc(arena, config.memSizes.miscMemSize, 8);
		MemSpan miscMem2 = ArenaAlloc(arena, 16 * TOTAL_FILTER_COUNT + 128, 4);

#ifdef DO_WORLDGEN
		for (int biomeNum = 0; biomeNum < config.biomeCount; biomeNum++) {
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
		SpawnParams p = {
			currentSeed, *config.biomeScopes[0], config.spawnableCfg, spawnableDat, spawnableOffset, spawnableCount};
		CheckMountains(p);
		CheckEyeRooms(p);
		CheckNightmareSpawnWands(p);
		threadSync();

		SpawnableBlock result =
			ParseSpawnableBlock(spawnableDat.ptr, spawnables, config.spawnableCfg, currentSeed, spawnableCount);
		threadSync();
		seedPassed &=
			SpawnablesPassed(result, config.filterCfg, output, miscMem2, true, config.precheckCfg.precheckUpwarps);

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
	uint32_t recountIntervals = 2;

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
		if (currentSeed < config.generalCfg.seedEnd) {
			for (int j = 0; j < WorkerAppetite; j++) {
				if (currentSeed >= config.generalCfg.seedEnd) {
					params[j] = {0, 0};
					continue;
				}
				uint32_t nextSeed = currentSeed;
#ifdef REALTIME_SEEDS
				uint8_t* output = hOutput + (i * WorkerAppetite + j) * config.memSizes.outputSize;
				int _ = 0;
				writeInt({output, 4}, _, currentSeed);
				nextSeed = pick_world_seed(startTime + currentSeed);
#endif
				uint32_t length = std::min(config.generalCfg.seedBlockSize, config.generalCfg.seedEnd - currentSeed);
				params[j] = {(int)nextSeed, (int)length};
				currentSeed += length;
			}
			DispatchJob(*workers[i], params);
		} else {
			stoppedBlocks++;
			stopped[i] = true;
			printf("Sleeping worker %i.\n", index);
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
		if (currentSeed >= config.generalCfg.seedEnd && dbg_seed_loop_ctr < dbg_seed_loop_max) {
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
			uint64_t timescale1 = recountIntervals * recountIntervals * recountIntervals * 3;
			if (timescale1 < milliseconds) {
				recountIntervals++;
				uint64_t timescale2 = recountIntervals * recountIntervals * recountIntervals * 3;
				double expected = DispatchRate * (timescale2 - timescale1) / 1000.0;
				double fraction = returnedBlocksThisIter / expected;
				//printf("%ims: Recalculating seed block size. Current size %i, current fraction %.2f\n",
				//	milliseconds, config.generalCfg.seedBlockSize, fraction);
				returnedBlocksThisIter = 0;
#ifndef REALTIME_SEEDS
				if (!config.generalCfg.seedBlockOverride && fraction < 0.1 && config.generalCfg.seedBlockSize > 1)
					config.generalCfg.seedBlockSize *= 0.5;
				else if (!config.generalCfg.seedBlockOverride && fraction < 0.5 && config.generalCfg.seedBlockSize > 1)
					config.generalCfg.seedBlockSize *= 0.8;
				else if (!config.generalCfg.seedBlockOverride && fraction > 1.5)
					config.generalCfg.seedBlockSize *= 1.5 + 0.5 * (config.generalCfg.seedBlockSize == 1);
#endif
			}
			if (displayIntervals * config.outputCfg.printInterval * 1000 < milliseconds) {
				lastDiff = checkedSeeds - lastSeed;
				lastSeed = checkedSeeds;
				displayIntervals++;
				float percentComplete =
					((float)(checkedSeeds) / (config.generalCfg.seedEnd - config.generalCfg.seedStart));
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
			if (inputIdx > 0 || currentSeed < config.generalCfg.seedEnd) {
				for (int i = 0; i < WorkerAppetite; i++) {
					if (currentSeed >= config.generalCfg.seedEnd || returns[i].seedFound)
						continue;
					uint32_t nextSeed = currentSeed;
#ifdef REALTIME_SEEDS
					uint8_t* output = hOutput + (index * WorkerAppetite + i) * config.memSizes.outputSize;
					int _ = 0;
					writeInt({output, 4}, _, currentSeed);
					nextSeed = pick_world_seed(startTime + currentSeed);
#endif
					uint32_t length =
						std::min(config.generalCfg.seedBlockSize, config.generalCfg.seedEnd - currentSeed);
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

void InstantiateSector(
	BiomeWangScope** scopes, int& biomeCount, int& maxMapArea, const char* path, BiomeSector partialSector) {
	Vec2i tileDims = GetBufferImageDimensions((uint8_t*)get_wak_file(path).c_str());

	partialSector.tiles_w = tileDims.x;
	partialSector.tiles_h = tileDims.y;
	partialSector.map_w = GetWidthFromPix(partialSector.worldX, partialSector.worldX + partialSector.worldW);
	partialSector.map_h = GetWidthFromPix(partialSector.worldY, partialSector.worldY + partialSector.worldH);

	uint8_t* hTileData = (uint8_t*)malloc(3 * tileDims.x * tileDims.y);
	ReadBufferImage((uint8_t*)get_wak_file(path).c_str(), hTileData, false);

	BiomeWangScope scope;
	scope.ts = stbhw_build_tileset_from_image(hTileData, partialSector.b, 3 * tileDims.x, tileDims.x, tileDims.y);
	partialSector.wang_w = (partialSector.map_w + scope.ts.short_side_len - 1) / scope.ts.short_side_len;
	partialSector.wang_h = (partialSector.map_h + scope.ts.short_side_len + 3) / scope.ts.short_side_len;
	maxMapArea = max(maxMapArea, (int)(partialSector.map_w * partialSector.map_h));

	scope.ts.tileData = (uint8_t*)UploadToDevice(scope.ts.tileData, 3 * tileDims.x * tileDims.y);
	scope.bSec = partialSector;
	free(hTileData);
	BiomeWangScope* dScope = (BiomeWangScope*)UploadToDevice(&scope, sizeof(BiomeWangScope));

	scopes[biomeCount++] = dScope;
}
void InstantiateBiome(int biome, BiomeWangScope** ss, int& bC, int& mA) {
	switch (biome) {
	case B_COALMINE: InstantiateSector(ss, bC, mA, "data/wang_tiles/coalmine.png", {B_COALMINE, 34, 14, 5, 2}); break;
	case B_COALMINE_ALT:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/coalmine_alt.png", {B_COALMINE_ALT, 32, 15, 2, 1});
		break;
	case B_EXCAVATIONSITE:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/excavationsite.png", {B_EXCAVATIONSITE, 31, 17, 8, 2});
		break;
	case B_FUNGICAVE:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/fungicave.png", {B_FUNGICAVE, 28, 17, 3, 2});
		//InstantiateSector(ss, bC, mA, "data/wang_tiles/fungicave.png", { B_FUNGICAVE, 34, 28, 1, 1 });
		//InstantiateSector(ss, bC, mA, "data/wang_tiles/fungicave.png", { B_FUNGICAVE, 39, 31, 1, 1 });
		break;
	case B_SNOWCAVE: InstantiateSector(ss, bC, mA, "data/wang_tiles/snowcave.png", {B_SNOWCAVE, 30, 20, 10, 3}); break;
	case B_SNOWCASTLE:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/snowcastle.png", {B_SNOWCASTLE, 31, 24, 7, 2});
		break;
	case B_RAINFOREST:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/rainforest.png", {B_RAINFOREST, 30, 27, 9, 2});
		break;
	case B_RAINFOREST_OPEN:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/rainforest_open.png", {B_RAINFOREST_OPEN, 30, 28, 9, 2});
		break;
	case B_RAINFOREST_DARK:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/rainforest_dark.png", {B_RAINFOREST_DARK, 25, 26, 5, 8});
		break;
	case B_VAULT: InstantiateSector(ss, bC, mA, "data/wang_tiles/vault.png", {B_VAULT, 29, 31, 11, 3}); break;
	case B_CRYPT: InstantiateSector(ss, bC, mA, "data/wang_tiles/crypt.png", {B_CRYPT, 26, 35, 14, 4}); break;
	case B_WANDCAVE:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wandcave.png", {B_WANDCAVE, 27, 21, 3, 1});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wandcave.png", {B_WANDCAVE, 47, 35, 4, 4});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wandcave.png", {B_WANDCAVE, 41, 36, 6, 1});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wandcave.png", {B_WANDCAVE, 53, 36, 2, 2});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wandcave.png", {B_WANDCAVE, 53, 39, 5, 1});
		break;
	case B_VAULT_FROZEN:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/vault_frozen.png", {B_VAULT_FROZEN, 12, 15, 7, 5});
		break;
	case B_WIZARDCAVE:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wizardcave.png", {B_WIZARDCAVE, 23, 25, 3, 2});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wizardcave.png", {B_WIZARDCAVE, 47, 36, 2, 2});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wizardcave.png", {B_WIZARDCAVE, 51, 36, 8, 3});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wizardcave.png", {B_WIZARDCAVE, 59, 39, 1, 1});
		InstantiateSector(ss, bC, mA, "data/wang_tiles/wizardcave.png", {B_WIZARDCAVE, 53, 40, 6, 6});
		break;
	case B_FUNGIFOREST:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/fungiforest.png", {B_FUNGIFOREST, 59, 16, 7, 9});
		//InstantiateSector(ss, bC, mA, "data/wang_tiles/fungiforest.png", { B_FUNGIFOREST, 58, 35, 4, 6 });
		break;
	case B_ROBOBASE: InstantiateSector(ss, bC, mA, "data/wang_tiles/robobase.png", {B_ROBOBASE, 59, 29, 7, 9}); break;
	case B_LIQUIDCAVE:
		InstantiateSector(ss, bC, mA, "data/wang_tiles/liquidcave.png", {B_LIQUIDCAVE, 26, 14, 5, 2});
		break;
	case B_MEAT:
		//InstantiateSector(ss, bC, mA, "data/wang_tiles/meat.png", { B_MEAT, 47, 25, 3, 7 });
		InstantiateSector(ss, bC, mA, "data/wang_tiles/meat.png", {B_MEAT, 62, 38, 4, 8});
		break;
	}
}

void SearchMain(OutputProgressData& progress, void (*appendOutput)(char*, char*)) {
	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	UploadBiomeData();

	FILE* f = fopen("output.txt", "wb");

	time_t startTime = time(NULL) - 100000;
	Vec2i seedCounts = OutputLoop(f, startTime, progress, appendOutput);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	std::chrono::nanoseconds duration = time2 - time1;

	printf("Search finished in %ims. Checked %i seeds, found %i valid seeds (%.3f%%).\n",
		(int)(duration.count() / 1000000), seedCounts.x, seedCounts.y, (double)seedCounts.y / seedCounts.x * 100);

	fclose(f);
}
