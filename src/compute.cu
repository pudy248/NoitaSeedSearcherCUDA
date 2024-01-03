#include "../platforms/platform_implementation.h"

#include "../include/compute.h"
#include "../include/misc_funcs.h"

#include <cstdio>
#include <fstream>
#include <chrono>
#include <cstring>
#include <vector>

_compute SpanRet PLATFORM_API::EvaluateSpan(SearchConfig config, SpanParams span, void* threadMemBlock, void* outputPtr)
{
	if (span.seedCount == 0) return { 0, 0, false, 0 };

	MemoryArena arena = { (uint8_t*)threadMemBlock, 0 };
	for (int currentSeed = span.seedStart; currentSeed < span.seedStart + span.seedCount; currentSeed++)
	{
		arena.offset = 0;
		bool seedPassed = true;
		uint8_t* output = ArenaAlloc(arena, config.memSizes.outputSize, 4);

		seedPassed &= PrecheckSeed(currentSeed, config.precheckCfg);
		if (!seedPassed) continue;

		if (config.spawnableCfg.staticUpwarps)
		{
			uint8_t* upwarps = ArenaAlloc(arena, config.memSizes.spawnableMemSize, 4);
			uint8_t* miscMem2 = ArenaAlloc(arena, config.memSizes.miscMemSize, 8);
			int offset = 0;
			int _ = 0;
			spawnChest(315, 17, { currentSeed, {}, &config.spawnableCfg, upwarps, offset, _ });
			uint8_t* ptr1 = upwarps + offset;
			spawnChest(75, 117, { currentSeed, {}, &config.spawnableCfg, upwarps, offset, _ });
			Spawnable* spawnables[] = { (Spawnable*)upwarps, (Spawnable*)ptr1 };
			SpawnableBlock b = { currentSeed, 2, spawnables };

			seedPassed &= SpawnablesPassed(b, config.filterCfg, NULL, miscMem2, false);
			ArenaSetOffset(arena, upwarps);
			if (!seedPassed) continue;
		}

		int spawnableCount = 0;
		int spawnableOffset = 8;
		uint8_t* mapMem = ArenaAlloc(arena, config.memSizes.mapDataSize, 8);
		uint8_t* visited = ArenaAlloc(arena, config.memSizes.visitedMemSize);
		uint8_t* spawnableDat = ArenaAlloc(arena, config.memSizes.spawnableMemSize, 4);
		uint8_t* spawnables = ArenaAlloc(arena, config.memSizes.spawnableMemSize, 4);
		uint8_t* miscMem = ArenaAlloc(arena, config.memSizes.miscMemSize, 8);

		*(int*)spawnableDat = currentSeed;
#ifdef DO_WORLDGEN
		for (int biomeNum = 0; biomeNum < config.biomeCount; biomeNum++)
		{
			GenerateMap(currentSeed, config.biomeScopes[biomeNum], output, mapMem, visited, miscMem);
			threadSync();
			CheckSpawnables((WangFuncIndex*)mapMem, config.biomeScopes[biomeNum].tileSet,
				{ currentSeed, &config.biomeScopes[biomeNum], &config.spawnableCfg, spawnableDat, spawnableOffset, spawnableCount }, config.memSizes.spawnableMemSize);
			threadSync();
		}
#endif

		//CheckMountains(currentSeed, &config.spawnableCfg, spawnableDat, spawnableOffset, spawnableCount);
		//CheckEyeRooms(currentSeed, &config.spawnableCfg, spawnableDat, spawnableOffset, spawnableCount);
		//threadSync();

		((int*)spawnableDat)[1] = spawnableCount;
		SpawnableBlock result = ParseSpawnableBlock(spawnableDat, spawnables, config.spawnableCfg, config.memSizes.mapDataSize);
		seedPassed &= SpawnablesPassed(result, config.filterCfg, output, miscMem, true);

		if (!seedPassed)
		{
			continue;
		}
		
		memcpy(output, &currentSeed, 4);
		memcpy((uint8_t*)outputPtr, output, config.memSizes.outputSize);

		int extraSeeds = span.seedStart + span.seedCount - currentSeed - 1;
		return { span.seedStart, span.seedCount, true, extraSeeds };
	}
	return { span.seedStart, span.seedCount, false, 0 };
}

using namespace API_INTERNAL;
Vec2i OutputLoop(FILE* outputFile, time_t startTime, OutputProgressData& progress, void(*appendOutput)(char*, char*))
{
	int displayIntervals = 0;
	int recountIntervals = 0;

	uint32_t lastDiff = 0;
	uint32_t lastSeed = 0;

	uint32_t checkedSeeds = 0;
	uint32_t passedSeeds = 0;

	uint32_t currentSeed = config.generalCfg.seedStart;
	int index = 0;
	int stoppedBlocks = 0;

	int returnedBlocksThisIter = 0;

	std::vector<Worker> workers(NumWorkers);
	SpanParams* params = (SpanParams*)malloc(WorkerAppetite * sizeof(SpanParams));
	uint8_t* hOutput = (uint8_t*)malloc(NumWorkers * WorkerAppetite * config.memSizes.outputSize);

	//initial dispatch
	for (int i = 0; i < NumWorkers; i++)
	{
		workers[i] = CreateWorker();
	}
	for (int i = 0; i < NumWorkers; i++)
	{
		if (currentSeed < config.generalCfg.endSeed)
		{
			for (int j = 0; j < WorkerAppetite; j++)
			{
				if (currentSeed >= config.generalCfg.endSeed)
				{
					params[j] = { 0, 0 };
					continue;
				}
				uint32_t nextSeed = currentSeed;
#ifdef REALTIME_SEEDS
				uint8_t* output = pointers.hPointers.hOutput + (index * BLOCKSIZE + inputIdx) * config.memSizes.outputSize;
				int _ = 0;
				writeInt(output, _, currentSeed);
				nextSeed = GenerateSeed(startTime + currentSeed);
#endif
				uint32_t length = min(config.generalCfg.seedBlockSize, config.generalCfg.endSeed - currentSeed);
				params[j] = { (int)nextSeed, (int)length };
				currentSeed += length;
			}
			DispatchJob(workers[i], params);
		}
		else stoppedBlocks++;
	}

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	//loop
	while (stoppedBlocks < NumWorkers)
	{
		if (progress.abort)
		{
			for (int i = 0; i < NumWorkers; i++)
				AbortJob(workers[i]);
			break;
		}

		if (index == 0)
		{
			std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
			std::chrono::nanoseconds duration = time2 - time1;
			uint64_t milliseconds = (uint64_t)(duration.count() / 1000000);
			progress.elapsedMillis = milliseconds;
			progress.searchedSeeds = checkedSeeds;
			progress.validSeeds = passedSeeds;
			if (recountIntervals * 50 < milliseconds)
			{
				recountIntervals++;
#ifndef REALTIME_SEEDS
				if (!config.generalCfg.seedBlockOverride && returnedBlocksThisIter < 2 && config.generalCfg.seedBlockSize > 1)
					config.generalCfg.seedBlockSize *= 0.7f;
				else if (!config.generalCfg.seedBlockOverride && returnedBlocksThisIter > 5)
					config.generalCfg.seedBlockSize *= 2;
#endif
				returnedBlocksThisIter = 0;
			}
			if (displayIntervals * config.outputCfg.printInterval * 1000 < milliseconds)
			{
				lastDiff = checkedSeeds - lastSeed;
				lastSeed = checkedSeeds;
				displayIntervals++;
				float percentComplete = ((float)(checkedSeeds) / (config.generalCfg.endSeed - config.generalCfg.seedStart));
				progress.progressPercent = percentComplete;
				int seconds = (displayIntervals - 1) * config.outputCfg.printInterval;
				int minutes = seconds / 60;
				int hours = minutes / 60;
				if (config.outputCfg.printProgressLog)
					printf("[%02ih %02im %02is]: %2.3f%% complete. Searched %i (+%i this interval), found %i valid seeds. %i\n",
						hours, minutes % 60, seconds % 60, percentComplete * 100, checkedSeeds, lastDiff, passedSeeds, currentSeed);
			}
		}

		if (QueryWorker(workers[index]))
		{
			SpanRet* returns = SubmitJob(workers[index]);
			returnedBlocksThisIter++;

			int* times = (int*)malloc(4 * WorkerAppetite);
			bool* hasOutput = (bool*)malloc(WorkerAppetite);
			for (int i = 0; i < WorkerAppetite; i++) hasOutput[i] = false;
			int inputIdx = 0;
			memset(params, 0, sizeof(SpanParams) * WorkerAppetite);
			for (int i = 0; i < WorkerAppetite; i++)
			{
				checkedSeeds += returns[i].seedCount - returns[i].leftoverSeeds;
				if (!returns[i].seedFound) continue;
				passedSeeds++;

				uint8_t* uOutput = (uint8_t*)returns[i].outputPtr;
				uint8_t* output = hOutput + (index * WorkerAppetite + i) * config.memSizes.outputSize;
				int _ = 0;
				times[i] = readInt(output, _);
				memcpy(output, uOutput, config.memSizes.outputSize);
				hasOutput[i] = true;

				if (returns[i].leftoverSeeds > 0)
				{
					params[inputIdx++] = { returns[i].seedStart + returns[i].seedCount - returns[i].leftoverSeeds, returns[i].leftoverSeeds };
				};
			}
			if (inputIdx > 0 || currentSeed < config.generalCfg.endSeed)
			{
				for (int i = 0; i < WorkerAppetite; i++)
				{
					if (currentSeed >= config.generalCfg.endSeed || returns[i].seedFound)
						continue;
					uint32_t nextSeed = currentSeed;
#ifdef REALTIME_SEEDS
					uint8_t* output = pointers.hPointers.hOutput + (index * BLOCKSIZE + inputIdx) * config.memSizes.outputSize;
					int _ = 0;
					writeInt(output, _, currentSeed);
					nextSeed = GenerateSeed(startTime + currentSeed);
#endif
					uint32_t length = min(config.generalCfg.seedBlockSize, config.generalCfg.endSeed - currentSeed);
					params[inputIdx++] = { (int)nextSeed, (int)length };
					currentSeed += length;
				}
				DispatchJob(workers[index], params);
			}
			else
				stoppedBlocks++;

			for (int i = 0; i < WorkerAppetite; i++)
			{
				if (!hasOutput[i]) continue;
				uint8_t* output = hOutput + (index * WorkerAppetite + i) * config.memSizes.outputSize;

				PrintOutputBlock(output, outputFile, config.outputCfg, appendOutput);
			}
			free(times);
			free(hasOutput);

		}
		index = (index + 1) % NumWorkers;
	}
	for (int i = 0; i < NumWorkers; i++) DestroyWorker(workers[i]);
	workers.clear();
	free(params);
	free(hOutput);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	std::chrono::nanoseconds duration = time2 - time1;
	uint64_t milliseconds = (uint64_t)(duration.count() / 1000000);
	progress.elapsedMillis = milliseconds;
	progress.searchedSeeds = checkedSeeds;
	progress.validSeeds = passedSeeds;

	return { (int)checkedSeeds, (int)passedSeeds };
}

void InstantiateSector(BiomeWangScope* scopes, int& biomeCount, int& maxMapArea, const char* path, BiomeSector partialSector)
{
	Vec2i tileDims = GetImageDimensions(path);

	partialSector.tiles_w = tileDims.x;
	partialSector.tiles_h = tileDims.y;
	partialSector.map_w = GetWidthFromPix(partialSector.worldX, partialSector.worldX + partialSector.worldW);
	partialSector.map_h = GetWidthFromPix(partialSector.worldY, partialSector.worldY + partialSector.worldH);

	uint8_t* hTileData = (uint8_t*)malloc(3 * tileDims.x * tileDims.y);
	ReadImage(path, hTileData);
	//blockOutRooms(hTileData, tileDims.x, tileDims.y, COLOR_WHITE);
	WangTileset* tileSet = (WangTileset*)malloc(sizeof(WangTileset));
	BiomeSpawnFunctions* fns[2] = { GetSpawnFunc(B_NONE), GetSpawnFunc(partialSector.b)};
	stbhw_build_tileset_from_image(hTileData, tileSet, fns, 3 * tileDims.x, tileDims.x, tileDims.y);
	free(hTileData);
	free(fns[0]);
	free(fns[1]);
	WangTileset* dTileSet = (WangTileset*)UploadToDevice(tileSet, sizeof(WangTileset));

	partialSector.wang_w = (partialSector.map_w + tileSet->short_side_len - 1) / tileSet->short_side_len;
	partialSector.wang_h = (partialSector.map_h + tileSet->short_side_len + 3) / tileSet->short_side_len;
	free(tileSet);

	maxMapArea = max(maxMapArea, (int)(partialSector.wang_w * partialSector.wang_h));

	scopes[biomeCount++] = { dTileSet, partialSector };
}
void InstantiateBiome(const char* path, BiomeWangScope* ss, int& bC, int& mA)
{
	{
		if (strcmp(path, "resources/wang_tiles/coalmine.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_COALMINE, 34, 14, 5, 2 });
		}
		else if (strcmp(path, "resources/wang_tiles/coalmine_alt.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_COALMINE_ALT, 32, 15, 2, 1 });
		}
		else if (strcmp(path, "resources/wang_tiles/excavationsite.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_EXCAVATIONSITE, 31, 17, 8, 2 });
		}
		else if (strcmp(path, "resources/wang_tiles/fungicave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_FUNGICAVE, 28, 17, 3, 1 });
		}
		else if (strcmp(path, "resources/wang_tiles/snowcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_SNOWCAVE, 30, 20, 10, 3 });
		}
		else if (strcmp(path, "resources/wang_tiles/snowcastle.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_SNOWCASTLE, 31, 24, 7, 2 });
		}
		else if (strcmp(path, "resources/wang_tiles/rainforest.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_RAINFOREST, 30, 27, 9, 2 });
		}
		else if (strcmp(path, "resources/wang_tiles/rainforest_open.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_RAINFOREST_OPEN, 30, 28, 9, 2 });
		}
		else if (strcmp(path, "resources/wang_tiles/rainforest_dark.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_RAINFOREST_DARK, 25, 26, 5, 8 });
		}
		else if (strcmp(path, "resources/wang_tiles/vault.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_VAULT, 29, 31, 11, 3 });
		}
		else if (strcmp(path, "resources/wang_tiles/crypt.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_CRYPT, 26, 35, 14, 4 });
		}
		else if (strcmp(path, "resources/wang_tiles/wandcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_WANDCAVE, 47, 35, 4, 4 });
		}
		else if (strcmp(path, "resources/wang_tiles/vault_frozen.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_VAULT_FROZEN, 12, 15, 7, 5 });
		}
		else if (strcmp(path, "resources/wang_tiles/wizardcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_WIZARDCAVE, 53, 40, 6, 6 });
		}
		else if (strcmp(path, "resources/wang_tiles/fungiforest.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_FUNGIFOREST, 59, 16, 7, 9 });
		}
		else if (strcmp(path, "resources/wang_tiles/robobase.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_ROBOBASE, 59, 29, 7, 9 });
		}
		else if (strcmp(path, "resources/wang_tiles/liquidcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_LIQUIDCAVE, 26, 14, 5, 2 });
		}
		else if (strcmp(path, "resources/wang_tiles/meat.png") == 0)
		{
			InstantiateSector(ss, bC, mA, path, { B_MEAT, 62, 38, 4, 8 });
		}

		else
		{
			printf("Invalid biome path: %s\n", path);
		}
	}
}

void SearchMain(OutputProgressData& progress, void(*appendOutput)(char*, char*))
{
	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	InitializePlatform();
	AllocateComputeMemory();
	FILE* f = fopen("output.txt", "wb");

	time_t startTime = _time64(NULL);
	Vec2i seedCounts = OutputLoop(f, startTime, progress, appendOutput);

	std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
	std::chrono::nanoseconds duration = time2 - time1;

	printf("Search finished in %ims. Checked %i seeds, found %i valid seeds.\n", (int)(duration.count() / 1000000), seedCounts.x, seedCounts.y);

	FreeComputeMemory();
	DestroyPlatform();
	fclose(f);
}
