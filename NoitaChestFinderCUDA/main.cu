#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/pngutils.h"
#include "misc/utilities.h"

#include "misc/error.h"
#include "misc/databaseutils.h"
#include "worldSeedGeneration.h"

#include "Configuration.h"
#include "Precheckers.h"
#include "Worldgen.h"
#include "WorldgenSearch.h"
#include "Filters.h"

#include "biomes/allBiomes.h"

#include "Windows.h"
#include "gui/guiMain.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <stdlib.h>
#include <cstring>

int global_iters = 1;

//tired of seeing an error for these being undefined
__device__ int atomicAdd(int* address, int val);

struct DeviceConfig
{
	int NumBlocks;
	MemSizeConfig memSizes;
	GeneralConfig generalCfg;
	StaticPrecheckConfig precheckCfg;
	SpawnableConfig spawnableCfg;
	FilterConfig filterCfg;

	int biomeCount;
	BiomeWangScope biomeScopes[20];
};

struct DevicePointers
{
	volatile int* __restrict__ numActiveThreads;
	BlockIO* __restrict__ uThreads;
	uint8_t* __restrict__ uOutput;
	uint8_t* __restrict__ dArena;
};

struct HostPointers
{
	volatile int* numActiveThreads;
	BlockIO* uThreads;
	uint8_t* uOutput;
	uint8_t* hOutput;
};

struct AllPointers
{
	DevicePointers dPointers;
	HostPointers hPointers;
};

__device__ ThreadRet DispatchThread(DeviceConfig dConfig, DevicePointers dPointers, int memIdx, ThreadInput input)
{
	//printf("idx: %i [%i - %i]\n", memIdx, input.startSeed, input.startSeed + input.seedCount - 1);
	if (input.seedCount == 0) return { input, false, 0 };

	uint8_t* uOutput = dPointers.uOutput + memIdx * dConfig.memSizes.outputSize;

	uint8_t* threadMemPtr = dPointers.dArena + memIdx * dConfig.memSizes.threadMemTotal;
	MemoryArena arena = { threadMemPtr, 0 };

	for (int currentSeed = input.startSeed; currentSeed < input.startSeed + input.seedCount; currentSeed++)
	{
		arena.offset = 0;
		bool seedPassed = true;

		uint8_t* output = ArenaAlloc(arena, dConfig.memSizes.outputSize, 4);
		SQLRow outputRow;
		memset(&outputRow, 0, sizeof(SQLRow));
		outputRow.SEED = currentSeed;

		seedPassed &= PrecheckSeed(outputRow, currentSeed, dConfig.precheckCfg);

		if (!seedPassed) continue;

		if (dConfig.spawnableCfg.staticUpwarps)
		{
			uint8_t* upwarps = ArenaAlloc(arena, dConfig.memSizes.spawnableMemSize, 4);
			int offset = 0;
			int _ = 0;
			spawnChest(315, 17, { currentSeed, {}, dConfig.spawnableCfg, upwarps, offset, _ });
			byte* ptr1 = upwarps + offset;
			spawnChest(75, 117, { currentSeed, {}, dConfig.spawnableCfg, upwarps, offset, _ });
			Spawnable* spawnables[] = { (Spawnable*)upwarps, (Spawnable*)ptr1 };
			SpawnableBlock b = { currentSeed, 2, spawnables };

			seedPassed &= SpawnablesPassed(b, dConfig.filterCfg, NULL, false);
			ArenaSetOffset(arena, upwarps);

			if (!seedPassed) continue;
		}
		__syncthreads();

		int spawnableCount = 0;
		int spawnableOffset = 8;
#ifdef DO_WORLDGEN
		uint8_t* mapMem = ArenaAlloc(arena, dConfig.memSizes.mapDataSize, 8); //Pointers are 8 bytes
		uint8_t* miscMem = ArenaAlloc(arena, dConfig.memSizes.miscMemSize, 4);
		uint8_t* visited = ArenaAlloc(arena, dConfig.memSizes.visitedMemSize);
		uint8_t* spawnables = ArenaAlloc(arena, dConfig.memSizes.spawnableMemSize, 4);
		*(int*)spawnables = currentSeed;
		for (int biomeNum = 0; biomeNum < dConfig.biomeCount; biomeNum++)
		{
			GenerateMap(currentSeed, dConfig.biomeScopes[biomeNum], output, mapMem, visited, miscMem);
			__syncthreads();
			SetFunctionPointerSetterFunctionPointerArrayPointers();
			CheckSpawnables(mapMem, { currentSeed, dConfig.biomeScopes[biomeNum].bSec, dConfig.spawnableCfg, spawnables, spawnableOffset, spawnableCount }, dConfig.memSizes.spawnableMemSize);
			__syncthreads();
		}

		CheckMountains(currentSeed, dConfig.spawnableCfg, spawnables, spawnableOffset, spawnableCount);
		CheckEyeRooms(currentSeed, dConfig.spawnableCfg, spawnables, spawnableOffset, spawnableCount);
		__syncthreads();

		((int*)spawnables)[1] = spawnableCount;
		SpawnableBlock result = ParseSpawnableBlock(spawnables, mapMem, dConfig.spawnableCfg, dConfig.memSizes.mapDataSize);
		seedPassed &= SpawnablesPassed(result, dConfig.filterCfg, output, true);

		if (!seedPassed)
		{
			continue;
		}
#endif

#ifdef SQL_OUTPUT
		memcpy(output, &currentSeed, 4);
		memcpy(output + 4, &outputRow, sizeof(SQLRow));
#else
		memcpy(output, &currentSeed, 4);
		//memcpy(output + 4, &spawnableCount, 4);
#endif
		memcpy(uOutput, output, dConfig.memSizes.outputSize);

		uint32_t extraSeeds = input.startSeed + input.seedCount - currentSeed - 1;
		return { input, true, extraSeeds };
	}
	return { input, false, 0 };
}

__global__ void DispatchBlock(DeviceConfig dConfig, DevicePointers dPointers, int memIdx)
{
	atomicAdd((int*)dPointers.numActiveThreads, 1);
	uint32_t hwIdx = blockIdx.x * blockDim.x + threadIdx.x;
	BlockIO* ioPtr = dPointers.uThreads + memIdx;
	ThreadRet ret = DispatchThread(dConfig, dPointers, memIdx * BLOCKSIZE + hwIdx, ioPtr->inputs.inputs[hwIdx]);
	memcpy(&(ioPtr->ret.threads[hwIdx]), &ret, sizeof(ThreadRet));
	atomicAdd(&ioPtr->ret.returned, 1);
	atomicAdd((int*)dPointers.numActiveThreads, -1);
}

void hDispatchBlock(DeviceConfig dConfig, DevicePointers dPointers, int memIdx, cudaStream_t stream)
{
	//printf("Dispatching block %i\n", memIdx);
	DispatchBlock << <BLOCKDIV, BLOCKSIZE / BLOCKDIV, 0, stream >> > (dConfig, dPointers, memIdx);
}

void InstantiateSector(BiomeWangScope* scopes, int& biomeCount, int& maxMapArea, uint8_t* tileData, Vec2i tileDims, BiomeSector partialSector)
{
	partialSector.tiles_w = tileDims.x;
	partialSector.tiles_h = tileDims.y;
	partialSector.map_w = GetWidthFromPix(partialSector.worldX, partialSector.worldX + partialSector.worldW);
	partialSector.map_h = GetWidthFromPix(partialSector.worldY, partialSector.worldY + partialSector.worldH);

	maxMapArea = max(maxMapArea, (int)(partialSector.map_w * (partialSector.map_h + 4)));

	scopes[biomeCount++] = { tileData, partialSector };
}

void InstantiateBiome(const char* path, BiomeWangScope* ss, int& bC, int& mA)
{
	Vec2i tileDims = GetImageDimensions(path);
	uint64_t tileDataSize = 3 * tileDims.x * tileDims.y;

	uint8_t* dTileData;
	uint8_t* dTileSet;

	uint8_t* hTileData = (uint8_t*)malloc(tileDataSize);
	ReadImage(path, hTileData);
	blockOutRooms(hTileData, tileDims.x, tileDims.y, COLOR_WHITE);

	checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
	checkCudaErrors(cudaMalloc(&dTileSet, tileDataSize));

	checkCudaErrors(cudaMemcpy(dTileData, hTileData, tileDataSize, cudaMemcpyHostToDevice));
	buildTS << <1, 1 >> > (dTileData, dTileSet, tileDims.x, tileDims.y);
	checkCudaErrors(cudaDeviceSynchronize());

	free(hTileData);
	checkCudaErrors(cudaFree(dTileData));

	{
		if (strcmp(path, "wang_tiles/coalmine.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_COALMINE, 34, 14, 5, 2 });
		}
		else if (strcmp(path, "wang_tiles/coalmine_alt.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_COALMINE_ALT, 32, 15, 2, 1 });
		}
		else if (strcmp(path, "wang_tiles/excavationsite.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_EXCAVATIONSITE, 31, 17, 8, 2 });
		}
		else if (strcmp(path, "wang_tiles/fungicave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_FUNGICAVE, 28, 17, 3, 1 });
		}
		else if (strcmp(path, "wang_tiles/snowcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_SNOWCAVE, 30, 20, 10, 3 });
		}
		else if (strcmp(path, "wang_tiles/snowcastle.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_SNOWCASTLE, 31, 24, 7, 2 });
		}
		else if (strcmp(path, "wang_tiles/rainforest.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_RAINFOREST, 30, 27, 9, 2 });
		}
		else if (strcmp(path, "wang_tiles/rainforest_open.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_RAINFOREST_OPEN, 30, 28, 9, 2 });
		}
		else if (strcmp(path, "wang_tiles/rainforest_dark.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_RAINFOREST_DARK, 25, 26, 5, 8 });
		}
		else if (strcmp(path, "wang_tiles/vault.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_VAULT, 29, 31, 11, 3 });
		}
		else if (strcmp(path, "wang_tiles/crypt.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_CRYPT, 26, 35, 14, 4 });
		}
		else if (strcmp(path, "wang_tiles/wandcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_WANDCAVE, 47, 35, 4, 4 });
		}
		else if (strcmp(path, "wang_tiles/vault_frozen.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_VAULT_FROZEN, 12, 15, 7, 5 });
		}
		else if (strcmp(path, "wang_tiles/wizardcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_WIZARDCAVE, 53, 40, 6, 6 });
		}
		else if (strcmp(path, "wang_tiles/fungiforest.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_FUNGIFOREST, 59, 16, 7, 9 });
		}
		else if (strcmp(path, "wang_tiles/robobase.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_ROBOBASE, 59, 29, 7, 9 });
		}
		else if (strcmp(path, "wang_tiles/liquidcave.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_LIQUIDCAVE, 26, 14, 5, 2 });
		}
		else if (strcmp(path, "wang_tiles/meat.png") == 0)
		{
			InstantiateSector(ss, bC, mA, dTileSet, tileDims, { B_MEAT, 62, 38, 4, 8 });
		}

		else
		{
			printf("Invalid biome path: %s\n", path);
		}
	}
}

DeviceConfig CreateConfigs(int maxMapArea, int biomeCount)
{
	int DEVICE;
	cudaGetDevice(&DEVICE);
	int NUMBLOCKS = 16;

	MemSizeConfig memSizes = {
#ifdef DO_WORLDGEN
		3 * maxMapArea + 4096,
#else
		512,
#endif
		3 * maxMapArea + 128,
		4 * maxMapArea,
		maxMapArea,
		4096,
	};

	GeneralConfig generalCfg = { 40_GB, 1, INT_MAX, 1, false, 5 };
#ifdef REALTIME_SEEDS
	generalCfg.seedBlockSize = 1;
#endif
	SpawnableConfig spawnableCfg = { {0, 0}, {0, 0}, 0, 7,
		false, //greed
		false, //pacifist
		false, //shop spells
		false, //shop wands
		false, //eye rooms
		false, //upwarp check
		false, //biome chests
		false, //biome pedestals
		false, //biome altars
		true, //biome pixelscenes
		false, //enemies
		false, //hell shops
		false, //potion contents
		false, //chest spells
		false, //wand contents
	};

	ItemFilter iFilters[] = { ItemFilter({REFRESH_MIMIC}, 1), ItemFilter({MIMIC_SIGN}) };
	MaterialFilter mFilters[] = { MaterialFilter({CREEPY_LIQUID}) };
	SpellFilter sFilters[] = {
		SpellFilter({SPELL_LUMINOUS_DRILL, SPELL_LASER_LUMINOUS_DRILL, SPELL_BLACK_HOLE, SPELL_BLACK_HOLE_DEATH_TRIGGER}, 1),
		SpellFilter({SPELL_LIGHT_BULLET, SPELL_LIGHT_BULLET_TRIGGER, SPELL_SPITTER}),
		SpellFilter({SPELL_CURSE_WITHER_PROJECTILE}) };
	PixelSceneFilter psFilters[] = { PixelSceneFilter({PS_NONE}, {MAGIC_GAS_HP_REGENERATION}) };

	FilterConfig filterCfg = FilterConfig(false, 0, iFilters, 0, mFilters, 0, sFilters, 1, psFilters, false, 10);

	StaticPrecheckConfig precheckCfg = {
		{false, SKATEBOARD},
		{false, GOLD},
		{false, SPELL_LIGHT_BULLET, SPELL_DYNAMITE},
		{false, WATER},
		{false, AlchemyOrdering::ONLY_CONSUMED, {MATERIAL_NONE, MATERIAL_NONE, MATERIAL_NONE}, {MUD, WATER, SOIL}},
		{false, {BM_GOLD_VEIN_SUPER, BM_NONE}},
		{false, {FungalShift(SS_GOLD, SD_FLASK, 0, 1)}},
		{false, //Example: Searches for perk lottery + extra perk in first HM, then any 4 perks in 2nd HM as long as they all are lottery picks
			{{PERK_PERKS_LOTTERY, true, 0, 3}, {PERK_EXTRA_PERK, true, 0, 3}, {PERK_NONE, true, 3, 4}, {PERK_NONE, true, 4, 5}, {PERK_NONE, true, 5, 6}, {PERK_NONE, true, 6, 7}},
			{3, 4, 4, 4, 4, 4, 4} //Also, XX_NONE is considered to be equal to everything for like 90% of calculations
		},
	};

	memSizes.spawnableMemSize *= spawnableCfg.pwWidth.x * 2 + 1;
	memSizes.spawnableMemSize *= spawnableCfg.pwWidth.y * 2 + 1;
	memSizes.spawnableMemSize *= biomeCount;

	uint64_t freeMem;
	uint64_t totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	printf("memory free: %lli of %lli bytes\n", freeMem, totalMem);
	freeMem *= 0.9f; //leave a bit of extra
	freeMem = min(freeMem, generalCfg.requestedMemory);

#ifdef DO_WORLDGEN
	uint64_t minMemoryPerThread = memSizes.outputSize * 2 + memSizes.mapDataSize + memSizes.miscMemSize + memSizes.visitedMemSize + memSizes.spawnableMemSize;
#else
	uint64_t minMemoryPerThread = memSizes.outputSize * 2 + memSizes.spawnableMemSize;
#endif
	printf("each thread requires %lli bytes of block memory\n", minMemoryPerThread);
	memSizes.threadMemTotal = minMemoryPerThread;

	int numThreads = min((uint64_t)(generalCfg.endSeed - generalCfg.startSeed), freeMem / minMemoryPerThread);
	int numBlocks = numThreads / BLOCKSIZE;
	int numBlocksRounded = max(min(NUMBLOCKS, numBlocks - numBlocks % 1), 1);
	generalCfg.requestedMemory = minMemoryPerThread * numBlocksRounded * BLOCKSIZE;
	generalCfg.seedBlockSize = min((uint32_t)generalCfg.seedBlockSize, (generalCfg.endSeed - generalCfg.startSeed) / (numBlocksRounded * BLOCKSIZE) + 1);
	printf("creating %ix%i threads\n", numBlocksRounded, BLOCKSIZE);

	return { numBlocksRounded, memSizes, generalCfg, precheckCfg, spawnableCfg, filterCfg, biomeCount };
}

AllPointers AllocateMemory(DeviceConfig config)
{
	BiomeData hBiomeData[30];
	SetBiomeData(hBiomeData);
	BiomeData* dBiomeData;
	checkCudaErrors(cudaMalloc(&dBiomeData, sizeof(hBiomeData)));
	checkCudaErrors(cudaMemcpy(dBiomeData, hBiomeData, sizeof(hBiomeData), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(AllBiomeData, &dBiomeData, sizeof(void*)));

	cudaSetDeviceFlags(cudaDeviceMapHost);

	uint64_t outputSize = config.NumBlocks * BLOCKSIZE * config.memSizes.outputSize;

	//Host
	uint8_t* hOutput;
	checkCudaErrors(cudaHostAlloc(&hOutput, outputSize, cudaHostAllocDefault));

	//Device
	uint8_t* dArena;
	uint8_t* dOverlayMem;

	checkCudaErrors(cudaMalloc(&dArena, config.generalCfg.requestedMemory));
	checkCudaErrors(cudaMalloc(&dOverlayMem, 3 * 256 * 103));

	uint8_t* hPtr = (uint8_t*)malloc(3 * 256 * 103);
	ReadImage("wang_tiles/coalmine_overlay.png", hPtr);
	checkCudaErrors(cudaMemcpy(dOverlayMem, hPtr, 3 * 256 * 103, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(coalmine_overlay, &dOverlayMem, sizeof(void*), 0));
	free(hPtr);

	//Unified
	volatile int* hActiveThreads, * dActiveThreads;
	BlockIO* hThreads, * dThreads;
	uint8_t* hUnifiedOutput, * dUnifiedOutput;
	checkCudaErrors(cudaHostAlloc((void**)&hActiveThreads, 4, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dActiveThreads, (void*)hActiveThreads, 0));
	*hActiveThreads = 0;

	checkCudaErrors(cudaHostAlloc((void**)&hThreads, sizeof(BlockIO) * config.NumBlocks, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dThreads, (void*)hThreads, 0));

	for (int i = 0; i < config.NumBlocks; i++)
	{
		memset((void*)&hThreads[i], 0, sizeof(BlockIO));
	}

	checkCudaErrors(cudaHostAlloc((void**)&hUnifiedOutput, outputSize, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dUnifiedOutput, (void*)hUnifiedOutput, 0));

	printf("Allocated %lliMB of host and %lliMB of device memory\n", (2 * outputSize + sizeof(BlockIO) * config.NumBlocks) / 1_MB, config.generalCfg.requestedMemory / 1_MB);


	return { {dActiveThreads, dThreads, dUnifiedOutput, dArena}, {hActiveThreads, hThreads, hUnifiedOutput, hOutput} };
}

Vec2i OutputLoop(DeviceConfig config, AllPointers pointers, FILE* outputFile, time_t startTime, sqlite3* db)
{
	int displayIntervals = 0;
	int recountIntervals = 0;
	int counter = 0;

	int miscCounter = 0;

	uint32_t lastDiff = 0;
	uint32_t lastSeed = 0;

	uint32_t checkedSeeds = 0;
	uint32_t passedSeeds = 0;

	uint32_t currentSeed = config.generalCfg.startSeed;
	int index = 0;
	int stoppedBlocks = 0;

	int returnedBlocksThisIter = 0;

	cudaStream_t* kernelStreams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * config.NumBlocks);

#ifdef SQL_OUTPUT
	constexpr int rowBufferSize = 1024 * 1024;
	SQLRow* rowBuffer = (SQLRow*)malloc(sizeof(SQLRow) * rowBufferSize);
	int rowCounter;
#endif

	//initial dispatch
	for (int block = 0; block < config.NumBlocks; block++)
	{
		checkCudaErrors(cudaStreamCreateWithFlags(kernelStreams + block, cudaStreamNonBlocking));

		if (currentSeed < config.generalCfg.endSeed)
		{
			BlockIO* blockIO = pointers.hPointers.uThreads + block;
			for (int inputIdx = 0; inputIdx < BLOCKSIZE; inputIdx++)
			{
				if (currentSeed >= config.generalCfg.endSeed)
				{
					blockIO->inputs.inputs[inputIdx] = { 0, 0 };
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
				blockIO->inputs.inputs[inputIdx] = { nextSeed, length };
				currentSeed += length;
			}
			hDispatchBlock(config, pointers.dPointers, block, kernelStreams[block]);
		}
		else
			stoppedBlocks++;
	}

	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

	//loop
	while (stoppedBlocks < config.NumBlocks)
	{
		if (index == 0)
		{
			counter++;

			if (counter == 4)
			{
				counter = 0;
				std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
				std::chrono::nanoseconds duration = time2 - time1;
				uint64_t milliseconds = (uint64_t)(duration.count() / 1000000);
				if (recountIntervals * 20 < milliseconds)
				{
					recountIntervals++;
#ifndef REALTIME_SEEDS
					if (!config.generalCfg.seedBlockOverride && returnedBlocksThisIter < 2 && config.generalCfg.seedBlockSize > 1)
						config.generalCfg.seedBlockSize *= 0.5f;
					else if (!config.generalCfg.seedBlockOverride && returnedBlocksThisIter > 10)
						config.generalCfg.seedBlockSize *= 2;
#endif
					returnedBlocksThisIter = 0;
				}
				if (displayIntervals * config.generalCfg.printInterval * 1000 < milliseconds)
				{
					lastDiff = checkedSeeds - lastSeed;
					lastSeed = checkedSeeds;
					displayIntervals++;
					float percentComplete = ((float)(checkedSeeds) / (config.generalCfg.endSeed - config.generalCfg.startSeed));
					printf(">%i: %2.3f%% complete. Searched %i (+%i this interval), found %i valid seeds. (%i/%i active threads, size %i) (counter %i)\n",
						displayIntervals, percentComplete * 100, checkedSeeds, lastDiff, passedSeeds, *pointers.hPointers.numActiveThreads, BLOCKSIZE * config.NumBlocks, config.generalCfg.seedBlockSize, miscCounter);
				}
			}
		}

		BlockIO* blockIO = pointers.hPointers.uThreads + index;
		//printf("%i: %i\n", index, blockIO->ret.returned);
		if (blockIO->ret.returned == BLOCKSIZE)
		{
			returnedBlocksThisIter++;
			cudaStreamSynchronize(kernelStreams[index]);
			blockIO->ret.returned = 0;

			int times[BLOCKSIZE];
			bool hasOutput[BLOCKSIZE];
			memset(hasOutput, 0, BLOCKSIZE);
			int inputIdx = 0;
			for (int i = 0; i < BLOCKSIZE; i++)
			{
				//printf("[%i - %i] -- %i %i\n", blockIO->ret.threads[i].input.startSeed, blockIO->ret.threads[i].input.startSeed + blockIO->ret.threads[i].input.seedCount, blockIO->ret.threads[i].input.seedCount, blockIO->ret.threads[i].leftoverSeeds);
				checkedSeeds += blockIO->ret.threads[i].input.seedCount - blockIO->ret.threads[i].leftoverSeeds;
				if (!blockIO->ret.threads[i].seedFound) continue;
				passedSeeds++;

				uint8_t* uOutput = pointers.hPointers.uOutput + (index * BLOCKSIZE + i) * config.memSizes.outputSize;
				uint8_t* output = pointers.hPointers.hOutput + (index * BLOCKSIZE + i) * config.memSizes.outputSize;
				int _ = 0;
				times[i] = readInt(output, _);
				memcpy(output, uOutput, config.memSizes.outputSize);
				hasOutput[i] = true;

				if (blockIO->ret.threads[i].leftoverSeeds > 0)
				{
					blockIO->inputs.inputs[inputIdx++] = { blockIO->ret.threads[i].input.startSeed + blockIO->ret.threads[i].input.seedCount - blockIO->ret.threads[i].leftoverSeeds, blockIO->ret.threads[i].leftoverSeeds };
				};
			}
			if (inputIdx > 0 || currentSeed < config.generalCfg.endSeed)
			{
				for (; inputIdx < BLOCKSIZE; inputIdx++)
				{
					if (currentSeed >= config.generalCfg.endSeed)
					{
						blockIO->inputs.inputs[inputIdx] = { 0, 0 };
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
					blockIO->inputs.inputs[inputIdx] = { nextSeed, length };
					currentSeed += length;
				}
				hDispatchBlock(config, pointers.dPointers, index, kernelStreams[index]);
			}
			else
				stoppedBlocks++;

			for (int i = 0; i < BLOCKSIZE; i++)
			{
				if (!hasOutput[i]) continue;
				uint8_t* output = pointers.hPointers.hOutput + (index * BLOCKSIZE + i) * config.memSizes.outputSize;

				//write output
				{
#ifdef IMAGE_OUTPUT
					int memOffset = 0;
					int seed = readInt(output, memOffset);
					int w = readInt(output, memOffset);
					int h = readInt(output, memOffset);
					char buffer[30];
					int bufOffset = 0;
					_putstr_offset("outputs/", buffer, bufOffset);
					_itoa_offset(seed, 10, buffer, bufOffset);
					_putstr_offset(".png", buffer, bufOffset);
					buffer[bufOffset++] = '\0';
					WriteImage(buffer, output + memOffset, w, h);
#else
#ifdef SPAWNABLE_OUTPUT
					char buffer[4096];
					int bufOffset = 0;
					int memOffset = 0;
					int seed = readInt(output, memOffset);
					int sCount = readInt(output, memOffset);
					if (seed == 0) continue;

#ifdef REALTIME_SEEDS
					_itoa_offset(time, 10, buffer, bufOffset);
					_putstr_offset(" secs, seed ", buffer, bufOffset);
#endif
					_itoa_offset(seed, 10, buffer, bufOffset);
					if (sCount > 0)
					{
						_putstr_offset(": ", buffer, bufOffset);
						Spawnable* sPtr;
						for (int i = 0; i < sCount; i++)
						{
							sPtr = (Spawnable*)(output + memOffset);
							Spawnable s = readMisalignedSpawnable(sPtr);
							Vec2i chunkCoords = GetLocalPos(s.x, s.y);

							_putstr_offset(" ", buffer, bufOffset);
							_putstr_offset(SpawnableTypeNames[s.sType - TYPE_CHEST], buffer, bufOffset);
							_putstr_offset("(", buffer, bufOffset);
							_itoa_offset(s.x, 10, buffer, bufOffset);
							if (abs(chunkCoords.x - 35) > 35)
							{
								_putstr_offset("[", buffer, bufOffset);
								_putstr_offset(s.x > 0 ? "E" : "W", buffer, bufOffset);
								int pwPos = abs((int)rintf((chunkCoords.x - 35) / 70.0f));
								_itoa_offset(pwPos, 10, buffer, bufOffset);
								_putstr_offset("]", buffer, bufOffset);
							}
							_putstr_offset(", ", buffer, bufOffset);
							_itoa_offset(s.y, 10, buffer, bufOffset);
							if (abs(chunkCoords.y - 24) > 24)
							{
								_putstr_offset("[", buffer, bufOffset);
								_putstr_offset(s.y > 0 ? "H" : "S", buffer, bufOffset);
								int pwPos = abs((int)rintf((chunkCoords.y - 24) / 48.0f));
								_itoa_offset(pwPos, 10, buffer, bufOffset);
								_putstr_offset("]", buffer, bufOffset);
							}
							_putstr_offset("){", buffer, bufOffset);

							for (int n = 0; n < s.count; n++)
							{
								Item item = *(&sPtr->contents + n);
								if (item == DATA_MATERIAL)
								{
									int offset2 = n + 1;
									short m = readShort((uint8_t*)(&sPtr->contents), offset2);
									_putstr_offset("POTION_", buffer, bufOffset);
									_putstr_offset(MaterialNames[m], buffer, bufOffset);
									n += 2;
								}
								else if (item == DATA_SPELL)
								{
									int offset2 = n + 1;
									short m = readShort((uint8_t*)(&sPtr->contents), offset2);
									_putstr_offset("SPELL_", buffer, bufOffset);
									_putstr_offset(SpellNames[m], buffer, bufOffset);
									n += 2;
								}
								else if (item == DATA_PIXEL_SCENE)
								{
									int offset2 = n + 1;
									short ps = readShort((uint8_t*)(&sPtr->contents), offset2);
									short m = readShort((uint8_t*)(&sPtr->contents), offset2);
									_putstr_offset(PixelSceneNames[ps], buffer, bufOffset);
									if (m != MATERIAL_NONE)
									{
										_putstr_offset("[", buffer, bufOffset);
										_putstr_offset(MaterialNames[m], buffer, bufOffset);
										_putstr_offset("]", buffer, bufOffset);
									}
									n += 4;
								}
								else if (item == DATA_WAND)
								{
									n++;
									WandData dat = readMisalignedWand((WandData*)(&sPtr->contents + n));
									_putstr_offset("[", buffer, bufOffset);

									_itoa_offset_decimal((int)(dat.capacity * 100), 10, 2, buffer, bufOffset);
									_putstr_offset(" CAPACITY, ", buffer, bufOffset);

									_itoa_offset(dat.multicast, 10, buffer, bufOffset);
									_putstr_offset(" MULTI, ", buffer, bufOffset);

									_itoa_offset(dat.delay, 10, buffer, bufOffset);
									_putstr_offset(" DELAY, ", buffer, bufOffset);

									_itoa_offset(dat.reload, 10, buffer, bufOffset);
									_putstr_offset(" RELOAD, ", buffer, bufOffset);

									_itoa_offset(dat.mana, 10, buffer, bufOffset);
									_putstr_offset(" MANA, ", buffer, bufOffset);

									_itoa_offset(dat.regen, 10, buffer, bufOffset);
									_putstr_offset(" REGEN, ", buffer, bufOffset);

									//speed... float?

									_itoa_offset(dat.spread, 10, buffer, bufOffset);
									_putstr_offset(" SPREAD, ", buffer, bufOffset);

									_putstr_offset(dat.shuffle ? "SHUFFLE] AC_" : "NON-SHUFFLE] AC_", buffer, bufOffset);
									n += 33;
									continue;
								}
								else if (GOLD_NUGGETS > item || item > TRUE_ORB)
								{
									_putstr_offset("0x", buffer, bufOffset);
									_itoa_offset_zeroes(item, 16, 2, buffer, bufOffset);
								}
								else
								{
									int idx = item - GOLD_NUGGETS;
									_putstr_offset(ItemNames[idx], buffer, bufOffset);
								}

								if (n < s.count - 1)
									_putstr_offset(" ", buffer, bufOffset);
							}
							_putstr_offset("}", buffer, bufOffset);
							memOffset += s.count + 13;
						}
					}
					buffer[bufOffset++] = '\n';
					buffer[bufOffset++] = '\0';
					fprintf(outputFile, "%s", buffer);
					printf("%s", buffer);
#else
#ifdef SQL_OUTPUT
					SQLRow r = *(SQLRow*)(output + 4);
					rowBuffer[rowCounter++] = r;
					if (rowCounter == rowBufferSize)
					{
						rowCounter = 0;
						InsertRowBlock(db, rowBuffer, rowBufferSize);
					}
#else
#ifdef REALTIME_SEEDS
					printf("in %i seconds [UNIX %i]: seed %i\n", times[i], (int)(startTime + times[i]), GenerateSeed(startTime + times[i]));
#else
					char buffer[12];
					int bufOffset = 0;
					int seed = *(int*)output;
					_itoa_offset(seed, 10, buffer, bufOffset);
					buffer[bufOffset++] = '\n';
					buffer[bufOffset++] = '\0';
					fprintf(outputFile, "%s", buffer);
					printf("%s", buffer);
#endif
#endif
#endif
#endif
				}
			}

		}
		index = (index + 1) % config.NumBlocks;
	}
#ifdef SQL_OUTPUT
	if (rowCounter > 0)
		InsertRowBlock(db, rowBuffer, rowCounter);
	free(rowBuffer);
#endif
	printf("final misc. counter tally: %i\n", miscCounter);
	for (int i = 0; i < config.NumBlocks; i++) cudaStreamDestroy(kernelStreams[i]);
	free(kernelStreams);
	return { (int)checkedSeeds, (int)passedSeeds };
}

void FreeMemory(AllPointers pointers)
{
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dArena));
}

void FindInterestingColors(const char* path, int initialPathLen)
{
	char path2[100];
	int i = 0;
	while (path[i] != '/')
	{
		path2[i] = toupper(path[i]);
		i++;
	}
	path2[i++] = '_';
	while (path[i] != '.')
	{
		path2[i] = toupper(path[i]);
		i++;
	}
	path2[i++] = '\0';

#if 1
	printf("PS_%s,\n", path2);
#else
	const uint32_t interestingColors[] = { 0x78ffff, 0x55ff8c, 0x50a000, 0x00ff00, 0xff0000, 0x800000 };
	const char* typeEnum[] = { "PSST_SpawnHeart", "PSST_SpawnChest", "PSST_SpawnFlask", "PSST_SpawnItem", "PSST_SmallEnemy", "PSST_LargeEnemy" };

	png_byte color_type = GetColorType(path);
	if (color_type != PNG_COLOR_TYPE_RGB) ConvertRGBAToRGB(path);
	Vec2i dims = GetImageDimensions(path);
	uint8_t* data = (uint8_t*)malloc(3 * dims.x * dims.y);
	ReadImage(path, data);

	printf("%s = {\n", path2);

	for (int x = 0; x < dims.x; x++)
	{
		for (int y = 0; y < dims.y; y++)
		{
			int idx = 3 * (y * dims.x + x);
			uint32_t pix = createRGB(data[idx], data[idx + 1], data[idx + 2]);
			for (int i = 0; i < 6; i++)
			{
				if (pix == interestingColors[i])
				{
					printf("	\"PixelSceneSpawn(%s, %i, %i),\",\n", typeEnum[i], x, y);
				}
			}
		}
	}
	printf("},\n");
#endif
}

void GetAllInterestingPixelsInFolder(const char* path)
{
	WIN32_FIND_DATA fd;

	char buffer[50];
	int offset = 0;
	_putstr_offset(path, buffer, offset);
	_putstr_offset("*.*", buffer, offset);
	buffer[offset] = '\0';

	HANDLE hFind = ::FindFirstFile(buffer, &fd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				offset = 0;
				_putstr_offset(path, buffer, offset);
				_putstr_offset(fd.cFileName, buffer, offset);
				buffer[offset] = '\0';
				FindInterestingColors(buffer, strlen(path));
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
}

void GenerateSpellData()
{
	printf("__device__ const static bool spellSpawnableInChests[] = {\n");
	for (int j = 0; j < SpellCount; j++)
	{
		bool passed = false;
		for (int t = 0; t < 11; t++)
		{
			if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL || allSpells[j].s == SPELL_SEA_SWAMP)
			{
				passed = true;
				break;
			}
		}
		printf(passed ? "true" : "false");
		printf(",\n");
	}
	printf("};\n");

	printf("__device__ const static bool spellSpawnableInBoxes[] = {\n");
	for (int j = 0; j < SpellCount; j++)
	{
		bool passed = false;
		if (allSpells[j].type == MODIFIER || allSpells[j].type == UTILITY)
		{
			for (int t = 0; t < 11; t++)
			{
				if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL || allSpells[j].s == SPELL_SEA_SWAMP)
				{
					passed = true;
					break;
				}
			}
		}
		printf(passed ? "true" : "false");
		printf(",\n");
	}
	printf("};\n");

	int counters2[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
	double sums[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
	for (int t = 0; t < 11; t++)
	{
		printf("__device__ const static SpellProb spellProbs_%i[] = {\n", t);
		for (int j = 0; j < SpellCount; j++)
		{
			if (allSpells[j].spawn_probabilities[t] > 0)
			{
				counters2[t]++;
				sums[t] += allSpells[j].spawn_probabilities[t];
				printf("{%f,SPELL_%s},\n", sums[t], SpellNames[j + 1]);
			}
		}
		printf("};\n");
	}

	printf("__device__ const static int spellTierCounts[] = {\n");
	for (int t = 0; t < 11; t++)
	{
		printf("%i,\n", counters2[t]);
	}
	printf("};\n");

	printf("__device__ const static float spellTierSums[] = {\n");
	for (int t = 0; t < 11; t++)
	{
		printf("%f,\n", sums[t]);
	}
	printf("};\n\n");


	for (int tier = 0; tier < 11; tier++)
	{
		int counters[8] = { 0,0,0,0,0,0,0,0 };
		for (int t = 0; t < 8; t++)
		{
			for (int j = 0; j < SpellCount; j++)
			{
				if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
				{
					counters[t]++;
				}
			}
		}
		for (int t = 0; t < 8; t++)
		{
			if (counters[t] > 0)
			{
				double sum = 0;
				printf("__device__ const static SpellProb spellProbs_%i_T%i[] = {\n", tier, t);
				for (int j = 0; j < SpellCount; j++)
				{
					if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
					{
						sum += allSpells[j].spawn_probabilities[tier];
						printf("{%f,SPELL_%s},\n", sum, SpellNames[j + 1]);
					}
				}
				printf("};\n");
			}
		}
		printf("__device__ const static SpellProb* spellProbs_%i_Types[] = {\n", tier);
		for (int t = 0; t < 8; t++)
		{
			if (counters[t] > 0)
				printf("spellProbs_%i_T%i,\n", tier, t);
			else
				printf("NULL,\n");
		}
		printf("};\n");

		printf("__device__ const static int spellProbs_%i_Counts[] = {\n", tier);
		for (int t = 0; t < 8; t++)
		{
			printf("%i,\n", counters[t]);
		}
		printf("};\n\n");
	}
}

int main()
{
#ifdef SFML
	SfmlMain();
	return;
#else

	//sqlite3* db = OpenDatabase("D:/testDatabase.db");
	//SelectFromDB(db);
	//CloseDatabase(db);
	//return;

	//GenerateSpellData();
	//return;
	//GetAllInterestingPixelsInFolder("excavationsite/");
	//return;

	for (global_iters = 0; global_iters < 1; global_iters++)
	{
		time_t startTime = _time64(NULL);
		std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

		BiomeWangScope biomes[20];
		int biomeCount = 0;
		int maxMapArea = 0;
		InstantiateBiome("wang_tiles/coalmine.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/coalmine_alt.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/excavationsite.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/fungicave.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/snowcave.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/snowcastle.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/rainforest.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/rainforest_open.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/vault.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/crypt.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/fungiforest.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/vault_frozen.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/liquidcave.png", biomes, biomeCount, maxMapArea);
		//InstantiateBiome("wang_tiles/meat.png", biomes, biomeCount, maxMapArea);

		DeviceConfig config = CreateConfigs(maxMapArea, biomeCount);

		memcpy(&config.biomeScopes, biomes, sizeof(BiomeWangScope) * 20);

		AllPointers pointers = AllocateMemory(config);
		sqlite3* db = NULL;
#ifdef DATABASE_OUTPUT
		sqlite3* db = OpenDatabase("D:/testDatabase.db");
		CreateTable(db);
#endif
		FILE* f = fopen("output.txt", "wb");

		Vec2i seedCounts = OutputLoop(config, pointers, f, startTime, db);
		checkCudaErrors(cudaDeviceSynchronize());
#ifdef DATABASE_OUTPUT
		SelectFromDB(db);
#endif

		std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
		std::chrono::nanoseconds duration = time2 - time1;

		printf("Search finished in %ims. Checked %i seeds, found %i valid seeds.\n", (int)(duration.count() / 1000000), seedCounts.x, seedCounts.y);

		FreeMemory(pointers);
#ifdef DATABASE_OUTPUT
		CloseDatabase(db);
#endif
		fclose(f);
	}
#endif
}

/*
cudaSetDeviceFlags(cudaDeviceMapHost);

volatile int* h_buckets;
volatile int* d_buckets;

cudaHostAlloc((void**)&h_buckets, sizeof(volatile int) * 70, cudaHostAllocMapped);

cudaHostGetDevicePointer((void**)&d_buckets, (void*)h_buckets, 0);

for (int i = 0; i < 70; i++) h_buckets[i] = 0;
wandExperiment<<<256,64>>>((int*)d_buckets, 6, true);
cudaDeviceSynchronize();
for (int i = 0; i < 70; i++) {
	printf("multicast %i: %i wands\n", i, h_buckets[i]);
}
return;*/

/*
__global__ void wandExperiment(const int level, const bool nonShuffle)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	constexpr int radius = 100;
	constexpr int seed = 913380622;
	constexpr int center_x = 5061;
	constexpr int center_y = 11119;
	for (int x = -radius + index + center_x; x < radius + center_x; x += stride) {
		for (int y = -radius + center_y; y < radius + center_y; y++) {
			Wand w = GetWandWithLevel(seed, x, y, level, nonShuffle, false);
			bool found = false;
			for (int i = 0; i < w.spellIdx; i++)
			{
				if (w.spells[i] == SPELL_SWAPPER_PROJECTILE)
					found = true;
			}

			if (found) printf("%i %i\n", x, y);
		}
	}
}
*/
