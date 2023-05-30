#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/error.h"
#include "misc/pngutils.h"

#include "Configuration.h"
#include "Precheckers.h"
#include "Worldgen.h"
#include "WorldgenSearch.h"
#include "Filters.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

int global_iters = 0;

//tired of seeing an error for it being undefined
__device__ int atomicAdd(int* address, int val);

struct DeviceConfig
{
	int NumBlocks;
	int BlockSize;
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
	volatile int* uCheckedSeeds;
	volatile int* uPassedSeeds;
	volatile UnifiedOutputFlags* uFlags;
	uint8_t* uOutput;
	uint8_t* dArena;
};

struct HostPointers
{
	volatile int* hCheckedSeeds;
	volatile int* hPassedSeeds;
	volatile UnifiedOutputFlags* hFlags;
	uint8_t* uOutput;
	uint8_t* hOutput;
};

struct AllPointers
{
	DevicePointers dPointers;
	HostPointers hPointers;
};


__global__ void Kernel(DeviceConfig dConfig, DevicePointers dPointers)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	volatile UnifiedOutputFlags* flags = dPointers.uFlags + index;
	uint8_t* uOutput = dPointers.uOutput + index * dConfig.memSizes.outputSize;

	uint8_t* threadMemPtr = dPointers.dArena + index * dConfig.memSizes.threadMemTotal;
	MemoryArena arena = { threadMemPtr, 0 };

	int checkedCtr = 0;
	int passedCtr = 0;
	int seedIndex = -1;

	bool pollState = true;
	uint32_t startSeed = 0;

	while (true)
	{
		arena.offset = 0;

		if (pollState)
		{
			ThreadState state = flags->state;
			if (state == Running) pollState = false;
			if (state == ThreadStop) break; //End execution
			if (state == HostLock || state == SeedFound || state == QueueEmpty) continue; //Stall until host updates
		}

		if (seedIndex >= dConfig.generalCfg.seedBlockSize)
		{
			flags->state = QueueEmpty;
			pollState = true;
			seedIndex = -1;
			startSeed = 0;
			continue;
		}

		if (seedIndex == -1)
		{
			startSeed = flags->seed;
			seedIndex++;
		}

		int currentSeed = startSeed + seedIndex;
		seedIndex++;

		if (currentSeed >= dConfig.generalCfg.endSeed || flags->seed == 0)
		{
			flags->state = QueueEmpty;
			pollState = true;
			seedIndex = -1;
			startSeed = 0;
			continue;
		}

		checkedCtr++;
		bool seedPassed = true;

		uint8_t* output = ArenaAlloc(arena, dConfig.memSizes.outputSize, 4);

		seedPassed &= PrecheckSeed(currentSeed, dConfig.precheckCfg);

		if (!seedPassed)
		{
			if (checkedCtr > dConfig.generalCfg.atomicGranularity)
			{
				atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.generalCfg.atomicGranularity);
				checkedCtr -= dConfig.generalCfg.atomicGranularity;
			}
			continue;
		}

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
			CheckSpawnables(mapMem, currentSeed, spawnables, spawnableOffset, spawnableCount, dConfig.biomeScopes[biomeNum].cfg, dConfig.spawnableCfg, dConfig.memSizes.spawnableMemSize);
		}

		CheckMountains(currentSeed, dConfig.spawnableCfg, spawnables, spawnableOffset, spawnableCount);
		CheckEyeRooms(currentSeed, dConfig.spawnableCfg, spawnables, spawnableOffset, spawnableCount);

		//printf("%i spawnables\n", spawnableCount);

		((int*)spawnables)[1] = spawnableCount;
		SpawnableBlock result = ParseSpawnableBlock(spawnables, mapMem, dConfig.spawnableCfg, dConfig.memSizes.mapDataSize);
		seedPassed &= SpawnablesPassed(result, dConfig.filterCfg, output, true);

		if (!seedPassed)
		{
			if (checkedCtr > dConfig.generalCfg.atomicGranularity)
			{
				atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.generalCfg.atomicGranularity);
				checkedCtr -= dConfig.generalCfg.atomicGranularity;
			}
			continue;
		}
#endif
		flags->state = DeviceLock;
		memcpy(output, &currentSeed, 4);
		memcpy(output + 4, &spawnableCount, 4);
		memcpy(uOutput, output, dConfig.memSizes.outputSize);
		flags->state = SeedFound;
		pollState = true;

		passedCtr++;
		if (checkedCtr >= dConfig.generalCfg.atomicGranularity)
		{
			atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.generalCfg.atomicGranularity);
			checkedCtr -= dConfig.generalCfg.atomicGranularity;
		}
		if (passedCtr >= dConfig.generalCfg.passedGranularity)
		{
			atomicAdd((int*)dPointers.uPassedSeeds, dConfig.generalCfg.passedGranularity);
			passedCtr -= dConfig.generalCfg.passedGranularity;
		}
	}
	atomicAdd((int*)dPointers.uCheckedSeeds, checkedCtr);
	atomicAdd((int*)dPointers.uPassedSeeds, passedCtr);
}

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

BiomeWangScope InstantiateBiome(const char* path, int& maxMapArea)
{
	MapConfig mapCfg;
	int minX = INT_MIN;
	int maxX = INT_MAX;
	int minY = INT_MIN;
	int maxY = INT_MAX;
	int spawnableMemMult; //to be used Later
	{
		if (strcmp(path, "wang_tiles/coalmine.png") == 0)
		{
			mapCfg = { 34, 14, 5, 2, 0 };
			spawnableMemMult = 4;
			minX = -500;
			maxX = 2000;
			minY = 10;
			maxY = 1000;
		}
		else if (strcmp(path, "wang_tiles/coalmine_alt.png") == 0)
		{
			mapCfg = { 32, 15, 2, 1, 1 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/excavationsite.png") == 0)
		{
			mapCfg = { 31, 17, 8, 2, 2 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/fungicave.png") == 0)
		{
			mapCfg = { 28, 17, 3, 1, 3 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/snowcave.png") == 0)
		{
			mapCfg = { 30, 20, 10, 3, 4 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/snowcastle.png") == 0)
		{
			mapCfg = { 31, 24, 7, 2, 5 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/rainforest.png") == 0)
		{
			mapCfg = { 30, 27, 9, 2, 6 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/rainforest_open.png") == 0)
		{
			mapCfg = { 30, 28, 9, 2, 7 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/rainforest_dark.png") == 0)
		{
			mapCfg = { 25, 26, 5, 8, 8 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/vault.png") == 0)
		{
			mapCfg = { 29, 31, 11, 3, 9 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/crypt.png") == 0)
		{
			mapCfg = { 26, 35, 14, 4, 10 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/wandcave.png") == 0)
		{
			mapCfg = { 47, 35, 4, 4, 11 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/vault_frozen.png") == 0)
		{
			mapCfg = { 12, 15, 7, 5, 12 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/wizardcave.png") == 0)
		{
			mapCfg = { 53, 40, 6, 6, 13 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/fungiforest.png") == 0)
		{
			mapCfg = { 59, 16, 7, 9, 15 };
			spawnableMemMult = 4;
		}
		else if (strcmp(path, "wang_tiles/robobase.png") == 0)
		{
			mapCfg = { 59, 29, 7, 9, 25 };
			spawnableMemMult = 4;
		}

		else
		{
			printf("Invalid biome path: %s\n", path);
			return { NULL, {} };
		}
	}

	mapCfg.minX = minX;
	mapCfg.maxX = maxX;
	mapCfg.minY = minY;
	mapCfg.maxY = maxY;

	mapCfg.maxTries = 100;
	mapCfg.isCoalMine = strcmp(path, "wang_tiles/coalmine.png") == 0;

	mapCfg.map_w = GetWidthFromPix(mapCfg.worldX, mapCfg.worldX + mapCfg.worldW);
	mapCfg.map_h = GetWidthFromPix(mapCfg.worldY, mapCfg.worldY + mapCfg.worldH);

	IntPair tileDims = GetImageDimensions(path);
	mapCfg.tiles_w = tileDims.x;
	mapCfg.tiles_h = tileDims.y;

	maxMapArea = std::max(maxMapArea, (int)(mapCfg.map_w * (mapCfg.map_h + 4)));
	uint64_t tileDataSize = 3 * mapCfg.tiles_w * mapCfg.tiles_h;

	uint8_t* dTileData;
	uint8_t* dTileSet;

	uint8_t* hTileData = (uint8_t*)malloc(3 * mapCfg.tiles_w * mapCfg.tiles_h);
	ReadImage(path, hTileData);

	checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
	checkCudaErrors(cudaMalloc(&dTileSet, tileDataSize));

	checkCudaErrors(cudaMemcpy(dTileData, hTileData, 3 * mapCfg.tiles_w * mapCfg.tiles_h, cudaMemcpyHostToDevice));
	buildTS<<<1, 1>>>(dTileData, dTileSet, mapCfg.tiles_w, mapCfg.tiles_h);
	checkCudaErrors(cudaDeviceSynchronize());
	
	free(hTileData);
	checkCudaErrors(cudaFree(dTileData));

	return { dTileSet, mapCfg };
}

DeviceConfig CreateConfigs(int maxMapArea, int biomeCount)
{
	constexpr auto NUMBLOCKS = 256;
	constexpr auto BLOCKSIZE = 64;

	MemSizeConfig memSizes = {
#ifdef DO_WORLDGEN
		3 * maxMapArea + 4096,
#else
		64,
#endif
		3 * maxMapArea + 128,
		4 * maxMapArea,
		maxMapArea,
		64,
	};

	GeneralConfig generalCfg = { 40_GB, 1, INT_MAX, 1024*128, 5, 1, 1 };
	SpawnableConfig spawnableCfg = { {0, 0}, {0, 0}, 0, 7, 
		false, //greed
		false, //pacifist
		false, //shop spells
		false, //shop wands
		false, //eye rooms
		false, //biome chests
		false, //biome pedestals
		false, //biome altars
		false, //biome pixelscenes
		false, //hell shops
		false, //potion contents
		false, //chest spells
		false, //wand contents
	};

	Item iF1[FILTER_OR_COUNT] = { PAHA_SILMA };
	Item iF2[FILTER_OR_COUNT] = { MIMIC };
	Material mF1[FILTER_OR_COUNT] = { BRASS };
	Spell sF1[FILTER_OR_COUNT] = { SPELL_SUMMON_WANDGHOST };
	Spell sF2[FILTER_OR_COUNT] = { SPELL_CASTER_CAST };
	Spell sF3[FILTER_OR_COUNT] = { SPELL_CURSE_WITHER_PROJECTILE };

	ItemFilter iFilters[] = { ItemFilter(iF1, 1), ItemFilter(iF2) };
	MaterialFilter mFilters[] = { MaterialFilter(mF1) };
	SpellFilter sFilters[] = { SpellFilter(sF1, 1), SpellFilter(sF2), SpellFilter(sF3) };

	FilterConfig filterCfg = FilterConfig(false, 0, iFilters, 0, mFilters, 0, sFilters, false, 27);

	StaticPrecheckConfig precheckCfg = {
		{true, SKATEBOARD},
		{false, URINE},
		{false, SPELL_LIGHT_BULLET, SPELL_DYNAMITE},
		{false, BLOOD},
		{false, AlchemyOrdering::ONLY_CONSUMED, {MUD, WATER, SOIL}, {MUD, WATER, SOIL}},
		{false, {BM_GOLD_VEIN_SUPER, BM_NONE, BM_GOLD_VEIN_SUPER}},
		{false, {FungalShift(SS_FLASK, SD_CHEESE_STATIC, 0, 4), FungalShift(), FungalShift(), FungalShift()}},
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
	freeMem *= 0.9f; //leave a bit of extra
	printf("memory free: %lli of %lli bytes\n", freeMem, totalMem);

#ifdef DO_WORLDGEN
	uint64_t minMemoryPerThread = memSizes.outputSize * 2 + memSizes.mapDataSize + memSizes.miscMemSize + memSizes.visitedMemSize + memSizes.spawnableMemSize;
#else
	uint64_t minMemoryPerThread = memSizes.outputSize * 2 + memSizes.spawnableMemSize;
#endif
	printf("each thread requires %lli bytes of block memory\n", minMemoryPerThread);
	memSizes.threadMemTotal = minMemoryPerThread;

	int numThreads = std::min((uint64_t)(generalCfg.endSeed - generalCfg.startSeed), freeMem / minMemoryPerThread);
	int blockSize = std::min(numThreads, BLOCKSIZE);
	int numBlocks = numThreads / BLOCKSIZE;
	int numBlocksRounded = std::max(std::min(NUMBLOCKS, numBlocks - numBlocks % 8), 1);
	generalCfg.requestedMemory = minMemoryPerThread * numBlocksRounded * BLOCKSIZE;
	generalCfg.seedBlockSize = std::min((uint32_t)generalCfg.seedBlockSize, (generalCfg.endSeed - generalCfg.startSeed) / (numBlocksRounded * blockSize) + 1);
	printf("creating %ix%i threads\n", numBlocksRounded, blockSize);

	return { numBlocksRounded, blockSize, memSizes, generalCfg, precheckCfg, spawnableCfg, filterCfg, biomeCount };
}

AllPointers AllocateMemory(DeviceConfig config)
{
	cudaSetDeviceFlags(cudaDeviceMapHost);

	uint64_t outputSize = config.NumBlocks * config.BlockSize * config.memSizes.outputSize;

	//Host
	uint8_t* hOutput;

	hOutput = (uint8_t*)malloc(outputSize);

	//Device
	uint8_t* dArena;
	uint8_t* dOverlayMem;

	checkCudaErrors(cudaMalloc(&dArena, config.generalCfg.requestedMemory));
	checkCudaErrors(cudaMalloc(&dOverlayMem, 3 * 256 * 103));

	uint8_t* hPtr = (uint8_t*)malloc(3 * 256 * 103);
	ReadImage("wang_tiles/coalmine_overlay.png", hPtr);
	checkCudaErrors(cudaMemcpy(dOverlayMem, hPtr, 3 * 256 * 103, cudaMemcpyHostToDevice));
	cudaMemcpyToSymbol(coalmine_overlay, &dOverlayMem, sizeof(void*), 0);
	free(hPtr);

	//Unified
	volatile int* hCheckedSeeds, *dCheckedSeeds;
	volatile int* hPassedSeeds, *dPassedSeeds;

	volatile UnifiedOutputFlags* hFlags, *dFlags;
	uint8_t* hUnifiedOutput, *dUnifiedOutput;

	checkCudaErrors(cudaHostAlloc((void**)&hCheckedSeeds, 4, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void**)&hPassedSeeds, 4, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dCheckedSeeds, (void*)hCheckedSeeds, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dPassedSeeds, (void*)hPassedSeeds, 0));
	*hCheckedSeeds = 0;
	*hPassedSeeds = 0;

	checkCudaErrors(cudaHostAlloc((void**)&hFlags, sizeof(UnifiedOutputFlags) * config.NumBlocks * config.BlockSize, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dFlags, (void*)hFlags, 0));
	
	for (int i = 0; i < config.NumBlocks * config.BlockSize; i++)
	{
		hFlags[i].state = ThreadState::QueueEmpty;
	}

	checkCudaErrors(cudaHostAlloc((void**)&hUnifiedOutput, outputSize, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&dUnifiedOutput, (void*)hUnifiedOutput, 0));

	return { {dCheckedSeeds, dPassedSeeds, dFlags, dUnifiedOutput, dArena}, {hCheckedSeeds, hPassedSeeds, hFlags, hUnifiedOutput, hOutput} };
}

void OutputLoop(DeviceConfig config, HostPointers pointers, cudaEvent_t _event, FILE* outputFile)
{
	std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
	int intervals = 0;
	int counter = 0;
	uint32_t lastDiff = 0;
	uint32_t lastSeed = 0;

	uint32_t currentSeed = config.generalCfg.startSeed;
	int index = -1;
	int stoppedThreads = 0;
	int foundSeeds = 0;
	while (cudaEventQuery(_event) == cudaErrorNotReady)
	{
		if (index == 0)
		{
			counter++;

			if (counter == 8)
			{
				counter = 0;
				std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
				std::chrono::nanoseconds duration = time2 - time1;
				uint64_t milliseconds = (uint64_t)(duration.count() / 1000000);
				if (intervals * config.generalCfg.printInterval * 1000 < milliseconds)
				{
					lastDiff = *pointers.hCheckedSeeds - lastSeed;
					lastSeed = *pointers.hCheckedSeeds;
					intervals++;
					float percentComplete = ((float)(*pointers.hCheckedSeeds) / (config.generalCfg.endSeed - config.generalCfg.startSeed));
					uint64_t freeMem;
					uint64_t totalMem;
					cudaMemGetInfo(&freeMem, &totalMem);
					printf(">%i: %2.3f%% complete. Searched %i (+%i this interval), found %i valid seeds. (%lli/%lliMB used)\n", intervals, percentComplete * 100, *pointers.hCheckedSeeds, lastDiff, *pointers.hPassedSeeds, (totalMem - freeMem) / 1049576, totalMem / 1049576);
				}
			}
		}

		if (stoppedThreads == config.NumBlocks * config.BlockSize) break;

		index = (index + 1) % (config.NumBlocks * config.BlockSize);

		ThreadState state = pointers.hFlags[index].state;
		if (state == Running || state == DeviceLock || state == ThreadStop) continue;
		if (stoppedThreads == config.NumBlocks * config.BlockSize) break;

		if (state == QueueEmpty || pointers.hFlags[index].seed == 0)
		{
			if (currentSeed >= config.generalCfg.endSeed)
			{
				pointers.hFlags[index].state = ThreadStop;
				stoppedThreads++;
				//printf("Stopping thread %i\n", index);
				continue;
			}
			else
			{
				//printf("Assigning thread %i seed block %i\n", index, currentSeed);
				pointers.hFlags[index].seed = currentSeed;
				pointers.hFlags[index].state = Running;
				currentSeed += config.generalCfg.seedBlockSize;
				continue;
			}
		}

		if (state == SeedFound)
		{
			uint8_t* uOutput = pointers.uOutput + index * config.memSizes.outputSize;
			uint8_t* output = pointers.hOutput + index * config.memSizes.outputSize;

			pointers.hFlags[index].state = HostLock;
			memcpy(output, uOutput, config.memSizes.outputSize);
			pointers.hFlags[index].state = Running;
			foundSeeds++;

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
#endif
#ifdef SPAWNABLE_OUTPUT
			char buffer[4096];
			int bufOffset = 0;
			int memOffset = 0;
			int seed = readInt(output, memOffset);
			int sCount = readInt(output, memOffset);
			if (seed == 0) continue;
			
			_itoa_offset(seed, 10, buffer, bufOffset);
			if (sCount > 0)
			{
				_putstr_offset(": ", buffer, bufOffset);
				Spawnable* sPtr;
				for (int i = 0; i < sCount; i++)
				{
					sPtr = (Spawnable*)(output + memOffset);
					Spawnable s = readMisalignedSpawnable(sPtr);
					IntPair chunkCoords = GetLocalPos(s.x, s.y);

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
					_putstr_offset(")", buffer, bufOffset);
					_putstr_offset(" [", buffer, bufOffset);

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
					_putstr_offset("]", buffer, bufOffset);
					memOffset += s.count + 13;
				}
			}
			buffer[bufOffset++] = '\n';
			buffer[bufOffset++] = '\0';
			fprintf(outputFile, "%s", buffer);
			printf("%s", buffer);
#else
			char buffer[12];
			int bufOffset = 0;
			_itoa_offset(*(int*)output, 10, buffer, bufOffset);
			buffer[bufOffset++] = '\n';
			buffer[bufOffset++] = '\0';
			fprintf(outputFile, "%s", buffer);
			//printf("%s", buffer);
#endif
		}
	}
}

void FreeMemory(AllPointers pointers)
{
	free(pointers.hPointers.hOutput);

	checkCudaErrors(cudaFree((void*)pointers.dPointers.dArena));
}

int main()
{
	for (global_iters = 0; global_iters < 1; global_iters++)
	{
		std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();

		BiomeWangScope biomes[20];
		int biomeCount = 0;
		int maxMapArea = 0;
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/coalmine.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/coalmine_alt.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/excavationsite.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/fungicave.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/snowcave.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/snowcastle.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/vault.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/vault_frozen.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/crypt.png", maxMapArea);
		//biomes[biomeCount++] = InstantiateBiome("wang_tiles/fungiforest.png", maxMapArea);

		DeviceConfig config = CreateConfigs(maxMapArea, biomeCount);

		memcpy(&config.biomeScopes, biomes, sizeof(BiomeWangScope) * 20);

		AllPointers pointers = AllocateMemory(config);

		FILE* f = fopen("output.txt", "wb");

		cudaEvent_t _event;
		checkCudaErrors(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming));
		Kernel<<<config.NumBlocks, config.BlockSize>>>(config, pointers.dPointers);
		checkCudaErrors(cudaEventRecord(_event));

		OutputLoop(config, pointers.hPointers, _event, f);
		checkCudaErrors(cudaDeviceSynchronize());

		std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
		std::chrono::nanoseconds duration = time2 - time1;

		printf("Search finished in %ims. Checked %i seeds, found %i valid seeds.\n", (int)(duration.count() / 1000000), *pointers.hPointers.hCheckedSeeds, *pointers.hPointers.hPassedSeeds);

		FreeMemory(pointers);
		fclose(f);
	}
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
printf("__device__ const static bool spellSpawnableInChests[] = {\n");
for (int j = 0; j < SpellCount; j++)
{
	bool passed = false;
	for (int t = 0; t < 11; t++)
	{
		if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL)
		{
			passed = true;
			break;
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
	for(int t = 0; t < 8; t++) {
		double sum = 0;
		printf("__device__ const static SpellProb spellProbs_%i_T%i[] = {\n", tier, t);
		for (int j = 0; j < SpellCount; j++)
		{
			if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
			{
				counters[t]++;
				sum += allSpells[j].spawn_probabilities[tier];
				printf("{%f,SPELL_%s},\n", sum, SpellNames[j + 1]);
			}
		}
		printf("};\n");
	}
	printf("__device__ const static SpellProb* spellProbs_%i_Types[] = {\n", tier);
	for (int t = 0; t < 8; t++)
	{
		if(counters[t] > 0)
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
return;*/
