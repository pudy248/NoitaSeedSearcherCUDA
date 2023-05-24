#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/error.h"

#include "Precheckers.h"
#include "Worldgen.h"
#include "WorldgenSearch.h"
#include "Filters.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

//#define SEED_OUTPUT

//tired of seeing an error for it being undefined
__device__ int atomicAdd(int* address, int val);

struct GlobalConfig
{
	uint startSeed;
	uint endSeed;
	int printInterval;
	int atomicGranularity;
	int passedGranularity;
};

struct MemBlockSizes
{
	size_t outputSize;
	size_t mapDataSize;
	size_t miscMemSize;
	size_t visitedMemSize;
};

struct DeviceConfig
{
	int NumBlocks;
	int BlockSize;
	MemBlockSizes memSizes;
	GlobalConfig globalCfg;
	PrecheckConfig precheckCfg;
	WorldgenConfig worldCfg;
	LootConfig lootCfg;
	FilterConfig filterCfg;

	//Unused in device code
	const char* wangPath;
};

struct DevicePointers
{
	volatile int* uCheckedSeeds;
	volatile int* uPassedSeeds;
	volatile UnifiedOutputFlags* uFlags;
	byte* uOutput;
	byte* dTileData;
	byte* dOutput;
	byte* dMapData;
	byte* dMiscMem;
	byte* dVisitedMem;
};

struct HostPointers
{
	volatile int* hCheckedSeeds;
	volatile int* hPassedSeeds;
	volatile UnifiedOutputFlags* hFlags;
	byte* uOutput;
	byte* hOutput;
	byte* hTileData;
};

struct AllPointers
{
	DevicePointers dPointers;
	HostPointers hPointers;
};

__global__ void Kernel(DeviceConfig dConfig, DevicePointers dPointers)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	volatile UnifiedOutputFlags* flags = dPointers.uFlags + index;
	byte* uOutput = dPointers.uOutput + index * dConfig.memSizes.outputSize;

	byte* output = dPointers.dOutput + index * dConfig.memSizes.outputSize;
#ifdef DO_WORLDGEN
	byte* map = dPointers.dMapData + index * dConfig.memSizes.mapDataSize;
	byte* miscMem = dPointers.dMiscMem + index * dConfig.memSizes.miscMemSize;
	byte* visited = dPointers.dVisitedMem + index * dConfig.memSizes.visitedMemSize;
	byte* spawnableMem = miscMem;
#endif

	int counter = 0;
	int counter2 = 0;
	int seedIndex = -1;

	bool pollState = true;
	uint startSeed = 0;

	while (true)
	{
		if (pollState)
		{
			ThreadState state = flags->state;
			if (state == Running) pollState = false;
			if (state == ThreadStop) break; //End execution
			if (state == HostLock || state == SeedFound || state == QueueEmpty) continue; //Stall until host updates
		}

		if (seedIndex >= seedBlockSize)
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

		if (currentSeed >= dConfig.globalCfg.endSeed || flags->seed == 0)
		{
			flags->state = QueueEmpty;
			pollState = true;
			seedIndex = -1;
			startSeed = 0;
			continue;
		}

		counter++;
		bool seedPassed = true;

		seedPassed &= PrecheckSeed(currentSeed, dConfig.precheckCfg);

		if (!seedPassed)
		{
#ifdef DO_ATOMICS
			if (counter > dConfig.globalCfg.atomicGranularity)
			{
				atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.globalCfg.atomicGranularity);
				counter -= dConfig.globalCfg.atomicGranularity;
			}
#endif
			continue;
		}

#ifdef DO_WORLDGEN
		GenerateMap(currentSeed, output, map, visited, miscMem, dConfig.worldCfg, dConfig.globalCfg.startSeed / 5);

		CheckSpawnables(map, currentSeed, spawnableMem, output, dConfig.worldCfg, dConfig.lootCfg, dConfig.memSizes.miscMemSize);
		
		SpawnableBlock result = ParseSpawnableBlock(spawnableMem, map, output, dConfig.lootCfg, dConfig.memSizes.mapDataSize);
		seedPassed &= SpawnablesPassed(result, dConfig.filterCfg);

		if (!seedPassed)
		{
#ifdef DO_ATOMICS
			if (counter > dConfig.globalCfg.atomicGranularity)
			{
				atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.globalCfg.atomicGranularity);
				counter -= dConfig.globalCfg.atomicGranularity;
			}
#endif
			continue;
		}
#endif

		flags->state = DeviceLock;


		memcpy(uOutput, &currentSeed, 4);


		flags->state = SeedFound;
		pollState = true;

#ifdef DO_ATOMICS
		counter2++;
		if (counter >= dConfig.globalCfg.atomicGranularity)
		{
			atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.globalCfg.atomicGranularity);
			counter -= dConfig.globalCfg.atomicGranularity;
		}
		if (counter2 >= dConfig.globalCfg.passedGranularity)
		{
			atomicAdd((int*)dPointers.uPassedSeeds, dConfig.globalCfg.passedGranularity);
			counter2 -= dConfig.globalCfg.passedGranularity;
		}
#endif
	}
#ifdef DO_ATOMICS
	atomicAdd((int*)dPointers.uCheckedSeeds, counter);
	atomicAdd((int*)dPointers.uPassedSeeds, counter2);
#endif
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

DeviceConfig CreateConfigs()
{
	//MINES
	WorldgenConfig worldCfg = { 348, 448, 256, 103, 34, 14, true, false, 100 };
	const char* fileName = "minesWang.bin";
	constexpr auto NUMBLOCKS = 128;
	constexpr auto BLOCKSIZE = 64;
	constexpr auto biomeIdx = 0;
	constexpr auto mapMemMult = 4;
	constexpr auto miscMemMult = 10;

	//EXCAVATION SITE
	//WorldgenConfig worldCfg = { 344, 440, 409, 102, 31, 17, false, false, 100 };
	//const char* fileName = "excavationsiteWang.bin";
	//constexpr auto NUMBLOCKS = 64;
	//constexpr auto BLOCKSIZE = 64;
	//constexpr auto biomeIdx = 1;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//SNOWCAVE
	//WorldgenConfig worldCfg = { 440, 560, 512, 153, 30, 20, false, false, 100 };
	//const char* fileName = "snowcaveWang.bin";
	//constexpr auto NUMBLOCKS = 64;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto biomeIdx = 1;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//CRYPT
	//WorldgenConfig worldCfg = { 282, 342, 717, 204, 26, 35, false, false, 100 };
	//const char* fileName = "cryptWang.bin";
	//constexpr auto NUMBLOCKS = 32;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto biomeIdx = 10;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//OVERGROWN CAVERNS
	//WorldgenConfig worldCfg = { 144, 235, 359, 461, 59, 16, false, false, 1 };
	//const char* fileName = "fungiforestWang.bin";
	//constexpr auto NUMBLOCKS = 32;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto biomeIdx = 15;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//HELL
	//WorldgenConfig worldCfg = { 156, 364, 921, 256, 25, 43, false, true, 100 };
	//const char* fileName = "hellWang.bin";
	//constexpr auto NUMBLOCKS = 32;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto biomeIdx = 0;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	MemBlockSizes memSizes = {
		4096,
		mapMemMult * 3 * worldCfg.map_w * (worldCfg.map_h + 4),
		miscMemMult * worldCfg.map_w * worldCfg.map_h,
		worldCfg.map_w * worldCfg.map_h
	};

	GlobalConfig globalCfg = { 1, INT_MAX, 1, 1, 1 };

	Item iF1[FILTER_OR_COUNT] = { HEART_MIMIC };
	Item iF2[FILTER_OR_COUNT] = { MIMIC };
	Material mF1[FILTER_OR_COUNT] = { BRASS };
	Spell sF1[FILTER_OR_COUNT] = { SPELL_REGENERATION_FIELD };
	Spell sF2[FILTER_OR_COUNT] = { SPELL_CASTER_CAST };
	Spell sF3[FILTER_OR_COUNT] = { SPELL_CURSE_WITHER_PROJECTILE };

	ItemFilter iFilters[] = { ItemFilter(iF1, 4), ItemFilter(iF2) };
	MaterialFilter mFilters[] = { MaterialFilter(mF1) };
	SpellFilter sFilters[] = { SpellFilter(sF1, 5), SpellFilter(sF2), SpellFilter(sF3) };

	FilterConfig filterCfg = FilterConfig(false, 0, iFilters, 0, mFilters, 1, sFilters, false, 27);
	LootConfig lootCfg = LootConfig(0, 0, true, false, false, false, false, filterCfg.materialFilterCount > 0, false, biomeIdx, true);

	PrecheckConfig precheckCfg = {
		false,
		false, URINE,
		false, SPELL_RUBBER_BALL, SPELL_GRENADE,
		true, WATER,
		false, AlchemyOrdering::ONLY_CONSUMED, {MUD, WATER, SOIL}, {MUD, WATER, SOIL},
		false, {BM_GOLD_VEIN_SUPER, BM_NONE, BM_GOLD_VEIN_SUPER},
		false, {FungalShift(SS_FLASK, SD_CHEESE_STATIC, 0, 4), FungalShift(), FungalShift(), FungalShift()},
		false, {{PERK_ANGRY_GHOST, false, 0, 3}},
		false, filterCfg, lootCfg,
		false, false, false, 1, 6, 5
	};

	if (precheckCfg.checkBiomeModifiers && !ValidateBiomeModifierConfig(precheckCfg))
	{
		printf("Impossible biome modifier set! Aborting...\n");
	}

	return { NUMBLOCKS, BLOCKSIZE, memSizes, globalCfg, precheckCfg, worldCfg, lootCfg, filterCfg, fileName };
}

AllPointers AllocateMemory(DeviceConfig config)
{
	cudaSetDeviceFlags(cudaDeviceMapHost);

	size_t outputSize = config.NumBlocks * config.BlockSize * config.memSizes.outputSize;
#ifdef DO_WORLDGEN
	size_t tileDataSize = 3 * config.worldCfg.tiles_w * config.worldCfg.tiles_h;
	size_t mapDataSize = config.NumBlocks * config.BlockSize * config.memSizes.mapDataSize;
	size_t miscMemSize = config.NumBlocks * config.BlockSize * config.memSizes.miscMemSize;
	size_t visitedMemSize = config.NumBlocks * config.BlockSize * config.memSizes.visitedMemSize;

	printf("Memory Usage Statistics:\n");
	printf("Output: %ziMB  Map data: %ziMB\n", outputSize / 1000000, mapDataSize / 1000000);
	printf("Misc memory: %ziMB  Visited cells: %ziMB\n", miscMemSize / 1000000, visitedMemSize / 1000000);
	printf("Total memory: %ziMB\n", (tileDataSize + outputSize + mapDataSize + miscMemSize + visitedMemSize) / 1000000);
#endif


	//Host
	byte* hOutput;
	byte* hTileData;

	hOutput = (byte*)malloc(outputSize);
	hTileData = (byte*)malloc(3 * config.worldCfg.tiles_w * config.worldCfg.tiles_h);
	std::ifstream source(config.wangPath, std::ios_base::binary);
	source.read((char*)hTileData, 3 * config.worldCfg.tiles_w * config.worldCfg.tiles_h);
	source.close();

	//Device
	byte* dOutput;
	byte* dTileData;
	byte* dMapData;
	byte* dMiscMem;
	byte* dVisitedMem;

	checkCudaErrors(cudaMalloc(&dOutput, outputSize));
#ifdef DO_WORLDGEN
	checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
	checkCudaErrors(cudaMalloc(&dMapData, mapDataSize));
	checkCudaErrors(cudaMalloc(&dMiscMem, miscMemSize));
	checkCudaErrors(cudaMalloc(&dVisitedMem, visitedMemSize));

	checkCudaErrors(cudaMemcpy(dTileData, hTileData, 3 * config.worldCfg.tiles_w * config.worldCfg.tiles_h, cudaMemcpyHostToDevice));
	buildTS<<<1, 1>>>(dTileData, config.worldCfg.tiles_w, config.worldCfg.tiles_h);
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	//Unified
	volatile int* hCheckedSeeds, *dCheckedSeeds;
	volatile int* hPassedSeeds, *dPassedSeeds;

	volatile UnifiedOutputFlags* hFlags, *dFlags;
	byte* hUnifiedOutput, *dUnifiedOutput;

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

	return { {dCheckedSeeds, dPassedSeeds, dFlags, dUnifiedOutput, dTileData, dOutput, dMapData, dMiscMem, dVisitedMem}, {hCheckedSeeds, hPassedSeeds, hFlags, hUnifiedOutput, hOutput, hTileData} };
}

void OutputLoop(DeviceConfig config, HostPointers pointers, cudaEvent_t _event, ofstream& outputStream)
{
	chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
	int intervals = 0;
	int counter = 0;
	uint lastDiff = 0;
	uint lastSeed = 0;

	uint currentSeed = config.globalCfg.startSeed;
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
				//printf("time poll\n");
				counter = 0;
				chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
				std::chrono::nanoseconds duration = time2 - time1;
				ulong milliseconds = (ulong)(duration.count() / 1000000);
				if (intervals * config.globalCfg.printInterval * 1000 < milliseconds)
				{
					lastDiff = *pointers.hCheckedSeeds - lastSeed;
					lastSeed = *pointers.hCheckedSeeds;
					intervals++;
					float percentComplete = ((float)(*pointers.hCheckedSeeds) / (config.globalCfg.endSeed - config.globalCfg.startSeed));
					printf(">%i: %2.3f%% complete. Searched %i (+%i this interval), found %i valid seeds.\n", intervals, percentComplete * 100, *pointers.hCheckedSeeds, lastDiff, *pointers.hPassedSeeds);
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
			if (currentSeed >= config.globalCfg.endSeed)
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
				currentSeed += seedBlockSize;
				continue;
			}
		}

		if (state == SeedFound)
		{
			byte* uOutput = pointers.uOutput + index * config.memSizes.outputSize;
			byte* output = pointers.hOutput + index * config.memSizes.outputSize;

			pointers.hFlags[index].state = HostLock;
			memcpy(output, uOutput, config.memSizes.outputSize);
			pointers.hFlags[index].state = Running;

			int seed = *(int*)output; //do other things otherwise
			char buffer[30];
			int offset = 0;
			_itoa_offset(seed, 10, buffer, offset);
			buffer[offset++] = '\n';
			outputStream.write(buffer, offset);
			foundSeeds++;
		}
	}
}

void FreeMemory(AllPointers pointers)
{
#ifdef DO_WORLDGEN
	freeTS << <1, 1 >> > ();
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	//free((void*)pointers.hPointers.hCheckedSeeds);
	//free((void*)pointers.hPointers.hPassedSeeds);
	//free((void*)pointers.hPointers.hFlags);
	//free((void*)pointers.hPointers.uOutput);
	free(pointers.hPointers.hOutput);
	free(pointers.hPointers.hTileData);


	checkCudaErrors(cudaFree((void*)pointers.dPointers.dOutput));
#ifdef DO_WORLDGEN
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dTileData));
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dMapData));
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dMiscMem));
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dVisitedMem));
#endif
}

int main()
{
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

	for (int global_iters = 0; global_iters < 1; global_iters++)
	{
		chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

		DeviceConfig config = CreateConfigs();
		AllPointers pointers = AllocateMemory(config);

		std::ofstream f = ofstream("output.txt", std::ios::binary);

		cudaEvent_t _event;
		checkCudaErrors(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming));
		Kernel<<<config.NumBlocks, config.BlockSize>>>(config, pointers.dPointers);
		checkCudaErrors(cudaEventRecord(_event));

		OutputLoop(config, pointers.hPointers, _event, f);
		checkCudaErrors(cudaDeviceSynchronize());

		chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
		std::chrono::nanoseconds duration = time2 - time1;

		printf("Search finished in %ims. Checked %i seeds, found %i valid seeds.\n", (int)(duration.count() / 1000000), *pointers.hPointers.hCheckedSeeds, *pointers.hPointers.hPassedSeeds);

		FreeMemory(pointers);
		f.close();
	}
}