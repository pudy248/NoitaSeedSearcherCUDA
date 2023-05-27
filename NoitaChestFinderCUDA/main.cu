#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/error.h"

#include "Configuration.h"
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

struct DeviceConfig
{
	int NumBlocks;
	int BlockSize;
	MemSizeConfig memSizes;
	GeneralConfig generalCfg;
	StaticPrecheckConfig precheckCfg;
	MapConfig mapCfg;
	SpawnableConfig spawnableCfg;
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
	byte* dTileSet;
	byte* dArena;
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

	size_t threadMemSize = dConfig.generalCfg.requestedMemory / (dConfig.NumBlocks * dConfig.BlockSize);
	byte* threadMemPtr = dPointers.dArena + index * threadMemSize;
	MemoryArena arena = { threadMemPtr, 0 };

	int checkedCtr = 0;
	int passedCtr = 0;
	int seedIndex = -1;

	bool pollState = true;
	uint startSeed = 0;

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

		byte* output = ArenaAlloc(arena, dConfig.memSizes.outputSize, 4);

		seedPassed &= PrecheckSeed(currentSeed, dConfig.precheckCfg);

		if (!seedPassed)
		{
#ifdef DO_ATOMICS
			if (checkedCtr > dConfig.generalCfg.atomicGranularity)
			{
				atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.generalCfg.atomicGranularity);
				checkedCtr -= dConfig.generalCfg.atomicGranularity;
			}
#endif
			continue;
		}

#ifdef DO_WORLDGEN
		byte* mapMem = ArenaAlloc(arena, dConfig.memSizes.mapDataSize, 8); //Pointers are 8 bytes
		byte* miscMem = ArenaAlloc(arena, dConfig.memSizes.miscMemSize, 4);
		byte* visited = ArenaAlloc(arena, dConfig.memSizes.miscMemSize);
		GenerateMap(currentSeed, dPointers.dTileSet, output, mapMem, visited, miscMem, dConfig.mapCfg, dConfig.generalCfg.startSeed / 5);

		ArenaSetOffset(arena, miscMem);
		byte* bufferMem = ArenaAlloc(arena, dConfig.memSizes.bufferSize);
		byte* spawnableMem = ArenaAlloc(arena, 4, 4);

		CheckSpawnables(mapMem, currentSeed, spawnableMem, output, dConfig.mapCfg, dConfig.spawnableCfg, dConfig.memSizes.miscMemSize);
		
		SpawnableBlock result = ParseSpawnableBlock(spawnableMem, mapMem, output, dConfig.mapCfg, dConfig.spawnableCfg, dConfig.memSizes.mapDataSize);
		seedPassed &= SpawnablesPassed(result, dConfig.mapCfg, dConfig.filterCfg, output, bufferMem, true);

		if (!seedPassed)
		{
#ifdef DO_ATOMICS
			if (checkedCtr > dConfig.generalCfg.atomicGranularity)
			{
				atomicAdd((int*)dPointers.uCheckedSeeds, dConfig.generalCfg.atomicGranularity);
				checkedCtr -= dConfig.generalCfg.atomicGranularity;
			}
#endif
			continue;
		}
#endif

		//flags->state = DeviceLock;
		memcpy(uOutput, output, dConfig.memSizes.outputSize);
		flags->state = SeedFound;
		pollState = true;

#ifdef DO_ATOMICS
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
#endif
	}
#ifdef DO_ATOMICS
	atomicAdd((int*)dPointers.uCheckedSeeds, checkedCtr);
	atomicAdd((int*)dPointers.uPassedSeeds, passedCtr);
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
	MapConfig mapCfg = { 348, 448, 256, 103, 34, 14, -500, 2000, 10, 1000, true, false, 0, 100 };
	const char* fileName = "wang/minesWang.bin";
	constexpr auto NUMBLOCKS = 128;
	constexpr auto BLOCKSIZE = 64;
	constexpr auto mapMemMult = 1;
	constexpr auto miscMemMult = 6;

	//EXCAVATION SITE
	//MapConfig mapCfg = { 344, 440, 409, 102, 31, 17, -100000, 100000, -100000, 100000, false, false, 1, 100 };
	//const char* fileName = "wang/excavationsiteWang.bin";
	//constexpr auto NUMBLOCKS = 64;
	//constexpr auto BLOCKSIZE = 64;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//SNOWCAVE
	//MapConfig mapCfg = { 440, 560, 512, 153, 30, 20, -100000, 100000, -100000, 100000, false, false, 1, 100 };
	//const char* fileName = "wang/snowcaveWang.bin";
	//constexpr auto NUMBLOCKS = 64;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//CRYPT
	//MapConfig mapCfg = { 282, 342, 717, 204, 26, 35, -100000, 100000, -100000, 100000, false, false, 10, 100 };
	//const char* fileName = "wang/cryptWang.bin";
	//constexpr auto NUMBLOCKS = 32;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	//OVERGROWN CAVERNS
	//MapConfig mapCfg = { 144, 235, 359, 461, 59, 16, -100000, 100000, -100000, 100000, false, false, 15, 100 };
	//const char* fileName = "wang/fungiforestWang.bin";
	//constexpr auto NUMBLOCKS = 16;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto mapMemMult = 3;
	//constexpr auto miscMemMult = 8;

	//HELL
	//MapConfig mapCfg = { 156, 364, 921, 256, 25, 43, -100000, 100000, -100000, 100000, false, true, 0, 100 };
	//const char* fileName = "wang/hellWang.bin";
	//constexpr auto NUMBLOCKS = 32;
	//constexpr auto BLOCKSIZE = 32;
	//constexpr auto mapMemMult = 4;
	//constexpr auto miscMemMult = 10;

	MemSizeConfig memSizes = {
		4096,
		mapMemMult * 3 * mapCfg.map_w * (mapCfg.map_h + 4),
		miscMemMult * mapCfg.map_w * mapCfg.map_h,
		mapCfg.map_w * mapCfg.map_h,
		4096
	};

	GeneralConfig generalCfg = { 0, 1, INT_MAX, 8, 60, 1, 1 };
	SpawnableConfig spawnableCfg = {0, 0, 0, 0, false, false, false, false, true, false, false, false, false, false, false};

	Item iF1[FILTER_OR_COUNT] = { SAMPO, TRUE_ORB };
	Item iF2[FILTER_OR_COUNT] = { MIMIC };
	Material mF1[FILTER_OR_COUNT] = { BRASS };
	Spell sF1[FILTER_OR_COUNT] = { SPELL_REGENERATION_FIELD };
	Spell sF2[FILTER_OR_COUNT] = { SPELL_CASTER_CAST };
	Spell sF3[FILTER_OR_COUNT] = { SPELL_CURSE_WITHER_PROJECTILE };

	ItemFilter iFilters[] = { ItemFilter(iF1, 1), ItemFilter(iF2) };
	MaterialFilter mFilters[] = { MaterialFilter(mF1) };
	SpellFilter sFilters[] = { SpellFilter(sF1, 5), SpellFilter(sF2), SpellFilter(sF3) };

	FilterConfig filterCfg = FilterConfig(false, 1, iFilters, 0, mFilters, 0, sFilters, false, 27);

	StaticPrecheckConfig precheckCfg = {
		{false, URINE},
		{false, SPELL_LIGHT_BULLET, SPELL_DYNAMITE},
		{false, BLOOD},
		{false, AlchemyOrdering::ONLY_CONSUMED, {MUD, WATER, SOIL}, {MUD, WATER, SOIL}},
		{false, {BM_GOLD_VEIN_SUPER, BM_NONE, BM_GOLD_VEIN_SUPER}},
		{false, {FungalShift(SS_FLASK, SD_CHEESE_STATIC, 0, 4), FungalShift(), FungalShift(), FungalShift()}},
		{false, {{PERK_ANGRY_GHOST, false, 0, 3}}},
	};

	size_t freeMem;
	size_t totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	generalCfg.requestedMemory = freeMem;

	printf("memory free: %lli of %lli bytes\n", freeMem, totalMem);

	size_t minMemoryPerThread = memSizes.outputSize * 2 + memSizes.mapDataSize + memSizes.miscMemSize + memSizes.visitedMemSize;
	int numThreads = freeMem / minMemoryPerThread;
	int numBlocks = numThreads / BLOCKSIZE;
	int numBlocksRounded = numBlocks - numBlocks % 8;
	printf("creating %i thread blocks\n", numBlocksRounded);

	return { numBlocksRounded, BLOCKSIZE, memSizes, generalCfg, precheckCfg, mapCfg, spawnableCfg, filterCfg, fileName };
}

AllPointers AllocateMemory(DeviceConfig config)
{

	cudaSetDeviceFlags(cudaDeviceMapHost);

	size_t outputSize = config.NumBlocks * config.BlockSize * config.memSizes.outputSize;
#ifdef DO_WORLDGEN
	size_t tileDataSize = 3 * config.mapCfg.tiles_w * config.mapCfg.tiles_h;
	size_t mapDataSize = config.NumBlocks * config.BlockSize * config.memSizes.mapDataSize;
	size_t miscMemSize = config.NumBlocks * config.BlockSize * config.memSizes.miscMemSize;
	size_t visitedMemSize = config.NumBlocks * config.BlockSize * config.memSizes.visitedMemSize;

	//printf("Memory Usage Statistics:\n");
	//printf("Output: %ziMB  Map data: %ziMB\n", outputSize / 1000000, mapDataSize / 1000000);
	//printf("Misc memory: %ziMB  Visited cells: %ziMB\n", miscMemSize / 1000000, visitedMemSize / 1000000);
	//printf("Total memory: %ziMB\n", (tileDataSize * 2 + outputSize + mapDataSize + miscMemSize + visitedMemSize) / 1000000);
#endif


	//Host
	byte* hOutput;
	byte* hTileData;

	hOutput = (byte*)malloc(outputSize);
	hTileData = (byte*)malloc(3 * config.mapCfg.tiles_w * config.mapCfg.tiles_h);
	std::ifstream source(config.wangPath, std::ios_base::binary);
	source.read((char*)hTileData, 3 * config.mapCfg.tiles_w * config.mapCfg.tiles_h);
	source.close();

	//Device
	byte* dArena;
	byte* dTileData;
	byte* dTileSet;

#ifdef DO_WORLDGEN
	checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
	checkCudaErrors(cudaMalloc(&dTileSet, tileDataSize));

	checkCudaErrors(cudaMemcpy(dTileData, hTileData, 3 * config.mapCfg.tiles_w * config.mapCfg.tiles_h, cudaMemcpyHostToDevice));
	buildTS<<<1, 1>>>(dTileData, dTileSet, config.mapCfg.tiles_w, config.mapCfg.tiles_h);
	checkCudaErrors(cudaDeviceSynchronize());
#endif

	checkCudaErrors(cudaMalloc(&dArena, config.generalCfg.requestedMemory - 2 * tileDataSize));

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

	return { {dCheckedSeeds, dPassedSeeds, dFlags, dUnifiedOutput, dTileData, dTileSet, dArena}, {hCheckedSeeds, hPassedSeeds, hFlags, hUnifiedOutput, hOutput, hTileData} };
}

void OutputLoop(DeviceConfig config, HostPointers pointers, cudaEvent_t _event, ofstream& outputStream)
{
	chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
	int intervals = 0;
	int counter = 0;
	uint lastDiff = 0;
	uint lastSeed = 0;

	uint currentSeed = config.generalCfg.startSeed;
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
				if (intervals * config.generalCfg.printInterval * 1000 < milliseconds)
				{
					lastDiff = *pointers.hCheckedSeeds - lastSeed;
					lastSeed = *pointers.hCheckedSeeds;
					intervals++;
					float percentComplete = ((float)(*pointers.hCheckedSeeds) / (config.generalCfg.endSeed - config.generalCfg.startSeed));
					size_t freeMem;
					size_t totalMem;
					cudaMemGetInfo(&freeMem, &totalMem);
					printf(">%i: %2.3f%% complete. Searched %i (+%i this interval), found %i valid seeds. (%lli/%lliMB used)\n", intervals, percentComplete * 100, *pointers.hCheckedSeeds, lastDiff, *pointers.hPassedSeeds, (totalMem - freeMem) / 1MB, totalMem / 1MB);
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


			char buffer[100];
			int bufOffset = 0;
			int memOffset = 0;
			int seed = readInt(output, memOffset);
			int sCount = readInt(output, memOffset);
			_itoa_offset(seed, 10, buffer, bufOffset);
			_putstr_offset(": ", buffer, bufOffset);

			for (int i = 0; i < sCount; i++)
			{
				int x = readInt(output, memOffset);
				int y = readInt(output, memOffset);
				Item item = (Item)readByte(output, memOffset);

				_putstr_offset(item == TRUE_ORB ? "ORB" : "SAMPO", buffer, bufOffset);
				buffer[bufOffset++] = '(';
				_itoa_offset(x, 10, buffer, bufOffset);
				buffer[bufOffset++] = ' ';
				_itoa_offset(y, 10, buffer, bufOffset);
				buffer[bufOffset++] = ')';
				if(i < sCount - 1)
					buffer[bufOffset++] = ' ';
			}

			buffer[bufOffset++] = '\n';
			outputStream.write(buffer, bufOffset);
			foundSeeds++;
		}
	}
}

void FreeMemory(AllPointers pointers)
{
	//free((void*)pointers.hPointers.hCheckedSeeds);
	//free((void*)pointers.hPointers.hPassedSeeds);
	//free((void*)pointers.hPointers.hFlags);
	//free((void*)pointers.hPointers.uOutput);
	free(pointers.hPointers.hOutput);
	free(pointers.hPointers.hTileData);


	checkCudaErrors(cudaFree((void*)pointers.dPointers.dArena));
#ifdef DO_WORLDGEN
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dTileData));
	checkCudaErrors(cudaFree((void*)pointers.dPointers.dTileSet));
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