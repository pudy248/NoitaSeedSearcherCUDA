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
};

struct MemBlockSizes
{
	size_t outputSize;
	size_t mapDataSize;
	size_t miscMemSize;
	size_t visitedMemSize;
};

__global__ void Kernel(byte* outputBlock, byte* dMapData, byte* dMiscMem, byte* dVisitedMem, MemBlockSizes memSizes, GlobalConfig globalCfg, PrecheckConfig precheckCfg, WorldgenConfig worldCfg, LootConfig lootCfg, FilterConfig filterCfg, int* checkedSeeds, int* passedSeeds)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	for (int seed = globalCfg.startSeed + index; seed < globalCfg.endSeed; seed += stride)
	{
#ifdef SEED_OUTPUT
		byte* output = outputBlock + (seed - globalCfg.startSeed) * memSizes.outputSize;
#else
		byte* output = outputBlock + index * memSizes.outputSize;
#endif

		byte* map = dMapData + index * memSizes.mapDataSize;
		byte* miscMem = dMiscMem + index * memSizes.miscMemSize;
		byte* visited = dVisitedMem + index * memSizes.visitedMemSize;
		byte* spawnableMem = miscMem;

		if (!PrecheckSeed(seed, precheckCfg))
		{
			atomicAdd(checkedSeeds, 1);
			continue;
		}

		GenerateMap(seed, output, map, visited, miscMem, worldCfg, globalCfg.startSeed / 5);

		CheckSpawnables(map, seed, spawnableMem, output, worldCfg, lootCfg, memSizes.miscMemSize);

		SpawnableBlock result = ParseSpawnableBlock(spawnableMem, map, output, lootCfg, memSizes.mapDataSize);
		bool seedPassed = SpawnablesPassed(result, filterCfg, true);

		atomicAdd(checkedSeeds, 1);
		if (seedPassed)
			atomicAdd(passedSeeds, 1);
	}
}

/*
__global__ void wandExperiment(int* buckets)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	constexpr int radius = 50000;
	for (int x = -radius + index; x < radius; x += stride) {
		if (x % 1000 == 0) printf("%i done\n", x);
		for (int y = -radius; y < radius; y++) {
			Wand w = GetWandWithLevel(1234, x, y, 11, false, false);
			if (((int)w.capacity) > 26) atomicAdd(&(buckets[(int)w.capacity]), 1);
		}
	}
}
*/

int main()
{
	/*cudaSetDeviceFlags(cudaDeviceMapHost);

	volatile int* h_buckets;
	volatile int* d_buckets;

	cudaHostAlloc((void**)&h_buckets, sizeof(volatile int) * 70, cudaHostAllocMapped);

	cudaHostGetDevicePointer((void**)&d_buckets, (void*)h_buckets, 0);

	for (int i = 0; i < 70; i++) h_buckets[i] = 0;
	wandExperiment<<<256,64>>>((int*)d_buckets);
	cudaDeviceSynchronize();
	for (int i = 0; i < 70; i++) {
		printf("size %i: %i wands\n", i, h_buckets[i]);
	}
	return;*/
	for (int global_iters = 0; global_iters < 1; global_iters++)
	{
		chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

		//MINES
		WorldgenConfig worldCfg = { 348, 448, 256, 103, 34, 14, true, false, 100 };
		const char* fileName = "minesDump.bin";
		constexpr auto NUMBLOCKS = 128;
		constexpr auto BLOCKSIZE = 64;
		constexpr auto biomeIdx = 0;

		//CRYPT
		//WorldgenConfig worldCfg = { 282, 342, 717, 204, 26, 35, false, false, 100 };
		//const char* fileName = "cryptDump.bin";
		//constexpr auto NUMBLOCKS = 64;
		//constexpr auto BLOCKSIZE = 32;
		//constexpr auto biomeIdx = 10;

		//OVERGROWN CAVERNS
		//WorldgenConfig worldCfg = { 144, 235, 359, 461, 59, 16, false, false, 1 };
		//const char* fileName = "fungiforestDump.bin";
		//constexpr auto NUMBLOCKS = 32;
		//constexpr auto BLOCKSIZE = 32;
		//constexpr auto biomeIdx = 15;

		//HELL
		//WorldgenConfig worldCfg = { 156, 364, 921, 256, 25, 43, false, true, 100 };
		//const char* fileName = "hellDump.bin";
		//constexpr auto NUMBLOCKS = 32;
		//constexpr auto BLOCKSIZE = 32;
		//constexpr auto biomeIdx = 0;

		MemBlockSizes memSizes = {
#ifdef SEED_OUTPUT
			3 * worldCfg.map_w * worldCfg.map_h,
#else
			256,
#endif
			3 * 3 * worldCfg.map_w * (worldCfg.map_h + 4),
			8 * worldCfg.map_w * worldCfg.map_h,
			worldCfg.map_w * worldCfg.map_h
		};

		GlobalConfig globalCfg = { 1, INT_MAX, 1 };

		Item iF1[FILTER_OR_COUNT] = { TRUE_ORB, SAMPO };
		Spell sF1[FILTER_OR_COUNT] = { SPELL_GAMMA, SPELL_ALPHA, SPELL_OMEGA };
		Spell sF2[FILTER_OR_COUNT] = { SPELL_EXPLOSIVE_PROJECTILE };
		Spell sF3[FILTER_OR_COUNT] = { SPELL_NUKE_GIGA };

		ItemFilter iFilters[] = { ItemFilter(iF1, 2) };
		Material mFilters[] = { FUNGUS_POWDER };
		SpellFilter sFilters[] = { SpellFilter(sF1), SpellFilter(sF2), SpellFilter(sF3) };

		FilterConfig filterCfg = FilterConfig(true, 0, iFilters, 0, mFilters, 2, sFilters, false, 26);
		LootConfig lootCfg = LootConfig(0, true, false, false, false, false, filterCfg.materialFilterCount > 0, false, biomeIdx, filterCfg.spellFilterCount > 0);

		PrecheckConfig precheckCfg = {
			false,
			false, ACID,
			false, MAGIC_LIQUID_MOVEMENT_FASTER,
			false, {MUD, WATER, SOIL}, {MUD, WATER, SOIL},
			false, true, {FungalShift(SS_ACID_GAS, false, SD_NONE, true)},
			false, {BM_GOLD_VEIN_SUPER, BM_NONE, BM_NONE},
			false, {PERK_PROTECTION_EXPLOSION, PERK_PROTECTION_FIRE, PERK_PROTECTION_RADIOACTIVITY },
			false, filterCfg, lootCfg
		};

		if (precheckCfg.checkBiomeModifiers && !ValidateBiomeModifierConfig(precheckCfg))
		{
			printf("Impossible biome modifier set! Aborting...\n");
			return;
		}

		int sharedMemSize = 0;
#ifdef SEED_OUTPUT
		size_t outputSize = (globalCfg.endSeed - globalCfg.startSeed) * memSizes.outputSize;
#else
		size_t outputSize = NUMBLOCKS * BLOCKSIZE * memSizes.outputSize;
#endif
		size_t tileDataSize = 3 * worldCfg.tiles_w * worldCfg.tiles_h;
		size_t mapDataSize = NUMBLOCKS * BLOCKSIZE * memSizes.mapDataSize;
		size_t miscMemSize = NUMBLOCKS * BLOCKSIZE * memSizes.miscMemSize;
		size_t visitedMemSize = NUMBLOCKS * BLOCKSIZE * memSizes.visitedMemSize;

		byte* output = (byte*)malloc(outputSize);
		byte* tileData = (byte*)malloc(3 * worldCfg.tiles_w * worldCfg.tiles_h);
		std::ifstream source(fileName, std::ios_base::binary);
		source.read((char*)tileData, 3 * worldCfg.tiles_w * worldCfg.tiles_h);
		source.close();

		byte* dTileData;
		byte* dOutput;
		byte* dMapData;
		byte* dMiscMem;
		byte* dVisitedMem;

		//printf("Memory Usage Statistics:\n");
		//printf("Output: %iMB  Map data: %iMB\n", outputSize / 1000000, mapDataSize / 1000000);
		//printf("Misc memory: %iMB  Visited cells: %iMB\n", miscMemSize / 1000000, visitedMemSize / 1000000);
		//printf("Total memory: %iMB\n",(tileDataSize + outputSize + mapDataSize + miscMemSize + visitedMemSize) / 1000000);

		cudaSetDeviceFlags(cudaDeviceMapHost);

		volatile int* h_checkedSeeds, * h_passedSeeds;
		volatile int* d_checkedSeeds, * d_passedSeeds;

		cudaHostAlloc((void**)&h_checkedSeeds, sizeof(volatile int), cudaHostAllocMapped);
		cudaHostAlloc((void**)&h_passedSeeds, sizeof(volatile int), cudaHostAllocMapped);

		cudaHostGetDevicePointer((void**)&d_checkedSeeds, (void*)h_checkedSeeds, 0);
		cudaHostGetDevicePointer((void**)&d_passedSeeds, (void*)h_passedSeeds, 0);

		*h_checkedSeeds = 0;
		*h_passedSeeds = 0;

		checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
		checkCudaErrors(cudaMalloc(&dOutput, outputSize));
		checkCudaErrors(cudaMalloc(&dMapData, mapDataSize));
		checkCudaErrors(cudaMalloc(&dMiscMem, miscMemSize));
		checkCudaErrors(cudaMalloc(&dVisitedMem, visitedMemSize));

		checkCudaErrors(cudaMemcpy(dTileData, tileData, 3 * worldCfg.tiles_w * worldCfg.tiles_h, cudaMemcpyHostToDevice));
		buildTS << <1, 1 >> > (dTileData, worldCfg.tiles_w, worldCfg.tiles_h);
		checkCudaErrors(cudaDeviceSynchronize());

		cudaEvent_t _event;
		checkCudaErrors(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming));
		Kernel << <NUMBLOCKS, BLOCKSIZE, sharedMemSize >> > (dOutput, dMapData, dMiscMem, dVisitedMem, memSizes, globalCfg, precheckCfg, worldCfg, lootCfg, filterCfg, (int*)d_checkedSeeds, (int*)d_passedSeeds);
		checkCudaErrors(cudaEventRecord(_event));

		int intervals = 0;
		if (globalCfg.printInterval > 0)
		{
			while (cudaEventQuery(_event) != cudaSuccess && (*h_checkedSeeds) < (globalCfg.endSeed - globalCfg.startSeed))
			{
				intervals++;
				float percentComplete = ((float)(*h_checkedSeeds) / (globalCfg.endSeed - globalCfg.startSeed));
				printf("Interval %i: %2.4f%% complete (%i seeds), found %i valid seeds.\n", intervals, percentComplete * 100, *h_checkedSeeds, *h_passedSeeds);
				this_thread::sleep_for(chrono::seconds(globalCfg.printInterval));
			}
		}
		checkCudaErrors(cudaDeviceSynchronize());

		printf("Intervals elapsed: %i. Checked %i seeds, found %i valid seeds.\n", intervals, *h_checkedSeeds, *h_passedSeeds);

		freeTS << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(output, dOutput, outputSize, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(dTileData));
		checkCudaErrors(cudaFree(dOutput));
		checkCudaErrors(cudaFree(dMapData));
		checkCudaErrors(cudaFree(dMiscMem));
		checkCudaErrors(cudaFree(dVisitedMem));

		free(tileData);
		chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
		std::chrono::nanoseconds duration = time2 - time1;
		printf("%i ms\n", (int)(duration.count() / 1000000));


		std::ofstream f = ofstream("output.bin", std::ios::binary);
		f.write((char*)output, outputSize);
		f.close();
		free(output);
	}
}