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
#ifdef DO_WORLDGEN
		byte* map = dMapData + index * memSizes.mapDataSize;
		byte* miscMem = dMiscMem + index * memSizes.miscMemSize;
		byte* visited = dVisitedMem + index * memSizes.visitedMemSize;
		byte* spawnableMem = miscMem;
#endif
		if (!PrecheckSeed(seed, precheckCfg))
		{
			atomicAdd(checkedSeeds, 1);
			continue;
		}

		bool seedPassed = true;

#ifdef DO_WORLDGEN
		GenerateMap(seed, output, map, visited, miscMem, worldCfg, globalCfg.startSeed / 5);

		CheckSpawnables(map, seed, spawnableMem, output, worldCfg, lootCfg, memSizes.miscMemSize);
		
		SpawnableBlock result = ParseSpawnableBlock(spawnableMem, map, output, lootCfg, memSizes.mapDataSize);
		seedPassed = SpawnablesPassed(result, filterCfg, true);
#endif

		atomicAdd(checkedSeeds, 1);
		if (seedPassed)
		{
			printf("Seed passed: %i\n", seed);
			atomicAdd(passedSeeds, 1);
		}
	}
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

	for (int global_iters = 0; global_iters < 1; global_iters++)
	{
		chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

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
#ifdef SEED_OUTPUT
			3 * worldCfg.map_w * worldCfg.map_h,
#else
			256,
#endif
			mapMemMult * 3 * worldCfg.map_w * (worldCfg.map_h + 4),
			miscMemMult * worldCfg.map_w * worldCfg.map_h,
			worldCfg.map_w * worldCfg.map_h
		};

		GlobalConfig globalCfg = { 1, INT_MAX, 5 };

		Item iF1[FILTER_OR_COUNT] = { PAHA_SILMA };
		Item iF2[FILTER_OR_COUNT] = { MIMIC };
		Material mF1[FILTER_OR_COUNT] = { BRASS };
		Spell sF1[FILTER_OR_COUNT] = { SPELL_OMEGA };
		Spell sF2[FILTER_OR_COUNT] = { SPELL_BLACK_HOLE_DEATH_TRIGGER, SPELL_BLACK_HOLE };
		//Spell sF3[FILTER_OR_COUNT] = { SPELL_BLACK_HOLE };

		ItemFilter iFilters[] = { ItemFilter(iF1, 4), ItemFilter(iF2) };
		MaterialFilter mFilters[] = { MaterialFilter(mF1) };
		SpellFilter sFilters[] = { SpellFilter(sF1), SpellFilter(sF2) };

		FilterConfig filterCfg = FilterConfig(true, 1, iFilters, 0, mFilters, 0, sFilters, false, 36);
		LootConfig lootCfg = LootConfig(0, 0, true, false, false, false, false, filterCfg.materialFilterCount > 0, false, biomeIdx, false);

		PrecheckConfig precheckCfg = {
			false,
			false, WATER,
			false, WATER,
			false, AlchemyOrdering::ONLY_CONSUMED, {MUD, WATER, SOIL}, {MUD, WATER, SOIL},
			false, {FungalShift(SS_ACID_GAS, SD_NONE, 0, 1)},
			false, {BM_GOLD_VEIN_SUPER, BM_NONE, BM_NONE},
			false, {{PERK_PERKS_LOTTERY, true, 0, 3}, {PERK_UNLIMITED_SPELLS, false, 0, 6}, {PERK_EDIT_WANDS_EVERYWHERE, false, 0, 3}, {PERK_PROTECTION_EXPLOSION, false, 0, 6}, {PERK_NO_MORE_SHUFFLE, false, 0, 6}},
			false, filterCfg, lootCfg,
			false, true, 0, 6
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

		printf("Memory Usage Statistics:\n");
		printf("Output: %ziMB  Map data: %ziMB\n", outputSize / 1000000, mapDataSize / 1000000);
		printf("Misc memory: %ziMB  Visited cells: %ziMB\n", miscMemSize / 1000000, visitedMemSize / 1000000);
		printf("Total memory: %ziMB\n",(tileDataSize + outputSize + mapDataSize + miscMemSize + visitedMemSize) / 1000000);

		cudaSetDeviceFlags(cudaDeviceMapHost);

		volatile uint* h_checkedSeeds, * h_passedSeeds;
		volatile uint* d_checkedSeeds, * d_passedSeeds;

		cudaHostAlloc((void**)&h_checkedSeeds, sizeof(volatile uint), cudaHostAllocMapped);
		cudaHostAlloc((void**)&h_passedSeeds, sizeof(volatile uint), cudaHostAllocMapped);

		cudaHostGetDevicePointer((void**)&d_checkedSeeds, (void*)h_checkedSeeds, 0);
		cudaHostGetDevicePointer((void**)&d_passedSeeds, (void*)h_passedSeeds, 0);

		*h_checkedSeeds = 0;
		*h_passedSeeds = 0;

		checkCudaErrors(cudaMalloc(&dOutput, outputSize));
#ifdef DO_WORLDGEN
		checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
		checkCudaErrors(cudaMalloc(&dMapData, mapDataSize));
		checkCudaErrors(cudaMalloc(&dMiscMem, miscMemSize));
		checkCudaErrors(cudaMalloc(&dVisitedMem, visitedMemSize));

		checkCudaErrors(cudaMemcpy(dTileData, tileData, 3 * worldCfg.tiles_w * worldCfg.tiles_h, cudaMemcpyHostToDevice));
		buildTS << <1, 1 >> > (dTileData, worldCfg.tiles_w, worldCfg.tiles_h);
		checkCudaErrors(cudaDeviceSynchronize());
#endif

		cudaEvent_t _event;
		checkCudaErrors(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming));
		Kernel << <NUMBLOCKS, BLOCKSIZE, sharedMemSize >> > (dOutput, dMapData, dMiscMem, dVisitedMem, memSizes, globalCfg, precheckCfg, worldCfg, lootCfg, filterCfg, (int*)d_checkedSeeds, (int*)d_passedSeeds);
		checkCudaErrors(cudaEventRecord(_event));

		int intervals = 0;
		if (globalCfg.printInterval > 0)
		{
			while (cudaEventQuery(_event) == cudaErrorNotReady && (*h_checkedSeeds) < (globalCfg.endSeed - globalCfg.startSeed))
			{
				intervals++;
				float percentComplete = ((float)(*h_checkedSeeds) / (globalCfg.endSeed - globalCfg.startSeed));
				printf("Interval %i: %2.4f%% complete (%i seeds), found %i valid seeds.\n", intervals, percentComplete * 100, *h_checkedSeeds, *h_passedSeeds);
				this_thread::sleep_for(chrono::seconds(globalCfg.printInterval));
			}
		}
		checkCudaErrors(cudaDeviceSynchronize());

		freeTS << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(output, dOutput, outputSize, cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(dOutput));

#ifdef DO_WORLDGEN
		checkCudaErrors(cudaFree(dTileData));
		checkCudaErrors(cudaFree(dMapData));
		checkCudaErrors(cudaFree(dMiscMem));
		checkCudaErrors(cudaFree(dVisitedMem));
#endif

		free(tileData);
		chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
		std::chrono::nanoseconds duration = time2 - time1;

		printf("Intervals elapsed: %i (%ims). Checked %i seeds, found %i valid seeds.\n", intervals, (int)(duration.count() / 1000000), *h_checkedSeeds, *h_passedSeeds);


		std::ofstream f = ofstream("output.bin", std::ios::binary);
		f.write((char*)output, outputSize);
		f.close();
		free(output);
	}
}