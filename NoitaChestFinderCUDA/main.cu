﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/error.h"

#include "Precheckers.h"
#include "Worldgen.h"
#include "WorldgenSearch.h"
#include "Filters.h"

#include <iostream>
#include <fstream>
#include <chrono>

struct GlobalConfig {
	uint startSeed;
	uint endSeed;
};

struct MemBlockSizes {
	size_t outputSize;
	size_t mapDataSize;
	size_t miscMemSize;
	size_t visitedMemSize;
};

__global__ void Kernel(byte* outputBlock, byte* dMapData, byte* dMiscMem, byte* dVisitedMem, MemBlockSizes memSizes, GlobalConfig globalCfg, PrecheckConfig precheckCfg, WorldgenConfig worldCfg, LootConfig lootCfg, FilterConfig filterCfg)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	for (int seed = globalCfg.startSeed + index; seed < globalCfg.endSeed; seed += stride) {

		//byte* output = outputBlock + (seed - globalCfg.startSeed) * memSizes.outputSize;
		byte* output = outputBlock + index * memSizes.outputSize;
		byte* map = dMapData + index * memSizes.mapDataSize;
		byte* miscMem = dMiscMem + index * memSizes.miscMemSize;
		byte* visited = dVisitedMem + index * memSizes.visitedMemSize;

		if (!PrecheckSeed(seed, precheckCfg)) continue;

		GenerateMap(seed, output, map, visited, miscMem, worldCfg, globalCfg.startSeed / 5);

		byte* localPtr1 = miscMem;
		CheckSpawnables(map, seed, &localPtr1, output, worldCfg, lootCfg);

		byte* localPtr2 = miscMem;
		SeedSpawnables result = ParseSpawnableBlock(&localPtr2, output, lootCfg);
		for (int i = 0; i < result.count; i++) {
			bool printSpawnable = SpawnablePassed(result.seed, result.spawnables[i], filterCfg);
			
			if (printSpawnable) {
				printf("%i @ (%i, %i) (#%i of %i): T%i, %i bytes: (", result.seed, result.spawnables[i].x, result.spawnables[i].y, i + 1, result.count, result.spawnables[i].sType, result.spawnables[i].count);
				for (int n = 0; n < result.spawnables[i].count; n++)
					if (result.spawnables[i].contents[n] >= SAMPO)
						printf("%x ", result.spawnables[i].contents[n]);
					else if (result.spawnables[i].contents[n] == DATA_MATERIAL) {
						byte* ptr = (byte*)result.spawnables[i].contents + n + 1;
						short m = readShort(&ptr);
						printf("%s ", MaterialNames[m]);
						n += 2;
						continue;
					}
					else {
						int idx = result.spawnables[i].contents[n] - GOLD_NUGGETS;
						printf("%s ", ItemStrings[idx]);
					}
				printf(")\n");
			}
		}
		freeSeedSpawnables(result);

		if (seed % 10'000'000 == 0) printf("Increment: %i\n", seed);
	}
}

int main() 
{
	for (int global_iters = 0; global_iters < 1; global_iters++) {
		chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

		//MINES
		//WorldgenConfig worldCfg = { 348, 448, 256, 103, 34, 14, true, 100 };
		//const char* fileName = "minesDump.bin";
		//constexpr auto NUMBLOCKS = 128;
		//constexpr auto BLOCKSIZE = 64;

		//CRYPT
		const char* fileName = "cryptDump.bin";
		WorldgenConfig worldCfg = { 282, 342, 717, 204, 26, 35, true, 100 };
		constexpr auto NUMBLOCKS = 64;
		constexpr auto BLOCKSIZE = 32;

		MemBlockSizes memSizes = {
			3 * worldCfg.map_w * worldCfg.map_h,
			3 * worldCfg.map_w * (worldCfg.map_h + 4),
			sizeof(IntPair) * worldCfg.map_w * worldCfg.map_h,
			worldCfg.map_w * worldCfg.map_h
		};

		GlobalConfig globalCfg = { 1, INT_MAX };
		LootConfig lootCfg = LootConfig(2, true, false, true, false, false, false, false, false);

		ItemFilter filters[] = { ItemFilter(WAND_T6NS) };
		Material mFilters[] = { MONSTER_POWDER_TEST };
		Spell sFilters[] = { SPELL_REGENERATION_FIELD, SPELL_GAMMA };

		FilterConfig filterCfg = { 1, filters, 0, mFilters, 0, sFilters, true, 40 };

		PrecheckConfig precheckCfg = {
			false,
			false, ACID,
			false, URINE,
			false, {MUD, WATER, SOIL}, {MUD, WATER, SOIL},
			false, false, {FungalShift(SS_NONE, true, SD_CHEESE_STATIC, false)},
			false, {BM_GOLD_VEIN_SUPER, BM_NONE, BM_NONE},
			false, {PERKS_LOTTERY, IRON_STOMACH, FOOD_CLOCK },
			false, filterCfg, lootCfg };

		if (precheckCfg.checkBiomeModifiers && !ValidateBiomeModifierConfig(precheckCfg)) {
			printf("Impossible biome modifier set! Aborting...\n");
			return;
		}

		size_t tileDataSize = 3 * worldCfg.tiles_w * worldCfg.tiles_h;
		//size_t outputSize = (globalCfg.endSeed - globalCfg.startSeed) * memSizes.outputSize;
		size_t outputSize = NUMBLOCKS * BLOCKSIZE * memSizes.outputSize;
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

		checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
		checkCudaErrors(cudaMalloc(&dOutput, outputSize));
		checkCudaErrors(cudaMalloc(&dMapData, mapDataSize));
		checkCudaErrors(cudaMalloc(&dMiscMem, miscMemSize));
		checkCudaErrors(cudaMalloc(&dVisitedMem, visitedMemSize));

		checkCudaErrors(cudaMemcpy(dTileData, tileData, 3 * worldCfg.tiles_w * worldCfg.tiles_h, cudaMemcpyHostToDevice));
		buildTS << <1, 1 >> > (dTileData, worldCfg.tiles_w, worldCfg.tiles_h);
		checkCudaErrors(cudaDeviceSynchronize());
		int sharedMemSize = 0;
		//printf("kernel shared mem: %i\n", sharedMemSize);
		Kernel << <NUMBLOCKS, BLOCKSIZE, sharedMemSize >> > (dOutput, dMapData, dMiscMem, dVisitedMem, memSizes, globalCfg, precheckCfg, worldCfg, lootCfg, filterCfg);
		checkCudaErrors(cudaDeviceSynchronize());
		//printf("exit kernel\n");
		freeTS << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());

		//checkCudaErrors(cudaMemcpy(output + 3 * map_w * map_h * (globalCfg.startSeed / 5), dOutput + 3 * map_w * map_h * (globalCfg.startSeed / 5), 3 * map_w * map_h, cudaMemcpyDeviceToHost));
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