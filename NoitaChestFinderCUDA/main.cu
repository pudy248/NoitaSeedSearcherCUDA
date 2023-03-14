#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/error.h"

#include "Precheckers.h"
#include "Worldgen.h"
#include "WorldgenSearch.h"

#include <iostream>
#include <fstream>
#include <chrono>

#define NUMBLOCKS 128
#define BLOCKSIZE 64

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

__global__ void Kernel(byte* outputBlock, byte* dMapData, byte* dMiscMem, byte* dVisitedMem, MemBlockSizes memSizes, GlobalConfig globalCfg, PrecheckConfig precheckCfg, WorldgenConfig worldCfg, LootConfig lootCfg)
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
		//else printf("Precheck passed: %i\n", seed);

		GenerateMap(seed, output, map, visited, miscMem, worldCfg, globalCfg.startSeed / 5);

		byte* localPtr1 = miscMem;
		CheckSpawnables(map, seed, &localPtr1, output, worldCfg, lootCfg);

		byte* localPtr2 = miscMem;
		SeedSpawnables result = ParseSpawnableBlock(&localPtr2, output, lootCfg);
		for (int i = 0; i < result.count; i++) {
			Spawnable s = result.spawnables[i];
			//if(s.count > 5) printf("%i @ (%i, %i) (#%i of %i): T%i #%i (0=%i)\n", result.seed, s.x, s.y, i + 1, result.count, s.sType, s.count, s.contents[0]);
			for (int j = 0; j < s.count - 1; j++) {
				if(s.contents[j] == MIMIC_SIGN) {
					if(s.contents[j + 1] == MIMIC_LEGGY || s.contents[j + 1] == HEART_MIMIC)
						printf("%i @ (%i, %i) (#%i of %i): T%i #%i (0=%i)\n", result.seed, s.x, s.y, i + 1, result.count, s.sType, s.count, s.contents[0]);
				}
			}
		}
		freeSeedSpawnables(result);
	}
}

int main() 
{
	chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

	const int tiles_w = 348;
	const int tiles_h = 448;
	const int map_w = 256;
	const int map_h = 103;

	MemBlockSizes memSizes = {
		256,
		3 * map_w * (map_h + 4),
		sizeof(IntPair) * map_w * map_h,
		map_w * map_h
	};

	GlobalConfig globalCfg = { 1, 1000000 };
	PrecheckConfig precheckCfg = {
		false, MATERIAL_NONE,
		false, MATERIAL_NONE,
		false, {}, {},
		false, {FungalShift(GOLD, false, CHEESE_STATIC, false)},
		false, {},
		false, {} };
	WorldgenConfig worldCfg = { tiles_w, tiles_h, map_w, map_h, 34, 14, true, 10 };
	LootConfig lootCfg = { 0, true };

	size_t tileDataSize = 3 * tiles_w * tiles_h;
	//size_t outputSize = (globalCfg.endSeed - globalCfg.startSeed) * memSizes.outputSize;
	size_t outputSize = NUMBLOCKS * BLOCKSIZE * memSizes.outputSize;
	size_t mapDataSize = NUMBLOCKS * BLOCKSIZE * memSizes.mapDataSize;
	size_t miscMemSize = NUMBLOCKS * BLOCKSIZE * memSizes.miscMemSize;
	size_t visitedMemSize = NUMBLOCKS * BLOCKSIZE * memSizes.visitedMemSize;

	byte* output = (byte*)malloc(outputSize);
	byte* tileData = (byte*)malloc(3 * tiles_w * tiles_h);
	std::ifstream source("minesDump.bin", std::ios_base::binary);
	source.read((char*)tileData, 3 * tiles_w * tiles_h);
	source.close();

	byte* dTileData;
	byte* dOutput;
	byte* dMapData;
	byte* dMiscMem;
	byte* dVisitedMem;

	printf("Memory Usage Statistics:\n");
	printf("Output: %iMB  Map data: %iMB\n", outputSize / 1000000, mapDataSize / 1000000);
	printf("Misc memory: %iMB  Visited cells: %iMB\n", miscMemSize / 1000000, visitedMemSize / 1000000);
	printf("Total memory: %iMB\n",(tileDataSize + outputSize + mapDataSize + miscMemSize + visitedMemSize) / 1000000);

	checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
	checkCudaErrors(cudaMalloc(&dOutput, outputSize));
	checkCudaErrors(cudaMalloc(&dMapData, mapDataSize));
	checkCudaErrors(cudaMalloc(&dMiscMem, miscMemSize));
	checkCudaErrors(cudaMalloc(&dVisitedMem, visitedMemSize));

	checkCudaErrors(cudaMemcpy(dTileData, tileData, 3 * tiles_w * tiles_h, cudaMemcpyHostToDevice));
	buildTS << <1, 1 >> > (dTileData, tiles_w, tiles_h);
	checkCudaErrors(cudaDeviceSynchronize());
	int sharedMemSize = 0;
	//printf("kernel shared mem: %i\n", sharedMemSize);
	Kernel << <NUMBLOCKS, BLOCKSIZE, sharedMemSize >> > (dOutput, dMapData, dMiscMem, dVisitedMem, memSizes, globalCfg, precheckCfg, worldCfg, lootCfg);
	checkCudaErrors(cudaDeviceSynchronize());
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