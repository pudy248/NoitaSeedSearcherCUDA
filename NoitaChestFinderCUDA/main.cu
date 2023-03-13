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

__global__ void Kernel(byte* outputBlock, byte* dMapData, byte* dMiscMem, byte* dVisitedMem, GlobalConfig globalCfg, PrecheckConfig precheckCfg, WorldgenConfig worldCfg, LootConfig lootCfg)
{
	extern __shared__ int dSharedMem[];

	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	uint stride = blockDim.x * gridDim.x;

	byte* output = outputBlock + index * 3 * worldCfg.map_w * worldCfg.map_h;
	byte* map = dMapData + index * 3 * worldCfg.map_w * (worldCfg.map_h + 4);
	byte* miscMem = dMiscMem + index * sizeof(IntPair) * worldCfg.map_w * worldCfg.map_h;
	byte* visited = dVisitedMem + index * (worldCfg.map_w * worldCfg.map_h + 1);

	//byte** sharedMem = (byte**)dSharedMem + threadIdx.x * sizeof(byte*);

	for (int seed = globalCfg.startSeed + index; seed < globalCfg.endSeed; seed += stride) {
		if (!PrecheckSeed(seed, precheckCfg)) continue;
		printf("starting %i\n", seed);

		GenerateMap(seed, output, map, visited, dMiscMem, worldCfg);

		byte* localPtr1 = miscMem;
		CheckSpawnables(dMiscMem, map, seed, &localPtr1, output, worldCfg, lootCfg);

		byte* localPtr2 = miscMem;
		SeedSpawnables result = ParseSpawnableBlock(&localPtr2, output, lootCfg);
		for (int i = 0; i < result.count; i++) {
			Spawnable s = result.spawnables[i];
			printf("%i @ %i,%i: T%i #%i (0=%i)\n", result.seed, s.x, s.y, s.sType, s.count, s.contents[0]);
			//for (int j = 0; j < s.count; j++) {
			//	if (s.contents[j] == HEART_MIMIC) {
			//		printf("%i @ %i,%i: T%i #%i (0=%i)\n", result.seed, s.x, s.y, s.sType, s.count, s.contents[0]);
			//	}
			//}
		}
		freeSeedSpawnables(result);
	}
}

int main() 
{
	const int tiles_w = 348;
	const int tiles_h = 448;
	const int map_w = 256;
	const int map_h = 103;

	GlobalConfig globalCfg = { 1, 10000 };
	PrecheckConfig precheckCfg = { false, MATERIAL_NONE, false, URINE, false, {}, {}, false, {}, false, {} };
	WorldgenConfig worldCfg = { tiles_w, tiles_h, map_w, map_h, 34, 14, true, 10 };
	LootConfig lootCfg = { 0, 10, true };

	chrono::steady_clock::time_point time1 = chrono::steady_clock::now();

	byte* tileData = (byte*)malloc(3 * tiles_w * tiles_h);
	std::ifstream source("minesDump.bin", std::ios_base::binary);
	source.read((char*)tileData, 3 * tiles_w * tiles_h);
	source.close();

	byte* dTileData;
	byte* dOutput;
	byte* dMapData;
	byte* dMiscMem;
	byte* dVisitedMem;

	checkCudaErrors(cudaMalloc(&dTileData, 3 * tiles_w * tiles_h));
	checkCudaErrors(cudaMalloc(&dOutput, 3 * NUMBLOCKS * BLOCKSIZE * map_w * map_h));
	checkCudaErrors(cudaMalloc(&dMapData, 3 * NUMBLOCKS * BLOCKSIZE * map_w * (map_h + 4)));
	checkCudaErrors(cudaMalloc(&dMiscMem, sizeof(IntPair) * NUMBLOCKS * BLOCKSIZE * map_w * map_h));
	checkCudaErrors(cudaMalloc(&dVisitedMem, NUMBLOCKS * BLOCKSIZE * (map_w * map_h + 1)));

	checkCudaErrors(cudaMemcpy(dTileData, tileData, 3 * tiles_w * tiles_h, cudaMemcpyHostToDevice));
	buildTS<<<1, 1>>> (dTileData, tiles_w, tiles_h);
	checkCudaErrors(cudaDeviceSynchronize());
	int sharedMemSize = 0;
	printf("kernel shared mem: %i\n", sharedMemSize);
	Kernel<<<NUMBLOCKS, BLOCKSIZE, sharedMemSize>>>(dOutput, dMapData, dMiscMem, dVisitedMem, globalCfg, precheckCfg, worldCfg, lootCfg);
	checkCudaErrors(cudaDeviceSynchronize());
	freeTS<<<1, 1>>>();
	checkCudaErrors(cudaDeviceSynchronize());

	byte* output = (byte*)malloc(3 * NUMBLOCKS * BLOCKSIZE * map_w * map_h);
	checkCudaErrors(cudaMemcpy(output, dOutput, 3 * NUMBLOCKS * BLOCKSIZE * map_w * map_h, cudaMemcpyDeviceToHost));
	std::ofstream f = ofstream("output.bin", std::ios::binary);
	f.write((char*)output, 3 * NUMBLOCKS * BLOCKSIZE * map_w * map_h);
	f.close();

	checkCudaErrors(cudaFree(dTileData));
	checkCudaErrors(cudaFree(dOutput));
	checkCudaErrors(cudaFree(dMapData));
	checkCudaErrors(cudaFree(dMiscMem));
	checkCudaErrors(cudaFree(dVisitedMem));

	free(tileData);
	free(output);
	chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
	std::chrono::nanoseconds duration = time2 - time1;
	printf("%i ms\n", (int)(duration.count() / 1000000));
}