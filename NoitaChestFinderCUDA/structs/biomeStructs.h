#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "primitives.h"
#include "enums.h"
#include "spawnableStructs.h"

#include "../Configuration.h"
#include "../WorldgenSearch.h"

struct BiomeData
{
	BiomeWands wandTiers;
	PixelSceneList pixel_scenes_01;
	PixelSceneList pixel_scenes_02;
	PixelSceneList pixel_scenes_03;
	EnemyList smallEnemies;
	EnemyList bigEnemies;

	__universal__ constexpr BiomeData() {};

	__universal__ constexpr BiomeData(BiomeWands _tiers, PixelSceneList _01, PixelSceneList _02, PixelSceneList _03, EnemyList _small, EnemyList _big)
		: wandTiers(_tiers), pixel_scenes_01(_01), pixel_scenes_02(_02), pixel_scenes_03(_03), smallEnemies(_small), bigEnemies(_big) {}
};

__device__ BiomeData* AllBiomeData;
__device__ void(*BiomeFnPtrs[30])(SpawnParams params);

struct BiomeSector
{
	Biome b;

	int worldX;
	int worldY;
	int worldW;
	int worldH;
	uint32_t tiles_w;
	uint32_t tiles_h;
	uint32_t map_w;
	uint32_t map_h;
};

struct SpawnParams
{
	int seed;
	BiomeSector currentSector;
	SpawnableConfig sCfg;
	uint8_t* bytes;
	int& offset;
	int& sCount;
};

__device__ void(*spawnPixelScene01)(int x, int y, SpawnParams params);
__device__ void(*spawnPixelScene02)(int x, int y, SpawnParams params);
__device__ void(*spawnPixelScene03)(int x, int y, SpawnParams params);
__device__ void(*spawnSmallEnemies)(int x, int y, SpawnParams params);
__device__ void(*spawnBigEnemies)(int x, int y, SpawnParams params);
__device__ bool(*spawnItem)(int x, int y, SpawnParams params);