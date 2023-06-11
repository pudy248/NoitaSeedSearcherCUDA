#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../structs/enums.h"
#include "../structs/spawnableStructs.h"

__device__ PixelSceneList coalmine_pixel_scenes_01 = { 0, 0, {} };
__device__ PixelSceneList pixel_scenes_02 = { 0, 0, {} };
__device__ PixelSceneList pixel_scenes_oiltank = { 0, 0, {} };

__device__ void fCOALPIT02(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnChest(x + 94, y + 224, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fCARTHILL(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnBigEnemies(x + 15, y + 169, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 75, y + 226, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fCOALPIT05(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 132, y + 125, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fLABORATORY(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 132, y + 125, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fOILTANK1(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 65, y + 236, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fOILTANK2(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 65, y + 236, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fOILTANK5(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 65, y + 236, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fOILTANK_ALT(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 30, y + 124, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fPHYSICS01(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnBigEnemies(x + 169, y + 105, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 185, y + 104, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.biomeChests) spawnHeart(x + 200, y + 107, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 201, y + 96, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 210, y + 106, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 225, y + 107, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 235, y + 100, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fPHYSICS02(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnSmallEnemies(x + 24, y + 100, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 34, y + 107, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 49, y + 106, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 58, y + 96, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.biomeChests) spawnHeart(x + 59, y + 107, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 74, y + 104, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 90, y + 105, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fPHYSICS03(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnBigEnemies(x + 54, y + 105, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 95, y + 108, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 130, y + 108, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.biomeChests) spawnHeart(x + 140, y + 113, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 167, y + 111, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 203, y + 108, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fRADIOACTIVECAVE(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 130, y + 44, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fRECEPTACLE_OIL(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnSmallEnemies(x + 26, y + 106, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 71, y + 113, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 238, y + 109, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fSHRINE01(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnSmallEnemies(x + 106, y + 97, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 122, y + 99, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 132, y + 100, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.biomeChests) spawnHeart(x + 133, y + 86, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 142, y + 99, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 153, y + 99, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnBigEnemies(x + 165, y + 100, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 176, y + 98, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fSHRINE02(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnSmallEnemies(x + 66, y + 103, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 195, y + 102, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fSLIMEPIT(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.biomeChests) spawnHeart(x + 127, y + 117, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.biomeChests) spawnHeart(x + 133, y + 107, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fSWARM(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnSmallEnemies(x + 111, y + 49, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 115, y + 104, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 127, y + 47, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.biomeChests) spawnHeart(x + 140, y + 51, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 146, y + 102, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 155, y + 43, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 169, y + 94, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 171, y + 45, seed, mCfg, sCfg, output, offset, sCount);
}
__device__ void fSYMBOLROOM(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount)
{
	if(sCfg.enemies) spawnSmallEnemies(x + 53, y + 106, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 101, y + 95, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 163, y + 95, seed, mCfg, sCfg, output, offset, sCount);
	if(sCfg.enemies) spawnSmallEnemies(x + 209, y + 99, seed, mCfg, sCfg, output, offset, sCount);
}

__global__ void InitPixelScenes()
{
	coalmine_pixel_scenes_01 =
	{
		6,
		3,
		200,
	{
		PixelSceneData(PS_COALPIT01, 0.5f),
		PixelSceneData(PS_COALPIT02, 0.5f, fCOALPIT02),
		PixelSceneData(PS_CARTHILL, 0.5f, fCARTHILL),
		PixelSceneData(PS_COALPIT03, 0.5f),
		PixelSceneData(PS_COALPIT04, 0.5f),
		PixelSceneData(PS_COALPIT05, 0.5f, fCOALPIT05),
	}
	};
	pixel_scenes_02 =
	{
		17,
		10.5f,
		100,
	{
		PixelSceneData(PS_SHRINE01, 0.5f, fSHRINE01),
		PixelSceneData(PS_SHRINE02, 0.5f, fSHRINE02),
		PixelSceneData(PS_SLIMEPIT, 0.5f, fSLIMEPIT),
		PixelSceneData(PS_LABORATORY, 0.5f, fLABORATORY),
		PixelSceneData(PS_SWARM, 0.5f, fSWARM),
		PixelSceneData(PS_SYMBOLROOM, 0.5f, fSYMBOLROOM),
		PixelSceneData(PS_PHYSICS_01, 0.5f, fPHYSICS01),
		PixelSceneData(PS_PHYSICS_02, 0.5f, fPHYSICS02),
		PixelSceneData(PS_PHYSICS_03, 0.5f, fPHYSICS03),
		PixelSceneData(PS_SHOP, 1.5f),
		PixelSceneData(PS_RADIOACTIVECAVE, 0.5f, fRADIOACTIVECAVE),
		PixelSceneData(PS_WANDTRAP_H_02, 0.75f),
		PixelSceneData(PS_WANDTRAP_H_04, 0.75f, {OIL, ALCOHOL, GUNPOWDER_EXPLOSIVE}),
		PixelSceneData(PS_WANDTRAP_H_06, 0.75f, {MAGIC_LIQUID_TELEPORTATION, MAGIC_LIQUID_POLYMORPH, MAGIC_LIQUID_RANDOM_POLYMORPH, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_WANDTRAP_H_07, 0.75f, {WATER, OIL, ALCOHOL, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_PHYSICS_SWING_PUZZLE, 0.5f),
		PixelSceneData(PS_RECEPTACLE_OIL, 0.5f, fRECEPTACLE_OIL),
	}
	};
	pixel_scenes_oiltank =
	{
		8,
		5.0604f,
		200,
	{
		PixelSceneData(PS_OILTANK_1, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, SAND, COAL, RADIOACTIVE_LIQUID}, fOILTANK1),
		PixelSceneData(PS_OILTANK_1, 0.0004f, {MAGIC_LIQUID_TELEPORTATION, MAGIC_LIQUID_POLYMORPH, MAGIC_LIQUID_RANDOM_POLYMORPH, MAGIC_LIQUID_BERSERK, MAGIC_LIQUID_CHARM,
			MAGIC_LIQUID_INVISIBILITY, MAGIC_LIQUID_HP_REGENERATION, SALT, BLOOD, GOLD, HONEY}),
		PixelSceneData(PS_OILTANK_2, 0.01f, {BLOOD_FUNGI, BLOOD_COLD, LAVA, POISON, SLIME, GUNPOWDER_EXPLOSIVE, SOIL, SALT, BLOOD, CEMENT}, fOILTANK2),
		PixelSceneData(PS_OILTANK_2, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, OIL, COAL, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_OILTANK_3, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, WATER, COAL, RADIOACTIVE_LIQUID, MAGIC_LIQUID_TELEPORTATION}),
		PixelSceneData(PS_OILTANK_4, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, SAND, COAL, RADIOACTIVE_LIQUID, MAGIC_LIQUID_RANDOM_POLYMORPH}),
		PixelSceneData(PS_OILTANK_5, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, RADIOACTIVE_LIQUID, COAL, RADIOACTIVE_LIQUID}, fOILTANK5),
		PixelSceneData(PS_OILTANK_PUZZLE, 0.05f),
	}
	};
}