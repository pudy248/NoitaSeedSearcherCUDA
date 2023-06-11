#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../structs/enums.h"
#include "../structs/spawnableStructs.h"

#include "../data/pixelscenes.h"

#include "noita_random.h"

#include "../Configuration.h"

__device__ void LoadPixelScene(int x, int y, uint32_t seed, PixelSceneList list, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	if (y > mCfg.maxY - list.extraHeightNeeded) return;
	NollaPRNG random = NollaPRNG(seed);
	float rnd2 = random.ProceduralRandomf(x, y, 0, list.probSum);

	PixelSceneData pickedScene;
	Material pickedMat = MATERIAL_NONE;
	for (int i = 0; i < list.count; i++)
	{
		if (rnd2 <= list.scenes[i].prob)
		{
			pickedScene = list.scenes[i];
			break;
		}
		rnd2 -= list.scenes[i].prob;
	}
	if (pickedScene.materialCount > 0)
	{
		int idx = (int)roundf(random.ProceduralRandomf(x + 11, y - 21, 0, pickedScene.materialCount - 1));
		pickedMat = pickedScene.extraMaterials[idx];
	}

	sCount++;
	writeInt(bytes, offset, x);
	writeInt(bytes, offset, y);
	writeByte(bytes, offset, TYPE_PIXEL_SCENE);
	writeInt(bytes, offset, 5);
	writeByte(bytes, offset, DATA_PIXEL_SCENE);
	writeShort(bytes, offset, pickedScene.scene);
	writeShort(bytes, offset, pickedMat);

	if (pickedScene.hasExtraFunction) pickedScene.extraFunction(x, y, seed, pickedScene.scene, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnPixelScene01(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	random.SetRandomSeed(x, y);
	int rnd = random.Random(1, 100);
	if (rnd <= 50)
		LoadPixelScene(x, y, seed, coalmine_pixel_scenes_01, mCfg, sCfg, bytes, offset, sCount);
	else
		LoadPixelScene(x, y, seed, pixel_scenes_oiltank, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnPixelScene02(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	LoadPixelScene(x, y, seed, pixel_scenes_02, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnOilTank(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	random.SetRandomSeed(x, y);
	int rnd = random.Random(1, 100);
	if(rnd > 50)
		LoadPixelScene(x, y, seed, coalmine_pixel_scenes_01, mCfg, sCfg, bytes, offset, sCount);
	else
		LoadPixelScene(x, y, seed, pixel_scenes_oiltank, mCfg, sCfg, bytes, offset, sCount);
}
