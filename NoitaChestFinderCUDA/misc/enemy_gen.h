#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../structs/enums.h"
#include "../structs/spawnableStructs.h"

#include "../data/enemies.h"

#include "noita_random.h"

#include "../Configuration.h"

__device__ void SpawnEnemies(int x, int y, uint32_t seed, EnemyList list, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	float rnd2 = random.ProceduralRandomf(x, y, 0, list.probSum);

	EnemyData pickedEnemy;
	for (int i = 0; i < list.count; i++)
	{
		if (rnd2 <= list.enemies[i].prob)
		{
			pickedEnemy = list.enemies[i];
			break;
		}
		rnd2 -= list.enemies[i].prob;
	}
	if (pickedEnemy.enemy == ENEMY_NONE) return;
	int enemyCount = random.ProceduralRandomi(x + 6 + (pickedEnemy.enemy == ENEMY_HAMIS ? 4 : 0), y + 5, pickedEnemy.minCount, pickedEnemy.maxCount);

	sCount++;
	writeInt(bytes, offset, x);
	writeInt(bytes, offset, y);
	writeByte(bytes, offset, TYPE_ENEMY);
	writeInt(bytes, offset, enemyCount);
	for(int i = 0; i < enemyCount; i++)
		writeByte(bytes, offset, pickedEnemy.enemy == ENEMY_HAMIS ? WAND_T1 : GOLD_NUGGETS);
	//writeShort(bytes, offset, pickedScene.scene);
	//writeShort(bytes, offset, pickedMat);*/
}

__device__ void spawnSmallEnemies(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	int topY = GetGlobalPos(mCfg.worldX, mCfg.worldY, 0, 0).y;
	float verticalPercent = (float)(y - topY) / (mCfg.map_h * 10);
	float spawnPercent = 2.1f * verticalPercent + 0.2f;
	NollaPRNG random(seed);
	if(random.ProceduralRandomf(x, y, 0, 1) <= spawnPercent)
		SpawnEnemies(x, y, seed, coalmine_small_enemies, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnBigEnemies(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	int topY = GetGlobalPos(mCfg.worldX, mCfg.worldY, 0, 0).y;
	float verticalPercent = (float)(y - topY) / (mCfg.map_h * 10);
	float spawnPercent = 1.75f * verticalPercent - 0.1f;
	NollaPRNG random(seed);
	if (random.ProceduralRandomf(x, y, 0, 1) <= spawnPercent)
		SpawnEnemies(x, y, seed, coalmine_large_enemies, mCfg, sCfg, bytes, offset, sCount);
}
