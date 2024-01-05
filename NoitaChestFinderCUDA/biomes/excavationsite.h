#pragma once
#include "../platforms/IPlatform.h"

#include "../structs/biomeStructs.h"
#include "../structs/spawnableStructs.h"
#include "../WorldgenSearch.h"

//TODO FINISH!!!

namespace FUNCS_EXCAVATIONSITE
{
	__compute__ void spawn_pixel_scene_01(int x, int y, SpawnParams params)
	{
		LoadPixelScene(x, y, AllBiomeData[params.currentSector.b].pixel_scenes_01, params);
	}
	__compute__ void spawn_pixel_scene_02(int x, int y, SpawnParams params)
	{
		LoadPixelScene(x, y, AllBiomeData[params.currentSector.b].pixel_scenes_02, params);
	}
	__compute__ void spawn_pixel_scene_03(int x, int y, SpawnParams params)
	{

	}

	__compute__ void spawn_small_enemies(int x, int y, SpawnParams params)
	{
		int topY = GetGlobalPos(params.currentSector.worldX, params.currentSector.worldY, 0, 0).y;
		float verticalPercent = (float)(y - topY) / (params.currentSector.map_h * 10);
		float spawnPercent = 2.5f * verticalPercent + 0.35f;
		NollaPRNG random(params.seed);
		if (random.ProceduralRandomf(x, y, 0, 1) <= spawnPercent)
			SpawnEnemies(x, y, AllBiomeData[params.currentSector.b].smallEnemies, params);
	}
	__compute__ void spawn_big_enemies(int x, int y, SpawnParams params)
	{
		int topY = GetGlobalPos(params.currentSector.worldX, params.currentSector.worldY, 0, 0).y;
		float verticalPercent = (float)(y - topY) / (params.currentSector.map_h * 10);
		float spawnPercent = 2.1f * verticalPercent;
		NollaPRNG random(params.seed);
		if (random.ProceduralRandomf(x, y, 0, 1) <= spawnPercent)
			SpawnEnemies(x, y, AllBiomeData[params.currentSector.b].bigEnemies, params);
	}
	__compute__ bool spawn_item(int x, int y, SpawnParams params)
	{
		NollaPRNG random(params.seed);
		float r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
		return r > 0.725;
	}

	__compute__ void SetFunctionPointers(SpawnParams params)
	{
		spawnPixelScene01 = spawn_pixel_scene_01;
		spawnPixelScene02 = spawn_pixel_scene_02;
		spawnPixelScene03 = spawn_pixel_scene_03;
		spawnSmallEnemies = spawn_small_enemies;
		spawnBigEnemies = spawn_big_enemies;
		spawnItem = spawn_item;
	}
};

const BiomeData DAT_EXCAVATIONSITE(
	BiomeWands(
		3,
		{{2, WAND_T1NS},
		{2, WAND_T2},
		{2, WAND_T2B}}
	),
	PixelSceneList(
		6,
		{
			PixelSceneData(PS_COALMINE_COALPIT01, 0.5f),
			PixelSceneData(PS_COALMINE_COALPIT02, 0.5f, {PixelSceneSpawn(PSST_SpawnChest, 94, 224),}),
			PixelSceneData(PS_COALMINE_CARTHILL, 0.5f, {PixelSceneSpawn(PSST_LargeEnemy, 15, 169),PixelSceneSpawn(PSST_SmallEnemy, 75, 226),}),
			PixelSceneData(PS_COALMINE_COALPIT03, 0.5f),
			PixelSceneData(PS_COALMINE_COALPIT04, 0.5f),
			PixelSceneData(PS_COALMINE_COALPIT05, 0.5f, {PixelSceneSpawn(PSST_SpawnHeart, 66, 215),}),
		}
	),
	PixelSceneList(
		9,
		{
			PixelSceneData(PS_COALMINE_SHRINE01_ALT, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 106, 97),PixelSceneSpawn(PSST_LargeEnemy, 122, 99),PixelSceneSpawn(PSST_SmallEnemy, 132, 100),PixelSceneSpawn(PSST_SpawnHeart, 133, 86),PixelSceneSpawn(PSST_LargeEnemy, 142, 99),PixelSceneSpawn(PSST_SmallEnemy, 153, 99),PixelSceneSpawn(PSST_LargeEnemy, 165, 100),PixelSceneSpawn(PSST_SmallEnemy, 176, 98),}),
			PixelSceneData(PS_COALMINE_SHRINE02_ALT, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 66, 103),PixelSceneSpawn(PSST_SmallEnemy, 195, 102),}),
			PixelSceneData(PS_COALMINE_SWARM_ALT, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 111, 49),PixelSceneSpawn(PSST_SmallEnemy, 115, 104),PixelSceneSpawn(PSST_SmallEnemy, 127, 47),PixelSceneSpawn(PSST_SpawnItem, 140, 51),PixelSceneSpawn(PSST_SmallEnemy, 146, 102),PixelSceneSpawn(PSST_SmallEnemy, 155, 43),PixelSceneSpawn(PSST_SmallEnemy, 169, 94),PixelSceneSpawn(PSST_SmallEnemy, 171, 45),}),
			PixelSceneData(PS_COALMINE_SYMBOLROOM_ALT, 1.2f, {PixelSceneSpawn(PSST_SmallEnemy, 24, 111),PixelSceneSpawn(PSST_SmallEnemy, 53, 106),PixelSceneSpawn(PSST_SmallEnemy, 101, 95),PixelSceneSpawn(PSST_SmallEnemy, 163, 95),PixelSceneSpawn(PSST_SmallEnemy, 209, 99),}),
			PixelSceneData(PS_COALMINE_PHYSICS_01_ALT, 1.2f, {PixelSceneSpawn(PSST_LargeEnemy, 169, 105),PixelSceneSpawn(PSST_SmallEnemy, 185, 104),PixelSceneSpawn(PSST_SpawnItem, 200, 107),PixelSceneSpawn(PSST_SmallEnemy, 201, 96),PixelSceneSpawn(PSST_SmallEnemy, 210, 106),PixelSceneSpawn(PSST_LargeEnemy, 225, 107),PixelSceneSpawn(PSST_SmallEnemy, 235, 100),}),
			PixelSceneData(PS_COALMINE_PHYSICS_02_ALT, 1.2f, {PixelSceneSpawn(PSST_SmallEnemy, 24, 100),PixelSceneSpawn(PSST_LargeEnemy, 34, 107),PixelSceneSpawn(PSST_SmallEnemy, 49, 106),PixelSceneSpawn(PSST_SmallEnemy, 58, 96),PixelSceneSpawn(PSST_SpawnItem, 59, 107),PixelSceneSpawn(PSST_SmallEnemy, 74, 104),PixelSceneSpawn(PSST_LargeEnemy, 90, 105),}),
			PixelSceneData(PS_COALMINE_PHYSICS_03_ALT, 1.2f, {PixelSceneSpawn(PSST_LargeEnemy, 54, 105),PixelSceneSpawn(PSST_LargeEnemy, 95, 108),PixelSceneSpawn(PSST_LargeEnemy, 130, 108),PixelSceneSpawn(PSST_SpawnHeart, 140, 113),PixelSceneSpawn(PSST_LargeEnemy, 167, 111),PixelSceneSpawn(PSST_LargeEnemy, 203, 108),}),
			PixelSceneData(PS_COALMINE_SHOP_ALT, 0.75f),
			PixelSceneData(PS_COALMINE_RADIOACTIVECAVE, 0.5f, {PixelSceneSpawn(PSST_SpawnHeart, 130, 44),}),
		}
	),
	PixelSceneList(),

	EnemyList(),
	EnemyList()
);