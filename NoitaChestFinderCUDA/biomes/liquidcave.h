#pragma once
#include "../platforms/IPlatform.h"

#include "../structs/biomeStructs.h"
#include "../structs/spawnableStructs.h"
#include "../WorldgenSearch.h"

//TODO FINISH!!!

namespace FUNCS_LIQUIDCAVE
{
	__compute__ void spawn_pixel_scene_01(int x, int y, SpawnParams params)
	{
		LoadPixelScene(x - 5, y - 3, AllBiomeData[params.currentSector.b].pixel_scenes_01, params);
	}
	__compute__ void spawn_pixel_scene_02(int x, int y, SpawnParams params)
	{

	}
	__compute__ void spawn_pixel_scene_03(int x, int y, SpawnParams params)
	{

	}

	__compute__ void spawn_small_enemies(int x, int y, SpawnParams params)
	{
		SpawnEnemies(x, y, AllBiomeData[params.currentSector.b].smallEnemies, params);
	}
	__compute__ void spawn_big_enemies(int x, int y, SpawnParams params)
	{
		SpawnEnemies(x, y, AllBiomeData[params.currentSector.b].bigEnemies, params);
	}
	__compute__ bool spawn_item(int x, int y, SpawnParams params)
	{
		return false;
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

const BiomeData DAT_LIQUIDCAVE(
	BiomeWands(),
	PixelSceneList(
		1,
		{
			PixelSceneData(PS_LIQUIDCAVE_CONTAINER_01, 0.5f, {OIL, ALCOHOL, LAVA, MAGIC_LIQUID_TELEPORTATION, MAGIC_LIQUID_PROTECTION_ALL, MATERIAL_CONFUSION, LIQUID_FIRE, MAGIC_LIQUID_WEAKNESS}),
		}
	),
	PixelSceneList(),
	PixelSceneList(),

	EnemyList(),
	EnemyList()
);