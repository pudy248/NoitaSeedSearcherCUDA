#include "../platforms/platform_implementation.h"

#include "../include/search_structs.h"
#include "../include/worldgen_structs.h"
#include "../include/compute.h"
#include "../include/compute.h"


namespace FUNCS_COALMINE
{

	_data BiomeWands wandLevels(
		2,
		{
			WandLevel(17, UNKNOWN_WAND),
			WandLevel(1.9f, WAND_T1)
		}
	);

	_data PixelSceneList pixel_scene_01(
		6,
		{
			PixelSceneData(PS_COALMINE_COALPIT01, 0.5f),
			PixelSceneData(PS_COALMINE_COALPIT02, 0.5f, {PixelSceneSpawn(PSST_SpawnChest, 94, 224),}),
			PixelSceneData(PS_COALMINE_CARTHILL, 0.5f, {PixelSceneSpawn(PSST_LargeEnemy, 15, 169),PixelSceneSpawn(PSST_SmallEnemy, 75, 226),}),
			PixelSceneData(PS_COALMINE_COALPIT03, 0.5f),
			PixelSceneData(PS_COALMINE_COALPIT04, 0.5f),
			PixelSceneData(PS_COALMINE_COALPIT05, 0.5f, {PixelSceneSpawn(PSST_SpawnHeart, 66, 215),}),
		}
	);

	_data PixelSceneList pixel_scene_02(
		17,
		{
				PixelSceneData(PS_COALMINE_SHRINE01, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 106, 97),PixelSceneSpawn(PSST_LargeEnemy, 122, 99),PixelSceneSpawn(PSST_SmallEnemy, 132, 100),PixelSceneSpawn(PSST_SpawnHeart, 133, 86),PixelSceneSpawn(PSST_LargeEnemy, 142, 99),PixelSceneSpawn(PSST_SmallEnemy, 153, 99),PixelSceneSpawn(PSST_LargeEnemy, 165, 100),PixelSceneSpawn(PSST_SmallEnemy, 176, 98),}),
				PixelSceneData(PS_COALMINE_SHRINE02, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 66, 103),PixelSceneSpawn(PSST_SmallEnemy, 195, 102),}),
				PixelSceneData(PS_COALMINE_SLIMEPIT, 0.5f, {PixelSceneSpawn(PSST_SpawnItem, 127, 117),PixelSceneSpawn(PSST_SpawnHeart, 133, 107),}),
				PixelSceneData(PS_COALMINE_LABORATORY, 0.5f, {PixelSceneSpawn(PSST_SpawnItem, 132, 125),}),
				PixelSceneData(PS_COALMINE_SWARM, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 111, 49),PixelSceneSpawn(PSST_SmallEnemy, 115, 104),PixelSceneSpawn(PSST_SmallEnemy, 127, 47),PixelSceneSpawn(PSST_SpawnItem, 140, 51),PixelSceneSpawn(PSST_SmallEnemy, 146, 102),PixelSceneSpawn(PSST_SmallEnemy, 155, 43),PixelSceneSpawn(PSST_SmallEnemy, 169, 94),PixelSceneSpawn(PSST_SmallEnemy, 171, 45),}),
				PixelSceneData(PS_COALMINE_SYMBOLROOM, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 53, 106),PixelSceneSpawn(PSST_SmallEnemy, 101, 95),PixelSceneSpawn(PSST_SmallEnemy, 163, 95),PixelSceneSpawn(PSST_SmallEnemy, 209, 99),}),
				PixelSceneData(PS_COALMINE_PHYSICS_01, 0.5f, {PixelSceneSpawn(PSST_LargeEnemy, 169, 105),PixelSceneSpawn(PSST_SmallEnemy, 185, 104),PixelSceneSpawn(PSST_SpawnItem, 200, 107),PixelSceneSpawn(PSST_SmallEnemy, 201, 96),PixelSceneSpawn(PSST_SmallEnemy, 210, 106),PixelSceneSpawn(PSST_LargeEnemy, 225, 107),PixelSceneSpawn(PSST_SmallEnemy, 235, 100),}),
				PixelSceneData(PS_COALMINE_PHYSICS_02, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 24, 100),PixelSceneSpawn(PSST_LargeEnemy, 34, 107),PixelSceneSpawn(PSST_SmallEnemy, 49, 106),PixelSceneSpawn(PSST_SmallEnemy, 58, 96),PixelSceneSpawn(PSST_SpawnItem, 59, 107),PixelSceneSpawn(PSST_SmallEnemy, 74, 104),PixelSceneSpawn(PSST_LargeEnemy, 90, 105),}),
				PixelSceneData(PS_COALMINE_PHYSICS_03, 0.5f, {PixelSceneSpawn(PSST_LargeEnemy, 54, 105),PixelSceneSpawn(PSST_LargeEnemy, 95, 108),PixelSceneSpawn(PSST_LargeEnemy, 130, 108),PixelSceneSpawn(PSST_SpawnHeart, 140, 113),PixelSceneSpawn(PSST_LargeEnemy, 167, 111),PixelSceneSpawn(PSST_LargeEnemy, 203, 108),}),
				PixelSceneData(PS_COALMINE_SHOP, 1.5f),
				PixelSceneData(PS_COALMINE_RADIOACTIVECAVE, 0.5f, {PixelSceneSpawn(PSST_SpawnHeart, 130, 44),}),
				PixelSceneData(PS_COALMINE_WANDTRAP_H_02, 0.75f),
				PixelSceneData(PS_COALMINE_WANDTRAP_H_04, 0.75f, {OIL,ALCOHOL,GUNPOWDER_EXPLOSIVE,}),
				PixelSceneData(PS_COALMINE_WANDTRAP_H_06, 0.75f, {MAGIC_LIQUID_TELEPORTATION,MAGIC_LIQUID_POLYMORPH,MAGIC_LIQUID_RANDOM_POLYMORPH,RADIOACTIVE_LIQUID,}),
				PixelSceneData(PS_COALMINE_WANDTRAP_H_07, 0.75f, {WATER,OIL,ALCOHOL,RADIOACTIVE_LIQUID,}),
				PixelSceneData(PS_COALMINE_PHYSICS_SWING_PUZZLE, 0.5f),
				PixelSceneData(PS_COALMINE_RECEPTACLE_OIL, 0.5f, {PixelSceneSpawn(PSST_SmallEnemy, 26, 106),PixelSceneSpawn(PSST_SmallEnemy, 71, 113),PixelSceneSpawn(PSST_SmallEnemy, 238, 109),}),
		}
	);

	_data PixelSceneList oiltank(
		8,
		{
			PixelSceneData(PS_COALMINE_OILTANK_1, 1.0f, {WATER,OIL,WATER,OIL,ALCOHOL,SAND,COAL,RADIOACTIVE_LIQUID,}, {PixelSceneSpawn(PSST_SpawnItem, 65, 236),}),
			PixelSceneData(PS_COALMINE_OILTANK_1, 0.0004f, {MAGIC_LIQUID_TELEPORTATION,MAGIC_LIQUID_POLYMORPH,MAGIC_LIQUID_RANDOM_POLYMORPH,MAGIC_LIQUID_BERSERK,MAGIC_LIQUID_CHARM,MAGIC_LIQUID_INVISIBILITY,MAGIC_LIQUID_HP_REGENERATION,SALT,BLOOD,GOLD,HONEY,}, {PixelSceneSpawn(PSST_SpawnItem, 65, 236),}),
			PixelSceneData(PS_COALMINE_OILTANK_2, 0.01f, {BLOOD_FUNGI,BLOOD_COLD,LAVA,POISON,SLIME,GUNPOWDER_EXPLOSIVE,SOIL,SALT,BLOOD,CEMENT,}, {PixelSceneSpawn(PSST_SpawnItem, 65, 236),}),
			PixelSceneData(PS_COALMINE_OILTANK_2, 1.0f, {WATER,OIL,WATER,OIL,ALCOHOL,OIL,COAL,RADIOACTIVE_LIQUID,}, {PixelSceneSpawn(PSST_SpawnItem, 65, 236),}),
			PixelSceneData(PS_COALMINE_OILTANK_3, 1.0f, {WATER,OIL,WATER,OIL,ALCOHOL,WATER,COAL,RADIOACTIVE_LIQUID,MAGIC_LIQUID_TELEPORTATION,}),
			PixelSceneData(PS_COALMINE_OILTANK_4, 1.0f, {WATER,OIL,WATER,OIL,ALCOHOL,SAND,COAL,RADIOACTIVE_LIQUID,MAGIC_LIQUID_POLYMORPH,}),
			PixelSceneData(PS_COALMINE_OILTANK_5, 1.0f, {WATER,OIL,WATER,OIL,ALCOHOL,RADIOACTIVE_LIQUID,COAL,RADIOACTIVE_LIQUID,}, {PixelSceneSpawn(PSST_SpawnItem, 65, 236),}),
			PixelSceneData(PS_COALMINE_OILTANK_PUZZLE, 0.05f),
		}
	);

	_compute void spawn_pixel_scene_01(int x, int y, SpawnParams params)
	{
		if (!params.sCfg->biomePixelScenes) return;
		NollaPRNG random = NollaPRNG(params.seed);
		random.SetRandomSeed(x, y);
		int rnd = random.Random(1, 100);
		if (rnd <= 50)
			LoadPixelScene(x, y, pixel_scene_01, params);
		else
			LoadPixelScene(x, y, oiltank, params);
	}
	_compute void spawn_pixel_scene_02(int x, int y, SpawnParams params)
	{
		if (!params.sCfg->biomePixelScenes) return;
		LoadPixelScene(x, y, pixel_scene_02, params);
	}
	_compute void spawn_oiltank(int x, int y, SpawnParams params)
	{
		if (!params.sCfg->biomePixelScenes) return;
		NollaPRNG random = NollaPRNG(params.seed);
		random.SetRandomSeed(x, y);
		int rnd = random.Random(1, 100);
		if (rnd > 50)
			LoadPixelScene(x, y, pixel_scene_01, params);
		else
			LoadPixelScene(x, y, oiltank, params);
	}

	_compute bool spawn_item(int x, int y, SpawnParams params)
	{
		NollaPRNG random(params.seed);
		float r = random.ProceduralRandomf(x, y, 0, 1);
		if (r < 0.47) return false;
		r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
		return r > 0.755;
	}

	_compute void SetFunctionPointers(SpawnParams& params)
	{
		params.spawnItem = spawn_item;
	}

	_data BiomeSpawnFunctions _CoalmineFuncs = BiomeSpawnFunctions(FUNCS_COALMINE::SetFunctionPointers, {
		SpawnFunction(0xff0aff, FUNCS_COALMINE::spawn_pixel_scene_01),
		SpawnFunction(0xff0080, FUNCS_COALMINE::spawn_pixel_scene_02),
		SpawnFunction(0xc35700, FUNCS_COALMINE::spawn_oiltank),
	});
};

_data static BiomeSpawnFunctions _DefaultFunctions(NULL, {
	SpawnFunction(0x78ffff, spawnHeart),
	SpawnFunction(0x55ff8c, spawnChest),
	SpawnFunction(0x50a000, spawnPotion),
	SpawnFunction(0x00ff00, spawnWand),
	//SpawnFunction(0xff0000, params.spawnSmallEnemies),
	//SpawnFunction(0x800000, params.spawnBigEnemies),
	//SpawnFunction(0x808000, spawnHellShop),
});

_compute void CopySpawnFuncs()
{
	AllSpawnFunctions[0] = &_DefaultFunctions;
	AllSpawnFunctions[B_COALMINE] = &FUNCS_COALMINE::_CoalmineFuncs;
}