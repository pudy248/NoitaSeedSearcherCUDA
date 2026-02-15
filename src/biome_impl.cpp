#include "../platforms/platform_implementation.h"

#include "../include/compute.h"
#include "../include/search_structs.h"
#include "../include/worldgen_structs.h"

// Things to do to add a biome:
// Add entry to Biome enum
// Get pixel scene list defs from data gen mod
// Add pixel scene enums and namelists
// Write biome-specific spawn functions
// Add spawn functions to biome-specific table
// Update SetBiomeData

static BiomeSpawnColors DefaultSpawnColors({0x78ffff, 0x55ff8c});
_data static BiomeSpawnFunctions DefaultSpawnFunctions(NULL, {spawnHeart, spawnChest});

_compute static void load_random_pixel_scene(int x, int y, const SpawnParams& params, int index, int dim) {
	LoadPixelScene(x, y, AllPixelSceneLists[params.currentBiome.bSec.b].lists[index], dim, params);
}

namespace FUNCS_COALMINE {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_COALMINE_COALPIT01, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT02, 0.5f),
		PixelSceneData(PS_COALMINE_CARTHILL, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT03, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT04, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT05, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_COALMINE_SHRINE01, 0.5f),
		PixelSceneData(PS_COALMINE_SHRINE02, 0.5f),
		PixelSceneData(PS_COALMINE_SLIMEPIT, 0.5f),
		PixelSceneData(PS_COALMINE_LABORATORY, 0.5f),
		PixelSceneData(PS_COALMINE_SWARM, 0.5f),
		PixelSceneData(PS_COALMINE_SYMBOLROOM, 0.5f),
		PixelSceneData(PS_COALMINE_PHYSICS_01, 0.5f),
		PixelSceneData(PS_COALMINE_PHYSICS_02, 0.5f),
		PixelSceneData(PS_COALMINE_PHYSICS_03, 0.5f),
		PixelSceneData(PS_COALMINE_SHOP, 1.5f),
		PixelSceneData(PS_COALMINE_RADIOACTIVECAVE, 0.5f),
		PixelSceneData(PS_COALMINE_WANDTRAP_H_02, 0.75f),
		PixelSceneData(PS_COALMINE_WANDTRAP_H_04, 0.75f, {OIL, ALCOHOL, GUNPOWDER_EXPLOSIVE}),
		PixelSceneData(PS_COALMINE_WANDTRAP_H_06, 0.75f,
			{MAGIC_LIQUID_TELEPORTATION, MAGIC_LIQUID_POLYMORPH, MAGIC_LIQUID_RANDOM_POLYMORPH, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_COALMINE_WANDTRAP_H_07, 0.75f, {WATER, OIL, ALCOHOL, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_COALMINE_PHYSICS_SWING_PUZZLE, 0.5f),
		PixelSceneData(PS_COALMINE_RECEPTACLE_OIL, 0.5f),
	}),
	PixelSceneList({
		// g_oiltank
		PixelSceneData(PS_COALMINE_OILTANK_1, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, SAND, COAL, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_COALMINE_OILTANK_1, 0.0004f,
			{MAGIC_LIQUID_TELEPORTATION, MAGIC_LIQUID_POLYMORPH, MAGIC_LIQUID_RANDOM_POLYMORPH, MAGIC_LIQUID_BERSERK,
				MAGIC_LIQUID_CHARM, MAGIC_LIQUID_INVISIBILITY, MAGIC_LIQUID_HP_REGENERATION, SALT, BLOOD, GOLD, HONEY}),
		PixelSceneData(PS_COALMINE_OILTANK_2, 0.01f,
			{BLOOD_FUNGI, BLOOD_COLD, LAVA, POISON, SLIME, GUNPOWDER_EXPLOSIVE, SOIL, SALT, BLOOD, CEMENT}),
		PixelSceneData(PS_COALMINE_OILTANK_2, 1.0f, {WATER, OIL, WATER, OIL, ALCOHOL, OIL, COAL, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_COALMINE_OILTANK_3, 1.0f,
			{WATER, OIL, WATER, OIL, ALCOHOL, WATER, COAL, RADIOACTIVE_LIQUID, MAGIC_LIQUID_TELEPORTATION}),
		PixelSceneData(PS_COALMINE_OILTANK_4, 1.0f,
			{WATER, OIL, WATER, OIL, ALCOHOL, SAND, COAL, RADIOACTIVE_LIQUID, MAGIC_LIQUID_POLYMORPH}),
		PixelSceneData(PS_COALMINE_OILTANK_5, 1.0f,
			{WATER, OIL, WATER, OIL, ALCOHOL, RADIOACTIVE_LIQUID, COAL, RADIOACTIVE_LIQUID}),
		PixelSceneData(PS_COALMINE_OILTANK_PUZZLE, 0.05f),
	}),
	PixelSceneList({
		// g_oiltank_alt
		PixelSceneData(PS_COALMINE_OILTANK_ALT, 1.0f,
			{WATER, OIL, WATER, OIL, ALCOHOL, SAND, RADIOACTIVE_LIQUID, RADIOACTIVE_LIQUID, MAGIC_LIQUID_BERSERK}),
	}),
});

_data BiomeWands wandLevels({WandLevel(17, UNKNOWN_WAND), WandLevel(1.9f, WAND_T1)});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	random.SetRandomSeedInt(x, y);
	int rnd = random.Random(1, 100);
	if (rnd <= 50)
		load_random_pixel_scene(x, y, params, 0, true);
	else
		load_random_pixel_scene(x, y, params, 2, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}
_compute void load_oiltank(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	random.SetRandomSeedInt(x, y);
	int rnd = random.Random(1, 100);
	if (rnd > 50)
		load_random_pixel_scene(x, y, params, 0, true);
	else
		load_random_pixel_scene(x, y, params, 2, true);
}
_compute void load_oiltank_alt(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	random.SetRandomSeedInt(x, y);
	int rnd = random.Random(1, 100);
	if (rnd > 50)
		load_random_pixel_scene(x, y, params, 0, true);
	else
		load_random_pixel_scene(x, y, params, 2, true);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x, y, 0, 1);
	if (r < 0.47)
		return;
	r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
	if (r < 0.755)
		return;
	spawnWand(x + 5, y - 9, params);
}

_compute void spawn_bbqbox(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	random.SetRandomSeed(x, y);
	if (random.Random(1, 100) <= 99) {
		params.sCount++;
		writeInt(params.bytes, params.offset, x + 10);
		writeInt(params.bytes, params.offset, y + 10);
		writeByte(params.bytes, params.offset, TYPE_ITEM_PEDESTAL);
		writeInt(params.bytes, params.offset, 1);
		writeByte(params.bytes, params.offset, HEART_NORMAL);
	} else {
		params.sCount++;
		writeInt(params.bytes, params.offset, x);
		writeInt(params.bytes, params.offset, y);
		writeByte(params.bytes, params.offset, TYPE_ITEM_PEDESTAL);
		writeInt(params.bytes, params.offset, 3);
		writeByte(params.bytes, params.offset, DATA_MATERIAL);
		writeShort(params.bytes, params.offset, URINE);
	}
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080, 0xc35700, 0x4e175e, 0xf12ab5});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02,
												 load_oiltank, load_oiltank_alt, spawn_bbqbox});
};

namespace FUNCS_COALMINE_ALT {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_COALMINE_COALPIT01, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT02, 0.5f),
		PixelSceneData(PS_COALMINE_CARTHILL, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT03, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT04, 0.5f),
		PixelSceneData(PS_COALMINE_COALPIT05, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_COALMINE_SHRINE01_ALT, 0.5f),
		PixelSceneData(PS_COALMINE_SHRINE02_ALT, 0.5f),
		PixelSceneData(PS_COALMINE_SWARM_ALT, 0.5f),
		PixelSceneData(PS_COALMINE_SYMBOLROOM_ALT, 1.2f),
		PixelSceneData(PS_COALMINE_PHYSICS_01_ALT, 1.2f),
		PixelSceneData(PS_COALMINE_PHYSICS_02_ALT, 1.2f),
		PixelSceneData(PS_COALMINE_PHYSICS_03_ALT, 1.2f),
		PixelSceneData(PS_COALMINE_SHOP_ALT, 0.75f),
		PixelSceneData(PS_COALMINE_RADIOACTIVECAVE, 0.5f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(1, UNKNOWN_WAND),
});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x, y, 0, 1);
	if (r < 0.47)
		return;
	r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
	if (r < 0.725)
		return;
	spawnWand(x + 5, y - 9, params);
}
static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02});
};

namespace FUNCS_EXCAVATIONSITE {
BiomePixelScenes Scenes(
	{PixelSceneList({
		 // g_pixel_scene_04
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_1, 0.5f),
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_2, 0.5f),
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_3B, 0.5f),
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_4, 0.5f),
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_5, 0.5f),
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_6, 0.5f),
		 PixelSceneData(PS_EXCAVATIONSITE_MACHINE_7, 0.3f),
		 PixelSceneData(PS_EXCAVATIONSITE_SHOP, 3.0f),
		 PixelSceneData(PS_EXCAVATIONSITE_OILTANK_1, 0.8f),
		 PixelSceneData(PS_EXCAVATIONSITE_LAKE, 0.8f),
	 }),
		PixelSceneList({
			// g_pixel_scene_04_alt
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_1_ALT, 0.5f),
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_2_ALT, 0.5f),
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_3B_ALT, 0.5f),
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_4_ALT, 0.5f),
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_5_ALT, 0.5f),
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_6_ALT, 0.5f),
			PixelSceneData(PS_EXCAVATIONSITE_MACHINE_7_ALT, 0.3f),
			PixelSceneData(PS_EXCAVATIONSITE_SHOP_ALT, 3.0f),
			PixelSceneData(
				PS_EXCAVATIONSITE_RECEPTACLE_STEAM, 0.7f),
			PixelSceneData(PS_EXCAVATIONSITE_LAKE_ALT, 0.8f),
		}),
		PixelSceneList({
			// g_puzzleroom
			PixelSceneData(PS_EXCAVATIONSITE_PUZZLEROOM_01, 1.5f),
			PixelSceneData(PS_EXCAVATIONSITE_PUZZLEROOM_02, 1.5f),
			PixelSceneData(PS_EXCAVATIONSITE_PUZZLEROOM_03, 1.5f),
		}),
		/*PixelSceneList({
			// g_gunpowderpool_01
			PixelSceneData(
				PS_EXCAVATIONSITE_GUNPOWDERPOOL_01, 1.5f),
		}),
		PixelSceneList({
			// g_gunpowderpool_02
			PixelSceneData(
				PS_EXCAVATIONSITE_GUNPOWDERPOOL_02, 1.5f),
		}),
		PixelSceneList({
			// g_gunpowderpool_03
			PixelSceneData(
				PS_EXCAVATIONSITE_GUNPOWDERPOOL_03, 1.5f),
		}),
		PixelSceneList({
			// g_gunpowderpool_04
			PixelSceneData(
				PS_EXCAVATIONSITE_GUNPOWDERPOOL_04, 1.5f),
		}),*/
		PixelSceneList({
			PixelSceneData(PS_EXCAVATIONSITE_MEDITATION_CUBE, 1),
		})});

_data BiomeWands wandLevels({
	WandLevel(2, WAND_T1NS),
	WandLevel(2, WAND_T2),
	WandLevel(2, WAND_T2B),
});

_compute void spawn_pixel_scene_04(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, false);
}
_compute void spawn_pixel_scene_04_alt(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}
_compute void spawn_puzzleroom(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 2, false);
}
/*_compute void spawn_gunpowderpool_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 3);
}
_compute void spawn_gunpowderpool_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 4);
}
_compute void spawn_gunpowderpool_03(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 5);
}
_compute void spawn_gunpowderpool_04(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 6);
}*/
_compute void spawn_meditation_cube(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	NollaPRNG random(params.seed);
	random.SetRandomSeedInt(x, y);
	int rnd = random.Random(1, 100);
	if (rnd > 96)
		load_random_pixel_scene(x - 20, y - 29, params, 3, 2);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
	if (r < 0.725)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors(
	{0x50a000, 0x00ff00, 0x00ac64, 0x00ac6e, 0x7868ff, /*0x70d79e, 0x70d79f, 0x70d7a0, 0x70d7a1,*/ 0xb09016});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_04, spawn_pixel_scene_04_alt,
												 spawn_puzzleroom, /*spawn_gunpowderpool_01,
			  spawn_gunpowderpool_02, spawn_gunpowderpool_03, spawn_gunpowderpool_04,*/
			  spawn_meditation_cube});
};

namespace FUNCS_SNOWCAVE {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_SNOWCAVE_VERTICALOBSERVATORY, 0.5f),
		PixelSceneData(PS_SNOWCAVE_VERTICALOBSERVATORY2, 0.5f),
		PixelSceneData(PS_SNOWCAVE_ICEBRIDGE2, 0.5f),
		PixelSceneData(PS_SNOWCAVE_PIPE, 0.5f),
		PixelSceneData(PS_SNOWCAVE_RECEPTACLE_WATER, 0.25f),
	}),
	PixelSceneList({
		// g_pixel_scene_01_alt
		PixelSceneData(PS_SNOWCAVE_VERTICALOBSERVATORY_ALT, 0.5f),
		PixelSceneData(PS_SNOWCAVE_VERTICALOBSERVATORY2_ALT, 0.5f),
		PixelSceneData(PS_SNOWCAVE_ICEBRIDGE2_ALT, 0.5f),
		PixelSceneData(PS_SNOWCAVE_PIPE_ALT, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_SNOWCAVE_CRATER, 0.4f),
		PixelSceneData(PS_SNOWCAVE_HORIZONTALOBSERVATORY, 0.5f),
		PixelSceneData(PS_SNOWCAVE_HORIZONTALOBSERVATORY2, 0.5f),
		PixelSceneData(PS_SNOWCAVE_HORIZONTALOBSERVATORY3, 0.3f),
		PixelSceneData(PS_SNOWCAVE_ICEBRIDGE, 0.4f),
		PixelSceneData(PS_SNOWCAVE_SNOWCASTLE, 0.4f),
		PixelSceneData(PS_SNOWCAVE_SYMBOLROOM, 0.0f),
		PixelSceneData(PS_SNOWCAVE_ICEPILLAR, 0.5f),
		PixelSceneData(PS_SNOWCAVE_SHOP, 1.5f),
		PixelSceneData(PS_SNOWCAVE_CAMP, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_03
		PixelSceneData(PS_NONE, 0.9f),
		PixelSceneData(PS_SNOWCAVE_TINYOBSERVATORY, 0.5f),
		PixelSceneData(PS_SNOWCAVE_TINYOBSERVATORY2, 0.5f),
		PixelSceneData(PS_SNOWCAVE_BURIED_EYE, 0.2f),
	}),
	/*
	PixelSceneList({
		// g_acidtank_right
		PixelSceneData(PS_NONE, 1.7f),
		PixelSceneData(PS_SNOWCAVE_ACIDTANK_2, 0.2f),
	}),
	PixelSceneList({
		// g_acidtank_left
		PixelSceneData(PS_NONE, 1.7f),
		PixelSceneData(PS_SNOWCAVE_ACIDTANK, 0.2f),
	}),
	PixelSceneList({
		// g_pixel_scene_04
		PixelSceneData(PS_NONE, 0.5f),
		PixelSceneData(PS_SNOWCAVE_ICICLES, 0.5f),
		PixelSceneData(PS_SNOWCAVE_ICICLES2, 0.5f),
		PixelSceneData(PS_SNOWCAVE_ICICLES3, 0.5f),
		PixelSceneData(PS_SNOWCAVE_ICICLES4, 0.5f),
	}),*/
	PixelSceneList({
		// g_puzzle_capsule
		PixelSceneData(PS_NONE, 9.0f),
		PixelSceneData(PS_SNOWCAVE_PUZZLE_CAPSULE, 1.0f),
	}),
	PixelSceneList({
		// g_puzzle_capsule_b
		PixelSceneData(PS_NONE, 9.0f),
		PixelSceneData(PS_SNOWCAVE_PUZZLE_CAPSULE_B, 1.0f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T2),
	WandLevel(5, WAND_T2B),
	WandLevel(5, WAND_T2NS),
});

_compute bool safe(int x, int y) { return !(x >= 125 && x <= 249 && y >= 3070 && y <= 3187); }

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_01_alt(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 2, false);
}
_compute void spawn_pixel_scene_03(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 3, true);
}
_compute void spawn_puzzle_capsule(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 4, false);
}
_compute void spawn_puzzle_capsule_b(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x - 50, y - 230, params, 5, false);
}
/*
_compute void spawn_pixel_scene_04(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 6, false);
}
_compute void spawn_acidtank_right(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	if (safe(x, y))
		load_random_pixel_scene(x - 12, y - 12, params, 4, false);
}
_compute void spawn_acidtank_left(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	if (safe(x, y))
		load_random_pixel_scene(x - 252, y - 12, params, 5, false);
}*/

_compute void spawn_potion(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePedestals)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 6, y - 3, params);
}
_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r >= 0.45)
		return;
	spawnWand(x - 5, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xc800ff, 0xff0080, 0x00ac33, 0x4691c7, 0x3691d7});
_data static BiomeSpawnFunctions Funcs(
	NULL, {spawn_potion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_01_alt, spawn_pixel_scene_02,
			  spawn_pixel_scene_03,
			  spawn_puzzle_capsule, spawn_puzzle_capsule_b});
};

namespace FUNCS_SNOWCASTLE {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_SNOWCASTLE_SHAFT, 0.5f),
		PixelSceneData(PS_SNOWCASTLE_BRIDGE, 0.5f),
		PixelSceneData(PS_SNOWCASTLE_DRILL, 0.5f),
		PixelSceneData(PS_SNOWCASTLE_GREENHOUSE, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_SNOWCASTLE_CARGOBAY, 0.4f),
		PixelSceneData(PS_SNOWCASTLE_BAR, 0.8f),
		PixelSceneData(PS_SNOWCASTLE_BEDROOM, 0.8f),
		PixelSceneData(PS_SNOWCASTLE_ACIDPOOL, 0.4f),
		PixelSceneData(PS_SNOWCASTLE_POLYMORPHROOM, 0.4f),
		PixelSceneData(PS_SNOWCASTLE_TELEROOM, 0.2f),
		PixelSceneData(PS_SNOWCASTLE_SAUNA, 0.3f),
		PixelSceneData(PS_SNOWCASTLE_KITCHEN, 0.3f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T3),
	WandLevel(5, WAND_T3B),
	WandLevel(5, WAND_T3NS),
});

_compute bool safe(int x, int y) { return !(x >= 125 && x <= 249 && y >= 5118 && y <= 5259) && !(y > 6100); }

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	if (safe(x, y))
		load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	if (safe(x, y))
		load_random_pixel_scene(x, y, params, 1, false);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.2)
		return;
	spawnWand(x + 5, y - 5, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02});

};

namespace FUNCS_RAINFOREST {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_RAINFOREST_PIT01, 0.5f),
		PixelSceneData(PS_RAINFOREST_PIT02, 0.5f),
		PixelSceneData(PS_RAINFOREST_PIT03, 0.5f),
		PixelSceneData(PS_RAINFOREST_OILTANK_01, 0.8f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_RAINFOREST_HUT01, 0.5f),
		PixelSceneData(PS_RAINFOREST_HUT02, 0.5f),
		PixelSceneData(PS_RAINFOREST_BASE, 0.4f),
		PixelSceneData(PS_RAINFOREST_HUT03, 0.5f),
		PixelSceneData(PS_RAINFOREST_SYMBOLROOM, 1.2f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T4),
	WandLevel(5, WAND_T5),
	WandLevel(3, WAND_T2NS),
	WandLevel(3, WAND_T3NS),
	WandLevel(5, WAND_T4B),
});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r >= 0.27)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02});
};

namespace FUNCS_RAINFOREST_OPEN {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_RAINFOREST_PIT01, 0.5f),
		PixelSceneData(PS_RAINFOREST_PIT02, 0.5f),
		PixelSceneData(PS_RAINFOREST_PIT03, 0.5f),
		PixelSceneData(PS_RAINFOREST_OILTANK_01, 0.8f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_RAINFOREST_HUT01, 0.5f),
		PixelSceneData(PS_RAINFOREST_HUT02, 0.5f),
		PixelSceneData(PS_RAINFOREST_BASE, 0.4f),
		PixelSceneData(PS_RAINFOREST_HUT03, 0.5f),
		PixelSceneData(PS_RAINFOREST_SYMBOLROOM, 1.2f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T4),
	WandLevel(5, WAND_T5),
	WandLevel(3, WAND_T2NS),
	WandLevel(3, WAND_T3NS),
	WandLevel(5, WAND_T4B),
});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.27)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02});
};

namespace FUNCS_VAULT {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_VAULT_ACIDTANK, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_VAULT_LAB, 0.5f, {RADIOACTIVE_LIQUID, RADIOACTIVE_LIQUID, ACID, ACID, ACID, ALCOHOL}),
		PixelSceneData(PS_VAULT_LAB2, 0.5f, {RADIOACTIVE_LIQUID, RADIOACTIVE_LIQUID, ACID, ACID, ACID, ALCOHOL}),
		PixelSceneData(PS_VAULT_LAB3, 0.5f, {RADIOACTIVE_LIQUID, RADIOACTIVE_LIQUID, ACID, ACID, ACID, ALCOHOL}),
		PixelSceneData(PS_VAULT_SYMBOLROOM, 1.2f),
		PixelSceneData(PS_VAULT_LAB_PUZZLE, 0.3f),
	}),
	PixelSceneList({
		// g_pixel_scene_wide
		PixelSceneData(PS_VAULT_BRAIN_ROOM, 0.5f),
		PixelSceneData(PS_VAULT_SHOP, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_tall
		PixelSceneData(PS_VAULT_ELECTRIC_TUNNEL_ROOM, 0.5f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T5),
	WandLevel(5, WAND_T5B),
	WandLevel(3, WAND_T3NS),
	WandLevel(2, WAND_T4NS),
});

_compute bool safe(int x, int y) { return !(x >= 125 && x <= 249 && y >= 8694 && y <= 8860); }

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}
_compute void spawn_pixel_scene_wide(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 2, false);
}
_compute void spawn_pixel_scene_tall(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 3, true);
}

_compute void spawn_potion(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePedestals)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 6, y - 4, params);
}
_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r >= 0.93)
		return;
	spawnWand(x + 5, y - 6, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080, 0x692e94, 0x822e5b});
_data static BiomeSpawnFunctions Funcs(NULL, {spawn_potion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02,
												 spawn_pixel_scene_wide, spawn_pixel_scene_tall});
};

namespace FUNCS_CRYPT {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_CRYPT_CATHEDRAL, 1.0f),
		PixelSceneData(PS_CRYPT_MINING, 1.0f),
		PixelSceneData(PS_CRYPT_POLYMORPHROOM, 1.0f),
	}),
	PixelSceneList({
		// g_pixel_scene_03
		PixelSceneData(PS_CRYPT_LAVAROOM, 1.0f),
		PixelSceneData(PS_CRYPT_PIT, 1.0f),
		PixelSceneData(PS_CRYPT_SYMBOLROOM, 1.0f),
		PixelSceneData(PS_CRYPT_WATER_LAVA, 1.0f),
	}),
	PixelSceneList({
		// g_pixel_scene_05
		PixelSceneData(PS_CRYPT_ROOM_LIQUID_FUNNEL, 1.0f),
		PixelSceneData(PS_CRYPT_ROOM_GATE_DROP, 1.0f),
		PixelSceneData(PS_CRYPT_SHOP, 1.0f),
	}),
	PixelSceneList({
		// g_pixel_scene_05b
		PixelSceneData(PS_CRYPT_ROOM_LIQUID_FUNNEL_B, 1.0f),
		PixelSceneData(PS_CRYPT_ROOM_GATE_DROP_B, 1.0f),
		PixelSceneData(PS_CRYPT_SHOP_B, 1.0f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T6),
	WandLevel(5, WAND_T6B),
	WandLevel(3, WAND_T5NS),
	WandLevel(2, WAND_T6NS),
});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_03(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 2, false);
}
_compute void spawn_pixel_scene_05(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 4, false);
}
_compute void spawn_pixel_scene_05b(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 5, false);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.38)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0x00ac33, 0x97ab00, 0xc9d959});
_data static BiomeSpawnFunctions Funcs(NULL,
	{spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_03, spawn_pixel_scene_05, spawn_pixel_scene_05b});
};

namespace FUNCS_FUNGICAVE {
BiomePixelScenes Scenes({});

_data BiomeWands wandLevels({
	WandLevel(0.5f, WAND_T2NS),
	WandLevel(0.5f, WAND_T1NS),
});

_compute void spawn_potion(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePedestals)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x, y - 6, params);
}
_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.06)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00});
_data static BiomeSpawnFunctions Funcs(NULL, {spawn_potion, spawn_item});
};

namespace FUNCS_FUNGIFOREST {
BiomePixelScenes Scenes({});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T3NS),
	WandLevel(5, WAND_T4NS),
	WandLevel(5, WAND_T5B),
});

_compute void spawn_potion(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePedestals)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x, y - 6, params);
}
_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.06)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00});
_data static BiomeSpawnFunctions Funcs(NULL, {spawn_potion, spawn_item});
};

namespace FUNCS_RAINFOREST_DARK {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_RAINFOREST_PIT01, 0.5f),
		PixelSceneData(PS_RAINFOREST_PIT02, 0.5f),
		PixelSceneData(PS_RAINFOREST_PIT03, 0.5f),
	}),
	PixelSceneList({
		// g_pixel_scene_02
		PixelSceneData(PS_RAINFOREST_HUT03, 0.5f),
		PixelSceneData(PS_RAINFOREST_SYMBOLROOM, 1.2f),
	}),
});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T4),
	WandLevel(5, WAND_T5),
	WandLevel(3, WAND_T2NS),
	WandLevel(3, WAND_T3NS),
	WandLevel(5, WAND_T4B),
});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, true);
}
_compute void spawn_pixel_scene_02(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 1, false);
}

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.27)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00, 0xff0aff, 0xff0080});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item, spawn_pixel_scene_01, spawn_pixel_scene_02});
};

namespace FUNCS_WIZARDCAVE {
BiomePixelScenes Scenes({});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T6),
	WandLevel(5, WAND_T6B),
	WandLevel(3, WAND_T5NS),
	WandLevel(2, WAND_T6NS),
});

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.38)
		return;
	spawnWand(x, y - 14, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item});
};

namespace FUNCS_LIQUIDCAVE {
BiomePixelScenes Scenes({
	PixelSceneList({
		// g_pixel_scene_01
		PixelSceneData(PS_LIQUIDCAVE_CONTAINER_01, 0.5f,
			{OIL, ALCOHOL, LAVA, MAGIC_LIQUID_TELEPORTATION, MAGIC_LIQUID_PROTECTION_ALL, MATERIAL_CONFUSION,
				LIQUID_FIRE, MAGIC_LIQUID_WEAKNESS}),
	}),
});

_data BiomeWands wandLevels({});

_compute void spawn_pixel_scene_01(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePixelSceneIndexing)
		return;
	load_random_pixel_scene(x, y, params, 0, 2);
}

static BiomeSpawnColors Colors({0x50a000, 0xff0aff});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_pixel_scene_01});

};

namespace FUNCS_ROBOBASE {
BiomePixelScenes Scenes({});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T5),
	WandLevel(5, WAND_T5B),
	WandLevel(3, WAND_T3NS),
	WandLevel(2, WAND_T4NS),
});

_compute void spawn_potion(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePedestals)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 6, y - 4, params);
}
_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r <= 0.93)
		return;
	spawnWand(x + 5, y - 6, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00});
_data static BiomeSpawnFunctions Funcs(NULL, {spawn_potion, spawn_item});
};

namespace FUNCS_VAULT_FROZEN {
BiomePixelScenes Scenes({});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T5),
	WandLevel(3, WAND_T3NS),
	WandLevel(2, WAND_T4NS),
});

_compute void spawn_potion(int x, int y, const SpawnParams& params) {
	if (!params.sCfg.biomePedestals)
		return;
	NollaPRNG random = NollaPRNG(params.seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 6, y - 4, params);
}
_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r < 0.83)
		spawnWand(x + 5, y - 6, params);
	else if (r < 0.93)
		CheckUtilityBoxLoot(x, y, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00});
_data static BiomeSpawnFunctions Funcs(NULL, {spawn_potion, spawn_item});
};

namespace FUNCS_MEAT {
BiomePixelScenes Scenes({});

_data BiomeWands wandLevels({
	WandLevel(5, WAND_T5),
	WandLevel(5, WAND_T6),
	WandLevel(5, WAND_T4NS),
	WandLevel(5, WAND_T5NS),
});

_compute void spawn_item(int x, int y, const SpawnParams& params) {
	NollaPRNG random(params.seed);
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	if (r < 0.3)
		spawnWand(x - 5, y - 14, params);
	else if (r < 0.55)
		CheckUtilityBoxLoot(x, y, params);
}

static BiomeSpawnColors Colors({0x50a000, 0x00ff00});
_data static BiomeSpawnFunctions Funcs(NULL, {spawnPotion, spawn_item});
};

namespace FUNCS_WANDCAVE {
BiomePixelScenes Scenes({});
_data BiomeWands wandLevels({});
static BiomeSpawnColors Colors({});
_data static BiomeSpawnFunctions Funcs(NULL, {});
};
namespace FUNCS_PYRAMID {
BiomePixelScenes Scenes({});
_data BiomeWands wandLevels({});
static BiomeSpawnColors Colors({});
_data static BiomeSpawnFunctions Funcs(NULL, {});
};
namespace FUNCS_THE_END {
BiomePixelScenes Scenes({});
_data BiomeWands wandLevels({});
static BiomeSpawnColors Colors({});
_data static BiomeSpawnFunctions Funcs(NULL, {});
};
namespace FUNCS_THE_SKY {
BiomePixelScenes Scenes({});
_data BiomeWands wandLevels({});
static BiomeSpawnColors Colors({});
_data static BiomeSpawnFunctions Funcs(NULL, {});
};
namespace FUNCS_SANDCAVE {
BiomePixelScenes Scenes({});
_data BiomeWands wandLevels({});
static BiomeSpawnColors Colors({});
_data static BiomeSpawnFunctions Funcs(NULL, {});
};

#define _SetBiomeDataHelper1(bName)                      \
	AllSpawnFunctions[B_##bName] = FUNCS_##bName::Funcs; \
	AllWandLevels[B_##bName] = FUNCS_##bName::wandLevels
#define _SetBiomeDataHelper2(bName)                             \
	HostSpawnColors[B_##bName] = FUNCS_##bName::Colors;         \
	if (B_##bName <= B_LIQUIDCAVE) {                            \
		HostPixelSceneLists[B_##bName] = FUNCS_##bName::Scenes; \
	}

_compute void SetBiomeData() {
	AllSpawnFunctions[0] = DefaultSpawnFunctions;
	_SetBiomeDataHelper1(COALMINE);
	_SetBiomeDataHelper1(COALMINE_ALT);
	_SetBiomeDataHelper1(EXCAVATIONSITE);
	_SetBiomeDataHelper1(SNOWCAVE);
	_SetBiomeDataHelper1(SNOWCASTLE);
	_SetBiomeDataHelper1(RAINFOREST);
	_SetBiomeDataHelper1(RAINFOREST_OPEN);
	_SetBiomeDataHelper1(VAULT);
	_SetBiomeDataHelper1(CRYPT);
	_SetBiomeDataHelper1(FUNGICAVE);
	_SetBiomeDataHelper1(FUNGIFOREST);
	_SetBiomeDataHelper1(RAINFOREST_DARK);
	_SetBiomeDataHelper1(LIQUIDCAVE);
	_SetBiomeDataHelper1(WANDCAVE);
	_SetBiomeDataHelper1(THE_END);
	_SetBiomeDataHelper1(THE_SKY);
	_SetBiomeDataHelper1(WIZARDCAVE);
	_SetBiomeDataHelper1(SANDCAVE);
	_SetBiomeDataHelper1(PYRAMID);
	_SetBiomeDataHelper1(ROBOBASE);
	_SetBiomeDataHelper1(VAULT_FROZEN);
	_SetBiomeDataHelper1(MEAT);
}
void SetBiomePixelScenes() {
	HostSpawnColors[0] = DefaultSpawnColors;
	HostPixelSceneLists[0] = {};
	_SetBiomeDataHelper2(COALMINE);
	_SetBiomeDataHelper2(COALMINE_ALT);
	_SetBiomeDataHelper2(EXCAVATIONSITE);
	_SetBiomeDataHelper2(SNOWCAVE);
	_SetBiomeDataHelper2(SNOWCASTLE);
	_SetBiomeDataHelper2(RAINFOREST);
	_SetBiomeDataHelper2(RAINFOREST_OPEN);
	_SetBiomeDataHelper2(VAULT);
	_SetBiomeDataHelper2(CRYPT);
	_SetBiomeDataHelper2(FUNGICAVE);
	_SetBiomeDataHelper2(FUNGIFOREST);
	_SetBiomeDataHelper2(RAINFOREST_DARK);
	_SetBiomeDataHelper2(LIQUIDCAVE);
	_SetBiomeDataHelper2(WANDCAVE);
	_SetBiomeDataHelper2(THE_END);
	_SetBiomeDataHelper2(THE_SKY);
	_SetBiomeDataHelper2(WIZARDCAVE);
	_SetBiomeDataHelper2(SANDCAVE);
	_SetBiomeDataHelper2(PYRAMID);
	_SetBiomeDataHelper2(ROBOBASE);
	_SetBiomeDataHelper2(VAULT_FROZEN);
	_SetBiomeDataHelper2(MEAT);
}

#undef _SetBiomeDataHelper1
#undef _SetBiomeDataHelper2