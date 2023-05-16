#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum Material : short {
	MATERIAL_NONE,
	
	//VARIABLES
	MATERIAL_VAR1,
	MATERIAL_VAR2,
	MATERIAL_VAR3,
	MATERIAL_VAR4,

	//STATIC
	WATERROCK,
	ICE_GLASS,
	ICE_GLASS_B2,
	GLASS_BRITTLE,
	WOOD_PLAYER_B2,
	WOOD,
	WAX_B2,
	FUSE,
	WOOD_LOOSE,
	ROCK_LOOSE,
	ICE_CEILING,
	BRICK,
	CONCRETE_COLLAPSED,
	TNT,
	TNT_STATIC,
	METEORITE,
	SULPHUR_BOX2D,
	METEORITE_TEST,
	METEORITE_GREEN,
	STEEL,
	STEEL_RUST,
	METAL_RUST_RUST,
	METAL_RUST_BARREL_RUST,
	PLASTIC,
	ALUMINIUM,
	ROCK_STATIC_BOX2D,
	ROCK_BOX2D,
	CRYSTAL,
	MAGIC_CRYSTAL,
	CRYSTAL_MAGIC,
	ALUMINIUM_OXIDE,
	MEAT,
	MEAT_SLIME,
	PHYSICS_THROW_MATERIAL_PART2,
	ICE_MELTING_PERF_KILLER,
	ICE_B2,
	GLASS_LIQUIDCAVE,
	GLASS,
	NEON_TUBE_PURPLE,
	SNOW_B2,
	NEON_TUBE_BLOOD_RED,
	NEON_TUBE_CYAN,
	MEAT_BURNED,
	MEAT_DONE,
	MEAT_HOT,
	MEAT_WARM,
	MEAT_FRUIT,
	CRYSTAL_SOLID,
	CRYSTAL_PURPLE,
	GOLD_B2,
	MAGIC_CRYSTAL_GREEN,
	BONE_BOX2D,
	METAL_RUST_BARREL,
	METAL_RUST,
	METAL_WIRE_NOHIT,
	METAL_CHAIN_NOHIT,
	METAL_NOHIT,
	GLASS_BOX2D,
	POTION_GLASS_BOX2D,
	GEM_BOX2D,
	ITEM_BOX2D_GLASS,
	ITEM_BOX2D,
	ROCK_BOX2D_NOHIT_HARD,
	ROCK_BOX2D_NOHIT,
	POOP_BOX2D_HARD,
	ROCK_BOX2D_HARD,
	TEMPLEBRICK_BOX2D_EDGETILES,
	METAL_HARD,
	METAL,
	METAL_PROP_LOOSE,
	METAL_PROP_LOW_RESTITUTION,
	METAL_PROP,
	ALUMINIUM_ROBOT,
	PLASTIC_PROP,
	METEORITE_CRACKABLE,
	WOOD_PROP_DURABLE,
	CLOTH_BOX2D,
	WOOD_PROP_NOPLAYERHIT,
	WOOD_PROP,
	FUNGUS_LOOSE_TRIPPY,
	FUNGUS_LOOSE_GREEN,
	FUNGUS_LOOSE,
	GRASS_LOOSE,
	CACTUS,
	WOOD_WALL,
	WOOD_TRAILER,
	TEMPLEBRICK_BOX2D,
	FUSE_HOLY,
	FUSE_TNT,
	FUSE_BRIGHT,
	WOOD_PLAYER_B2_VERTICAL,
	MEAT_CONFUSION,
	MEAT_POLYMORPH_PROTECTION,
	MEAT_POLYMORPH,
	MEAT_FAST,
	MEAT_TELEPORT,
	MEAT_SLIME_CURSED,
	MEAT_CURSED,
	MEAT_TRIPPY,
	MEAT_HELPLESS,
	MEAT_WORM,
	MEAT_SLIME_ORANGE,
	MEAT_SLIME_GREEN,
	ICE_SLIME_GLASS,
	ICE_BLOOD_GLASS,
	ICE_POISON_GLASS,
	ICE_RADIOACTIVE_GLASS,
	ICE_COLD_GLASS,
	ICE_ACID_GLASS,
	TUBE_PHYSICS,
	ROCK_ERODING,
	MEAT_PUMPKIN,
	MEAT_FROG,
	MEAT_CURSED_DRY,
	NEST_BOX2D,
	NEST_FIREBUG_BOX2D,
	COCOON_BOX2D,
	ITEM_BOX2D_MEAT,
	GEM_BOX2D_YELLOW_SUN,
	GEM_BOX2D_RED_FLOAT,
	GEM_BOX2D_YELLOW_SUN_GRAVITY,
	GEM_BOX2D_DARKSUN,
	GEM_BOX2D_PINK,
	GEM_BOX2D_RED,
	GEM_BOX2D_TURQUOISE,
	GEM_BOX2D_OPAL,
	GEM_BOX2D_WHITE,
	GEM_BOX2D_GREEN,
	GEM_BOX2D_ORANGE,
	GOLD_BOX2D,
	BLOODGOLD_BOX2D,

	//POWDERS
	SAND_STATIC,
	NEST_STATIC,
	BLUEFUNGI_STATIC,
	ROCK_STATIC,
	LAVAROCK_STATIC,
	METEORITE_STATIC,
	TEMPLEROCK_STATIC,
	STEEL_STATIC,
	ROCK_STATIC_GLOW,
	SNOW_STATIC,
	ICE_STATIC,
	ICE_ACID_STATIC,
	ICE_COLD_STATIC,
	ICE_RADIOACTIVE_STATIC,
	ICE_POISON_STATIC,
	ICE_METEOR_STATIC,
	TUBEMATERIAL,
	GLASS_STATIC,
	SNOWROCK_STATIC,
	CONCRETE_STATIC,
	WOOD_STATIC,
	CHEESE_STATIC,
	MUD,
	CONCRETE_SAND,
	SAND,
	BONE,
	SOIL,
	SANDSTONE,
	FUNGISOIL,
	HONEY,
	GLUE,
	EXPLOSION_DIRT,
	VINE,
	ROOT,
	SNOW,
	SNOW_STICKY,
	ROTTEN_MEAT,
	MEAT_SLIME_SAND,
	ROTTEN_MEAT_RADIOACTIVE,
	ICE,
	SAND_HERB,
	WAX,
	GOLD,
	SILVER,
	COPPER,
	BRASS,
	DIAMOND,
	COAL,
	SULPHUR,
	SALT,
	SODIUM_UNSTABLE,
	GUNPOWDER,
	GUNPOWDER_EXPLOSIVE,
	GUNPOWDER_TNT,
	GUNPOWDER_UNSTABLE,
	GUNPOWDER_UNSTABLE_BIG,
	MONSTER_POWDER_TEST,
	RAT_POWDER,
	FUNGUS_POWDER,
	ORB_POWDER,
	GUNPOWDER_UNSTABLE_BOSS_LIMBS,
	PLASTIC_RED,
	GRASS,
	GRASS_ICE,
	GRASS_DRY,
	FUNGI,
	SPORE,
	MOSS,
	PLANT_MATERIAL,
	PLANT_MATERIAL_RED,
	CEILING_PLANT_MATERIAL,
	MUSHROOM_SEED,
	PLANT_SEED,
	MUSHROOM,
	MUSHROOM_GIANT_RED,
	MUSHROOM_GIANT_BLUE,
	GLOWSHROOM,
	BUSH_SEED,
	WOOD_PLAYER,
	TRAILER_TEXT,
	POO,
	GLASS_BROKEN,
	BLOOD_THICK,
	SAND_STATIC_RAINFOREST,
	SAND_STATIC_RAINFOREST_DARK,
	BONE_STATIC,
	RUST_STATIC,
	SAND_STATIC_BRIGHT,
	SAND_STATIC_RED,
	MOSS_RUST,
	FUNGI_CREEPING_SECRET,
	FUNGI_CREEPING,
	GRASS_DARK,
	FUNGI_GREEN,
	SHOCK_POWDER,
	FUNGUS_POWDER_BAD,
	BURNING_POWDER,
	PURIFYING_POWDER,
	SODIUM,
	METAL_SAND,
	STEEL_SAND,
	GOLD_RADIOACTIVE,
	ROCK_STATIC_INTRO,
	ROCK_STATIC_TRIP_SECRET,
	ENDSLIME_BLOOD,
	ENDSLIME,
	ROCK_STATIC_TRIP_SECRET2,
	SANDSTONE_SURFACE,
	SOIL_DARK,
	SOIL_DEAD,
	SOIL_LUSH_DARK,
	SOIL_LUSH,
	SAND_PETRIFY,
	LAVASAND,
	SAND_SURFACE,
	SAND_BLUE,
	PLASMA_FADING_PINK,
	PLASMA_FADING_GREEN,
	CORRUPTION_STATIC,
	WOOD_STATIC_GAS,
	WOOD_STATIC_VERTICAL,
	GOLD_STATIC_DARK,
	GOLD_STATIC_RADIOACTIVE,
	GOLD_STATIC,
	CREEPY_LIQUID_EMITTER,
	WOOD_BURNS_FOREVER,
	ROOT_GROWTH,
	WOOD_STATIC_WET,
	ICE_SLIME_STATIC,
	ICE_BLOOD_STATIC,
	ROCK_STATIC_INTRO_BREAKABLE,
	STEEL_STATIC_UNMELTABLE,
	STEEL_STATIC_STRONG,
	STEELPIPE_STATIC,
	STEELSMOKE_STATIC,
	STEELMOSS_SLANTED,
	STEELFROST_STATIC,
	ROCK_STATIC_CURSED,
	ROCK_STATIC_PURPLE,
	STEELMOSS_STATIC,
	ROCK_HARD,
	TEMPLEBRICK_MOSS_STATIC,
	TEMPLEBRICK_RED,
	GLOWSTONE_POTION,
	GLOWSTONE_ALTAR_HDR,
	GLOWSTONE_ALTAR,
	GLOWSTONE,
	TEMPLEBRICK_STATIC_RUINED,
	TEMPLEBRICK_DIAMOND_STATIC,
	TEMPLEBRICK_GOLDEN_STATIC,
	WIZARDSTONE,
	TEMPLEBRICKDARK_STATIC,
	ROCK_STATIC_FUNGAL,
	WOOD_TREE,
	ROCK_STATIC_NOEDGE,
	ROCK_HARD_BORDER,
	TEMPLEROCK_SOFT,
	TEMPLEBRICK_NOEDGE_STATIC,
	TEMPLEBRICK_STATIC_SOFT,
	TEMPLEBRICK_STATIC_BROKEN,
	TEMPLEBRICK_STATIC,
	ROCK_STATIC_WET,
	ROCK_MAGIC_GATE,
	ROCK_STATIC_POISON,
	ROCK_STATIC_CURSED_GREEN,
	ROCK_STATIC_RADIOACTIVE,
	ROCK_STATIC_GREY,
	COAL_STATIC,
	ROCK_VAULT,
	ROCK_MAGIC_BOTTOM,
	TEMPLEBRICK_THICK_STATIC,
	TEMPLEBRICK_THICK_STATIC_NOEDGE,
	TEMPLESLAB_STATIC,
	TEMPLESLAB_CRUMBLING_STATIC,
	THE_END,
	STEEL_RUSTED_NO_HOLES,
	STEEL_GREY_STATIC,
	SKULLROCK,

	//LIQUIDS
	WATER_STATIC,
	ENDSLIME_STATIC,
	SLIME_STATIC,
	SPORE_POD_STALK,
	WATER,
	WATER_TEMP,
	WATER_ICE,
	WATER_SWAMP,
	OIL,
	ALCOHOL,
	SIMA,
	JUHANNUSSIMA,
	MAGIC_LIQUID,
	MATERIAL_CONFUSION,
	MATERIAL_DARKNESS,
	MATERIAL_RAINBOW,
	MAGIC_LIQUID_MOVEMENT_FASTER,
	MAGIC_LIQUID_FASTER_LEVITATION,
	MAGIC_LIQUID_FASTER_LEVITATION_AND_MOVEMENT,
	MAGIC_LIQUID_WORM_ATTRACTOR,
	MAGIC_LIQUID_PROTECTION_ALL,
	MAGIC_LIQUID_MANA_REGENERATION,
	MAGIC_LIQUID_UNSTABLE_TELEPORTATION,
	MAGIC_LIQUID_TELEPORTATION,
	MAGIC_LIQUID_HP_REGENERATION,
	MAGIC_LIQUID_HP_REGENERATION_UNSTABLE,
	MAGIC_LIQUID_POLYMORPH,
	MAGIC_LIQUID_RANDOM_POLYMORPH,
	MAGIC_LIQUID_UNSTABLE_POLYMORPH,
	MAGIC_LIQUID_BERSERK,
	MAGIC_LIQUID_CHARM,
	MAGIC_LIQUID_INVISIBILITY,
	CLOUD_RADIOACTIVE,
	CLOUD_BLOOD,
	CLOUD_SLIME,
	SWAMP,
	BLOOD,
	BLOOD_FADING,
	BLOOD_FUNGI,
	BLOOD_WORM,
	PORRIDGE,
	BLOOD_COLD,
	RADIOACTIVE_LIQUID,
	RADIOACTIVE_LIQUID_FADING,
	PLASMA_FADING,
	GOLD_MOLTEN,
	WAX_MOLTEN,
	SILVER_MOLTEN,
	COPPER_MOLTEN,
	BRASS_MOLTEN,
	GLASS_MOLTEN,
	GLASS_BROKEN_MOLTEN,
	STEEL_MOLTEN,
	CREEPY_LIQUID,
	CEMENT,
	SLIME,
	SLUSH,
	VOMIT,
	PLASTIC_RED_MOLTEN,
	ACID,
	LAVA,
	URINE,
	ROCKET_PARTICLES,
	PEAT,
	PLASTIC_PROP_MOLTEN,
	PLASTIC_MOLTEN,
	SLIME_YELLOW,
	SLIME_GREEN,
	ALUMINIUM_OXIDE_MOLTEN,
	STEEL_RUST_MOLTEN,
	METAL_PROP_MOLTEN,
	ALUMINIUM_ROBOT_MOLTEN,
	ALUMINIUM_MOLTEN,
	METAL_NOHIT_MOLTEN,
	METAL_RUST_MOLTEN,
	METAL_MOLTEN,
	METAL_SAND_MOLTEN,
	STEELSMOKE_STATIC_MOLTEN,
	STEELMOSS_STATIC_MOLTEN,
	STEELMOSS_SLANTED_MOLTEN,
	STEEL_STATIC_MOLTEN,
	PLASMA_FADING_BRIGHT,
	RADIOACTIVE_LIQUID_YELLOW,
	CURSED_LIQUID,
	POISON,
	BLOOD_FADING_SLOW,
	MIDAS,
	MIDAS_PRECURSOR,
	LIQUID_FIRE_WEAK,
	LIQUID_FIRE,
	VOID_LIQUID,
	WATER_SALT,
	WATER_FADING,
	PEA_SOUP,

	//GASSES
	SMOKE,
	CLOUD,
	CLOUD_LIGHTER,
	SMOKE_EXPLOSION,
	STEAM,
	ACID_GAS,
	ACID_GAS_STATIC,
	SMOKE_STATIC,
	BLOOD_COLD_VAPOR,
	SAND_HERB_VAPOR,
	RADIOACTIVE_GAS,
	RADIOACTIVE_GAS_STATIC,
	MAGIC_GAS_HP_REGENERATION,
	RAINBOW_GAS,
	ALCOHOL_GAS,
	POO_GAS,
	FUNGAL_GAS,
	POISON_GAS,
	STEAM_TRAILER,
	SMOKE_MAGIC,

	//FIRES
	FIRE,
	SPARK,
	SPARK_ELECTRIC,
	FLAME,
	FIRE_BLUE,
	SPARK_GREEN,
	SPARK_GREEN_BRIGHT,
	SPARK_BLUE,
	SPARK_BLUE_DARK,
	SPARK_RED,
	SPARK_RED_BRIGHT,
	SPARK_WHITE,
	SPARK_WHITE_BRIGHT,
	SPARK_YELLOW,
	SPARK_PURPLE,
	SPARK_PURPLE_BRIGHT,
	SPARK_PLAYER,
	SPARK_TEAL
};

__device__ const char* MaterialNames[] = {
	"MATERIAL_NONE",

	//VARIABLES
	"MATERIAL_VAR1",
	"MATERIAL_VAR2",
	"MATERIAL_VAR3",
	"MATERIAL_VAR4",

	//STATIC
	"WATERROCK",
	"ICE_GLASS",
	"ICE_GLASS_B2",
	"GLASS_BRITTLE",
	"WOOD_PLAYER_B2",
	"WOOD",
	"WAX_B2",
	"FUSE",
	"WOOD_LOOSE",
	"ROCK_LOOSE",
	"ICE_CEILING",
	"BRICK",
	"CONCRETE_COLLAPSED",
	"TNT",
	"TNT_STATIC",
	"METEORITE",
	"SULPHUR_BOX2D",
	"METEORITE_TEST",
	"METEORITE_GREEN",
	"STEEL",
	"STEEL_RUST",
	"METAL_RUST_RUST",
	"METAL_RUST_BARREL_RUST",
	"PLASTIC",
	"ALUMINIUM",
	"ROCK_STATIC_BOX2D",
	"ROCK_BOX2D",
	"CRYSTAL",
	"MAGIC_CRYSTAL",
	"CRYSTAL_MAGIC",
	"ALUMINIUM_OXIDE",
	"MEAT",
	"MEAT_SLIME",
	"PHYSICS_THROW_MATERIAL_PART2",
	"ICE_MELTING_PERF_KILLER",
	"ICE_B2",
	"GLASS_LIQUIDCAVE",
	"GLASS",
	"NEON_TUBE_PURPLE",
	"SNOW_B2",
	"NEON_TUBE_BLOOD_RED",
	"NEON_TUBE_CYAN",
	"MEAT_BURNED",
	"MEAT_DONE",
	"MEAT_HOT",
	"MEAT_WARM",
	"MEAT_FRUIT",
	"CRYSTAL_SOLID",
	"CRYSTAL_PURPLE",
	"GOLD_B2",
	"MAGIC_CRYSTAL_GREEN",
	"BONE_BOX2D",
	"METAL_RUST_BARREL",
	"METAL_RUST",
	"METAL_WIRE_NOHIT",
	"METAL_CHAIN_NOHIT",
	"METAL_NOHIT",
	"GLASS_BOX2D",
	"POTION_GLASS_BOX2D",
	"GEM_BOX2D",
	"ITEM_BOX2D_GLASS",
	"ITEM_BOX2D",
	"ROCK_BOX2D_NOHIT_HARD",
	"ROCK_BOX2D_NOHIT",
	"POOP_BOX2D_HARD",
	"ROCK_BOX2D_HARD",
	"TEMPLEBRICK_BOX2D_EDGETILES",
	"METAL_HARD",
	"METAL",
	"METAL_PROP_LOOSE",
	"METAL_PROP_LOW_RESTITUTION",
	"METAL_PROP",
	"ALUMINIUM_ROBOT",
	"PLASTIC_PROP",
	"METEORITE_CRACKABLE",
	"WOOD_PROP_DURABLE",
	"CLOTH_BOX2D",
	"WOOD_PROP_NOPLAYERHIT",
	"WOOD_PROP",
	"FUNGUS_LOOSE_TRIPPY",
	"FUNGUS_LOOSE_GREEN",
	"FUNGUS_LOOSE",
	"GRASS_LOOSE",
	"CACTUS",
	"WOOD_WALL",
	"WOOD_TRAILER",
	"TEMPLEBRICK_BOX2D",
	"FUSE_HOLY",
	"FUSE_TNT",
	"FUSE_BRIGHT",
	"WOOD_PLAYER_B2_VERTICAL",
	"MEAT_CONFUSION",
	"MEAT_POLYMORPH_PROTECTION",
	"MEAT_POLYMORPH",
	"MEAT_FAST",
	"MEAT_TELEPORT",
	"MEAT_SLIME_CURSED",
	"MEAT_CURSED",
	"MEAT_TRIPPY",
	"MEAT_HELPLESS",
	"MEAT_WORM",
	"MEAT_SLIME_ORANGE",
	"MEAT_SLIME_GREEN",
	"ICE_SLIME_GLASS",
	"ICE_BLOOD_GLASS",
	"ICE_POISON_GLASS",
	"ICE_RADIOACTIVE_GLASS",
	"ICE_COLD_GLASS",
	"ICE_ACID_GLASS",
	"TUBE_PHYSICS",
	"ROCK_ERODING",
	"MEAT_PUMPKIN",
	"MEAT_FROG",
	"MEAT_CURSED_DRY",
	"NEST_BOX2D",
	"NEST_FIREBUG_BOX2D",
	"COCOON_BOX2D",
	"ITEM_BOX2D_MEAT",
	"GEM_BOX2D_YELLOW_SUN",
	"GEM_BOX2D_RED_FLOAT",
	"GEM_BOX2D_YELLOW_SUN_GRAVITY",
	"GEM_BOX2D_DARKSUN",
	"GEM_BOX2D_PINK",
	"GEM_BOX2D_RED",
	"GEM_BOX2D_TURQUOISE",
	"GEM_BOX2D_OPAL",
	"GEM_BOX2D_WHITE",
	"GEM_BOX2D_GREEN",
	"GEM_BOX2D_ORANGE",
	"GOLD_BOX2D",
	"BLOODGOLD_BOX2D",

	//POWDERS
	"SAND_STATIC",
	"NEST_STATIC",
	"BLUEFUNGI_STATIC",
	"ROCK_STATIC",
	"LAVAROCK_STATIC",
	"METEORITE_STATIC",
	"TEMPLEROCK_STATIC",
	"STEEL_STATIC",
	"ROCK_STATIC_GLOW",
	"SNOW_STATIC",
	"ICE_STATIC",
	"ICE_ACID_STATIC",
	"ICE_COLD_STATIC",
	"ICE_RADIOACTIVE_STATIC",
	"ICE_POISON_STATIC",
	"ICE_METEOR_STATIC",
	"TUBEMATERIAL",
	"GLASS_STATIC",
	"SNOWROCK_STATIC",
	"CONCRETE_STATIC",
	"WOOD_STATIC",
	"CHEESE_STATIC",
	"MUD",
	"CONCRETE_SAND",
	"SAND",
	"BONE",
	"SOIL",
	"SANDSTONE",
	"FUNGISOIL",
	"HONEY",
	"GLUE",
	"EXPLOSION_DIRT",
	"VINE",
	"ROOT",
	"SNOW",
	"SNOW_STICKY",
	"ROTTEN_MEAT",
	"MEAT_SLIME_SAND",
	"ROTTEN_MEAT_RADIOACTIVE",
	"ICE",
	"SAND_HERB",
	"WAX",
	"GOLD",
	"SILVER",
	"COPPER",
	"BRASS",
	"DIAMOND",
	"COAL",
	"SULPHUR",
	"SALT",
	"SODIUM_UNSTABLE",
	"GUNPOWDER",
	"GUNPOWDER_EXPLOSIVE",
	"GUNPOWDER_TNT",
	"GUNPOWDER_UNSTABLE",
	"GUNPOWDER_UNSTABLE_BIG",
	"MONSTER_POWDER_TEST",
	"RAT_POWDER",
	"FUNGUS_POWDER",
	"ORB_POWDER",
	"GUNPOWDER_UNSTABLE_BOSS_LIMBS",
	"PLASTIC_RED",
	"GRASS",
	"GRASS_ICE",
	"GRASS_DRY",
	"FUNGI",
	"SPORE",
	"MOSS",
	"PLANT_MATERIAL",
	"PLANT_MATERIAL_RED",
	"CEILING_PLANT_MATERIAL",
	"MUSHROOM_SEED",
	"PLANT_SEED",
	"MUSHROOM",
	"MUSHROOM_GIANT_RED",
	"MUSHROOM_GIANT_BLUE",
	"GLOWSHROOM",
	"BUSH_SEED",
	"WOOD_PLAYER",
	"TRAILER_TEXT",
	"POO",
	"GLASS_BROKEN",
	"BLOOD_THICK",
	"SAND_STATIC_RAINFOREST",
	"SAND_STATIC_RAINFOREST_DARK",
	"BONE_STATIC",
	"RUST_STATIC",
	"SAND_STATIC_BRIGHT",
	"SAND_STATIC_RED",
	"MOSS_RUST",
	"FUNGI_CREEPING_SECRET",
	"FUNGI_CREEPING",
	"GRASS_DARK",
	"FUNGI_GREEN",
	"SHOCK_POWDER",
	"FUNGUS_POWDER_BAD",
	"BURNING_POWDER",
	"PURIFYING_POWDER",
	"SODIUM",
	"METAL_SAND",
	"STEEL_SAND",
	"GOLD_RADIOACTIVE",
	"ROCK_STATIC_INTRO",
	"ROCK_STATIC_TRIP_SECRET",
	"ENDSLIME_BLOOD",
	"ENDSLIME",
	"ROCK_STATIC_TRIP_SECRET2",
	"SANDSTONE_SURFACE",
	"SOIL_DARK",
	"SOIL_DEAD",
	"SOIL_LUSH_DARK",
	"SOIL_LUSH",
	"SAND_PETRIFY",
	"LAVASAND",
	"SAND_SURFACE",
	"SAND_BLUE",
	"PLASMA_FADING_PINK",
	"PLASMA_FADING_GREEN",
	"CORRUPTION_STATIC",
	"WOOD_STATIC_GAS",
	"WOOD_STATIC_VERTICAL",
	"GOLD_STATIC_DARK",
	"GOLD_STATIC_RADIOACTIVE",
	"GOLD_STATIC",
	"CREEPY_LIQUID_EMITTER",
	"WOOD_BURNS_FOREVER",
	"ROOT_GROWTH",
	"WOOD_STATIC_WET",
	"ICE_SLIME_STATIC",
	"ICE_BLOOD_STATIC",
	"ROCK_STATIC_INTRO_BREAKABLE",
	"STEEL_STATIC_UNMELTABLE",
	"STEEL_STATIC_STRONG",
	"STEELPIPE_STATIC",
	"STEELSMOKE_STATIC",
	"STEELMOSS_SLANTED",
	"STEELFROST_STATIC",
	"ROCK_STATIC_CURSED",
	"ROCK_STATIC_PURPLE",
	"STEELMOSS_STATIC",
	"ROCK_HARD",
	"TEMPLEBRICK_MOSS_STATIC",
	"TEMPLEBRICK_RED",
	"GLOWSTONE_POTION",
	"GLOWSTONE_ALTAR_HDR",
	"GLOWSTONE_ALTAR",
	"GLOWSTONE",
	"TEMPLEBRICK_STATIC_RUINED",
	"TEMPLEBRICK_DIAMOND_STATIC",
	"TEMPLEBRICK_GOLDEN_STATIC",
	"WIZARDSTONE",
	"TEMPLEBRICKDARK_STATIC",
	"ROCK_STATIC_FUNGAL",
	"WOOD_TREE",
	"ROCK_STATIC_NOEDGE",
	"ROCK_HARD_BORDER",
	"TEMPLEROCK_SOFT",
	"TEMPLEBRICK_NOEDGE_STATIC",
	"TEMPLEBRICK_STATIC_SOFT",
	"TEMPLEBRICK_STATIC_BROKEN",
	"TEMPLEBRICK_STATIC",
	"ROCK_STATIC_WET",
	"ROCK_MAGIC_GATE",
	"ROCK_STATIC_POISON",
	"ROCK_STATIC_CURSED_GREEN",
	"ROCK_STATIC_RADIOACTIVE",
	"ROCK_STATIC_GREY",
	"COAL_STATIC",
	"ROCK_VAULT",
	"ROCK_MAGIC_BOTTOM",
	"TEMPLEBRICK_THICK_STATIC",
	"TEMPLEBRICK_THICK_STATIC_NOEDGE",
	"TEMPLESLAB_STATIC",
	"TEMPLESLAB_CRUMBLING_STATIC",
	"THE_END",
	"STEEL_RUSTED_NO_HOLES",
	"STEEL_GREY_STATIC",
	"SKULLROCK",

	//LIQUIDS
	"WATER_STATIC",
	"ENDSLIME_STATIC",
	"SLIME_STATIC",
	"SPORE_POD_STALK",
	"WATER",
	"WATER_TEMP",
	"WATER_ICE",
	"WATER_SWAMP",
	"OIL",
	"ALCOHOL",
	"SIMA",
	"JUHANNUSSIMA",
	"MAGIC_LIQUID",
	"MATERIAL_CONFUSION",
	"MATERIAL_DARKNESS",
	"MATERIAL_RAINBOW",
	"MAGIC_LIQUID_MOVEMENT_FASTER",
	"MAGIC_LIQUID_FASTER_LEVITATION",
	"MAGIC_LIQUID_FASTER_LEVITATION_AND_MOVEMENT",
	"MAGIC_LIQUID_WORM_ATTRACTOR",
	"MAGIC_LIQUID_PROTECTION_ALL",
	"MAGIC_LIQUID_MANA_REGENERATION",
	"MAGIC_LIQUID_UNSTABLE_TELEPORTATION",
	"MAGIC_LIQUID_TELEPORTATION",
	"MAGIC_LIQUID_HP_REGENERATION",
	"MAGIC_LIQUID_HP_REGENERATION_UNSTABLE",
	"MAGIC_LIQUID_POLYMORPH",
	"MAGIC_LIQUID_RANDOM_POLYMORPH",
	"MAGIC_LIQUID_UNSTABLE_POLYMORPH",
	"MAGIC_LIQUID_BERSERK",
	"MAGIC_LIQUID_CHARM",
	"MAGIC_LIQUID_INVISIBILITY",
	"CLOUD_RADIOACTIVE",
	"CLOUD_BLOOD",
	"CLOUD_SLIME",
	"SWAMP",
	"BLOOD",
	"BLOOD_FADING",
	"BLOOD_FUNGI",
	"BLOOD_WORM",
	"PORRIDGE",
	"BLOOD_COLD",
	"RADIOACTIVE_LIQUID",
	"RADIOACTIVE_LIQUID_FADING",
	"PLASMA_FADING",
	"GOLD_MOLTEN",
	"WAX_MOLTEN",
	"SILVER_MOLTEN",
	"COPPER_MOLTEN",
	"BRASS_MOLTEN",
	"GLASS_MOLTEN",
	"GLASS_BROKEN_MOLTEN",
	"STEEL_MOLTEN",
	"CREEPY_LIQUID",
	"CEMENT",
	"SLIME",
	"SLUSH",
	"VOMIT",
	"PLASTIC_RED_MOLTEN",
	"ACID",
	"LAVA",
	"URINE",
	"ROCKET_PARTICLES",
	"PEAT",
	"PLASTIC_PROP_MOLTEN",
	"PLASTIC_MOLTEN",
	"SLIME_YELLOW",
	"SLIME_GREEN",
	"ALUMINIUM_OXIDE_MOLTEN",
	"STEEL_RUST_MOLTEN",
	"METAL_PROP_MOLTEN",
	"ALUMINIUM_ROBOT_MOLTEN",
	"ALUMINIUM_MOLTEN",
	"METAL_NOHIT_MOLTEN",
	"METAL_RUST_MOLTEN",
	"METAL_MOLTEN",
	"METAL_SAND_MOLTEN",
	"STEELSMOKE_STATIC_MOLTEN",
	"STEELMOSS_STATIC_MOLTEN",
	"STEELMOSS_SLANTED_MOLTEN",
	"STEEL_STATIC_MOLTEN",
	"PLASMA_FADING_BRIGHT",
	"RADIOACTIVE_LIQUID_YELLOW",
	"CURSED_LIQUID",
	"POISON",
	"BLOOD_FADING_SLOW",
	"MIDAS",
	"MIDAS_PRECURSOR",
	"LIQUID_FIRE_WEAK",
	"LIQUID_FIRE",
	"VOID_LIQUID",
	"WATER_SALT",
	"WATER_FADING",
	"PEA_SOUP",

	//GASSES
	"SMOKE",
	"CLOUD",
	"CLOUD_LIGHTER",
	"SMOKE_EXPLOSION",
	"STEAM",
	"ACID_GAS",
	"ACID_GAS_STATIC",
	"SMOKE_STATIC",
	"BLOOD_COLD_VAPOR",
	"SAND_HERB_VAPOR",
	"RADIOACTIVE_GAS",
	"RADIOACTIVE_GAS_STATIC",
	"MAGIC_GAS_HP_REGENERATION",
	"RAINBOW_GAS",
	"ALCOHOL_GAS",
	"POO_GAS",
	"FUNGAL_GAS",
	"POISON_GAS",
	"STEAM_TRAILER",
	"SMOKE_MAGIC",

	//FIRES
	"FIRE",
	"SPARK",
	"SPARK_ELECTRIC",
	"FLAME",
	"FIRE_BLUE",
	"SPARK_GREEN",
	"SPARK_GREEN_BRIGHT",
	"SPARK_BLUE",
	"SPARK_BLUE_DARK",
	"SPARK_RED",
	"SPARK_RED_BRIGHT",
	"SPARK_WHITE",
	"SPARK_WHITE_BRIGHT",
	"SPARK_YELLOW",
	"SPARK_PURPLE",
	"SPARK_PURPLE_BRIGHT",
	"SPARK_PLAYER",
	"SPARK_TEAL"
};

constexpr int materialVarEntryCount = 10;

__device__ static bool MaterialRefEquals(Material reference, Material test)
{
	if (reference == MATERIAL_NONE) return true;
	if (test == MATERIAL_NONE) return false;
	return reference == test;
}

__device__ static bool MaterialEquals(Material reference, Material test, bool writeRef, int* ptrs, Material* variables)
{
	if (reference == MATERIAL_NONE) return true;
	else if ((int)reference <= MATERIAL_VAR4)
	{
		int idx = (int)reference - 1;
		if (writeRef)
		{
			if (ptrs[idx] >= materialVarEntryCount) printf("Material variable %i space ran out!\n", idx);
			else variables[idx * materialVarEntryCount + ptrs[idx]++] = test;
			return true;
		}
		else
		{
			bool foundVar = false;
			for (int i = 0; i < ptrs[idx]; i++)
			{
				if (MaterialRefEquals(variables[idx * materialVarEntryCount + i], test))
					foundVar = true;
			}
			return foundVar;
		}
	}

	if (test == MATERIAL_NONE) return false;
	/*if ((int)test <= MATERIAL_VAR4)
	{
		int idx = (int)test - 1;
		if (writeRef)
		{
			if (ptrs[idx] >= materialVarEntryCount) printf("Material variable %i space ran out!\n", idx);
			else variables[idx * materialVarEntryCount + ptrs[idx]++] = reference;
			return true;
		}
		else
		{
			bool foundVar = false;
			for (int i = 0; i < ptrs[idx]; i++)
			{
				if (MaterialEquals(reference, variables[idx * materialVarEntryCount + i], writeRef, ptrs, variables))
					foundVar = true;
			}
			return foundVar;
		}
	}*/

	return reference == test;
}