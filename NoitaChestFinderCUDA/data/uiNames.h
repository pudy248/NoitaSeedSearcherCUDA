#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const char* SpawnableTypeNames[] = {
	"CHEST",
	"GREAT_CHEST",
	"ITEM_PEDESTAL",
	"WAND_PEDESTAL",
	"NIGHTMARE_WAND",
	"EOE_DROP",
	"HM_SHOP",
	"HELL_SHOP",
	"EYE_ROOM"
};

const char* ItemNames[] = {
	"GOLD_NUGGETS",
	"CHEST_TO_GOLD",
	"RAIN_GOLD",
	"BOMB",
	"POWDER",
	"POTION_NORMAL",
	"POTION_SECRET",
	"POTION_RANDOM_MATERIAL",
	"KAMMI",
	"KUU",
	"PAHA_SILMA",
	"CHAOS_DIE",
	"SHINY_ORB",
	"UKKOSKIVI",
	"KIUASKIVI",
	"VUOKSIKIVI",
	"KAKKAKIKKARE",
	"RUNESTONE_LIGHT",
	"RUNESTONE_FIRE",
	"RUNESTONE_MAGMA",
	"RUNESTONE_WEIGHT",
	"RUNESTONE_EMPTINESS",
	"RUNESTONE_EDGES",
	"RUNESTONE_METAL",
	"RANDOM_SPELL",
	"SPELL_REFRESH",
	"HEART_NORMAL",
	"HEART_MIMIC",
	"HEART_BIGGER",
	"FULL_HEAL",
	"WAND_T1",
	"WAND_T1NS",
	"WAND_T1B",
	"WAND_T2",
	"WAND_T2NS",
	"WAND_T2B",
	"WAND_T3",
	"WAND_T3NS",
	"WAND_T3B",
	"WAND_T4",
	"WAND_T4NS",
	"WAND_T4B",
	"WAND_T5",
	"WAND_T5NS",
	"WAND_T5B",
	"WAND_T6",
	"WAND_T6NS",
	"WAND_T6B",

	"WAND_T10",
	"WAND_T10NS",

	"EGG_PURPLE",
	"EGG_SLIME",
	"EGG_MONSTER",
	"BROKEN_WAND",
	"UNKNOWN_WAND",

	"MIMIC",
	"MIMIC_LEGGY",
	"MIMIC_SIGN",

	"SAMPO",
	"TRUE_ORB",
};

const char* MaterialNames[] = {
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

const char* BiomeNames[] = {
	"NONE",
	"COALMINE",
	"COALMINE_ALT",
	"EXCAVATIONSITE",
	"FUNGICAVE",
	"SNOWCAVE",
	"SNOWCASTLE",
	"RAINFOREST",
	"VAULT",
	"CRYPT"
};

const char* BiomeModifierNames[] = {
	"NONE",
	"MOIST",
	"FOG_OF_WAR_REAPPEARS",
	"HIGH_GRAVITY",
	"LOW_GRAVITY",
	"CONDUCTIVE",
	"HOT",
	"GOLD_VEIN",
	"GOLD_VEIN_SUPER",
	"PLANT_INFESTED",
	"FURNISHED",
	"BOOBY_TRAPPED",
	"PERFORATED",
	"SPOOKY",
	"GRAVITY_FIELDS",
	"FUNGAL",
	"FLOODED",
	"GAS_FLOODED",
	"SHIELDED",
	"PROTECTION_FIELDS",
	"OMINOUS",
	"INVISIBILITY",
	"WORMY"
};

const char* PerkNames[] = {
	"CRITICAL_HIT",
	"BREATH_UNDERWATER",
	"EXTRA_MONEY",
	"EXTRA_MONEY_TRICK_KILL",
	"GOLD_IS_FOREVER",
	"TRICK_BLOOD_MONEY",
	"EXPLODING_GOLD",
	"HOVER_BOOST",
	"FASTER_LEVITATION",
	"MOVEMENT_FASTER",
	"STRONG_KICK",
	"TELEKINESIS",
	"REPELLING_CAPE",
	"EXPLODING_CORPSES",
	"SAVING_GRACE",
	"INVISIBILITY",
	"GLOBAL_GORE",
	"REMOVE_FOG_OF_WAR",
	"LEVITATION_TRAIL",
	"VAMPIRISM",
	"EXTRA_HP",
	"HEARTS_MORE_EXTRA_HP",
	"GLASS_CANNON",
	"LOW_HP_DAMAGE_BOOST",
	"RESPAWN",
	"WORM_ATTRACTOR",
	"RADAR_ENEMY",
	"FOOD_CLOCK",
	"IRON STOMACH",
	"WAND_RADAR",
	"ITEM_RADAR",
	"MOON_RADAR",
	"MAP",
	"PROTECTION_FIRE",
	"PROTECTION_RADIOACTIVITY",
	"PROTECTION_EXPLOSION",
	"PROTECTION_MELEE",
	"PROTECTION_ELECTRICITY",
	"TELEPORTITIS",
	"TELEPORTITIS_DODGE",
	"STAINLESS_ARMOUR",
	"EDIT_WANDS_EVERYWHERE",
	"NO_WAND_EDITING",
	"WAND_EXPERIMENTER",
	"ADVENTURER",
	"ABILITY_ACTIONS_MATERIALIZED",
	"PROJECTILE_HOMING",
	"PROJECTILE_HOMING_SHOOTER",
	"UNLIMITED_SPELLS",
	"FREEZE_FIELD",
	"FIRE_GAS",
	"DISSOLVE_POWDERS",
	"BLEED_SLIME",
	"BLEED_OIL",
	"BLEED_GAS",
	"SHIELD",
	"REVENGE_EXPLOSION",
	"REVENGE_TENTACLE",
	"REVENGE_RATS",
	"REVENGE_BULLET",
	"ATTACK_FOOT",
	"LEGGY_FEET",
	"PLAGUE_RATS",
	"VOMIT_RATS",
	"CORDYCEPS",
	"MOLD",
	"WORM_SMALLER_HOLES",
	"PROJECTILE_REPULSION",
	"RISKY_CRITICAL",
	"FUNGAL_DISEASE",
	"PROJECTILE_SLOW_FIELD",
	"PROJECTILE_REPULSION_SECTOR",
	"PROJECTILE_EATER_SECTOR",
	"ORBIT",
	"ANGRY_GHOST",
	"HUNGRY_GHOST",
	"DEATH_GHOST",
	"HOMUNCULUS",
	"LUKKI_MINION",
	"ELECTRICITY",
	"ATTRACT_ITEMS",
	"EXTRA_KNOCKBACK",
	"LOWER_SPREAD",
	"LOW_RECOIL",
	"BOUNCE",
	"FAST_PROJECTILES",
	"ALWAYS_CAST",
	"EXTRA_MANA",
	"NO_MORE_SHUFFLE",
	"NO_MORE_KNOCKBACK",
	"DUPLICATE_PROJECTILE",
	"FASTER_WANDS",
	"EXTRA_SLOTS",
	"CONTACT_DAMAGE",
	"EXTRA_PERK",
	"PERKS_LOTTERY",
	"GAMBLE",
	"EXTRA_SHOP_ITEM",
	"GENOME_MORE_HATRED",
	"GENOME_MORE_LOVE",
	"PEACE_WITH_GODS",
	"MANA_FROM_KILLS",
	"ANGRY_LEVITATION",
	"LASER_AIM",
	"PERSONAL_LASER",
	"MEGA_BEAM_STONE",
};

const char* SpellNames[] = {
	"NONE",
	"BOMB",
"LIGHT_BULLET",
"LIGHT_BULLET_TRIGGER",
"LIGHT_BULLET_TRIGGER_2",
"LIGHT_BULLET_TIMER",
"BULLET",
"BULLET_TRIGGER",
"BULLET_TIMER",
"HEAVY_BULLET",
"HEAVY_BULLET_TRIGGER",
"HEAVY_BULLET_TIMER",
"AIR_BULLET",
"SLOW_BULLET",
"SLOW_BULLET_TRIGGER",
"SLOW_BULLET_TIMER",
"HOOK",
"BLACK_HOLE",
"BLACK_HOLE_DEATH_TRIGGER",
"BLACK_HOLE_BIG",
"WHITE_HOLE_BIG",
"BLACK_HOLE_GIGA",
"TENTACLE_PORTAL",
"SPITTER",
"SPITTER_TIMER",
"SPITTER_TIER_2",
"SPITTER_TIER_2_TIMER",
"SPITTER_TIER_3",
"SPITTER_TIER_3_TIMER",
"BUBBLESHOT",
"BUBBLESHOT_TRIGGER",
"DISC_BULLET",
"DISC_BULLET_BIG",
"DISC_BULLET_BIGGER",
"BOUNCY_ORB",
"BOUNCY_ORB_TIMER",
"RUBBER_BALL",
"ARROW",
"POLLEN",
"LANCE",
"ROCKET",
"ROCKET_TIER_2",
"ROCKET_TIER_3",
"GRENADE",
"GRENADE_TRIGGER",
"GRENADE_TIER_2",
"GRENADE_TIER_3",
"GRENADE_ANTI",
"GRENADE_LARGE",
"MINE",
"MINE_DEATH_TRIGGER",
"PIPE_BOMB",
"PIPE_BOMB_DEATH_TRIGGER",
"EXPLODING_DEER",
"EXPLODING_DUCKS",
"WORM_SHOT",
"BOMB_DETONATOR",
"LASER",
"MEGALASER",
"LIGHTNING",
"BALL_LIGHTNING",
"LASER_EMITTER",
"LASER_EMITTER_FOUR",
"LASER_EMITTER_CUTTER",
"DIGGER",
"POWERDIGGER",
"CHAINSAW",
"LUMINOUS_DRILL",
"LASER_LUMINOUS_DRILL",
"TENTACLE",
"TENTACLE_TIMER",
"HEAL_BULLET",
"SPIRAL_SHOT",
"MAGIC_SHIELD",
"BIG_MAGIC_SHIELD",
"CHAIN_BOLT",
"FIREBALL",
"METEOR",
"FLAMETHROWER",
"ICEBALL",
"SLIMEBALL",
"DARKFLAME",
"MISSILE",
"FUNKY_SPELL",
"PEBBLE",
"DYNAMITE",
"GLITTER_BOMB",
"BUCKSHOT",
"FREEZING_GAZE",
"GLOWING_BOLT",
"SPORE_POD",
"GLUE_SHOT",
"BOMB_HOLY",
"BOMB_HOLY_GIGA",
"PROPANE_TANK",
"BOMB_CART",
"CURSED_ORB",
"EXPANDING_ORB",
"CRUMBLING_EARTH",
"SUMMON_ROCK",
"SUMMON_EGG",
"SUMMON_HOLLOW_EGG",
"TNTBOX",
"TNTBOX_BIG",
"SWARM_FLY",
"SWARM_FIREBUG",
"SWARM_WASP",
"FRIEND_FLY",
"ACIDSHOT",
"THUNDERBALL",
"FIREBOMB",
"SOILBALL",
"DEATH_CROSS",
"DEATH_CROSS_BIG",
"INFESTATION",
"WALL_HORIZONTAL",
"WALL_VERTICAL",
"WALL_SQUARE",
"TEMPORARY_WALL",
"TEMPORARY_PLATFORM",
"PURPLE_EXPLOSION_FIELD",
"DELAYED_SPELL",
"LONG_DISTANCE_CAST",
"TELEPORT_CAST",
"SUPER_TELEPORT_CAST",
"CASTER_CAST",
"MIST_RADIOACTIVE",
"MIST_ALCOHOL",
"MIST_SLIME",
"MIST_BLOOD",
"CIRCLE_FIRE",
"CIRCLE_ACID",
"CIRCLE_OIL",
"CIRCLE_WATER",
"MATERIAL_WATER",
"MATERIAL_OIL",
"MATERIAL_BLOOD",
"MATERIAL_ACID",
"MATERIAL_CEMENT",
"TELEPORT_PROJECTILE",
"TELEPORT_PROJECTILE_SHORT",
"TELEPORT_PROJECTILE_STATIC",
"SWAPPER_PROJECTILE",
"TELEPORT_PROJECTILE_CLOSER",
"NUKE",
"NUKE_GIGA",
"FIREWORK",
"SUMMON_WANDGHOST",
"TOUCH_GOLD",
"TOUCH_WATER",
"TOUCH_OIL",
"TOUCH_ALCOHOL",
"TOUCH_BLOOD",
"TOUCH_SMOKE",
"DESTRUCTION",
"BURST_2",
"BURST_3",
"BURST_4",
"BURST_8",
"BURST_X",
"SCATTER_2",
"SCATTER_3",
"SCATTER_4",
"I_SHAPE",
"Y_SHAPE",
"T_SHAPE",
"W_SHAPE",
"CIRCLE_SHAPE",
"PENTAGRAM_SHAPE",
"SPREAD_REDUCE",
"HEAVY_SPREAD",
"RECHARGE",
"LIFETIME",
"LIFETIME_DOWN",
"NOLLA",
"SLOW_BUT_STEADY",
"EXPLOSION_REMOVE",
"EXPLOSION_TINY",
"LASER_EMITTER_WIDER",
"MANA_REDUCE",
"BLOOD_MAGIC",
"MONEY_MAGIC",
"BLOOD_TO_POWER",
"DUPLICATE",
"QUANTUM_SPLIT",
"GRAVITY",
"GRAVITY_ANTI",
"SINEWAVE",
"CHAOTIC_ARC",
"PINGPONG_PATH",
"AVOIDING_ARC",
"FLOATING_ARC",
"FLY_DOWNWARDS",
"FLY_UPWARDS",
"HORIZONTAL_ARC",
"LINE_ARC",
"ORBIT_SHOT",
"SPIRALING_SHOT",
"PHASING_ARC",
"TRUE_ORBIT",
"BOUNCE",
"REMOVE_BOUNCE",
"HOMING",
"HOMING_SHORT",
"HOMING_ROTATE",
"HOMING_SHOOTER",
"AUTOAIM",
"HOMING_ACCELERATING",
"HOMING_CURSOR",
"HOMING_AREA",
"PIERCING_SHOT",
"CLIPPING_SHOT",
"DAMAGE",
"DAMAGE_RANDOM",
"BLOODLUST",
"DAMAGE_FOREVER",
"CRITICAL_HIT",
"AREA_DAMAGE",
"SPELLS_TO_POWER",
"ESSENCE_TO_POWER",
"ZERO_DAMAGE",
"HEAVY_SHOT",
"LIGHT_SHOT",
"KNOCKBACK",
"RECOIL",
"RECOIL_DAMPER",
"SPEED",
"ACCELERATING_SHOT",
"DECELERATING_SHOT",
"EXPLOSIVE_PROJECTILE",
"WATER_TO_POISON",
"BLOOD_TO_ACID",
"LAVA_TO_BLOOD",
"LIQUID_TO_EXPLOSION",
"TOXIC_TO_ACID",
"STATIC_TO_SAND",
"TRANSMUTATION",
"RANDOM_EXPLOSION",
"NECROMANCY",
"LIGHT",
"EXPLOSION",
"EXPLOSION_LIGHT",
"FIRE_BLAST",
"POISON_BLAST",
"ALCOHOL_BLAST",
"THUNDER_BLAST",
"BERSERK_FIELD",
"POLYMORPH_FIELD",
"CHAOS_POLYMORPH_FIELD",
"ELECTROCUTION_FIELD",
"FREEZE_FIELD",
"REGENERATION_FIELD",
"TELEPORTATION_FIELD",
"LEVITATION_FIELD",
"SHIELD_FIELD",
"PROJECTILE_TRANSMUTATION_FIELD",
"PROJECTILE_THUNDER_FIELD",
"PROJECTILE_GRAVITY_FIELD",
"VACUUM_POWDER",
"VACUUM_LIQUID",
"VACUUM_ENTITIES",
"SEA_LAVA",
"SEA_ALCOHOL",
"SEA_OIL",
"SEA_WATER",
"SEA_ACID",
"SEA_ACID_GAS",
"CLOUD_WATER",
"CLOUD_OIL",
"CLOUD_BLOOD",
"CLOUD_ACID",
"CLOUD_THUNDER",
"ELECTRIC_CHARGE",
"MATTER_EATER",
"FREEZE",
"HITFX_BURNING_CRITICAL_HIT",
"HITFX_CRITICAL_WATER",
"HITFX_CRITICAL_OIL",
"HITFX_CRITICAL_BLOOD",
"HITFX_TOXIC_CHARM",
"HITFX_EXPLOSION_SLIME",
"HITFX_EXPLOSION_SLIME_GIGA",
"HITFX_EXPLOSION_ALCOHOL",
"HITFX_EXPLOSION_ALCOHOL_GIGA",
"HITFX_PETRIFY",
"ROCKET_DOWNWARDS",
"ROCKET_OCTAGON",
"FIZZLE",
"BOUNCE_EXPLOSION",
"BOUNCE_SPARK",
"BOUNCE_LASER",
"BOUNCE_LASER_EMITTER",
"BOUNCE_LARPA",
"BOUNCE_SMALL_EXPLOSION",
"BOUNCE_LIGHTNING",
"BOUNCE_HOLE",
"FIREBALL_RAY",
"LIGHTNING_RAY",
"TENTACLE_RAY",
"LASER_EMITTER_RAY",
"FIREBALL_RAY_LINE",
"FIREBALL_RAY_ENEMY",
"LIGHTNING_RAY_ENEMY",
"TENTACLE_RAY_ENEMY",
"GRAVITY_FIELD_ENEMY",
"CURSE",
"CURSE_WITHER_PROJECTILE",
"CURSE_WITHER_EXPLOSION",
"CURSE_WITHER_MELEE",
"CURSE_WITHER_ELECTRICITY",
"ORBIT_DISCS",
"ORBIT_FIREBALLS",
"ORBIT_NUKES",
"ORBIT_LASERS",
"ORBIT_LARPA",
"CHAIN_SHOT",
"ARC_ELECTRIC",
"ARC_FIRE",
"ARC_GUNPOWDER",
"ARC_POISON",
"CRUMBLING_EARTH_PROJECTILE",
"X_RAY",
"UNSTABLE_GUNPOWDER",
"ACID_TRAIL",
"POISON_TRAIL",
"OIL_TRAIL",
"WATER_TRAIL",
"GUNPOWDER_TRAIL",
"FIRE_TRAIL",
"BURN_TRAIL",
"TORCH",
"TORCH_ELECTRIC",
"ENERGY_SHIELD",
"ENERGY_SHIELD_SECTOR",
"ENERGY_SHIELD_SHOT",
"TINY_GHOST",
"OCARINA_A",
"OCARINA_B",
"OCARINA_C",
"OCARINA_D",
"OCARINA_E",
"OCARINA_F",
"OCARINA_GSHARP",
"OCARINA_A2",
"KANTELE_A",
"KANTELE_D",
"KANTELE_DIS",
"KANTELE_E",
"KANTELE_G",
"RANDOM_SPELL",
"RANDOM_PROJECTILE",
"RANDOM_MODIFIER",
"RANDOM_STATIC_PROJECTILE",
"DRAW_RANDOM",
"DRAW_RANDOM_X3",
"DRAW_3_RANDOM",
"ALL_NUKES",
"ALL_DISCS",
"ALL_ROCKETS",
"ALL_DEATHCROSSES",
"ALL_BLACKHOLES",
"ALL_ACID",
"ALL_SPELLS",
"SUMMON_PORTAL",
"ADD_TRIGGER",
"ADD_TIMER",
"ADD_DEATH_TRIGGER",
"LARPA_CHAOS",
"LARPA_DOWNWARDS",
"LARPA_UPWARDS",
"LARPA_CHAOS_2",
"LARPA_DEATH",
"ALPHA",
"GAMMA",
"TAU",
"OMEGA",
"MU",
"PHI",
"SIGMA",
"ZETA",
"DIVIDE_2",
"DIVIDE_3",
"DIVIDE_4",
"DIVIDE_10",
"METEOR_RAIN",
"WORM_RAIN",
"RESET",
"IF_ENEMY",
"IF_PROJECTILE",
"IF_HP",
"IF_HALF",
"IF_END",
"IF_ELSE",
"COLOUR_RED",
"COLOUR_ORANGE",
"COLOUR_GREEN",
"COLOUR_YELLOW",
"COLOUR_PURPLE",
"COLOUR_BLUE",
"COLOUR_RAINBOW",
"COLOUR_INVIS",
"RAINBOW_TRAIL",
};
