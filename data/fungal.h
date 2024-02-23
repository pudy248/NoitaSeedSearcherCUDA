#pragma once
#include "../platforms/platform_implementation.h"
#include "../include/enums.h"

_data constexpr int fungalMaterialsFromCount = 18;
_data const ShiftSource fungalMaterialsFrom[] = {
	SS_WATER,
	SS_LAVA,
	SS_RADIOACTIVE_LIQUID,
	SS_OIL,
	SS_BLOOD,
	SS_FUNGI,
	SS_BLOOD_WORM,
	SS_ACID,
	SS_ACID_GAS,
	SS_MAGIC_LIQUID_POLYMORPH,
	SS_MAGIC_LIQUID_BERSERK,
	SS_DIAMOND,
	SS_SILVER,
	SS_STEAM,
	SS_SAND,
	SS_SNOW_STICKY,
	SS_ROCK_STATIC,
	SS_GOLD
};
_data constexpr float fungalSumFrom = 11.4503f;
_data const float fungalProbsFrom[] = {
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	0.4f,
	0.4f,
	0.4f,
	0.6f,
	0.6f,
	0.2f,
	0.4f,
	0.4f,
	0.05f,
	0.0003f,
};

_data constexpr int fungalMaterialsToCount = 31;
_data const ShiftDest fungalMaterialsTo[] = {
	SD_WATER,
	SD_LAVA,
	SD_RADIOACTIVE_LIQUID,
	SD_OIL,
	SD_BLOOD,
	SD_BLOOD_FUNGI,
	SD_ACID,
	SD_WATER_SWAMP,
	SD_ALCOHOL,
	SD_SIMA,
	SD_BLOOD_WORM,
	SD_POISON,
	SD_VOMIT,
	SD_PEA_SOUP,
	SD_FUNGI,
	SD_SAND,
	SD_DIAMOND,
	SD_SILVER,
	SD_STEAM,
	SD_ROCK_STATIC,
	SD_GUNPOWDER,
	SD_MATERIAL_DARKNESS,
	SD_MATERIAL_CONFUSION,
	SD_ROCK_STATIC_RADIOACTIVE,
	SD_MAGIC_LIQUID_POLYMORPH,
	SD_MAGIC_LIQUID_RANDOM_POLYMORPH,
	SD_MAGIC_LIQUID_TELEPORTATION,
	SD_URINE,
	SD_POO,
	SD_VOID_LIQUID,
	SD_CHEESE_STATIC,
};
_data constexpr float fungalSumTo = 20.63f;
_data const float fungalProbsTo[] = {
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	1,
	0.8f,
	0.8f,
	0.8f,
	0.8f,
	0.5f,
	0.5f,
	0.5f,
	0.5f,
	0.2f,
	0.02f,
	0.02f,
	0.15f,
	0.01f,
	0.01f,
	0.01f,
	0.01f,
};