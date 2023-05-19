#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "materials.h"

enum ShiftSource : short {
	SS_FLASK = -1,
	SS_NONE = MATERIAL_NONE,
	SS_VAR1 = MATERIAL_VAR1,
	SS_VAR2 = MATERIAL_VAR2,
	SS_VAR3 = MATERIAL_VAR3,
	SS_VAR4 = MATERIAL_VAR4,
	SS_WATER = Material::WATER,
	SS_LAVA = Material::LAVA,
	SS_RADIOACTIVE_LIQUID = Material::RADIOACTIVE_LIQUID,
	SS_OIL = Material::OIL,
	SS_BLOOD = Material::BLOOD,
	SS_FUNGI = Material::FUNGI,
	SS_BLOOD_WORM = Material::BLOOD_WORM,
	SS_ACID = Material::ACID,
	SS_ACID_GAS = Material::ACID_GAS,
	SS_MAGIC_LIQUID_POLYMORPH = Material::MAGIC_LIQUID_POLYMORPH,
	SS_MAGIC_LIQUID_BERSERK = Material::MAGIC_LIQUID_BERSERK,
	SS_DIAMOND = Material::DIAMOND,
	SS_SILVER = Material::SILVER,
	SS_STEAM = Material::STEAM,
	SS_SAND = Material::SAND,
	SS_SNOW_STICKY = Material::SNOW_STICKY,
	SS_ROCK_STATIC = Material::ROCK_STATIC,
	SS_GOLD = Material::GOLD
};

enum ShiftDest : short {
	SD_FLASK = -1,
	SD_NONE = MATERIAL_NONE,
	SD_VAR1 = MATERIAL_VAR1,
	SD_VAR2 = MATERIAL_VAR2,
	SD_VAR3 = MATERIAL_VAR3,
	SD_VAR4 = MATERIAL_VAR4,
	SD_WATER = Material::WATER,
	SD_LAVA = Material::LAVA,
	SD_RADIOACTIVE_LIQUID = Material::RADIOACTIVE_LIQUID,
	SD_OIL = Material::OIL,
	SD_BLOOD = Material::BLOOD,
	SD_BLOOD_FUNGI = Material::BLOOD_FUNGI,
	SD_ACID = Material::ACID,
	SD_WATER_SWAMP = Material::WATER_SWAMP,
	SD_ALCOHOL = Material::ALCOHOL,
	SD_SIMA = Material::SIMA,
	SD_BLOOD_WORM = Material::BLOOD_WORM,
	SD_POISON = Material::POISON,
	SD_VOMIT = Material::VOMIT,
	SD_PEA_SOUP = Material::PEA_SOUP,
	SD_FUNGI = Material::FUNGI,
	SD_SAND = Material::SAND,
	SD_DIAMOND = Material::DIAMOND,
	SD_SILVER = Material::SILVER,
	SD_STEAM = Material::STEAM,
	SD_ROCK_STATIC = Material::ROCK_STATIC,
	SD_GUNPOWDER = Material::GUNPOWDER,
	SD_MATERIAL_DARKNESS = Material::MATERIAL_DARKNESS,
	SD_MATERIAL_CONFUSION = Material::MATERIAL_CONFUSION,
	SD_ROCK_STATIC_RADIOACTIVE = Material::ROCK_STATIC_RADIOACTIVE,
	SD_MAGIC_LIQUID_POLYMORPH = Material::MAGIC_LIQUID_POLYMORPH,
	SD_MAGIC_LIQUID_RANDOM_POLYMORPH = Material::MAGIC_LIQUID_RANDOM_POLYMORPH,
	SD_MAGIC_LIQUID_TELEPORTATION = Material::MAGIC_LIQUID_TELEPORTATION,
	SD_URINE = Material::URINE,
	SD_POO = Material::POO,
	SD_VOID_LIQUID = Material::VOID_LIQUID,
	SD_CHEESE_STATIC = Material::CHEESE_STATIC,
};

struct FungalShift {
	ShiftSource from;
	ShiftDest to;
	bool fromFlask;
	bool toFlask;
	int minIdx;
	int maxIdx;
	__host__ __device__ constexpr FungalShift()
		: from(SS_NONE),to(SD_NONE),fromFlask(false),toFlask(false),minIdx(0),maxIdx(0) {}
	__host__ __device__ FungalShift(ShiftSource _from, ShiftDest _to, int _minIdx, int _maxIdx)
	{
		if (_from == SS_FLASK) { from = SS_NONE; fromFlask = true; }
		else { from = _from; fromFlask = false; }
		if (_to == SD_FLASK) { to = SD_NONE; toFlask = true; }
		else { to = _to; toFlask = false; }
		minIdx = _minIdx;
		maxIdx = _maxIdx;
	}

	__device__ static bool Equals(FungalShift reference, FungalShift test, int ptrs[4], Material vars[materialVarEntryCount * 4]) {
		if (reference.fromFlask && !test.fromFlask) return false;
		if (reference.toFlask && !test.toFlask) return false;
		if (!MaterialEquals((Material)reference.from, (Material)test.from, false, ptrs, vars)) return false;
		if (!MaterialEquals((Material)reference.to, (Material)test.to, false, ptrs, vars)) return false;
		return true;
	}
};

constexpr auto maxFungalShifts = 3;

constexpr auto fungalMaterialsFromCount = 18;
__device__ const ShiftSource fungalMaterialsFrom[] = {
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
constexpr auto fungalSumFrom = 11.4503f;
__device__ const float fungalProbsFrom[] = {
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

constexpr auto fungalMaterialsToCount = 31;
__device__ const ShiftDest fungalMaterialsTo[] = {
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
constexpr auto fungalSumTo = 20.63f;
__device__ const float fungalProbsTo[] = {
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