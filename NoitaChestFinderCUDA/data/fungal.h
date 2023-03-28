#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "materials.h"

enum ShiftSource : short {
	SS_NONE = MATERIAL_NONE,
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
	SD_NONE = MATERIAL_NONE,
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
	bool fromFlask;
	ShiftDest to;
	bool toFlask;
	__host__ __device__ FungalShift() {
		from = SS_NONE;
		to = SD_NONE;
		fromFlask = false;
		toFlask = false;
	}
	__host__ __device__ FungalShift(ShiftSource _from, bool _fromFlask, ShiftDest _to, bool _toFlask) {
		from = _from;
		fromFlask = _fromFlask;
		to = _to;
		toFlask = _toFlask;
	}

	__host__ __device__ bool operator==(FungalShift other) {
		if (other.from != SS_NONE && other.from != from) return false;
		if (other.to != SD_NONE && other.to != to) return false;
		if (other.fromFlask && !fromFlask) return false;
		if (other.toFlask && !toFlask) return false;
		return true;
	}
};

#define maxFungalShifts 3

#define fungalMaterialsFromCount 18
__device__ __constant__ ShiftSource fungalMaterialsFrom[] = {
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
#define fungalSumFrom 11.4503f
__device__ __constant__ const float fungalProbsFrom[] = {
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

#define fungalMaterialsToCount 31
__device__ __constant__ ShiftDest fungalMaterialsTo[] = {
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
#define fungalSumTo 20.63f
__device__ __constant__ float fungalProbsTo[] = {
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