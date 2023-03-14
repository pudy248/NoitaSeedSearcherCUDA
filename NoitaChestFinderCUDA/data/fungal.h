#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "materials.h"

struct FungalShift {
	Material from;
	bool fromFlask;
	Material to;
	bool toFlask;
	__host__ __device__ FungalShift() {
		from = MATERIAL_NONE;
		to = MATERIAL_NONE;
		fromFlask = false;
		toFlask = false;
	}
	__host__ __device__ FungalShift(Material _from, bool _fromFlask, Material _to, bool _toFlask) {
		from = _from;
		fromFlask = _fromFlask;
		to = _to;
		toFlask = _toFlask;
	}
};

#define maxFungalShifts 20

#define fungalMaterialsFromCount 18
__device__ __constant__ Material fungalMaterialsFrom[] = {
	Material::WATER,
	Material::LAVA,
	Material::RADIOACTIVE_LIQUID,
	Material::OIL,
	Material::BLOOD,
	Material::FUNGI,
	Material::BLOOD_WORM,
	Material::ACID,
	Material::ACID_GAS,
	Material::MAGIC_LIQUID_POLYMORPH,
	Material::MAGIC_LIQUID_BERSERK,
	Material::DIAMOND,
	Material::SILVER,
	Material::STEAM,
	Material::SAND,
	Material::SNOW_STICKY,
	Material::ROCK_STATIC,
	Material::GOLD
};
#define fungalSumFrom 11.4053f
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
	0.005f,
	0.0003f,
};

#define fungalMaterialsToCount 31
__device__ __constant__ Material fungalMaterialsTo[] = {
	Material::WATER,
	Material::LAVA,
	Material::RADIOACTIVE_LIQUID,
	Material::OIL,
	Material::BLOOD,
	Material::BLOOD_FUNGI,
	Material::ACID,
	Material::WATER_SWAMP,
	Material::ALCOHOL,
	Material::SIMA,
	Material::BLOOD_WORM,
	Material::POISON,
	Material::VOMIT,
	Material::PEA_SOUP,
	Material::FUNGI,
	Material::SAND,
	Material::DIAMOND,
	Material::SILVER,
	Material::STEAM,
	Material::ROCK_STATIC,
	Material::GUNPOWDER,
	Material::MATERIAL_DARKNESS,
	Material::MATERIAL_CONFUSION,
	Material::ROCK_STATIC_RADIOACTIVE,
	Material::MAGIC_LIQUID_POLYMORPH,
	Material::MAGIC_LIQUID_RANDOM_POLYMORPH,
	Material::MAGIC_LIQUID_TELEPORTATION,
	Material::URINE,
	Material::POO,
	Material::VOID_LIQUID,
	Material::CHEESE_STATIC,
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