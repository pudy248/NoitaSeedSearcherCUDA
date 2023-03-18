#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "materials.h"
#include "../misc/datatypes.h"

struct AlchemyRecipe {
	Material mats[4];
	float prob;
	uint iseed;

	__host__ __device__ AlchemyRecipe() {}
	__host__ __device__ AlchemyRecipe(Material mat1, Material mat2, Material mat3) {
		mats[0] = mat1;
		mats[1] = mat2;
		mats[2] = mat3;
	}

	__host__ __device__ bool operator==(AlchemyRecipe other) {
		bool passed1 = other.mats[0] == MATERIAL_NONE || (other.mats[0] == mats[0] || other.mats[0] == mats[2]);
		bool passed2 = other.mats[1] == MATERIAL_NONE || (other.mats[1] == mats[1]);
		bool passed3 = other.mats[2] == MATERIAL_NONE || (other.mats[2] == mats[0] || other.mats[2] == mats[2]);

		return passed1 && passed2 && passed3;
	}
};

#define alchemyLiquidCount 30
__device__ __constant__ Material alchemyLiquids[] = {
	Material::ACID,
	Material::ALCOHOL,
	Material::BLOOD,
	Material::BLOOD_FUNGI,
	Material::BLOOD_WORM,
	Material::CEMENT,
	Material::LAVA,
	Material::MAGIC_LIQUID_BERSERK,
	Material::MAGIC_LIQUID_CHARM,
	Material::MAGIC_LIQUID_FASTER_LEVITATION,
	Material::MAGIC_LIQUID_FASTER_LEVITATION_AND_MOVEMENT,
	Material::MAGIC_LIQUID_INVISIBILITY,
	Material::MAGIC_LIQUID_MANA_REGENERATION,
	Material::MAGIC_LIQUID_MOVEMENT_FASTER,
	Material::MAGIC_LIQUID_PROTECTION_ALL,
	Material::MAGIC_LIQUID_TELEPORTATION,
	Material::MAGIC_LIQUID_UNSTABLE_POLYMORPH,
	Material::MAGIC_LIQUID_UNSTABLE_TELEPORTATION,
	Material::MAGIC_LIQUID_WORM_ATTRACTOR,
	Material::MATERIAL_CONFUSION,
	Material::MUD,
	Material::OIL,
	Material::POISON,
	Material::RADIOACTIVE_LIQUID,
	Material::SWAMP,
	Material::URINE,
	Material::WATER,
	Material::WATER_ICE,
	Material::WATER_SWAMP,
	Material::MAGIC_LIQUID_RANDOM_POLYMORPH
};

#define alchemySolidCount 18
__device__ __constant__ Material alchemySolids[] = {
	Material::BONE,
	Material::BRASS,
	Material::COAL,
	Material::COPPER,
	Material::DIAMOND,
	Material::FUNGI,
	Material::GOLD,
	Material::GRASS,
	Material::GUNPOWDER,
	Material::GUNPOWDER_EXPLOSIVE,
	Material::ROTTEN_MEAT,
	Material::SAND,
	Material::SILVER,
	Material::SLIME,
	Material::SNOW,
	Material::SOIL,
	Material::WAX,
	Material::HONEY
};

__device__ uint* alchemyLGMRandom(uint* iseed, uint count) {
	while (count > 0) {
		*iseed = 16807 * (*iseed % 127773) - 2836 * (*iseed / 127773);
		//if (*iseed < 0) {
		//	*iseed = *iseed + 2147483647U;
		//};
		count--;
	}
	return iseed;
}

__device__ uint alchemyInit(uint seed) {
	uint iseed = (uint)((double)seed * 0.17127 + 1323.5903);
	alchemyLGMRandom(&iseed, 6);
	return iseed;
}

__device__ void alchemyShuffle(AlchemyRecipe* recipe, uint seed) {
	uint nseed = (seed >> 1) + 12534;
	alchemyLGMRandom(&nseed, 1);
	int index[4] = { 0,0,0,0 };
	for (int i = 0; i < 3; i++) {
		alchemyLGMRandom(&nseed, 1);
		index[i] = (int)((float)nseed / 2147483647.0f * ((float)(3 - i) + 1.0f));
	}
	int x = 3;
	for (int i = 0; i < 3; i++) {
		Material temp = recipe->mats[x];
		recipe->mats[x] = recipe->mats[index[i]];
		recipe->mats[index[i]] = temp;
		x -= 1;
	}
}

__device__ AlchemyRecipe alchemyGetRecipe(uint seed, uint iseed) {
	AlchemyRecipe recipe;
	int index[] = { 0, 0, 0, 0 };

	int i = 0;
	int x = 0;
	while (x < 3 && i < 1000) {
		alchemyLGMRandom(&iseed, 1);
		int temp = (int)(((float)iseed / 2147483647.0f) * alchemyLiquidCount);
		if (index[0] != temp && index[1] != temp && index[2] != temp && index[3] != temp) {
			index[x] = temp;
			x++;
		}
		i++;
	}
	if (i >= 1000) memset(recipe.mats, 0, 3);
	alchemyLGMRandom(&iseed, 1);
	index[3] = (int)(((float)iseed / 2147483647.0f) * alchemySolidCount);
	for (int n = 0; n < 3; n++) {
		recipe.mats[n] = alchemyLiquids[index[n]];
	}
	recipe.mats[3] = alchemySolids[index[3]];
	alchemyShuffle(&recipe, seed);
	alchemyLGMRandom(&iseed, 1);
	recipe.prob = 10 - (int)((float)iseed / 2147483647.0f * -91.0f);
	alchemyLGMRandom(&iseed, 1);
	recipe.iseed = iseed;
	return recipe;
}