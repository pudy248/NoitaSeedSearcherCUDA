#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "materials.h"
#include "../misc/datatypes.h"

enum AlchemyOrdering
{
	UNORDERED,
	ONLY_CONSUMED,
	STRICT_ORDERED
};

struct AlchemyRecipe {
	Material mats[4];

	__host__ __device__ AlchemyRecipe() {}
	__host__ __device__ AlchemyRecipe(Material mat1, Material mat2, Material mat3) {
		mats[0] = mat1;
		mats[1] = mat2;
		mats[2] = mat3;
	}

	__host__ __device__ static bool Equals(AlchemyRecipe reference, AlchemyRecipe test, AlchemyOrdering ordered) {
		if (ordered == STRICT_ORDERED)
		{
			bool passed1 = reference.mats[0] == MATERIAL_NONE || reference.mats[0] == test.mats[0];
			bool passed2 = reference.mats[1] == MATERIAL_NONE || reference.mats[1] == test.mats[1];
			bool passed3 = reference.mats[2] == MATERIAL_NONE || reference.mats[2] == test.mats[2];

			return passed1 && passed2 && passed3;
		}
		else if (ordered == ONLY_CONSUMED)
		{
			bool passed1 = reference.mats[0] == MATERIAL_NONE || (reference.mats[0] == test.mats[0] || reference.mats[0] == test.mats[2]);
			bool passed2 = reference.mats[1] == MATERIAL_NONE || (reference.mats[1] == test.mats[1]);
			bool passed3 = reference.mats[2] == MATERIAL_NONE || (reference.mats[2] == test.mats[0] || reference.mats[2] == test.mats[2]);

			return passed1 && passed2 && passed3;
		}
		else
		{
			bool passed1 = reference.mats[0] == MATERIAL_NONE || (reference.mats[0] == test.mats[0] || reference.mats[0] == test.mats[1] || reference.mats[0] == test.mats[2]);
			bool passed2 = reference.mats[1] == MATERIAL_NONE || (reference.mats[1] == test.mats[0] || reference.mats[1] == test.mats[1] || reference.mats[1] == test.mats[2]);
			bool passed3 = reference.mats[2] == MATERIAL_NONE || (reference.mats[2] == test.mats[0] || reference.mats[2] == test.mats[1] || reference.mats[2] == test.mats[2]);

			return passed1 && passed2 && passed3;
		}
	}
};

#define alchemyLiquidCount 30
__device__ const Material alchemyLiquids[] = {
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
__device__ const Material alchemySolids[] = {
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

__device__ AlchemyRecipe MaterialPicker(NollaPRNG& prng, uint worldSeed)
{
	AlchemyRecipe result;
	int counter = 0;
	int failed = 0;
	while (counter < 3 && failed < 99999)
	{
		int r = (int)(prng.Next() * alchemyLiquidCount);
		Material picked = alchemyLiquids[r];
		bool duplicate = false;
		for (int i = 0; i < counter; i++)
		{
			if (picked == result.mats[i]) duplicate = true;
		}
		if (duplicate) failed++;
		else
		{
			result.mats[counter++] = picked;
		}
	}
	failed = 0;
	while (counter < 4 && failed < 99999)
	{
		int r = (int)(prng.Next() * alchemySolidCount);
		Material picked = alchemySolids[r];
		bool duplicate = false;
		for (int i = 0; i < counter; i++)
		{
			if (picked == result.mats[i]) duplicate = true;
		}
		if (duplicate) failed++;
		else
		{
			result.mats[counter++] = picked;
		}
	}

	NollaPRNG prng2((worldSeed >> 1) + 12534);
	for (int i = 3; i >= 0; i--)
	{
		int r = (int)(prng2.Next() * (i + 1));
		Material temp = result.mats[i];
		result.mats[i] = result.mats[r];
		result.mats[r] = temp;
	}

	prng.Next();
	prng.Next();
	return result;
}