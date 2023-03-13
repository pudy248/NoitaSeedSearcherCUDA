#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "misc/datatypes.h"
#include "misc/noita_random.h"
#include "misc/utilities.h"

#include "data/rains.h"
#include "data/materials.h"
#include "data/fungal.h"
#include "data/alchemy.h"

#include <iostream>

struct PrecheckConfig {
	bool checkRain;
	Material rain;
	bool checkStartingFlask;
	Material startingFlask;
	bool checkAlchemy;
	AlchemyRecipe LC;
	AlchemyRecipe AP;
	bool checkFungalShifts;
	FungalShift shifts[20];
	bool checkBiomeModifiers;
	byte biomeModifiers[9];
	bool checkPerks;
	sbyte perks[130];
};

__device__ bool CheckRain(NoitaRandom* random, Material rain) {
	float rainfall_chance = 1.0f / 15;
	IntPair rnd = { 7893434, 3458934 };
	if (random_next(0, 1, random, &rnd) <= rainfall_chance) {
		int seedRainIndex = pick_random_from_table_backwards(rainProbs, rainCount, random, &rnd);
		Material seedRain = rainMaterials[seedRainIndex];
		//printf("wanted %i, got %i\n", rain, seedRain);
		return (Material)seedRain == rain;
	}
	//printf("wanted %i, got %i\n", rain, MATERIAL_NONE);
	return rain == MATERIAL_NONE;
}
__device__ bool CheckStartingFlask(NoitaRandom* random, Material starting_flask) {
	random->SetRandomSeed(-4.5, -4);
	Material material = Material::MATERIAL_NONE;
	int res = random->Random(1, 100);
	if (res <= 65)
	{
		res = random->Random(1, 100);
		if (res <= 10)
			material = Material::MUD;
		else if (res <= 20)
			material = Material::WATER_SWAMP;
		else if (res <= 30)
			material = Material::WATER_SALT;
		else if (res <= 40)
			material = Material::SWAMP;
		else if (res <= 50)
			material = Material::SNOW;
		else
			material = Material::WATER;
	}
	else if (res <= 70)
		material = Material::BLOOD;
	else if (res <= 99)
	{
		res = random->Random(0, 100);
		Material magic[] = {
			Material::ACID,
			Material::MAGIC_LIQUID_POLYMORPH,
			Material::MAGIC_LIQUID_RANDOM_POLYMORPH,
			Material::MAGIC_LIQUID_BERSERK,
			Material::MAGIC_LIQUID_CHARM,
			Material::MAGIC_LIQUID_MOVEMENT_FASTER
		};
		material = magic[random->Random(0, 5)];
	}
	else
	{
		res = random->Random(0, 100000);
		if (res == 666) material = Material::URINE;
		else if (res == 79) material = Material::GOLD;
		else if (random->Random(0, 1) == 0) material = Material::SLIME;
		else material = Material::GUNPOWDER_UNSTABLE;
	}
	return material == starting_flask;
}
__device__ bool CheckAlchemy(NoitaRandom* random, AlchemyRecipe LC, AlchemyRecipe AP) {
	AlchemyRecipe lc = alchemyGetRecipe(random->world_seed, alchemyInit(random->world_seed));
	AlchemyRecipe ap = alchemyGetRecipe(random->world_seed, lc.iseed);
	for (int i = 0; i < 3; i++) {
		bool lcFound = false;
		bool apFound = false;
		for (int j = 0; j < 3; j++) {
			if (lc.mats[j] == LC.mats[i]) lcFound = true;
			if (ap.mats[j] == AP.mats[i]) apFound = true;
		}
		if (LC.mats[i] != MATERIAL_NONE && !lcFound) return false;
		if (AP.mats[i] != MATERIAL_NONE && !apFound) return false;
	}
	return true;
}
__device__ bool CheckFungalShifts(NoitaRandom* random, FungalShift shift[maxFungalShifts]) {
	for (int i = 0; i < maxFungalShifts; i++) {
		FungalShift result = { MATERIAL_NONE, false, MATERIAL_NONE, false };
		random->SetRandomSeed(89346, 42345 + i);
		IntPair rnd = { 9123,58925 + i };
		result.from = (Material)pick_random_from_table_weighted(fungalProbsFrom, fungalSumFrom, fungalMaterialsFromCount, random, &rnd);
		result.to = (Material)pick_random_from_table_weighted(fungalProbsTo, fungalSumTo, fungalMaterialsToCount, random, &rnd);
		if (random_nexti(1, 100, random, &rnd) <= 75) {
			if (random_nexti(1, 100, random, &rnd) <= 50)
				result.fromFlask = true;
			else
				result.toFlask = true;
		}

		if (shift[i].from != MATERIAL_NONE && shift[i].from != result.from) return false;
		if (shift[i].to != MATERIAL_NONE && shift[i].to != result.to) return false;
		if (shift[i].fromFlask && !result.fromFlask) return false;
		if (shift[i].toFlask && !result.toFlask) return false;
	}
	return false;
}
__device__ bool CheckBiomeModifiers(NoitaRandom* random, byte biomeModifiers[9]) {
	return true;
}
__device__ bool CheckPerks(NoitaRandom* random, sbyte perks[130]) {
	return true;
}
__device__ bool PrecheckSeed(uint seed, PrecheckConfig config) {
	NoitaRandom sharedRandom = NoitaRandom(seed);
	//printf("precheck %i\n", seed);
	if (config.checkRain)
		if (!CheckRain(&sharedRandom, config.rain)) return false;
	if (config.checkStartingFlask)
		if (!CheckStartingFlask(&sharedRandom, config.startingFlask)) return false;
	if (config.checkAlchemy)
		if (!CheckAlchemy(&sharedRandom, config.LC, config.AP)) return false;
	if (config.checkFungalShifts)
		if (!CheckFungalShifts(&sharedRandom, config.shifts)) return false;
	if (config.checkBiomeModifiers)
		if (!CheckBiomeModifiers(&sharedRandom, config.biomeModifiers)) return false;
	if (config.checkPerks)
		if (!CheckPerks(&sharedRandom, config.perks)) return false;
	return true;
}