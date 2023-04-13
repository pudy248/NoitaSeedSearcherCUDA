#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "misc/datatypes.h"
#include "misc/noita_random.h"
#include "misc/utilities.h"

#include "data/rains.h"
#include "data/materials.h"
#include "data/alchemy.h"
#include "data/fungal.h"
#include "data/modifiers.h"
#include "data/perks.h"

#include "WorldgenSearch.h"
#include "Filters.h"

#include <iostream>

struct PrecheckConfig
{
	bool printPassed;

	bool checkRain;
	Material rain;
	bool checkStartingFlask;
	Material startingFlask;
	bool checkAlchemy;
	AlchemyRecipe LC;
	AlchemyRecipe AP;
	bool checkFungalShifts;
	bool orderedShifts;
	FungalShift shifts[maxFungalShifts];
	bool checkBiomeModifiers;
	BiomeModifier biomeModifiers[9];
	bool checkPerks;
	byte perks[130];
	bool checkUpwarps;
	FilterConfig fCfg;
	LootConfig lCfg;
};

__device__ bool CheckRain(NoitaRandom* random, Material rain)
{
	float rainfall_chance = 1.0f / 15;
	IntPair rnd = { 7893434, 3458934 };
	if (random_next(0, 1, random, &rnd) <= rainfall_chance)
	{
		int seedRainIndex = pick_random_from_table_backwards(rainProbs, rainCount, random, &rnd);
		Material seedRain = rainMaterials[seedRainIndex];
		//printf("wanted %i, got %i\n", rain, seedRain);
		return (Material)seedRain == rain;
	}
	//printf("wanted %i, got %i\n", rain, MATERIAL_NONE);
	return rain == MATERIAL_NONE;
}
__device__ bool CheckStartingFlask(NoitaRandom* random, Material starting_flask)
{
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
__device__ bool CheckAlchemy(NoitaRandom* random, AlchemyRecipe LC, AlchemyRecipe AP)
{
	AlchemyRecipe lc = alchemyGetRecipe(random->world_seed, alchemyInit(random->world_seed));
	AlchemyRecipe ap = alchemyGetRecipe(random->world_seed, lc.iseed);
	return lc == LC && ap == AP;
}
__device__ bool CheckFungalShifts(NoitaRandom* random, bool orderedShifts, FungalShift shift[maxFungalShifts])
{
	bool shiftFound[maxFungalShifts];
	for (int i = 0; i < maxFungalShifts; i++) shiftFound[i] = false;
	for (int i = 0; i < maxFungalShifts; i++)
	{
		FungalShift result;
		random->SetRandomSeed(89346, 42345 + i);
		IntPair rnd = { 9123,58925 + i };
		result.from = fungalMaterialsFrom[pick_random_from_table_weighted(fungalProbsFrom, fungalSumFrom, fungalMaterialsFromCount, random, &rnd)];
		result.to = fungalMaterialsTo[pick_random_from_table_weighted(fungalProbsTo, fungalSumTo, fungalMaterialsToCount, random, &rnd)];
		if (random_nexti(1, 100, random, &rnd) <= 75)
		{
			if (random_nexti(1, 100, random, &rnd) <= 50)
				result.fromFlask = true;
			else
				result.toFlask = true;
		}
		if (orderedShifts && !(result == shift[i])) return false;
		else
		{
			for (int j = 0; j < maxFungalShifts; j++)
			{
				if (!shiftFound[j] && result == shift[j])
				{
					shiftFound[j] = true;
					break;
				}
			}
		}
	}
	if (!orderedShifts)
		for (int i = 0; i < maxFungalShifts; i++) if (!shiftFound[i]) return false;
	return true;
}
__device__ bool CheckBiomeModifiers(NoitaRandom* random, BiomeModifier biomeModifiers[9])
{
	BiomeModifier modifiers[9];
	memset(modifiers, 0, 9);
	IntPair rnd = { 347893,90734 };
	for (int i = 0; i < 9; i++)
	{
		float chance = 0.1f;
		if (i == 0) chance = 0.2f;
		if (i == 1) chance = 0.15f;
		if (random_next(0, 1, random, &rnd) > chance) continue;
		modifiers[i] = (BiomeModifier)(pick_random_from_table_weighted(biomeModifierProbs, biomeModifierProbSum, biomeModifierCount, random, &rnd) + 1);
	}
	for (int i = 0; i < 9; i++) if (biomeModifiers[i] != BM_NONE && modifiers[i] != biomeModifiers[i]) return false;
	return true;
}
__device__ bool CheckPerks(NoitaRandom* random, byte perks[130])
{
	const int MIN_DISTANCE_BETWEEN_DUPLICATE_PERKS = 4;
	const short DEFAULT_MAX_STACKABLE_PERK_COUNT = 128;

	random->SetRandomSeed(1, 2);

	byte* perkDeck = (byte*)malloc(130 * sizeof(sbyte));
	short* stackable_distances = (short*)malloc(perkCount * sizeof(short));
	short* stackable_count = (short*)malloc(perkCount * sizeof(short));

	int perkDeckIdx = 0;
	for (int i = 0; i < 130; i++) perkDeck[i] = PERK_NONE;
	for (int i = 0; i < perkCount; i++) stackable_distances[i] = -1;
	for (int i = 0; i < perkCount; i++) stackable_count[i] = -1;


	for (int i = 0; i < perkCount; i++)
	{
		PerkData perkData = perkAttrs[i];
		if (perkData.not_default) continue;

		int how_many_times = 1;
		stackable_distances[i] = -1;
		stackable_count[i] = -1;

		if (perkData.stackable)
		{
			byte max_perks = random->Random(1, 2);
			if (perkData.max_in_pool != 0)
			{
				max_perks = random->Random(1, perkData.max_in_pool);
			}


			if (perkData.stackable_max != 0)
			{
				stackable_count[i] = perkData.stackable_max;
			}
			else
			{
				stackable_count[i] = DEFAULT_MAX_STACKABLE_PERK_COUNT;
			}

			if (perkData.stackable_rare)
			{
				max_perks = 1;
			}

			if (perkData.stackable_how_often_reappears != 0)
			{
				stackable_distances[i] = perkData.stackable_how_often_reappears;
			}
			else
			{
				stackable_distances[i] = MIN_DISTANCE_BETWEEN_DUPLICATE_PERKS;
			}

			how_many_times = random->Random(1, max_perks);
		}

		for (int j = 0; j < how_many_times; j++)
		{
			perkDeck[perkDeckIdx++] = i + 1;
		}
	}

	shuffle_table(perkDeck, random, perkDeckIdx - 1);

	for (int i = perkDeckIdx - 1; i >= 0; i--)
	{
		byte perk = perkDeck[i];
		if (stackable_distances[perk] != -1)
		{
			short min_distance = stackable_distances[perk];
			bool remove_me = false;

			for (int ri = i - min_distance; ri < i; ri++)
			{
				if (ri >= 0 && perkDeck[ri] == perk)
				{
					remove_me = true;
					break;
				}
			}

			if (remove_me) perkDeck[i] = 0;
		}
	}

	perkDeckIdx = 0;
	for (int i = 0; i < 130; i++)
	{
		if (perkDeck[i] != 0) perkDeck[perkDeckIdx++] = perkDeck[i];
	}
	for (int i = perkDeckIdx; i < 130; i++)
	{
		perkDeck[i] = 0;
	}

	free(stackable_count);
	free(stackable_distances);

	bool passed = true;
	for (int i = 0; i < 130; i += 3)
	{
		bool found[3] = { false, false, false };
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				if (perks[i + j] != PERK_NONE && perks[i + j] != perkDeck[i + k])
					found[j] = true;
			}
		}
		if (!(found[0] && found[1] && found[2])) passed = false;
	}
	free(perkDeck);
	return passed;
}
__device__ bool CheckUpwarps(NoitaRandom* random, FilterConfig fCfg, LootConfig lCfg)
{
	byte bytes[1000];
	int offset = 0;
	int tmp = 0;
	spawnChest(315, 17, random->world_seed, lCfg, bytes, offset, tmp);
	Spawnable* s = (Spawnable*)(bytes + 1);
	SpawnableBlock b = { random->world_seed, 1, &s };

	bool passed = SpawnablesPassed(b, fCfg, false);

	offset = 0;
	spawnChest(75, 117, random->world_seed, lCfg, bytes, offset, tmp);
	bool passed2 = SpawnablesPassed(b, fCfg, false);

	return passed || passed2;
}

__device__ bool PrecheckSeed(uint seed, PrecheckConfig config)
{
	NoitaRandom sharedRandom = NoitaRandom(seed);

	//Keep ordered by total runtime, so faster checks are run first and long checks can be skipped
	if (config.checkRain)
		if (!CheckRain(&sharedRandom, config.rain)) return false;
	if (config.checkStartingFlask)
		if (!CheckStartingFlask(&sharedRandom, config.startingFlask)) return false;
	//if (config.checkAlchemy)
	//	if (!CheckAlchemy(&sharedRandom, config.LC, config.AP)) return false;
	//if (config.checkFungalShifts)
	//	if (!CheckFungalShifts(&sharedRandom, config.orderedShifts, config.shifts)) return false;
	//if (config.checkBiomeModifiers)
	//	if (!CheckBiomeModifiers(&sharedRandom, config.biomeModifiers)) return false;
	if (config.checkUpwarps)
		if (!CheckUpwarps(&sharedRandom, config.fCfg, config.lCfg)) return false;
	if (config.checkPerks)
		if (!CheckPerks(&sharedRandom, config.perks)) return false;
	if (config.printPassed) printf("Precheck passed: %i\n", seed);
	return true;
}



bool ValidateBiomeModifierConfig(PrecheckConfig c)
{
	const BiomeBlacklist biomeBlacklists[] = {
		{}, //BM_NONE
		{}, //BM_MOIST
		{}, //BM_FOG_OF_WAR_REAPPEARS
		{}, //BM_HIGH_GRAVITY
		{}, //BM_LOW_GRAVITY
		{}, //BM_CONDUCTIVE
		{}, //BM_HOT
		{B_SNOWCASTLE}, //BM_GOLD_VEIN
		{B_FUNGICAVE, B_SNOWCASTLE, B_RAINFOREST, B_VAULT, B_CRYPT}, //BM_GOLD_VEIN_SUPER
		{B_SNOWCAVE, B_SNOWCASTLE, B_RAINFOREST}, //BM_PLANT_INFESTED
		{}, //BM_FURNISHED
		{}, //BM_BOOBY_TRAPPED
		{B_SNOWCAVE, B_VAULT, B_CRYPT}, //BM_PERFORATED
		{}, //BM_SPOOKY
		{}, //BM_GRAVITY_FIELDS
		{B_SNOWCASTLE, B_SNOWCAVE, B_FUNGICAVE}, //BM_FUNGAL
		{B_SNOWCAVE, B_RAINFOREST, B_VAULT}, ///BM_FLOODED
		{B_EXCAVATIONSITE, B_SNOWCAVE, B_SNOWCASTLE, B_VAULT, B_CRYPT}, //BM_GAS_FLOODED
		{B_EXCAVATIONSITE, B_SNOWCAVE}, //BM_SHIELDED
		{}, //BM_PROTECTION_FIELDS
		{B_COALMINE, B_EXCAVATIONSITE}, //BM_OMINOUS
		{B_COALMINE}, //BM_INVISIBILITY
		{B_COALMINE} //BM_WORMY
	};
	bool passed = true;
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			if (biomeBlacklists[c.biomeModifiers[i]].blacklist[j] == (Biome)(i + 1))
			{
				printf("Invalid biome modifier: %s cannot occur in biome %s\n", biomeModifierNames[c.biomeModifiers[i]], biomeNames[i + 1]);
				passed = false;
			}
		}
	}
	return passed;
}