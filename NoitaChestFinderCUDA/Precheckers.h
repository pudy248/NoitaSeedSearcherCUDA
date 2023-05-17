#pragma once

#include "defines.h"

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
#include "data/temples.h"

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
	AlchemyOrdering orderedAlchemy;
	AlchemyRecipe LC;
	AlchemyRecipe AP;
	bool checkFungalShifts;
	FungalShift shifts[maxFungalShifts];
	bool checkBiomeModifiers;
	BiomeModifier biomeModifiers[9];
	bool checkPerks;
	PerkInfo perks[20];
	bool checkUpwarps;
	FilterConfig fCfg;
	LootConfig lCfg;
	bool checkShops;
	int minHMidx;
	int maxHMidx;
};


__device__ bool CheckRain(NoitaRandom* random, Material rain)
{
	float rainfall_chance = 1.0f / 15;
	IntPair rnd = { 7893434, 3458934 };
	if (random_next(0, 1, random, &rnd) <= rainfall_chance)
	{
		int seedRainIndex = pick_random_from_table_backwards(rainProbs, rainCount, random, &rnd);
		Material seedRain = rainMaterials[seedRainIndex];
		return (Material)seedRain == rain;
	}
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

__device__ bool CheckAlchemy(NoitaRandom* random, AlchemyRecipe LC, AlchemyRecipe AP, AlchemyOrdering ordered)
{
	NollaPrng prng(random->world_seed * 0.17127 + 1323.5903);
	for (int i = 0; i < 5; i++) prng.Next();
	AlchemyRecipe lc = MaterialPicker(prng, random->world_seed);
	AlchemyRecipe ap = MaterialPicker(prng, random->world_seed);
	return AlchemyRecipe::Equals(LC, lc, ordered) && AlchemyRecipe::Equals(AP, ap, ordered);
}

__device__ bool CheckFungalShifts(NoitaRandom* random, FungalShift shiftFilters[maxFungalShifts])
{
	//return true;
	
	FungalShift generatedShifts[maxFungalShifts];
	for (int i = 0; i < maxFungalShifts; i++)
	{
		random->SetRandomSeed(89346, 42345 + i);
		IntPair rnd = { 9123,58925 + i };
		generatedShifts[i].from = fungalMaterialsFrom[pick_random_from_table_weighted(fungalProbsFrom, fungalSumFrom, fungalMaterialsFromCount, random, &rnd)];
		generatedShifts[i].to = fungalMaterialsTo[pick_random_from_table_weighted(fungalProbsTo, fungalSumTo, fungalMaterialsToCount, random, &rnd)];
		if (random_nexti(1, 100, random, &rnd) <= 75)
		{
			if (random_nexti(1, 100, random, &rnd) <= 50)
				generatedShifts[i].fromFlask = true;
			else
				generatedShifts[i].toFlask = true;
		}
	}

	Material variables[4 * materialVarEntryCount];
	int ptrs[4] = { 0,0,0,0 };
	
	//populate vars
	for (int i = 0; i < maxFungalShifts; i++)
	{
		if (shiftFilters[i].to > SD_NONE && (int)shiftFilters[i].to <= SD_VAR4)
		{
			for (int j = shiftFilters[i].minIdx; j < shiftFilters[i].maxIdx; j++)
			{
				if (MaterialEquals((Material)shiftFilters[i].from, (Material)generatedShifts[j].from, false, ptrs, variables))
					MaterialEquals((Material)shiftFilters[i].to, (Material)generatedShifts[j].to, true, ptrs, variables);
					//this has side effects when the bool is true, it looks kinda funny out of context tho
			}
		}
	}

	for (int i = 0; i < maxFungalShifts; i++)
	{
		bool found = false;
		for (int j = shiftFilters[i].minIdx; j < shiftFilters[i].maxIdx; j++)
		{
			if (FungalShift::Equals(shiftFilters[i], generatedShifts[j], ptrs, variables))
			{
				found = true;
				break;
			}
		}
		if (!found) return false;
	}
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


__device__ bool CheckPerks(NoitaRandom* random, PerkInfo perks[20])
{
	const int MIN_DISTANCE_BETWEEN_DUPLICATE_PERKS = 4;
	const short DEFAULT_MAX_STACKABLE_PERK_COUNT = 128;

	random->SetRandomSeed(1, 2);

	byte perkDeck[130];
	short stackable_distances[perkCount];
	short stackable_count[perkCount];

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

	NoitaRandom rnd = NoitaRandom(random->world_seed);
	for (int i = 0; i < 20; i++)
	{
		PerkInfo perkToCkeck = perks[i];
		bool found = perks[i].p == PERK_NONE;
		for (int j = perkToCkeck.minPosition; j < perkToCkeck.maxPosition; j++)
		{
			if (perkToCkeck.p == perkDeck[j])
			{
				if (perkToCkeck.lottery)
				{
					int x = temple_perk_x[(j / 3)] + (int)roundf(((j % 3) + 0.5f) * 20);
					int y = temple_perk_y[(j / 3)];
					rnd.SetRandomSeed(x, y);
					if (rnd.Random(1, 100) > 50)
						found = true;
				}
				else
					found = true;
			}
		}
		if (!found) return false;
	}
	return true;
}


__device__ bool CheckUpwarps(NoitaRandom* random, FilterConfig fCfg, LootConfig lCfg)
{
	byte bytes[1000];
	int offset = 0;
	int tmp = 0;
	spawnChest(315, 17, random->world_seed, lCfg, bytes, offset, tmp);
	spawnChest(75, 117, random->world_seed, lCfg, bytes, offset, tmp);
	Spawnable* s = (Spawnable*)bytes;
	SpawnableBlock b = { random->world_seed, 2, &s };

	return SpawnablesPassed(b, fCfg, false);
}

__device__ bool CheckShops(NoitaRandom* random, FilterConfig fCfg, LootConfig lCfg, int minIdx, int maxIdx)
{
	int width = 132;
	constexpr int itemCount = 5;
	float stepSize = width / (float)itemCount;
	bool passed = false;

	byte bytes[3000];
	Spawnable* mountains[7];
	int offset = 0;
	for (int hm_level = minIdx; hm_level <= maxIdx; hm_level++)
	{
		int x = temple_perk_x[hm_level] + shopOffsetX;
		int y = temple_perk_y[hm_level] + shopOffsetY;
		random->SetRandomSeed(x, y);
		int sale_item = random->Random(0, itemCount - 1);
		bool wands = random->Random(0, 100) > 50;

		if (wands)
		{
#ifdef DO_WANDGEN
			byte* mountainStart = bytes + offset;
			writeInt(bytes, offset, x);
			writeInt(bytes, offset, y);
			writeByte(bytes, offset, TYPE_HM_SHOP);
			byte* countPtr = getIntPtr(bytes, offset);
			int startOffset = offset;

			for (int i = 0; i < itemCount; i++)
			{
				Wand w = GetShopWand(random, roundf(x + i * stepSize), y, max(1, hm_level));
				writeByte(bytes, offset, DATA_WAND); //-1
				writeInt(bytes, offset, *(int*)&w.capacity); //0
				writeInt(bytes, offset, w.multicast); //4
				writeInt(bytes, offset, w.mana); //8
				writeInt(bytes, offset, w.regen); //12
				writeInt(bytes, offset, w.delay); //16
				writeInt(bytes, offset, w.reload); //20
				writeInt(bytes, offset, *(int*)&w.speed); //24
				writeInt(bytes, offset, w.spread); //28
				writeByte(bytes, offset, (byte)w.shuffle); //32
				writeByte(bytes, offset, (byte)w.spellIdx); //33
				writeByte(bytes, offset, DATA_SPELL);
				writeShort(bytes, offset, (short)w.alwaysCast); //34
				for (int i = 0; i < w.spellIdx; i++)
				{
					writeByte(bytes, offset, DATA_SPELL);
					writeShort(bytes, offset, (short)w.spells[i]);
				}
			}
			int tmp = 0;
			writeInt(countPtr, tmp, offset - startOffset);
			mountains[hm_level] = mountainStart;
#endif
		}
		else
		{
			byte* mountainStart = bytes + offset;
			writeInt(bytes, offset, x);
			writeInt(bytes, offset, y);
			writeByte(bytes, offset, TYPE_HM_SHOP);
			writeInt(bytes, offset, 30);

			for (int i = 0; i < itemCount; i++)
			{
				writeByte(bytes, offset, DATA_SPELL);
				writeShort(bytes, offset, GetRandomAction(random->world_seed, x + i * stepSize, y - 30, hm_level, 0));
				writeByte(bytes, offset, DATA_SPELL);
				writeShort(bytes, offset, GetRandomAction(random->world_seed, x + i * stepSize, y, hm_level, 0));
			}
			mountains[hm_level] = (Spawnable*)mountainStart;
		}
	}

	SpawnableBlock b = { random->world_seed, 7, mountains };
	passed = SpawnablesPassed(b, fCfg, false);
	return passed;
}

__device__ bool PrecheckSeed(uint seed, PrecheckConfig config)
{
	NoitaRandom sharedRandom = NoitaRandom(seed);
	/*for (int max_safe_polymorphs = 0; max_safe_polymorphs < 100; max_safe_polymorphs++)
	{
		sharedRandom.SetRandomSeed(64687, max_safe_polymorphs);
		if (sharedRandom.Random(1, 100) <= 50) return max_safe_polymorphs;
	}
	return false;*/

	//Keep ordered by total runtime, so faster checks are run first and long checks can be skipped
	if (config.checkRain)
		if (!CheckRain(&sharedRandom, config.rain)) return false;
	if (config.checkStartingFlask)
		if (!CheckStartingFlask(&sharedRandom, config.startingFlask)) return false;
	if (config.checkAlchemy)
		if (!CheckAlchemy(&sharedRandom, config.LC, config.AP, config.orderedAlchemy)) return false;
	if (config.checkFungalShifts)
		if (!CheckFungalShifts(&sharedRandom, config.shifts)) return false;
	if (config.checkBiomeModifiers)
		if (!CheckBiomeModifiers(&sharedRandom, config.biomeModifiers)) return false;
#ifdef DO_WORLDGEN
	if (config.checkUpwarps)
		if (!CheckUpwarps(&sharedRandom, config.fCfg, config.lCfg)) return false;
#endif
	if (config.checkShops)//config.lCfg.checkCards)
		if (!CheckShops(&sharedRandom, config.fCfg, config.lCfg, config.minHMidx, config.maxHMidx)) return false;
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