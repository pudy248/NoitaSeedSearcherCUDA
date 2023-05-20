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

	bool checkStartingFlask;
	Material startingFlask;
	bool checkStartingWands;
	Spell projectileSpell;
	Spell bombSpell;
	bool checkRain;
	Material rain;
	bool checkAlchemy;
	AlchemyOrdering orderedAlchemy;
	AlchemyRecipe LC;
	AlchemyRecipe AP;
	bool checkBiomeModifiers;
	BiomeModifier biomeModifiers[9];
	bool checkFungalShifts;
	FungalShift shifts[maxFungalShifts];
	bool checkPerks;
	PerkInfo perks[20];
	bool checkUpwarps;
	FilterConfig fCfg;
	LootConfig lCfg;
	bool checkShopSpells;
	bool checkShopWands;
	bool checkPacifists;
	int minHMidx;
	int maxHMidx;
	int itemCount;
};


__device__ bool CheckRain(NollaPRNG* random, Material rain)
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

__device__ bool CheckStartingFlask(NollaPRNG* random, Material starting_flask)
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

__device__ bool CheckStartingWands(NollaPRNG* random, Spell projectileSpell, Spell bombSpell)
{
	if (projectileSpell != SPELL_NONE)
	{
		Spell selectedProj = SPELL_NONE;
		random->SetRandomSeed(0, -11);
		random->Next();
		random->Next();
		random->Next();
		random->Next();
		random->Next();
		random->Next();
		random->Next();
		int rnd = random->Random(1, 100);
		if (rnd < 50)
		{
			const Spell spells[] = { SPELL_LIGHT_BULLET, SPELL_SPITTER, SPELL_RUBBER_BALL, SPELL_BOUNCY_ORB };
			int idx = random->Random(0, 3);
			selectedProj = spells[idx];
		}
		else selectedProj = SPELL_BOMB;
		if (selectedProj != projectileSpell) return false;
	}

	if (bombSpell != SPELL_NONE)
	{
		Spell selectedBomb = SPELL_NONE;
		random->SetRandomSeed(-1, 0);
		random->Next();
		random->Next();
		random->Next();
		random->Next();
		random->Next();
		int rnd = random->Random(1, 100);
		if (rnd < 50)
		{
			const Spell spells[] = { SPELL_BOMB, SPELL_DYNAMITE, SPELL_MINE, SPELL_ROCKET, SPELL_GRENADE };
			int idx = random->Random(0, 4);
			selectedBomb = spells[idx];
		}
		else selectedBomb = SPELL_BOMB;
		if (selectedBomb != bombSpell) return false;
	}
	return true;
}

__device__ bool CheckAlchemy(NollaPRNG* random, AlchemyRecipe LC, AlchemyRecipe AP, AlchemyOrdering ordered)
{
	NollaPRNG prng((int)(random->world_seed * 0.17127 + 1323.5903));
	for (int i = 0; i < 5; i++) prng.Next();
	AlchemyRecipe lc = MaterialPicker(prng, random->world_seed);
	AlchemyRecipe ap = MaterialPicker(prng, random->world_seed);
	//printf("%i %i %i, %i %i %i\n", lc.mats[0], lc.mats[1], lc.mats[2], ap.mats[0], ap.mats[1], ap.mats[2]);
	//if (lc.mats[0] == MUD && lc.mats[1] == WATER && lc.mats[2] == SOIL) return true;
	return AlchemyRecipe::Equals(LC, lc, ordered) && AlchemyRecipe::Equals(AP, ap, ordered);
}

__device__ bool CheckFungalShifts(NollaPRNG* random, FungalShift shiftFilters[maxFungalShifts])
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

__device__ bool CheckBiomeModifiers(NollaPRNG* random, BiomeModifier biomeModifiers[9])
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

__device__ bool CheckPerks(NollaPRNG* random, PerkInfo perks[20])
{
	random->SetRandomSeed(1, 2);

	constexpr int maxPerkCount = 130;
	byte perkDeck[maxPerkCount];

	int perkDeckIdx = 0;
	for (int i = 0; i < maxPerkCount; i++) perkDeck[i] = PERK_NONE;


	for (int i = 0; i < perkCount; i++)
	{
		PerkData perkData = perkAttrs[i];
		if (perkData.not_default) continue;

		int how_many_times = 1;

		if (perkData.stackable)
		{
			byte max_perks = random->Random(1, 2);
			if (perkData.max_in_pool != 0)
			{
				max_perks = random->Random(1, perkData.max_in_pool);
			}

			if (perkData.stackable_rare)
			{
				max_perks = 1;
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
		if (perkStackableDistances[perk - 1] != -1)
		{
			short min_distance = perkStackableDistances[perk - 1];
			bool remove_me = false;

			for (int ri = max(0, i - min_distance); ri < i; ri++)
			{
				if (perkDeck[ri] == perk)
				{
					remove_me = true;
					break;
				}
			}

			if (remove_me) perkDeck[i] = PERK_NONE;
		}
	}

	perkDeckIdx = 0;
	for (int i = 0; i < maxPerkCount; i++)
	{
		if (perkDeck[i] != 0) perkDeck[perkDeckIdx++] = perkDeck[i];
	}

	NollaPRNG rnd = NollaPRNG(random->world_seed);
	for (int i = 0; i < 20; i++)
	{
		PerkInfo perkToCkeck = perks[i];
		if (perks[i].p == PERK_NONE) break;
		bool found = false;
		for (int j = (perkToCkeck.minPosition + perkDeckIdx) % perkDeckIdx; j < (perkToCkeck.maxPosition + perkDeckIdx) % perkDeckIdx; j++)
		{
			if (perkToCkeck.p == perkDeck[j])
			{
				if (perkToCkeck.lottery)
				{
					int x = temple_x[(j / 3)] + (int)roundf(((j % 3) + 0.5f) * 20);
					int y = temple_y[(j / 3)];
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

__device__ bool CheckUpwarps(NollaPRNG* random, FilterConfig fCfg, LootConfig lCfg)
{
	byte bytes[1000];
	int offset = 0;
	int _ = 0;
	spawnChest(315, 17, random->world_seed, lCfg, bytes, offset, _);
	byte* ptr1 = bytes + offset;
	spawnChest(75, 117, random->world_seed, lCfg, bytes, offset, _);
	Spawnable* spawnables[] = { (Spawnable*)bytes, (Spawnable*)ptr1 };
	SpawnableBlock b = { random->world_seed, 2, spawnables };

	return SpawnablesPassed(b, fCfg, false);
}

__device__ bool CheckPacifists(NollaPRNG* random, FilterConfig fCfg, LootConfig lCfg, int minIdx, int maxIdx)
{
	byte bytes[3000];
	Spawnable* mountains[7] = { NULL,NULL,NULL,NULL,NULL,NULL,NULL };
	int offset = 0;
	int _ = 0;
	for (int hm_level = minIdx; hm_level < maxIdx; hm_level++)
	{
		int x = temple_x[hm_level] + chestOffsetX;
		int y = temple_y[hm_level] + chestOffsetY;
		byte* mountainStart = bytes + offset;
		CheckNormalChestLoot(x, y, random->world_seed, lCfg, false, bytes, offset, _);
		mountains[hm_level] = (Spawnable*)mountainStart;
	}
	SpawnableBlock b = { random->world_seed, 7, mountains };
	return SpawnablesPassed(b, fCfg, false);
}

__device__ bool CheckShops(NollaPRNG* random, FilterConfig fCfg, LootConfig lCfg, int minIdx, int maxIdx, int _itemCount, bool checkSpells, bool checkWands)
{
	int width = 132;
	constexpr int itemCount = 5;
	float stepSize = width / (float)itemCount;
	for (int pw = lCfg.pwCenter - lCfg.pwWidth; pw <= lCfg.pwCenter + lCfg.pwWidth; pw++)
	{
		bool passed = false;

		byte bytes[3000];
		Spawnable* mountains[7] = { NULL,NULL,NULL,NULL,NULL,NULL,NULL };
		int offset = 0;
		for (int hm_level = minIdx; hm_level < min(maxIdx, pw == 0 ? 7 : 6); hm_level++)
		{
			int x = temple_x[hm_level] + shopOffsetX + 70 * 512 * pw;
			int y = temple_y[hm_level] + shopOffsetY;
			random->SetRandomSeed(x, y);
			int sale_item = random->Random(0, itemCount - 1);
			bool wands = random->Random(0, 100) > 50;

			if (wands)
			{
				if (!checkWands) continue;
#ifdef DO_WANDGEN
				byte* mountainStart = bytes + offset;
				writeInt(bytes, offset, x);
				writeInt(bytes, offset, y);
				writeByte(bytes, offset, TYPE_HM_SHOP);
				int countOffset = offset;
				offset += 4;

				for (int i = 0; i < itemCount; i++)
				{
					Wand w = GetShopWand(random, round(x + i * stepSize), y, max(1, hm_level));
					writeByte(bytes, offset, DATA_WAND); //-1
					memcpy(bytes + offset, &w.capacity, 37);
					offset += 37;
					memcpy(bytes + offset, w.spells, w.spellCount * 3);
					offset += w.spellCount * 3;
		}
				writeInt(bytes, countOffset, offset - countOffset - 4);
				mountains[hm_level] = (Spawnable*)mountainStart;
#endif
	}
			else
			{
				if(!checkSpells) continue;
				byte* mountainStart = bytes + offset;
				writeInt(bytes, offset, x);
				writeInt(bytes, offset, y);
				writeByte(bytes, offset, TYPE_HM_SHOP);
				writeInt(bytes, offset, 6 * itemCount);

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
		passed = SpawnablesPassed(b, fCfg, true);
		if (passed) return true;
	}
	return false;
}

__device__ bool PrecheckSeed(uint seed, PrecheckConfig config)
{
	NollaPRNG sharedRandom = NollaPRNG(seed);
	/*for (int max_safe_polymorphs = 0; max_safe_polymorphs < 100; max_safe_polymorphs++)
	{
		sharedRandom.SetRandomSeed(64687, max_safe_polymorphs);
		if (sharedRandom.Random(1, 100) <= 50) return max_safe_polymorphs;
	}
	return false;*/

	//Keep ordered by total runtime, so faster checks are run first and long checks can be skipped
	if (config.checkStartingFlask)
		if (!CheckStartingFlask(&sharedRandom, config.startingFlask)) return false;										//555ms
	if (config.checkAlchemy)
		if (!CheckAlchemy(&sharedRandom, config.LC, config.AP, config.orderedAlchemy)) return false;					//805ms
	if (config.checkStartingWands)
		if (!CheckStartingWands(&sharedRandom, config.projectileSpell, config.bombSpell)) return false;					//1014ms
	if (config.checkRain)
		if (!CheckRain(&sharedRandom, config.rain)) return false;														//1450ms
	if (config.checkBiomeModifiers)
		if (!CheckBiomeModifiers(&sharedRandom, config.biomeModifiers)) return false;									//5817ms
	if (config.checkFungalShifts)
		if (!CheckFungalShifts(&sharedRandom, config.shifts)) return false;												//4500ms (3 shifts, 1 filter)
	//#ifdef DO_WORLDGEN
	if (config.checkUpwarps)
		if (!CheckUpwarps(&sharedRandom, config.fCfg, config.lCfg)) return false;										//7600ms
	//#endif
	if (config.checkPacifists)
		if (!CheckPacifists(&sharedRandom, config.fCfg, config.lCfg, config.minHMidx, config.maxHMidx)) return false;	//13800ms (7 HMs, 1 filter)
	if (config.checkPerks)
		if (!CheckPerks(&sharedRandom, config.perks)) return false;														//38000ms
	if (config.checkShopSpells || config.checkShopWands)
		if (!CheckShops(&sharedRandom, config.fCfg, config.lCfg, config.minHMidx, config.maxHMidx, config.itemCount, config.checkShopSpells, config.checkShopWands)) return false;		//50sec, 6500sec (without/with wands, 7 HMs, 1 filter)
	//Worldgen (Wang only): mines, temple	//260,000 sec, 440,000 sec
	//Worldgen + Spawnables: mines, temple	//360,000 sec, 750,000 sec
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