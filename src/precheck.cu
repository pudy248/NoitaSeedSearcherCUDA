#include "../platforms/platform_implementation.h"
#include "../include/search_structs.h"
#include "../include/noita_random.h"

#include "../data/alchemy.h"
#include "../data/fungal.h"
#include "../data/modifiers.h"
#include "../data/perks.h"
#include "../data/temples.h"
#include "../data/rains.h"

#include <cstdio>
#include <cmath>

_compute static AlchemyRecipe MaterialPicker(NollaPRNG& prng, uint32_t worldSeed)
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
	prng2.Next();
	for (int i = 3; i >= 0; i--)
	{
		int r = (int)(prng2.Next() * (i + 1));
		if (r == 4) {
			r = 3;
		}
		Material temp = result.mats[i];
		result.mats[i] = result.mats[r];
		result.mats[r] = temp;
	}

	prng.Next();
	prng.Next();
	return result;
}

constexpr int materialVarEntryCount = 10;
_compute static bool MaterialRefEquals(Material reference, Material test)
{
	if (reference == MATERIAL_NONE) return true;
	if (test == MATERIAL_NONE) return false;
	return reference == test;
}
_compute static bool MaterialEquals(Material reference, Material test, bool writeRef, int* ptrs, Material* variables)
{
	if (reference == MATERIAL_NONE) return true;
	else if ((int)reference <= MATERIAL_VAR4)
	{
		int idx = (int)reference - 1;
		if (writeRef)
		{
			if (ptrs[idx] >= materialVarEntryCount) printf("Material variable %i space ran out!\n", idx);
			else variables[idx * materialVarEntryCount + ptrs[idx]++] = test;
			return true;
		}
		else
		{
			bool foundVar = false;
			for (int i = 0; i < ptrs[idx]; i++)
			{
				if (MaterialRefEquals(variables[idx * materialVarEntryCount + i], test))
					foundVar = true;
			}
			return foundVar;
		}
	}

	if (test == MATERIAL_NONE) return false;
	/*if ((int)test <= MATERIAL_VAR4)
	{
		int idx = (int)test - 1;
		if (writeRef)
		{
			if (ptrs[idx] >= materialVarEntryCount) printf("Material variable %i space ran out!\n", idx);
			else variables[idx * materialVarEntryCount + ptrs[idx]++] = reference;
			return true;
		}
		else
		{
			bool foundVar = false;
			for (int i = 0; i < ptrs[idx]; i++)
			{
				if (MaterialEquals(reference, variables[idx * materialVarEntryCount + i], writeRef, ptrs, variables))
					foundVar = true;
			}
			return foundVar;
		}
	}*/

	return reference == test;
}

_compute static bool FungalShiftEquals(FungalShift reference, FungalShift test, int ptrs[4], Material vars[materialVarEntryCount * 4])
{
	if (reference.fromFlask && !test.fromFlask) return false;
	if (reference.toFlask && !test.toFlask) return false;
	if (!MaterialEquals((Material)reference.from, (Material)test.from, false, ptrs, vars)) return false;
	if (!MaterialEquals((Material)reference.to, (Material)test.to, false, ptrs, vars)) return false;
	return true;
}

_compute static void shuffle_table(uint8_t* perk_deck, NollaPRNG& rng, int iters)
{
	for (int i = iters; i > 0; i--)
	{
		int j = rng.Random(0, i);
		int8_t tmp = perk_deck[i];
		perk_deck[i] = perk_deck[j];
		perk_deck[j] = tmp;
	}
}

_compute static bool CheckCart(NollaPRNG& random, StartingCartConfig c)
{
	float r = random.ProceduralRandomf(673, -100, 0, 1) * 0.505f;
	CartType cart = SKATEBOARD;
	if (r < 0.25f) cart = MINECART;
	else if (r < 0.5f) cart = WOODCART;
	return c.cart == cart;
}

_compute static bool CheckRain(NollaPRNG& random, RainConfig c)
{
	float rainfall_chance = 1.0f / 15;
	Vec2i rnd = { 7893434, 3458934 };
	Material rain = MATERIAL_NONE;
	if (random_next(0, 1, random, rnd) <= rainfall_chance)
	{
		int seedRainIndex = pick_random_from_table_backwards(rainProbs, rainCount, random, rnd);
		rain = rainMaterials[seedRainIndex];
	}
	return c.rain == rain;
}

_compute static bool CheckStartingFlask(NollaPRNG& random, StartingFlaskConfig c)
{
	random.SetRandomSeed(-4.5, -4);
	Material material = Material::MATERIAL_NONE;
	int res = random.Random(1, 100);
	if (res <= 65)
	{
		res = random.Random(1, 100);
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
		res = random.Random(0, 100);
		Material magic[] = {
			Material::ACID,
			Material::MAGIC_LIQUID_POLYMORPH,
			Material::MAGIC_LIQUID_RANDOM_POLYMORPH,
			Material::MAGIC_LIQUID_BERSERK,
			Material::MAGIC_LIQUID_CHARM,
			Material::MAGIC_LIQUID_MOVEMENT_FASTER
		};
		material = magic[random.Random(0, 5)];
	}
	else
	{
		res = random.Random(0, 100000);
		if (res == 666) material = Material::URINE;
		else if (res == 79) material = Material::GOLD;
		else if (random.Random(0, 1) == 0) material = Material::SLIME;
		else material = Material::GUNPOWDER_UNSTABLE;
	}
	return material == c.flask;
}

_compute static bool CheckStartingWands(NollaPRNG& random, StartingWandConfig c)
{
	if (c.projectile != SPELL_NONE)
	{
		Spell selectedProj = SPELL_NONE;
		random.SetRandomSeedInt(0, -11);
		for (int i = 0; i < 7; i++) random.Next();
		int rnd = random.Random(1, 100);
		if (rnd < 50)
		{
			const Spell spells[] = { SPELL_SPITTER, SPELL_RUBBER_BALL, SPELL_BOUNCY_ORB };
			int idx = random.Random(0, 2);
			selectedProj = spells[idx];
		}
		else selectedProj = SPELL_LIGHT_BULLET;
		if (selectedProj != c.projectile) return false;
	}

	if (c.bomb != SPELL_NONE)
	{
		Spell selectedBomb = SPELL_NONE;
		random.SetRandomSeedInt(-1, 0);
		for (int i = 0; i < 5; i++) random.Next();
		int rnd = random.Random(1, 100);
		if (rnd < 50)
		{
			const Spell spells[] = { SPELL_BOMB, SPELL_DYNAMITE, SPELL_MINE, SPELL_ROCKET, SPELL_GRENADE };
			int idx = random.Random(0, 4);
			selectedBomb = spells[idx];
		}
		else selectedBomb = SPELL_BOMB;
		if (selectedBomb != c.bomb) return false;
	}
	return true;
}

_compute static bool CheckAlchemy(NollaPRNG& random, AlchemyConfig c)
{
	NollaPRNG prng((int)(random.world_seed * 0.17127 + 1323.5903));
	for (int i = 0; i < 6; i++) prng.Next();
	AlchemyRecipe lc = MaterialPicker(prng, random.world_seed);
	AlchemyRecipe ap = MaterialPicker(prng, random.world_seed);
	return lc.Equals(c.LC, lc, c.ordering) && ap.Equals(c.AP, ap, c.ordering);
}

_compute static bool CheckFungalShifts(NollaPRNG& random, FungalShiftConfig c)
{
	FungalShift generatedShifts[maxFungalShifts];
	for (int i = 0; i < maxFungalShifts; i++)
	{
		random.SetRandomSeedInt(89346, 42345 + i);
		Vec2i rnd = { 9123,58925 + i };
		generatedShifts[i].from = fungalMaterialsFrom[pick_random_from_table_weighted(fungalProbsFrom, fungalSumFrom, fungalMaterialsFromCount, random, rnd)];
		generatedShifts[i].to = fungalMaterialsTo[pick_random_from_table_weighted(fungalProbsTo, fungalSumTo, fungalMaterialsToCount, random, rnd)];
		if (random_nexti(1, 100, random, rnd) <= 75)
		{
			if (random_nexti(1, 100, random, rnd) <= 50)
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
		if (c.shifts[i].to > SD_NONE && (int)c.shifts[i].to <= SD_VAR4)
		{
			for (int j = c.shifts[i].minIdx; j < c.shifts[i].maxIdx; j++)
			{
				if (MaterialEquals((Material)c.shifts[i].from, (Material)generatedShifts[j].from, false, ptrs, variables))
					MaterialEquals((Material)c.shifts[i].to, (Material)generatedShifts[j].to, true, ptrs, variables);
				//this has side effects when the bool is true, it looks kinda funny out of context tho
			}
		}
	}

	for (int i = 0; i < maxFungalShifts; i++)
	{
		if (c.shifts[i].minIdx == c.shifts[i].maxIdx) continue;
		bool found = false;
		for (int j = c.shifts[i].minIdx; j < c.shifts[i].maxIdx; j++)
		{
			if (FungalShiftEquals(c.shifts[i], generatedShifts[j], ptrs, variables))
			{
				found = true;
				break;
			}
		}
		if (!found) return false;
	}
	return true;
}

_compute static bool CheckBiomeModifiers(NollaPRNG& random, BiomeModifierConfig c)
{
	BiomeModifier modifiers[9];
	memset(modifiers, 0, 9);
	Vec2i rnd = { 347893,90734 };
	for (int i = 0; i < 9; i++)
	{
		float chance = 0.1f;
		if (i == 0) chance = 0.2f;
		if (i == 1) chance = 0.15f;
		if (random_next(0, 1, random, rnd) > chance) continue;
		modifiers[i] = (BiomeModifier)(pick_random_from_table_weighted(biomeModifierProbs, biomeModifierProbSum, biomeModifierCount, random, rnd) + 1);
	}
	for (int i = 0; i < 9; i++) if (c.modifiers[i] != BM_NONE && modifiers[i] != c.modifiers[i]) return false;
	return true;
}

_compute static bool CheckPerks(NollaPRNG& random, PerkConfig c)
{
	random.SetRandomSeedInt(1, 2);

	constexpr int maxPerkCount = 130;
	uint8_t perkDeck[maxPerkCount];

	int perkDeckIdx = 0;
	for (int i = 0; i < maxPerkCount; i++) perkDeck[i] = PERK_NONE;


	for (int i = 0; i < perkCount; i++)
	{
		PerkData perkData = perkAttrs[i];
		if (perkData.not_default) continue;

		int how_many_times = 1;

		if (perkData.stackable)
		{
			uint8_t max_perks = random.Random(1, 2);
			if (perkData.max_in_pool != 0)
			{
				max_perks = random.Random(1, perkData.max_in_pool);
			}

			if (perkData.stackable_rare)
			{
				max_perks = 1;
			}

			how_many_times = random.Random(1, max_perks);
		}

		for (int j = 0; j < how_many_times; j++)
		{
			perkDeck[perkDeckIdx++] = i + 1;
		}
	}

	shuffle_table(perkDeck, random, perkDeckIdx - 1);

	for (int i = perkDeckIdx - 1; i >= 0; i--)
	{
		uint8_t perk = perkDeck[i];
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
		if (perkDeck[i] != 0) perkDeck[perkDeckIdx++] = perkDeck[i];

	NollaPRNG rnd = NollaPRNG(random.world_seed);
	for (int i = 0; i < maxPerkFilters; i++)
	{
		PerkInfo perkToCkeck = c.perks[i];
		if (c.perks[i].minPosition >= c.perks[i].maxPosition) continue;
		bool found = false;
		for (int j = (perkToCkeck.minPosition + perkDeckIdx) % perkDeckIdx; j < (perkToCkeck.maxPosition + perkDeckIdx) % perkDeckIdx; j++)
		{
			if (perkToCkeck.p == perkDeck[j] || perkToCkeck.p == PERK_NONE)
			{
				if (perkToCkeck.lottery)
				{
					int tmp = j;
					int templeIdx = 0;
					while (templeIdx < 6 && tmp >= c.perksPerMountain[templeIdx])
					{
						tmp -= c.perksPerMountain[templeIdx];
						templeIdx++;
					}

					int x = temple_x[templeIdx] + (int)rintf((tmp + 0.5f) * (60.0f / c.perksPerMountain[templeIdx]));
					int y = temple_y[templeIdx];
					rnd.SetRandomSeed(x, y);
					if (rnd.Random(1, 100) > 50)
					{
						found = true;
					}
				}
				else
				{
					found = true;
				}
			}
		}
		if (!found) return false;
	}
	return true;
}

_compute bool PrecheckSeed(uint32_t seed, StaticPrecheckConfig c)
{
	NollaPRNG sharedRandom = NollaPRNG(seed);
	/*for (int max_safe_polymorphs = 0; max_safe_polymorphs < 100; max_safe_polymorphs++)
	{
		sharedRandom.SetRandomSeed(64687, max_safe_polymorphs);
		if (sharedRandom.Random(1, 100) <= 50) return max_safe_polymorphs;
	}
	return false;*/

	//Keep ordered by total runtime, so faster checks are run first and long checks can be skipped

	if (c.cart.check)
		if (!CheckCart(sharedRandom, c.cart)) return false;

	if (c.flask.check)
		if (!CheckStartingFlask(sharedRandom, c.flask)) return false;

	if (c.wands.check)
		if (!CheckStartingWands(sharedRandom, c.wands)) return false;

	if (c.alchemy.check)
		if (!CheckAlchemy(sharedRandom, c.alchemy)) return false;

	if (c.rain.check)
		if (!CheckRain(sharedRandom, c.rain)) return false;

	if (c.biomes.check)
		if (!CheckBiomeModifiers(sharedRandom, c.biomes)) return false;

	if (c.fungal.check)
		if (!CheckFungalShifts(sharedRandom, c.fungal)) return false;

	if (c.perks.check)
		if (!CheckPerks(sharedRandom, c.perks)) return false;

	return true;
}