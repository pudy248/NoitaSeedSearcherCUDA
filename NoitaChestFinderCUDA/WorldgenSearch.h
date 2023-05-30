#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"
#include "structs/enums.h"
#include "structs/spawnableStructs.h"
#include "structs/staticPrecheckStructs.h"

#include "data/potions.h"
#include "data/spells.h"
#include "data/wand_levels.h"

#include "defines.h"
#include "misc/noita_random.h"
#include "misc/utilities.h"
#include "misc/wandgen.h"

#include "Configuration.h"

__device__ void createPotion(double x, double y, Item type, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset)
{
	if (!sCfg.genPotions) writeByte(bytes, offset, type);
	else
	{
		writeByte(bytes, offset, DATA_MATERIAL);
		NollaPRNG rnd = NollaPRNG(seed);
		rnd.SetRandomSeed(x - 4.5, y - 4);
		switch (type)
		{
		case POTION_NORMAL:
			if (rnd.Random(0, 100) <= 75)
			{
				if (rnd.Random(0, 100000) <= 50)
					writeShort(bytes, offset, MAGIC_LIQUID_HP_REGENERATION);
				else if (rnd.Random(200, 100000) <= 250)
					writeShort(bytes, offset, PURIFYING_POWDER);
				else
					writeShort(bytes, offset, potionMaterialsMagic[rnd.Random(0, magicMaterialCount - 1)]);
			}
			else
				writeShort(bytes, offset, potionMaterialsStandard[rnd.Random(0, standardMaterialCount - 1)]);

			break;
		case POTION_SECRET:
			writeShort(bytes, offset, potionMaterialsSecret[rnd.Random(0, secretMaterialCount - 1)]);
			break;
		case POTION_RANDOM_MATERIAL:
			if (rnd.Random(0, 100) <= 50)
				writeShort(bytes, offset, potionLiquids[rnd.Random(0, liquidMaterialCount - 1)]);
			else
				writeShort(bytes, offset, potionSands[rnd.Random(0, sandMaterialCount - 1)]);
			break;
		}
	}
}

__device__ void createWand(double x, double y, Item type, bool addOffset, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset)
{
	writeByte(bytes, offset, type);

	int wandNum = (int)type - (int)WAND_T1;
	int tier = wandNum / 3;
	bool nonshuffle = wandNum % 3 == 1;
	bool better = wandNum % 3 == 2;

#ifdef DO_WANDGEN
	if (!sCfg.genWands || better) return;
	else
	{
		int rand_x = (int)x;
		int rand_y = (int)y;

		if (addOffset)
		{
			rand_x += 510;
			rand_y += 683;
		}

		Wand w = GetWandWithLevel(seed, rand_x, rand_y, tier, nonshuffle, better);
		writeByte(bytes, offset, DATA_WAND); //-1
		memcpy(bytes + offset, &w.capacity, 37);
		offset += 37;
		memcpy(bytes + offset, w.spells, w.spellCount * 3);
		offset += w.spellCount * 3;
	}
#endif
}

__device__ Spell MakeRandomCard(NollaPRNG& random)
{
	Spell res = SPELL_NONE;
	char valid = 0;
	while (valid == 0)
	{
		int itemno = random.Random(0, SpellCount - 1);
		if (spellSpawnableInChests[itemno])
		{
			return (Spell)(itemno + 1);
		}
	}
	return res;
}

__device__ __noinline__ void CheckNormalChestLoot(int x, int y, uint32_t worldSeed, MapConfig mCfg, SpawnableConfig sCfg, bool hasMimicSign, uint8_t* bytes, int& offset, int& sCount)
{
	if (x < mCfg.minX || x > mCfg.maxX || y < mCfg.minY || y > mCfg.maxY) return;
	sCount++;
	writeInt(bytes, offset, x);
	writeInt(bytes, offset, y);
	writeByte(bytes, offset, TYPE_CHEST);
	int countOffset = offset;
	offset += 4;

	if (hasMimicSign)
		writeByte(bytes, offset, MIMIC_SIGN);

	NollaPRNG random = NollaPRNG(worldSeed);
	random.SetRandomSeed(roundRNGPos(x) + 509.7, y + 683.1);

	int count = 1;
	while (count > 0)
	{
		count--;
		int rnd = random.Random(1, 100);
		if (rnd <= 7) writeByte(bytes, offset, BOMB);
		else if (rnd <= 40)
		{
			rnd = random.Random(0, 100);

			rnd = random.Random(0, 100);
			if (rnd > 99)
			{
				int tamount = random.Random(1, 3);
				for (int i = 0; i < tamount; i++)
				{
					random.Random(-10, 10);
					random.Random(-10, 5);
				}

				if (random.Random(0, 100) > 50)
				{
					tamount = random.Random(1, 3);
					for (int i = 0; i < tamount; i++)
					{
						random.Random(-10, 10);
						random.Random(-10, 5);
					}
				}
				if (random.Random(0, 100) > 80)
				{
					tamount = random.Random(1, 3);
					for (int i = 0; i < tamount; i++)
					{
						random.Random(-10, 10);
						random.Random(-10, 5);
					}
				}
			}
			else
			{
				random.Random(-10, 10);
				random.Random(-10, 5);
			}
			writeByte(bytes, offset, GOLD_NUGGETS);
		}
		else if (rnd <= 50)
		{
			rnd = random.Random(1, 100);
			if (rnd <= 94)
				createPotion(roundRNGPos(x) + 510, y + 683, POTION_NORMAL, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 98) writeByte(bytes, offset, POWDER);
			else
			{
				rnd = random.Random(0, 100);
				if (rnd <= 98) createPotion(roundRNGPos(x) + 510, y + 683, POTION_SECRET, worldSeed, mCfg, sCfg, bytes, offset);
				else createPotion(roundRNGPos(x) + 510, y + 683, POTION_RANDOM_MATERIAL, worldSeed, mCfg, sCfg, bytes, offset);
			}
		}
		else if (rnd <= 54) writeByte(bytes, offset, Item::SPELL_REFRESH);
		else if (rnd <= 60)
		{
			Item opts[8] = { KAMMI, KUU, UKKOSKIVI, PAHA_SILMA, KIUASKIVI, (Item)127, CHAOS_DIE, SHINY_ORB };
			rnd = random.Random(0, 7);
			Item opt = opts[rnd];
			if ((int)opt == 127)
			{
				Item r_opts[7] = { RUNESTONE_LIGHT, RUNESTONE_FIRE, RUNESTONE_MAGMA, RUNESTONE_WEIGHT, RUNESTONE_EMPTINESS, RUNESTONE_EDGES, RUNESTONE_METAL };
				rnd = random.Random(0, 6);
				Item r_opt = r_opts[rnd];
				writeByte(bytes, offset, r_opt);
			}
			else
			{
				writeByte(bytes, offset, opt);
			}
		}
		else if (rnd <= 65)
		{
			int amount = 1;
			int rnd2 = random.Random(0, 100);
			if (rnd2 <= 50) amount = 1;
			else if (rnd2 <= 70) amount += 1;
			else if (rnd2 <= 80) amount += 2;
			else if (rnd2 <= 90) amount += 3;
			else amount += 4;

			for (int i = 0; i < amount; i++)
			{
				random.Next();
				Spell s = MakeRandomCard(random);
				if (sCfg.genSpells)
				{
					writeByte(bytes, offset, DATA_SPELL);
					writeShort(bytes, offset, s);
				}
			}
			if (!sCfg.genSpells)
				writeByte(bytes, offset, RANDOM_SPELL);
		}
		else if (rnd <= 84)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) createWand(x, y, WAND_T1, true, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 50) createWand(x, y, WAND_T1NS, true, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 75) createWand(x, y, WAND_T2, true, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 90) createWand(x, y, WAND_T2NS, true, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 96) createWand(x, y, WAND_T3, true, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 98) createWand(x, y, WAND_T3NS, true, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 99)createWand(x, y, WAND_T4, true, worldSeed, mCfg, sCfg, bytes, offset);
			else createWand(x, y, WAND_T4NS, true, worldSeed, mCfg, sCfg, bytes, offset);
		}
		else if (rnd <= 95)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 88) writeByte(bytes, offset, HEART_NORMAL);
			else if (rnd <= 89) writeByte(bytes, offset, HEART_MIMIC);
			else if (rnd <= 99) writeByte(bytes, offset, HEART_BIGGER);
			else writeByte(bytes, offset, FULL_HEAL);
		}
		else if (rnd <= 98) writeByte(bytes, offset, CHEST_TO_GOLD);
		else if (rnd <= 99)
			count += 2;
		else
			count += 3;
	}
	writeInt(bytes, countOffset, offset - countOffset - 4);
}

__device__ __noinline__ void CheckGreatChestLoot(int x, int y, uint32_t worldSeed, MapConfig mCfg, SpawnableConfig sCfg, bool hasMimicSign, uint8_t* bytes, int& offset, int& sCount)
{
	if (x < mCfg.minX || x > mCfg.maxX || y < mCfg.minY || y > mCfg.maxY) return;
	sCount++;
	writeInt(bytes, offset, x);
	writeInt(bytes, offset, y);
	writeByte(bytes, offset, TYPE_CHEST_GREATER);
	int countOffset = offset;
	offset += 4;

	if (hasMimicSign)
		writeByte(bytes, offset, MIMIC_SIGN);

	NollaPRNG random = NollaPRNG(worldSeed);
	random.SetRandomSeed(roundRNGPos(x), y);

	int count = 1;

	if (random.Random(0, 100000) >= 100000)
	{
		count = 0;
		if (random.Random(0, 1000) == 999) writeByte(bytes, offset, TRUE_ORB);
		else writeByte(bytes, offset, SAMPO);
	}

	while (count != 0)
	{
		count--;
		int rnd = random.Random(1, 100);

		if (rnd <= 30)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 30)
			{
				createPotion(x, y, POTION_NORMAL, worldSeed, mCfg, sCfg, bytes, offset);
				createPotion(x, y, POTION_NORMAL, worldSeed, mCfg, sCfg, bytes, offset);
				createPotion(x, y, POTION_SECRET, worldSeed, mCfg, sCfg, bytes, offset);
			}
			else
			{
				createPotion(x, y, POTION_SECRET, worldSeed, mCfg, sCfg, bytes, offset);
				createPotion(x, y, POTION_SECRET, worldSeed, mCfg, sCfg, bytes, offset);
				createPotion(x, y, POTION_RANDOM_MATERIAL, worldSeed, mCfg, sCfg, bytes, offset);
			}
		}
		else if (rnd <= 33)
		{
			writeByte(bytes, offset, RAIN_GOLD);
		}
		else if (rnd <= 38)
		{
			rnd = random.Random(1, 30);
			if (rnd == 30)
				writeByte(bytes, offset, KAKKAKIKKARE);
			else writeByte(bytes, offset, VUOKSIKIVI);
		}
		else if (rnd <= 39)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) createWand(x, y, WAND_T3, false, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 50) createWand(x, y, WAND_T3NS, false, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 75) createWand(x, y, WAND_T4, false, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 90) createWand(x, y, WAND_T4NS, false, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 96) createWand(x, y, WAND_T5, false, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 98) createWand(x, y, WAND_T5NS, false, worldSeed, mCfg, sCfg, bytes, offset);
			else if (rnd <= 99)createWand(x, y, WAND_T6, false, worldSeed, mCfg, sCfg, bytes, offset);
			else createWand(x, y, WAND_T6NS, false, worldSeed, mCfg, sCfg, bytes, offset);
		}
		else if (rnd <= 60)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 89) writeByte(bytes, offset, HEART_NORMAL);
			else if (rnd <= 99) writeByte(bytes, offset, HEART_BIGGER);
			else writeByte(bytes, offset, FULL_HEAL);
		}
		else if (rnd <= 99)
			count += 2;
		else
			count += 3;
	}
	writeInt(bytes, countOffset, offset - countOffset - 4);
}

__device__ __noinline__ void CheckItemPedestalLoot(int x, int y, uint32_t worldSeed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	if (x < mCfg.minX || x > mCfg.maxX || y < mCfg.minY || y > mCfg.maxY) return;
	sCount++;
	writeInt(bytes, offset, x);
	writeInt(bytes, offset, y);
	writeByte(bytes, offset, TYPE_ITEM_PEDESTAL);
	int countOffset = offset;
	offset += 4;

	NollaPRNG random = NollaPRNG(worldSeed);
	random.SetRandomSeed(x + 425, y - 243);

	int rnd = random.Random(1, 91);

	if (rnd <= 65)
		createPotion(x, y, POTION_NORMAL, worldSeed, mCfg, sCfg, bytes, offset);
	else if (rnd <= 70)
		writeByte(bytes, offset, POWDER);
	else if (rnd <= 71)
		writeByte(bytes, offset, CHAOS_DIE);
	else if (rnd <= 72)
	{
		uint8_t r_opts[7] = { RUNESTONE_LIGHT, RUNESTONE_FIRE, RUNESTONE_MAGMA, RUNESTONE_WEIGHT, RUNESTONE_EMPTINESS, RUNESTONE_EDGES, RUNESTONE_METAL };
		rnd = random.Random(0, 6);
		uint8_t r_opt = r_opts[rnd];
		writeByte(bytes, offset, r_opt);
	}
	else if (rnd <= 73)
		writeByte(bytes, offset, EGG_PURPLE);
	else if (rnd <= 77)
		writeByte(bytes, offset, EGG_SLIME);
	else if (rnd <= 79)
		writeByte(bytes, offset, EGG_MONSTER);
	else if (rnd <= 83)
		writeByte(bytes, offset, KIUASKIVI);
	else if (rnd <= 85)
		writeByte(bytes, offset, UKKOSKIVI);
	else if (rnd <= 89)
		writeByte(bytes, offset, BROKEN_WAND);
	else
		writeByte(bytes, offset, SHINY_ORB);

	writeInt(bytes, countOffset, offset - countOffset - 4);
}

__device__ void spawnHeart(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	float r = random.ProceduralRandomf(x, y, 0, 1);
	float heart_spawn_percent = 0.7f;

	if (r <= heart_spawn_percent && r > 0.3)
	{
		random.SetRandomSeed(x + 45, y - 2123);
		int rnd = random.Random(1, 100);
		if (rnd <= 90 || y < 512 * 3)
		{
			rnd = random.Random(1, 1000);
			bool hasSign = false;
			if (random.Random(1, 300) == 1)
			{
				hasSign = true;
			}
			if (rnd >= 1000)
				CheckGreatChestLoot(x, y, seed, mCfg, sCfg, hasSign, bytes, offset, sCount);
			else
				CheckNormalChestLoot(x, y, seed, mCfg, sCfg, hasSign, bytes, offset, sCount);
		}
		else
		{
			if (x < mCfg.minX || x > mCfg.maxX || y < mCfg.minY || y > mCfg.maxY) return;
			sCount++;
			writeInt(bytes, offset, x);
			writeInt(bytes, offset, y);
			writeByte(bytes, offset, TYPE_CHEST);
			int countOffset = offset;
			offset += 4;
			int totalBytes = 1;

			rnd = random.Random(1, 100);
			if (random.Random(1, 30 == 1)) {
				writeByte(bytes, offset, MIMIC_SIGN);
				totalBytes++;
			}
			if(rnd <= 95) writeByte(bytes, offset, MIMIC);
			else writeByte(bytes, offset, MIMIC_LEGGY);
			writeInt(bytes, countOffset, totalBytes);
		}
	}
}

__device__ void spawnChest(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	random.SetRandomSeed(x, y);
	int super_chest_spawn_rate = sCfg.greedCurse ? 100 : 2000;
	int rnd = random.Random(1, super_chest_spawn_rate);

	if (rnd >= super_chest_spawn_rate - 1)
		CheckGreatChestLoot(x, y, seed, mCfg, sCfg, false, bytes, offset, sCount);
	else
		CheckNormalChestLoot(x, y, seed, mCfg, sCfg, false, bytes, offset, sCount);
}

__device__ void spawnPotion(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 5, y - 4, seed, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnPixelScene(int x, int y, uint32_t seed, uint8_t oiltank, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	random.SetRandomSeed(x, y);
	int rnd = random.Random(1, 100);
	if (rnd <= 50 && oiltank == 0 || rnd > 50 && oiltank > 0)
	{
		float rnd2 = random.ProceduralRandomf(x, y, 0, 1) * 3;
		if (0.5f < rnd2 && rnd2 < 1)
		{
			spawnChest(x + 94, y + 224, seed, mCfg, sCfg, bytes, offset, sCount);
		}
	}
}

__device__ void spawnPixelScene1(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	spawnPixelScene(x, y, seed, 0, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnOilTank(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	spawnPixelScene(x, y, seed, 1, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnWand(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);

	if (!wandChecks[mCfg.biomeIdx](random, x, y)) return;

	int nx = x - 5;
	int ny = y - 14;
	BiomeWands wandSet = wandLevels[mCfg.biomeIdx]; //biomeIndex
	int sum = 0;
	for (int i = 0; i < wandSet.count; i++) sum += wandSet.levels[i].prob;
	float r = random.ProceduralRandomf(nx, ny, 0, 1) * sum;
	for (int i = 0; i < wandSet.count; i++)
	{
		if (r <= wandSet.levels[i].prob)
		{
			if (nx + 5 < mCfg.minX || nx + 5 > mCfg.maxX || ny + 5 < mCfg.minY || ny + 5 > mCfg.maxY) return;
			sCount++;
			writeInt(bytes, offset, nx + 5);
			writeInt(bytes, offset, ny + 5);
			writeByte(bytes, offset, TYPE_WAND_PEDESTAL);
			int countOffset = offset;
			offset += 4;
			createWand(nx + 5, ny + 5, wandSet.levels[i].id, false, seed, mCfg, sCfg, bytes, offset);
			writeInt(bytes, countOffset, offset - countOffset - 4);
			return;
		}
		r -= wandSet.levels[i].prob;
	}
}

__device__ void spawnNightmareEnemy(int _x, int _y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
#ifdef DO_WANDGEN_
	//t10 wands only
	if (floorf(_y / (512 * 4.0f)) <= 7) return;
	if (x < mCfg.minX || x > mCfg.maxX || y < mCfg.minY || y > mCfg.maxY) return;

	NollaPRNG random = NollaPRNG(seed);

	//enemy spawns
	//if (random.ProceduralRandomf(_x, _y, 0, 1.115f) <= 0.4f) return;

	int x = _x + 11;
	int y = _y + 5;
	random.SetRandomSeed(x, y);
	random.Next();
	int rnd = random.Random(1, 20);
	//wand spawns
	if (rnd != 1) return;

	float fOffset = random.ProceduralRandomf(x + 1, y, -4, 4);
	int intOffset = (int)rintf(fOffset);
	int pos_x = x + intOffset;
	int pos_y = y + intOffset;

	random.SetRandomSeed(pos_x, pos_y);
	if (random.Random(1, 100) >= 50) return;

	sCount++;
	writeInt(bytes, offset, pos_x);
	writeInt(bytes, offset, pos_y);
	writeByte(bytes, offset, TYPE_NIGHTMARE_WAND);
	writeInt(bytes, offset, 1);
	writeByte(bytes, offset, WAND_T10);

	Wand w = GetWandWithLevel(seed, pos_x, pos_y, 11, false, false);
	if (w.capacity > 30)
	{
		for (int i = 0; i < w.spellIdx; i++)
		{
			printf("%s ", SpellNames[w.spells[i]]);
		}
		printf("\n");
		printf("WAND - %i @ (%i %i) --- %.1f %i %i %i %i %i %i %.4f\n", seed, pos_x, pos_y, w.capacity, w.multicast, w.delay, w.reload, w.mana, w.regen, w.spread, w.speed);
	}
#endif
}

__device__ void spawnHellShop(int x, int y, uint32_t seed, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	sCount++;
	writeInt(bytes, offset, x);
	writeInt(bytes, offset, y);
	writeByte(bytes, offset, TYPE_HELL_SHOP);
	writeInt(bytes, offset, 3);

	writeByte(bytes, offset, DATA_SPELL);
	writeShort(bytes, offset, GetRandomAction(seed, x, y, 10, 0));
}

__device__ Wand GetShopWand(NollaPRNG& random, double x, double y, int level)
{
	random.SetRandomSeed(x, y);
	bool shuffle = random.Random(0, 100) <= 50;
	return GetWandWithLevel(random.world_seed, x, y, level, shuffle, false);
}

__device__ void CheckMountains(uint32_t seed, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	MapConfig dummyMapCfg = { 0,0,0,0,0,0,INT_MIN,INT_MAX,INT_MIN,INT_MAX };

	if (sCfg.pacifist)
	{
		for (int pw = sCfg.pwCenter.x - sCfg.pwWidth.x; pw <= sCfg.pwCenter.x + sCfg.pwWidth.x; pw++)
		{
			for (int hm_level = sCfg.minHMidx; hm_level < min(sCfg.maxHMidx, pw == 0 ? 7 : 6); hm_level++)
			{
				int x = temple_x[hm_level] + chestOffsetX + 70 * 512 * pw;
				int y = temple_y[hm_level] + chestOffsetY;
				CheckNormalChestLoot(x, y, seed, dummyMapCfg, sCfg, false, bytes, offset, sCount);
			}
		}
	}

	if (sCfg.shopSpells || sCfg.shopWands)
	{
		NollaPRNG random(seed);
		int width = 132;
		constexpr int itemCount = 5;
		float stepSize = width / (float)itemCount;
		for (int pw = sCfg.pwCenter.x - sCfg.pwWidth.x; pw <= sCfg.pwCenter.x + sCfg.pwWidth.x; pw++)
		{
			for (int hm_level = sCfg.minHMidx; hm_level < min(sCfg.maxHMidx, pw == 0 ? 7 : 6); hm_level++)
			{
				int x = temple_x[hm_level] + shopOffsetX + 70 * 512 * pw;
				int y = temple_y[hm_level] + shopOffsetY;
				int tier = temple_tiers[hm_level];
				random.SetRandomSeed(x, y);
				int sale_item = random.Random(0, itemCount - 1);
				bool wands = random.Random(0, 100) > 50;

				if (wands)
				{
					if (!sCfg.shopWands) continue;
#ifdef DO_WANDGEN
					sCount++;
					writeInt(bytes, offset, x);
					writeInt(bytes, offset, y);
					writeByte(bytes, offset, TYPE_HM_SHOP);
					int countOffset = offset;
					offset += 4;

					for (int i = 0; i < itemCount; i++)
					{
						Wand w = GetShopWand(random, round(x + i * stepSize), y, max(1, tier));
						writeByte(bytes, offset, DATA_WAND);
						memcpy(bytes + offset, &w.capacity, 37);
						offset += 37;
						memcpy(bytes + offset, w.spells, w.spellCount * 3);
						offset += w.spellCount * 3;
					}
					writeInt(bytes, countOffset, offset - countOffset - 4);
#endif
				}
				else
				{
					if (!sCfg.shopSpells) continue;
					sCount++;
					writeInt(bytes, offset, x);
					writeInt(bytes, offset, y);
					writeByte(bytes, offset, TYPE_HM_SHOP);
					writeInt(bytes, offset, 6 * itemCount);

					for (int i = 0; i < itemCount; i++)
					{
						writeByte(bytes, offset, DATA_SPELL);
						writeShort(bytes, offset, GetRandomAction(random.world_seed, x + i * stepSize, y - 30, tier, 0));
						writeByte(bytes, offset, DATA_SPELL);
						writeShort(bytes, offset, GetRandomAction(random.world_seed, x + i * stepSize, y, tier, 0));
					}
				}
			}
		}
	}
}

__device__ void CheckEyeRooms(uint32_t seed, SpawnableConfig sCfg, uint8_t* bytes, int& offset, int& sCount)
{
	IntPair positions[8] = { {-3992, 5380}, {-3971, 5397}, {-3949, 5414}, {-3926, 5428}, {-3758, 5424}, {-3735, 5410}, {-3713, 5393}, {-3692, 5376} };
	if (sCfg.eyeRooms)
	{
		NollaPRNG random(seed);
		for (int pw = sCfg.pwCenter.x - sCfg.pwWidth.x; pw <= sCfg.pwCenter.x + sCfg.pwWidth.x; pw++)
		{
			int x = -3850 + pw * 70 * 512;
			int y = 5400;
			sCount++;
			writeInt(bytes, offset, x);
			writeInt(bytes, offset, y);
			writeByte(bytes, offset, TYPE_EYE_ROOM);
			writeInt(bytes, offset, 24);

			for (int i = 0; i < 8; i++)
			{
				IntPair pos = positions[i] + IntPair(pw * 70 * 512, 0);
				random.SetRandomSeedInt(pos.x, pos.y);
				writeByte(bytes, offset, DATA_SPELL);
				writeShort(bytes, offset, MakeRandomCard(random));
			}
		}
	}
}

__device__ void CheckSpawnables(uint8_t* res, uint32_t seed, uint8_t* bytes, int& offset, int& sCount, MapConfig mCfg, SpawnableConfig sCfg, int maxMemory)
{
	static void (*spawnFuncs[])(int, int, uint32_t, MapConfig, SpawnableConfig, uint8_t*, int&, int&) = { spawnHeart, spawnChest, spawnPixelScene1, spawnOilTank, spawnPotion, spawnWand, spawnNightmareEnemy, spawnHellShop };
	uint8_t* map = res + 4 * 3 * mCfg.map_w;

	for (int px = 0; px < mCfg.map_w; px++)
	{
		for (int py = 0; py < mCfg.map_h; py++)
		{
			int pixelPos = 3 * (px + py * mCfg.map_w);
			if (map[pixelPos] == 0 && map[pixelPos + 1] == 0)
				continue;
			if (map[pixelPos] == 255 && map[pixelPos + 1] == 255)
				continue;

			//avoids having to switch every loop
			auto func = spawnFuncs[0];
			uint32_t pix = createRGB(map[pixelPos], map[pixelPos + 1], map[pixelPos + 2]);
			for (int pwY = sCfg.pwCenter.y - sCfg.pwWidth.y; pwY <= sCfg.pwCenter.y + sCfg.pwWidth.y; pwY++)
			{
				bool check = false;
				switch (pix)
				{
				case 0x78ffff:
					if (sCfg.biomeChests && pwY == 0)
					{
						func = spawnFuncs[0];
						check = true;
					}
					else continue;
					break;
				case 0x55ff8c:
					if (sCfg.biomeChests && pwY == 0)
					{
						func = spawnFuncs[1];
						check = true;
					}
					else continue;
					break;
				case 0xff0aff:
					if (sCfg.pixelScenes && pwY == 0)
					{
						func = spawnFuncs[2];
						check = true;
					}
					else continue;
					break;
				case 0xc35700:
					if (sCfg.pixelScenes && pwY == 0)
					{
						func = spawnFuncs[3];
						check = true;
					}
					else continue;
					break;
				case 0x50a000:
					if (sCfg.biomePedestals && pwY == 0)
					{
						func = spawnFuncs[4];
						check = true;
					}
					else continue;
					break;
				case 0x00ff00:
					if (sCfg.biomeAltars && pwY == 0)
					{
						func = spawnFuncs[5];
						check = true;
					}
					else continue;
					break;
				case 0xff0000:
					func = spawnFuncs[6];
					check = true;
					break;
				case 0x808000:
					if (sCfg.hellShops && pwY != 0)
					{
						func = spawnFuncs[7];
						check = true;
					}
					else continue;
					break;
				default:
					continue;
				}

				if (check)
				{
					for (int pwX = sCfg.pwCenter.x - sCfg.pwWidth.x; pwX <= sCfg.pwCenter.x + sCfg.pwWidth.x; pwX++)
					{
						IntPair gp = GetGlobalPos(mCfg.worldX + 70 * pwX, mCfg.worldY + 48 * pwY, px * 10, py * 10 - (int)truncf((pwY * 3) / 5.0f) * 10);
						func(gp.x, gp.y, seed, mCfg, sCfg, bytes, offset, sCount);
					}
				}
			}

			if (offset > maxMemory) printf("ran out of misc memory: %i of %i bytes used\n", offset, maxMemory);
		}
	}
}

__device__ SpawnableBlock ParseSpawnableBlock(uint8_t* bytes, uint8_t* putSpawnablesHere, SpawnableConfig sCfg, int maxMemory)
{
	int offset = 0;
	uint32_t seed = readInt(bytes, offset);
	int sCount = readInt(bytes, offset);

	Spawnable** spawnables = (Spawnable**)putSpawnablesHere;
	if (sCount * sizeof(Spawnable*) > maxMemory) printf("ran out of map memory: %i of %i bytes used\n", (int)(sCount * sizeof(Spawnable*)), maxMemory);
	for (int i = 0; i < sCount; i++)
	{
		Spawnable* s = (Spawnable*)(bytes + offset);
		spawnables[i] = s;
		offset += 9;
		int count = readInt(bytes, offset);
		offset += count;
	}	

	SpawnableBlock ret{ seed, sCount, spawnables };
	return ret;
}