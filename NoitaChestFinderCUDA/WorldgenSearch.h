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

__device__ void createPotion(double x, double y, Item type, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset)
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

__device__ void createWand(double x, double y, Item type, bool addOffset, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset)
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
		int itemno = random.Random(0, 392);
		if (spellSpawnableInChests[itemno])
		{
			return (Spell)(itemno + 1);
		}
	}
	return res;
}

__device__ __noinline__ void CheckNormalChestLoot(int x, int y, uint worldSeed, MapConfig mCfg, SpawnableConfig sCfg, bool hasMimicSign, byte* bytes, int& offset, int& sCount)
{
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

__device__ __noinline__ void CheckGreatChestLoot(int x, int y, uint worldSeed, MapConfig mCfg, SpawnableConfig sCfg, bool hasMimicSign, byte* bytes, int& offset, int& sCount)
{
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

__device__ __noinline__ void CheckItemPedestalLoot(int x, int y, uint worldSeed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
{
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
		byte r_opts[7] = { RUNESTONE_LIGHT, RUNESTONE_FIRE, RUNESTONE_MAGMA, RUNESTONE_WEIGHT, RUNESTONE_EMPTINESS, RUNESTONE_EDGES, RUNESTONE_METAL };
		rnd = random.Random(0, 6);
		byte r_opt = r_opts[rnd];
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

__device__ void spawnHeart(int x, int y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
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

__device__ void spawnChest(int x, int y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
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

__device__ void spawnPotion(int x, int y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
{
	NollaPRNG random = NollaPRNG(seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 5, y - 4, seed, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnPixelScene(int x, int y, uint seed, byte oiltank, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
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

__device__ void spawnPixelScene1(int x, int y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
{
	spawnPixelScene(x, y, seed, 0, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnOilTank(int x, int y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
{
	spawnPixelScene(x, y, seed, 1, mCfg, sCfg, bytes, offset, sCount);
}

__device__ void spawnWand(int x, int y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
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

__device__ void spawnNightmareEnemy(int _x, int _y, uint seed, MapConfig mCfg, SpawnableConfig sCfg, byte* bytes, int& offset, int& sCount)
{
#ifdef DO_WANDGEN_
	//t10 wands only
	if (floorf(_y / (512 * 4.0f)) <= 7) return;

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


__device__ void CheckSpawnables(byte* res, uint seed, byte* bytes, byte* output, MapConfig mCfg, SpawnableConfig sCfg, int maxMemory)
{
	static void (*spawnFuncs[7])(int, int, uint, MapConfig, SpawnableConfig, byte*, int&, int&) = { spawnHeart, spawnChest, spawnPixelScene1, spawnOilTank, spawnPotion, spawnWand, spawnNightmareEnemy };
	int offset = 0;
	byte* map = res + 4 * 3 * mCfg.map_w;
	writeInt(bytes, offset, seed);
	int countOffset = offset;
	offset += 4;
	int sCount = 0;

	for (int px = 0; px < mCfg.map_w; px++)
	{
		for (int py = 0; py < mCfg.map_h; py++)
		{
			int pixelPos = 3 * (px + py * mCfg.map_w);
			if (map[pixelPos] == 0 && map[pixelPos + 1] == 0)
				continue;
			if (map[pixelPos] == 255 && map[pixelPos + 1] == 255)
				continue;

			IntPair gp = GetGlobalPos(mCfg.worldX, mCfg.worldY, px * 10, py * 10);

			const int PWSize = mCfg.isNightmare ? 64 * 512 : 70 * 512;

			//avoids having to switch every loop
			auto func = spawnFuncs[0];
			long pix = createRGB(map[pixelPos], map[pixelPos + 1], map[pixelPos + 2]);
			
			bool check = false;
			switch (pix)
			{
			case 0x78ffff:
				if (sCfg.biomeChests)
				{
					func = spawnFuncs[0];
					check = true;
				}
				else continue;
				break;
			case 0x55ff8c:
				if (sCfg.biomeChests)
				{
					func = spawnFuncs[1];
					check = true;
				}
				else continue;
				break;
			case 0xff0aff:
				if (sCfg.pixelScenes)
				{
					func = spawnFuncs[2];
					check = true;
				}
				else continue;
				break;
			case 0xc35700:
				if (sCfg.pixelScenes)
				{
					func = spawnFuncs[3];
					check = true;
				}
				else continue;
				break;
			case 0x50a000:
				if (sCfg.biomePedestals)
				{
					func = spawnFuncs[4];
					check = true;
				}
				else continue;
				break;
			case 0x00ff00:
				if (sCfg.biomeAltars)
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
			default:
				continue;
			}

			if (check)
				for (int i = sCfg.pwCenter - sCfg.pwWidth; i <= sCfg.pwCenter + sCfg.pwWidth; i++)
					func(gp.x + PWSize * i, gp.y, seed, mCfg, sCfg, bytes, offset, sCount);
		}
	}

	writeInt(bytes, countOffset, sCount);
	if (offset > maxMemory) printf("ran out of misc memory: %i of %i bytes used\n", offset, maxMemory);
	//printf("%i, %i\n", offset, maxMemory);
	//memcpy(output, origin, *bytes - origin);
}

__device__ SpawnableBlock ParseSpawnableBlock(byte* bytes, byte* putSpawnablesHere, byte* output, MapConfig mCfg, SpawnableConfig sCfg, int maxMemory)
{
	int offset = 0;
	uint seed = readInt(bytes, offset);
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