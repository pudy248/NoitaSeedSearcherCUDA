#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/noita_random.h"
#include "misc/datatypes.h"
#include "misc/utilities.h"
#include "data/items.h"
#include "data/potions.h"
#include "data/spells.h"
#include "data/wand_levels.h"

#include <iostream>

struct LootConfig {
	int pwCount;

	bool searchChests;
	bool searchPedestals;
	bool searchWandAltars;
	bool searchPixelScenes;

	bool greedCurse;
	bool checkPotions;
	bool checkWands;
	bool checkCards;
};

__device__ void createPotion(int x, int y, Item type, uint worldSeed, LootConfig cfg, byte** bytes) {
	if (!cfg.checkPotions) writeByte(bytes, type);
	else {
		writeByte(bytes, DATA_MATERIAL);
		NoitaRandom rnd = NoitaRandom(worldSeed);
		rnd.SetRandomSeed(x - 4.5, y - 4);
		switch (type) {
		case POTION_NORMAL:
			if (rnd.Random(0, 100) <= 75) {
				if (rnd.Random(0, 100000) <= 50)
					writeMaterial(bytes, MAGIC_LIQUID_HP_REGENERATION);
				else if (rnd.Random(200, 100000) <= 250)
					writeMaterial(bytes, PURIFYING_POWDER);
				else
					writeMaterial(bytes, potionMaterialsMagic[rnd.Random(0, magicMaterialCount)]);
			}
			else
				writeMaterial(bytes, potionMaterialsStandard[rnd.Random(0, standardMaterialCount)]);

			break;
		case POTION_SECRET:
			writeMaterial(bytes, potionMaterialsSecret[rnd.Random(0, secretMaterialCount)]);
			break;
		case POTION_RANDOM_MATERIAL:
			if (rnd.Random(0, 100) <= 50)
				writeMaterial(bytes, potionLiquids[rnd.Random(0, liquidMaterialCount)]);
			else
				writeMaterial(bytes, potionSands[rnd.Random(0, sandMaterialCount)]);
			break;
		}
	}
}

__device__ int MakeRandomCard(NoitaRandom* random) {
	int res = 0;
	char valid = 0;
	while (valid == 0) {
		int itemno = random->Random(0, 392);
		SpellData item = all_spells[itemno];
		double sum = 0;
		for (int i = 0; i < 11; i++) sum += item.spawn_probabilities[i];
		if (sum > 0) {
			valid = 1;
			res = itemno;
		}
	}
	return res;
}

__device__ void CheckNormalChestLoot(int x, int y, uint worldSeed, LootConfig cfg, bool hasMimicSign, byte** bytes)
{
	writeByte(bytes, START_SPAWNABLE);
	writeInt(bytes, x);
	writeInt(bytes, y);
	writeByte(bytes, TYPE_CHEST);
	if (hasMimicSign) writeByte(bytes, MIMIC_SIGN);

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x + 509.7, y + 683.1);

	int count = 1;
	while (count > 0)
	{
		count--;
		int rnd = random.Random(1, 100);
		if (rnd <= 7) writeByte(bytes, BOMB);
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
					for (int i = 0; i < tamount; i++) {
						random.Random(-10, 10);
						random.Random(-10, 5);
					}
				}
				if (random.Random(0, 100) > 80) {
					tamount = random.Random(1, 3);
					for (int i = 0; i < tamount; i++) {
						random.Random(-10, 10);
						random.Random(-10, 5);
					}
				}
			}
			else {
				random.Random(-10, 10);
				random.Random(-10, 5);
			}
			writeByte(bytes, GOLD_NUGGETS);
		}
		else if (rnd <= 50)
		{
			rnd = random.Random(1, 100);
			if (rnd <= 94) createPotion(x, y, POTION_NORMAL, worldSeed, cfg, bytes);
			else if (rnd <= 98) writeByte(bytes, POWDER);
			else
			{
				rnd = random.Random(0, 100);
				if (rnd <= 98) createPotion(x, y, POTION_SECRET, worldSeed, cfg, bytes);
				else createPotion(x, y, POTION_RANDOM_MATERIAL, worldSeed, cfg, bytes);
			}
		}
		else if (rnd <= 54) writeByte(bytes, SPELL_REFRESH);
		else if (rnd <= 60)
		{
			Item opts[8] = { KAMMI, KUU, UKKOSKIVI, PAHA_SILMA, KIUASKIVI, (Item)127, CHAOS_DIE, SHINY_ORB};
			rnd = random.Random(0, 7);
			Item opt = opts[rnd];
			if ((int)opt == 127)
			{
				Item r_opts[7] = { RUNESTONE_LIGHT, RUNESTONE_FIRE, RUNESTONE_MAGMA, RUNESTONE_WEIGHT, RUNESTONE_EMPTINESS, RUNESTONE_EDGES, RUNESTONE_METAL };
				rnd = random.Random(0, 6);
				Item r_opt = r_opts[rnd];
				writeByte(bytes, r_opt);
			}
			else
			{
				writeByte(bytes, opt);
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

			for (int i = 0; i < amount; i++) {
				random.Random(0, 1);
				if (false) {
					int randCTR = random.randomCTR;
					//writeByte(bytes, (randCTR << 1) | 0x80;
				}
				MakeRandomCard(&random);
			}

			if (true)
				writeByte(bytes, RANDOM_SPELL);
		}
		else if (rnd <= 84)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) writeByte(bytes, WAND_T1);
			else if (rnd <= 50) writeByte(bytes, WAND_T1NS);
			else if (rnd <= 75) writeByte(bytes, WAND_T2);
			else if (rnd <= 90) writeByte(bytes, WAND_T2NS);
			else if (rnd <= 96) writeByte(bytes, WAND_T3);
			else if (rnd <= 98) writeByte(bytes, WAND_T3NS);
			else if (rnd <= 99) writeByte(bytes, WAND_T4);
			else writeByte(bytes, WAND_T4NS);
		}
		else if (rnd <= 95)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 88) writeByte(bytes, HEART_NORMAL);
			else if (rnd <= 89) writeByte(bytes, HEART_MIMIC);
			else if (rnd <= 99) writeByte(bytes, HEART_BIGGER);
			else writeByte(bytes, FULL_HEAL);
		}
		else if (rnd <= 98) writeByte(bytes, CHEST_TO_GOLD);
		else if (rnd <= 99) count += 2;
		else count += 3;
	}
	writeByte(bytes, END_SPAWNABLE);
}

__device__ void CheckGreatChestLoot(int x, int y, uint worldSeed, LootConfig cfg, bool hasMimicSign, byte** bytes)
{
	writeByte(bytes, START_SPAWNABLE);
	writeInt(bytes, x);
	writeInt(bytes, y);
	writeByte(bytes, TYPE_CHEST_GREATER);
	if (hasMimicSign) writeByte(bytes, MIMIC_SIGN);

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x, y);

	int count = 1;

	if (random.Random(0, 100000) >= 100000)
	{
		count = 0;
		if (random.Random(0, 1000) == 999) writeByte(bytes, TRUE_ORB);
		else writeByte(bytes, SAMPO);
	}

	while (count != 0)
	{
		count--;
		int rnd = random.Random(1, 100);

		if (rnd <= 30)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 30) {
				createPotion(x, y, POTION_SECRET, worldSeed, cfg, bytes);
				createPotion(x, y, POTION_SECRET, worldSeed, cfg, bytes);
				createPotion(x, y, POTION_RANDOM_MATERIAL, worldSeed, cfg, bytes);
			}
			else {
				createPotion(x, y, POTION_NORMAL, worldSeed, cfg, bytes);
				createPotion(x, y, POTION_NORMAL, worldSeed, cfg, bytes);
				createPotion(x, y, POTION_SECRET, worldSeed, cfg, bytes);
			}
		}
		else if (rnd <= 33)
		{
			writeByte(bytes, RAIN_GOLD);
		}
		else if (rnd <= 38)
		{
			rnd = random.Random(1, 30);
			if (rnd == 30)
				writeByte(bytes, KAKKAKIKKARE);
			else writeByte(bytes, VUOKSIKIVI);
		}
		else if (rnd <= 39)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) writeByte(bytes, WAND_T3);
			else if (rnd <= 50) writeByte(bytes, WAND_T3NS);
			else if (rnd <= 75) writeByte(bytes, WAND_T4);
			else if (rnd <= 90) writeByte(bytes, WAND_T4NS);
			else if (rnd <= 96) writeByte(bytes, WAND_T5);
			else if (rnd <= 98) writeByte(bytes, WAND_T5NS);
			else if (rnd <= 99) writeByte(bytes, WAND_T6);
			else writeByte(bytes, WAND_T6NS);
		}
		else if (rnd <= 60)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 89) writeByte(bytes, HEART_NORMAL);
			else if (rnd <= 99) writeByte(bytes, HEART_BIGGER);
			else writeByte(bytes, FULL_HEAL);
		}
		else if (rnd <= 99) count += 2;
		else count += 3;
	}
	writeByte(bytes, END_SPAWNABLE);
}

__device__ void CheckItemPedestalLoot(int x, int y, uint worldSeed, LootConfig cfg, byte** bytes)
{
	writeByte(bytes, START_SPAWNABLE);
	writeInt(bytes, x);
	writeInt(bytes, y);
	writeByte(bytes, TYPE_ITEM_PEDESTAL);

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x + 425, y - 243);

	int rnd = random.Random(1, 91);

	if (rnd <= 65)
		createPotion(x, y, POTION_NORMAL, worldSeed, cfg, bytes);
	else if (rnd <= 70)
		writeByte(bytes, POWDER);
	else if (rnd <= 71)
		writeByte(bytes, CHAOS_DIE);
	else if (rnd <= 72) {
		Item r_opts[7] = { RUNESTONE_LIGHT, RUNESTONE_FIRE, RUNESTONE_MAGMA, RUNESTONE_WEIGHT, RUNESTONE_EMPTINESS, RUNESTONE_EDGES, RUNESTONE_METAL };
		rnd = random.Random(0, 6);
		Item r_opt = r_opts[rnd];
		writeByte(bytes, r_opt);
	}
	else if (rnd <= 73)
		writeByte(bytes, EGG_PURPLE);
	else if (rnd <= 77)
		writeByte(bytes, EGG_SLIME);
	else if (rnd <= 79)
		writeByte(bytes, EGG_MONSTER);
	else if (rnd <= 83)
		writeByte(bytes, KIUASKIVI);
	else if (rnd <= 85)
		writeByte(bytes, UKKOSKIVI);
	else if (rnd <= 89)
		writeByte(bytes, BROKEN_WAND);
	else
		writeByte(bytes, SHINY_ORB);
	writeByte(bytes, END_SPAWNABLE);
}

__device__ void spawnHeart(int x, int y, uint seed, LootConfig cfg, byte** bytes)
{
	NoitaRandom random = NoitaRandom(seed);
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
			if (random.Random(1, 300) == 1) {
				hasSign = true;
			}
			if (rnd >= 1000)
				CheckGreatChestLoot(x, y, seed, cfg, hasSign, bytes);
			else
				CheckNormalChestLoot(x, y, seed, cfg, hasSign, bytes);
		}
		else {
			writeByte(bytes, START_SPAWNABLE);
			writeInt(bytes, x);
			writeInt(bytes, y);
			writeByte(bytes, TYPE_CHEST);
			
			rnd = random.Random(1, 100);
			if(random.Random(1,30==1)) writeByte(bytes, MIMIC_SIGN);
			if(rnd <= 95) writeByte(bytes, MIMIC);
			else writeByte(bytes, MIMIC_LEGGY);
			writeByte(bytes, END_SPAWNABLE);
		}
	}
}

__device__ void spawnChest(int x, int y, uint seed, LootConfig cfg, byte** bytes)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	int super_chest_spawn_rate = cfg.greedCurse ? 100 : 2000;
	int rnd = random.Random(1, super_chest_spawn_rate);

	if (rnd >= super_chest_spawn_rate - 1)
		CheckGreatChestLoot(x, y, seed, cfg, false, bytes);
	else
		CheckNormalChestLoot(x, y, seed, cfg, false, bytes);
}

__device__ void spawnPotion(int x, int y, uint seed, LootConfig cfg, byte** bytes)
{
	NoitaRandom random = NoitaRandom(seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 5, y - 4, seed, cfg, bytes);
}

__device__ void spawnPixelScene(int x, int y, uint seed, byte oiltank, LootConfig cfg, byte** bytes)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	int rnd = random.Random(1, 100);
	if (rnd <= 50 && oiltank == 0 || rnd > 50 && oiltank > 0) {
		float rnd2 = random.ProceduralRandomf(x, y, 0, 1) * 3;
		if (0.5f < rnd2 && rnd2 < 1) {
			spawnChest(x + 94, y + 224, seed, cfg, bytes);
		}
	}
}

__device__ void spawnPixelScene1(int x, int y, uint seed, LootConfig cfg, byte** bytes) {
	spawnPixelScene(x, y, seed, 0, cfg, bytes);
}

__device__ void spawnOilTank(int x, int y, uint seed, LootConfig cfg, byte** bytes) {
	spawnPixelScene(x, y, seed, 1, cfg, bytes);
}

__device__ void spawnWand(int x, int y, uint seed, LootConfig cfg, byte** bytes) {
	NoitaRandom random = NoitaRandom(seed);
	float r = random.ProceduralRandomf(x, y, 0, 1);
	if (r < 0.47) return;
	r = random.ProceduralRandomf(x - 11.431, y + 10.5257, 0, 1);
	if (r < 0.755) return;

	int nx = x - 5;
	int ny = y - 14;
	BiomeWands wandSet = wandLevels[0]; //biomeIndex
	int sum = 0;
	for (int i = 0; i < wandSet.count; i++) sum += wandSet.levels[i].prob;
	r = random.ProceduralRandomf(nx, ny, 0, 1) * sum;
	for (int i = 0; i < wandSet.count; i++) {
		if (r <= wandSet.levels[i].prob) {

			writeByte(bytes, START_SPAWNABLE);
			writeInt(bytes, nx + 5);
			writeInt(bytes, ny + 5);
			writeByte(bytes, TYPE_WAND_PEDESTAL);
			writeByte(bytes, wandSet.levels[i].id);
			writeByte(bytes, END_SPAWNABLE);
			return;
		}
		r -= wandSet.levels[i].prob;
	}
}

__device__ void CheckSpawnables(byte* res, uint seed, byte** bytes, byte* output, WorldgenConfig wCfg, LootConfig lCfg) {
	static void (*spawnFuncs[5])(int, int, uint, LootConfig, byte**) = { spawnHeart, spawnChest, spawnPixelScene1, spawnOilTank, spawnPotion };
	byte* origin = *bytes;
	byte* map = res + 4 * 3 * wCfg.map_w;
	writeByte(bytes, START_BLOCK);
	writeInt(bytes, seed);

	for (int px = 0; px < wCfg.map_w; px++)
	{
		for (int py = 0; py < wCfg.map_h; py++)
		{
			int pixelPos = 3 * (px + py * wCfg.map_w);
			if (map[pixelPos] == 0 && map[pixelPos + 1] == 0)
				continue;
			if (map[pixelPos] == 255 && map[pixelPos + 1] == 255)
				continue;

			IntPair gp = GetGlobalPos(wCfg.worldX, wCfg.worldY, px * 10, py * 10);

			const int PWSize = 70 * 512;

			//avoids having to switch every loop
			auto func = spawnFuncs[0];
			long pix = createRGB(map[pixelPos], map[pixelPos + 1], map[pixelPos + 2]);

			switch (pix) {
			case 0x78ffff:
				if (lCfg.searchChests)
					func = spawnFuncs[0];
				else continue;
				break;
			case 0x55ff8c:
				if (lCfg.searchChests)
					func = spawnFuncs[1];
				else continue;
				break;
			case 0xff0aff:
				if (lCfg.searchPixelScenes)
					func = spawnFuncs[2];
				else continue;
				break;
			case 0xc35700:
				if (lCfg.searchPixelScenes)
					func = spawnFuncs[3];
				else continue;
				break;
			case 0x50a000:
				if (lCfg.searchPedestals)
					func = spawnFuncs[4];
				else continue;
				break;
			case 0x00ff00:
				if (lCfg.searchWandAltars)
					func = spawnFuncs[5];
				else continue;
				break;
			default:
				continue;
			}

			for (int i = -lCfg.pwCount; i <= lCfg.pwCount; i++)
				func(gp.x + PWSize * i, gp.y, seed, lCfg, bytes);
		}
	}
	writeByte(bytes, END_BLOCK);
	//memcpy(output, origin, *bytes - origin);
}

__device__ Spawnable DecodeSpawnable(byte** bytes) {
	Spawnable ret = {};
	ret.x = readInt(bytes);
	ret.y = readInt(bytes);
	ret.sType = (SpawnableMetadata)(readByte(bytes));

	int i = 0;
	while (*(*bytes + i) != END_SPAWNABLE) i++;
	ret.count = i;
	ret.contents = (Item*)malloc(ret.count);
	memcpy(ret.contents, *bytes, ret.count);
	(*bytes) += i + 1;
	return ret;
}

__device__ SeedSpawnables ParseSpawnableBlock(byte** bytes, byte* output, LootConfig cfg) {
	(*bytes)++;
	uint seed = readInt(bytes);
	//int i = 0;
	int spawnableCount = 100;
	//while (*(*bytes + i) != END_BLOCK) {
	//	if (*(*bytes + i) == START_SPAWNABLE) {
	//		spawnableCount++;
	//	}
	//	i++;
	//}
	//printf("%i spawnables in %i bytes\n", spawnableCount, i);
	Spawnable* spawnables = (Spawnable*)malloc(sizeof(Spawnable) * spawnableCount);
	int idx = 0;
	SpawnableMetadata b = *(SpawnableMetadata*)*bytes;
	while (b != END_BLOCK) {
		b = (SpawnableMetadata)readByte(bytes);
		if (b == START_SPAWNABLE) {
			if (idx >= spawnableCount) printf("Stack blown!");
			else spawnables[idx++] = DecodeSpawnable(bytes);
		}
	}
	//memcpy(output, bak, *bytes - bak);
	SeedSpawnables ret{ seed, idx, spawnables };
	return ret;
}