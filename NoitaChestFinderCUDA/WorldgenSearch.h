#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/noita_random.h"
#include "misc/datatypes.h"
#include "misc/utilities.h"
#include "data/items.h"
#include "data/spells.h"
#include "data/wand_levels.h"

#include <iostream>

struct LootConfig {
	int pwCount;
	int blockSize;

	bool searchChests;
	bool searchPedestals;
	bool searchWandAltars;
	bool searchPixelScenes;

	bool greedCurse;
	bool checkPotions;
	bool checkWands;
	bool checkCards;
};

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

__device__ void CheckNormalChestLoot(int x, int y, uint worldSeed, LootConfig cfg, byte** writeLoc)
{
	writeByte(writeLoc, START_SPAWNABLE);
	writeInt(writeLoc, x);
	writeInt(writeLoc, y);
	writeByte(writeLoc, TYPE_CHEST);

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x + 509.7, y + 683.1);

	int count = 1;
	while (count > 0)
	{
		count--;
		int rnd = random.Random(1, 100);
		if (rnd <= 7) writeByte(writeLoc, BOMB);
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
			writeByte(writeLoc, GOLD_NUGGETS);
		}
		else if (rnd <= 50)
		{
			rnd = random.Random(1, 100);
			if (rnd <= 94) writeByte(writeLoc, POTION_NORMAL);
			else if (rnd <= 98) writeByte(writeLoc, POWDER);
			else
			{
				rnd = random.Random(0, 100);
				if (rnd <= 98) writeByte(writeLoc, POTION_SECRET);
				else writeByte(writeLoc, POTION_RANDOM_MATERIAL);
			}
		}
		else if (rnd <= 54) writeByte(writeLoc, SPELL_REFRESH);
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
				writeByte(writeLoc, r_opt);
			}
			else
			{
				writeByte(writeLoc, opt);
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
					//writeByte(writeLoc, (randCTR << 1) | 0x80;
				}
				MakeRandomCard(&random);
			}

			if (true)
				writeByte(writeLoc, RANDOM_SPELL);
		}
		else if (rnd <= 84)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) writeByte(writeLoc, WAND_T1);
			else if (rnd <= 50) writeByte(writeLoc, WAND_T1NS);
			else if (rnd <= 75) writeByte(writeLoc, WAND_T2);
			else if (rnd <= 90) writeByte(writeLoc, WAND_T2NS);
			else if (rnd <= 96) writeByte(writeLoc, WAND_T3);
			else if (rnd <= 98) writeByte(writeLoc, WAND_T3NS);
			else if (rnd <= 99) writeByte(writeLoc, WAND_T4);
			else writeByte(writeLoc, WAND_T4NS);
		}
		else if (rnd <= 95)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 88) writeByte(writeLoc, HEART_NORMAL);
			else if (rnd <= 89) writeByte(writeLoc, HEART_MIMIC);
			else if (rnd <= 99) writeByte(writeLoc, HEART_BIGGER);
			else writeByte(writeLoc, FULL_HEAL);
		}
		else if (rnd <= 98) writeByte(writeLoc, CHEST_TO_GOLD);
		else if (rnd <= 99) count += 2;
		else count += 3;
	}
	writeByte(writeLoc, END_SPAWNABLE);
}

__device__ void CheckGreatChestLoot(int x, int y, uint worldSeed, LootConfig cfg, byte** writeLoc)
{
	writeByte(writeLoc, START_SPAWNABLE);
	writeInt(writeLoc, x);
	writeInt(writeLoc, y);
	writeByte(writeLoc, TYPE_CHEST_GREATER);

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x, y);

	int count = 1;

	if (random.Random(0, 100000) >= 100000)
	{
		count = 0;
		if (random.Random(0, 1000) == 999) writeByte(writeLoc, TRUE_ORB);
		else writeByte(writeLoc, SAMPO);
	}

	while (count != 0)
	{
		count--;
		int rnd = random.Random(1, 100);

		if (rnd <= 30)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 30) {
				writeByte(writeLoc, POTION_SECRET);
				writeByte(writeLoc, POTION_SECRET);
				writeByte(writeLoc, POTION_RANDOM_MATERIAL);
			}
			else {
				writeByte(writeLoc, POTION_NORMAL);
				writeByte(writeLoc, POTION_NORMAL);
				writeByte(writeLoc, POTION_SECRET);
			}
		}
		else if (rnd <= 33)
		{
			writeByte(writeLoc, RAIN_GOLD);
		}
		else if (rnd <= 38)
		{
			rnd = random.Random(1, 30);
			if (rnd == 30)
				writeByte(writeLoc, KAKKAKIKKARE);
			else writeByte(writeLoc, VUOKSIKIVI);
		}
		else if (rnd <= 39)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 25) writeByte(writeLoc, WAND_T3);
			else if (rnd <= 50) writeByte(writeLoc, WAND_T3NS);
			else if (rnd <= 75) writeByte(writeLoc, WAND_T4);
			else if (rnd <= 90) writeByte(writeLoc, WAND_T4NS);
			else if (rnd <= 96) writeByte(writeLoc, WAND_T5);
			else if (rnd <= 98) writeByte(writeLoc, WAND_T5NS);
			else if (rnd <= 99) writeByte(writeLoc, WAND_T6);
			else writeByte(writeLoc, WAND_T6NS);
		}
		else if (rnd <= 60)
		{
			rnd = random.Random(0, 100);
			if (rnd <= 89) writeByte(writeLoc, HEART_NORMAL);
			else if (rnd <= 99) writeByte(writeLoc, HEART_BIGGER);
			else writeByte(writeLoc, FULL_HEAL);
		}
		else if (rnd <= 99) count += 2;
		else count += 3;
	}
	writeByte(writeLoc, END_SPAWNABLE);
}

__device__ void CheckItemPedestalLoot(int x, int y, uint worldSeed, LootConfig cfg, byte** writeLoc)
{
	writeByte(writeLoc, START_SPAWNABLE);
	writeInt(writeLoc, x);
	writeInt(writeLoc, y);
	writeByte(writeLoc, TYPE_ITEM_PEDESTAL);

	NoitaRandom random = NoitaRandom(worldSeed);
	random.SetRandomSeed(x + 425, y - 243);

	int rnd = random.Random(1, 91);

	if (rnd <= 65)
		writeByte(writeLoc, POTION_NORMAL);
	else if (rnd <= 70)
		writeByte(writeLoc, POWDER);
	else if (rnd <= 71)
		writeByte(writeLoc, CHAOS_DIE);
	else if (rnd <= 72) {
		Item r_opts[7] = { RUNESTONE_LIGHT, RUNESTONE_FIRE, RUNESTONE_MAGMA, RUNESTONE_WEIGHT, RUNESTONE_EMPTINESS, RUNESTONE_EDGES, RUNESTONE_METAL };
		rnd = random.Random(0, 6);
		Item r_opt = r_opts[rnd];
		writeByte(writeLoc, r_opt);
	}
	else if (rnd <= 73)
		writeByte(writeLoc, EGG_PURPLE);
	else if (rnd <= 77)
		writeByte(writeLoc, EGG_SLIME);
	else if (rnd <= 79)
		writeByte(writeLoc, EGG_MONSTER);
	else if (rnd <= 83)
		writeByte(writeLoc, KIUASKIVI);
	else if (rnd <= 85)
		writeByte(writeLoc, UKKOSKIVI);
	else if (rnd <= 89)
		writeByte(writeLoc, BROKEN_WAND);
	else
		writeByte(writeLoc, SHINY_ORB);
	writeByte(writeLoc, END_SPAWNABLE);
}

__device__ void spawnHeart(int x, int y, uint seed, LootConfig cfg, byte** writeLoc)
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
			if (rnd >= 1000)
				CheckGreatChestLoot(x, y, seed, cfg, writeLoc);
			else
				CheckNormalChestLoot(x, y, seed, cfg, writeLoc);
		}
	}
}

__device__ void spawnChest(int x, int y, uint seed, LootConfig cfg, byte** writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	int super_chest_spawn_rate = cfg.greedCurse ? 100 : 2000;
	int rnd = random.Random(1, super_chest_spawn_rate);

	if (rnd >= super_chest_spawn_rate - 1)
		CheckGreatChestLoot(x, y, seed, cfg, writeLoc);
	else
		CheckNormalChestLoot(x, y, seed, cfg, writeLoc);
}

__device__ void spawnPotion(int x, int y, uint seed, LootConfig cfg, byte** writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
	float rnd = random.ProceduralRandomf(x, y, 0, 1);

	if (rnd > 0.65f)
		CheckItemPedestalLoot(x + 5, y - 4, seed, cfg, writeLoc);
}

__device__ void spawnPixelScene(int x, int y, uint seed, byte oiltank, LootConfig cfg, byte** writeLoc)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	int rnd = random.Random(1, 100);
	if (rnd <= 50 && oiltank == 0 || rnd > 50 && oiltank > 0) {
		float rnd2 = random.ProceduralRandomf(x, y, 0, 1) * 3;
		if (0.5f < rnd2 && rnd2 < 1) {
			spawnChest(x + 94, y + 224, seed, cfg, writeLoc);
		}
	}
}

__device__ void spawnPixelScene1(int x, int y, uint seed, LootConfig cfg, byte** writeLoc) {
	spawnPixelScene(x, y, seed, 0, cfg, writeLoc);
}

__device__ void spawnOilTank(int x, int y, uint seed, LootConfig cfg, byte** writeLoc) {
	spawnPixelScene(x, y, seed, 1, cfg, writeLoc);
}

__device__ void spawnWand(int x, int y, uint seed, LootConfig cfg, byte** writeLoc) {
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

			writeByte(writeLoc, START_SPAWNABLE);
			writeInt(writeLoc, nx + 5);
			writeInt(writeLoc, ny + 5);
			writeByte(writeLoc, TYPE_WAND_PEDESTAL);
			writeByte(writeLoc, wandSet.levels[i].id);
			writeByte(writeLoc, END_SPAWNABLE);
			return;
		}
		r -= wandSet.levels[i].prob;
	}
}

__device__ void CheckSpawnables(byte* origin, byte* res, uint seed, byte** writeLoc, byte* output, WorldgenConfig wCfg, LootConfig lCfg) {
	static void (*spawnFuncs[5])(int, int, uint, LootConfig, byte**) = { spawnHeart, spawnChest, spawnPixelScene1, spawnOilTank, spawnPotion };
	byte* map = res + 4 * 3 * wCfg.map_w;
	byte* bak = *writeLoc;
	//printf("origin offset: %x\n", bak - origin);
	writeByte(writeLoc, START_BLOCK);
	writeInt(writeLoc, seed);

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
				func(gp.x + PWSize * i, gp.y, seed, lCfg, writeLoc);
		}
	}
	writeByte(writeLoc, END_BLOCK);
	//printf("offset: %i\n", *writeLoc - bak);
}

__device__ Spawnable DecodeSpawnable(byte** bytes) {
	Spawnable ret;
	ret.x = readInt(bytes);
	ret.y = readInt(bytes);
	ret.sType = (SpawnableType)(readByte(bytes) - (byte)SpawnableMetadata::TYPE_CHEST);
	int i = 0;
	while (*(SpawnableMetadata*)(*bytes + i) != END_SPAWNABLE) i++;
	ret.count = i;
	ret.contents = (Item*)malloc(ret.count);
	memcpy(ret.contents, *bytes, ret.count);
	(*bytes) += i + 1;
	return ret;
}

__device__ SeedSpawnables ParseSpawnableBlock(byte** bytes, byte* output, LootConfig cfg) {
	byte* bak = *bytes;
	(*bytes)++;
	uint seed = readInt(bytes);
	Spawnable* spawnables = (Spawnable*)malloc(sizeof(Spawnable) * cfg.blockSize);
	int idx = 0;
	SpawnableMetadata b = *(SpawnableMetadata*)*bytes;
	while (b != END_BLOCK) {
		b = (SpawnableMetadata)readByte(bytes);
		//printf("offset %i: %i\n", (int)(*bytes - bak), (byte)b);
		if (b == START_SPAWNABLE) {
			spawnables[idx++] = DecodeSpawnable(bytes);
		}
	}
	memcpy(output, bak, *bytes - bak + 100);
	SeedSpawnables ret{ seed, idx, spawnables };
	return ret;
}