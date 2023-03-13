#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../misc/datatypes.h"

enum SpawnableMetadata : byte
{
	BYTE_NONE,
	START_BLOCK,
	END_BLOCK,
	START_SPAWNABLE,
	END_SPAWNABLE,
	TYPE_CHEST,
	TYPE_CHEST_GREATER,
	TYPE_ITEM_PEDESTAL,
	TYPE_WAND_PEDESTAL,
	TYPE_EOE_DROP,

	DATA_MATERIAL,
	DATA_WAND,
	DATA_SPELL,
};

enum Item : byte {
	GOLD_NUGGETS=16,
	CHEST_TO_GOLD,
	RAIN_GOLD,
	BOMB,
	POWDER,
	POTION_NORMAL,
	POTION_SECRET,
	POTION_RANDOM_MATERIAL,
	KAMMI,
	KUU,
	PAHA_SILMA,
	CHAOS_DIE,
	SHINY_ORB,
	UKKOSKIVI,
	KIUASKIVI,
	VUOKSIKIVI,
	KAKKAKIKKARE,
	RUNESTONE_LIGHT,
	RUNESTONE_FIRE,
	RUNESTONE_MAGMA,
	RUNESTONE_WEIGHT,
	RUNESTONE_EMPTINESS,
	RUNESTONE_EDGES,
	RUNESTONE_METAL,
	RANDOM_SPELL,
	SPELL_REFRESH,
	HEART_NORMAL,
	HEART_MIMIC,
	HEART_BIGGER,
	FULL_HEAL,
	WAND_T1,
	WAND_T1NS,
	WAND_T2,
	WAND_T2NS,
	WAND_T3,
	WAND_T3NS,
	WAND_T4,
	WAND_T4NS,
	WAND_T5,
	WAND_T5NS,
	WAND_T6,
	WAND_T6NS,

	EGG_PURPLE,
	EGG_SLIME,
	EGG_MONSTER,
	BROKEN_WAND,
	UNKNOWN_WAND,

	WAND_T10NS,
	WAND_T1B,
	WAND_T2B,
	WAND_T3B,
	WAND_T4B,
	WAND_T5B,
	WAND_T6B,

	SAMPO=253,
	TRUE_ORB,
	ERR
};

enum SpawnableType : byte
{
	Chest,
	GreatChest,
	ItemPedestal,
	WandPedestal,
	EoEDrop
};

struct Spawnable {
	int x;
	int y;
	SpawnableType sType;
	int count;
	Item* contents;
};

struct SeedSpawnables {
	uint seed;
	int count;
	Spawnable* spawnables;
};

__device__ void freeSeedSpawnables(SeedSpawnables s) {
	for (int i = 0; i < s.count; i++) free(s.spawnables[i].contents);
	free(s.spawnables);
}

__device__ byte readByte(byte** ptr) {
	return *(byte*)((*ptr)++);
}

__device__ void writeByte(byte** ptr, byte b) {
	*(byte*)((*ptr)++) = b;
}

__device__ int readInt(byte** ptr) {
	return (readByte(ptr)) | (readByte(ptr) << 8) | (readByte(ptr) << 16) | (readByte(ptr) << 24);
}

__device__ void writeInt(byte** ptr, int val) {
	writeByte(ptr, val & 0xff);
	writeByte(ptr, (val >> 8) & 0xff);
	writeByte(ptr, (val >> 16) & 0xff);
	writeByte(ptr, (val >> 24) & 0xff);
}
