#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../misc/datatypes.h"

enum SpawnableMetadata : byte
{
	BYTE_NONE,
	TYPE_CHEST = 8,
	TYPE_CHEST_GREATER,
	TYPE_ITEM_PEDESTAL,
	TYPE_WAND_PEDESTAL,
	TYPE_NIGHTMARE_WAND,
	TYPE_EOE_DROP,
	TYPE_HM_SHOP,

	DATA_MATERIAL,
	DATA_SPELL,
	DATA_WAND,
};

enum Item : byte
{
	ITEM_NONE,

	GOLD_NUGGETS = 32,
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

	WAND_T1B,
	WAND_T2B,
	WAND_T3B,
	WAND_T4B,
	WAND_T5B,
	WAND_T6B,

	WAND_T10,
	WAND_T10NS,

	EGG_PURPLE,
	EGG_SLIME,
	EGG_MONSTER,
	BROKEN_WAND,
	UNKNOWN_WAND,

	MIMIC,
	MIMIC_LEGGY,
	MIMIC_SIGN,

	SAMPO = 253,
	TRUE_ORB,
	ERR
};

__device__ const char* SpawnableTypeNames[] = {
	"CHEST",
	"GREAT_CHEST",
	"ITEM_PEDESTAL",
	"WAND_PEDESTAL",
	"NIGHTMARE_WAND",
	"EOE_DROP",
	"HM_SHOP"
};

__device__ const char* ItemNames[] = {
	"GOLD_NUGGETS",
	"CHEST_TO_GOLD",
	"RAIN_GOLD",
	"BOMB",
	"POWDER",
	"POTION_NORMAL",
	"POTION_SECRET",
	"POTION_RANDOM_MATERIAL",
	"KAMMI",
	"KUU",
	"PAHA_SILMA",
	"CHAOS_DIE",
	"SHINY_ORB",
	"UKKOSKIVI",
	"KIUASKIVI",
	"VUOKSIKIVI",
	"KAKKAKIKKARE",
	"RUNESTONE_LIGHT",
	"RUNESTONE_FIRE",
	"RUNESTONE_MAGMA",
	"RUNESTONE_WEIGHT",
	"RUNESTONE_EMPTINESS",
	"RUNESTONE_EDGES",
	"RUNESTONE_METAL",
	"RANDOM_SPELL",
	"SPELL_REFRESH",
	"HEART_NORMAL",
	"HEART_MIMIC",
	"HEART_BIGGER",
	"FULL_HEAL",
	"WAND_T1",
	"WAND_T1NS",
	"WAND_T2",
	"WAND_T2NS",
	"WAND_T3",
	"WAND_T3NS",
	"WAND_T4",
	"WAND_T4NS",
	"WAND_T5",
	"WAND_T5NS",
	"WAND_T6",
	"WAND_T6NS",

	"WAND_T1B",
	"WAND_T2B",
	"WAND_T3B",
	"WAND_T4B",
	"WAND_T5B",
	"WAND_T6B",

	"WAND_T10",
	"WAND_T10NS",

	"EGG_PURPLE",
	"EGG_SLIME",
	"EGG_MONSTER",
	"BROKEN_WAND",
	"UNKNOWN_WAND",

	"MIMIC",
	"MIMIC_LEGGY",
	"MIMIC_SIGN",
};

//DON'T REARRANGE CONTENTS! THE ORDER IS HARDCODED
#pragma pack(push, 1)
struct Spawnable
{
	int x;
	int y;
	SpawnableMetadata sType;
	int count;
	Item contents;
};
#pragma pack(pop)

//DITTO
struct SpawnableBlock
{
	uint seed;
	int count;
	Spawnable** spawnables;
};

__device__
byte readByte(byte* ptr, int& offset)
{
	return ptr[offset++];
}

__device__
void writeByte(byte* ptr, int& offset, byte b)
{
	ptr[offset++] = b;
}

__device__  
int readInt(byte* ptr, int& offset)
{
	int tmp;
	memcpy(&tmp, ptr + offset, 4);
	offset += 4;
	return tmp;
}

__device__  
void writeInt(byte* ptr, int& offset, int val)
{
	memcpy(ptr + offset, &val, 4);
	offset += 4;
}

__device__ 
void incrInt(byte* ptr)
{
	int offsetTmp = 0;
	int tmp = readInt(ptr, offsetTmp);
	offsetTmp = 0;
	writeInt(ptr, offsetTmp, tmp + 1);
}

__device__ 
short readShort(byte* ptr, int& offset)
{
	return (readByte(ptr, offset) | (readByte(ptr, offset) << 8));
}

__device__ 
void writeShort(byte* ptr, int& offset, short s)
{
	writeByte(ptr, offset, ((short)s) & 0xff);
	writeByte(ptr, offset, (((short)s) >> 8) & 0xff);
}

__device__ int readMisaligned(int* ptr2)
{
	byte* ptr = (byte*)ptr2;
	int offset = 0;
	return readInt(ptr, offset);
}