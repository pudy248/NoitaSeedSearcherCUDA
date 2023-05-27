#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "primitives.h"
#include "enums.h"

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
struct SpawnableBlock
{
	uint seed;
	int count;
	Spawnable** spawnables;
};


#pragma pack(push, 1)
struct LabelledSpell
{
	SpawnableMetadata d;
	Spell s;
};
#pragma pack(pop)
struct SpellData
{
	Spell s;
	ActionType type;
	double spawn_probabilities[11];
};
struct SpellProb
{
	float p;
	Spell s;
};

struct Wand
{
	int level;
	bool isBetter;
	float cost;

	float prob_unshuffle;
	float prob_draw_many;
	bool force_unshuffle;
	bool is_rare;

	float capacity;
	int multicast;
	int mana;
	int regen;
	int delay;
	int reload;
	float speed;
	int spread;
	bool shuffle;
	byte spellCount;
	LabelledSpell alwaysCast;
	LabelledSpell spells[67];
};
//DON'T REARRANGE CONTENTS! THE ORDER IS HARDCODED
#pragma pack(push, 1)
struct WandData
{
	float capacity;
	int multicast;
	int mana;
	int regen;
	int delay;
	int reload;
	float speed;
	int spread;
	bool shuffle;
	byte spellCount;
	LabelledSpell alwaysCast;
	LabelledSpell spells;
};
#pragma pack(pop)

struct wandLevel
{
	float prob;
	Item id;
};
struct BiomeWands
{
	int count;
	wandLevel levels[6];
};

struct WandSprite
{
	const char* name;
	int fileNum;
	sbyte grip_x;
	sbyte grip_y;
	sbyte tip_x;
	sbyte tip_y;
	sbyte fire_rate_wait;
	sbyte actions_per_round;
	bool shuffle_deck_when_empty;
	sbyte deck_capacity;
	sbyte spread_degrees;
	sbyte reload_time;
};
struct WandSpaceDat
{
	float fire_rate_wait;
	float actions_per_round;
	bool shuffle_deck_when_empty;
	float deck_capacity;
	float spread_degrees;
	float reload_time;
};


__host__ __device__
byte readByte(byte* ptr, int& offset)
{
	return ptr[offset++];
}

__host__ __device__
void writeByte(byte* ptr, int& offset, byte b)
{
	ptr[offset++] = b;
}

__host__ __device__
int readInt(byte* ptr, int& offset)
{
	int tmp;
	memcpy(&tmp, ptr + offset, 4);
	offset += 4;
	return tmp;
}

__host__ __device__
void writeInt(byte* ptr, int& offset, int val)
{
	memcpy(ptr + offset, &val, 4);
	offset += 4;
}

__host__ __device__
void incrInt(byte* ptr)
{
	int offsetTmp = 0;
	int tmp = readInt(ptr, offsetTmp);
	offsetTmp = 0;
	writeInt(ptr, offsetTmp, tmp + 1);
}

__host__ __device__
short readShort(byte* ptr, int& offset)
{
	return (readByte(ptr, offset) | (readByte(ptr, offset) << 8));
}

__host__ __device__
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

__device__ Spawnable readMisalignedSpawnable(Spawnable* sPtr)
{
	byte* bPtr = (byte*)sPtr;
	Spawnable s;
	int offset = 0;
	s.x = readInt(bPtr, offset);
	s.y = readInt(bPtr, offset);
	s.sType = (SpawnableMetadata)readByte(bPtr, offset);
	s.count = readInt(bPtr, offset);
	return s;
}

__device__ WandData readMisalignedWand(WandData* wPtr)
{
	WandData w = {};
	memcpy(&w, wPtr, 37);
	return w;
}
