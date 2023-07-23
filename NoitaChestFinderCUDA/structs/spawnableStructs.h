#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "primitives.h"
#include "enums.h"

#include <initializer_list>

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
	uint32_t seed;
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
	double p;
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
	uint8_t spellCount;
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
	uint8_t spellCount;
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
	int8_t grip_x;
	int8_t grip_y;
	int8_t tip_x;
	int8_t tip_y;
	int8_t fire_rate_wait;
	int8_t actions_per_round;
	bool shuffle_deck_when_empty;
	int8_t deck_capacity;
	int8_t spread_degrees;
	int8_t reload_time;
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

struct PixelSceneData
{
	PixelScene scene;
	float prob;
	int materialCount;
	Material extraMaterials[11];
	bool hasExtraFunction;
	void (*extraFunction)(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount);

	__device__ constexpr PixelSceneData() : scene(PS_NONE), prob(0), materialCount(0), hasExtraFunction(false), extraMaterials(), extraFunction(NULL)
	{

	}

	__device__ PixelSceneData(PixelScene _scene, float _prob)
	{
		scene = _scene;
		prob = _prob;
		materialCount = 0;
		hasExtraFunction = false;
	}

	__device__ PixelSceneData(PixelScene _scene, float _prob, void (*_func)(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount))
	{
		scene = _scene;
		prob = _prob;
		materialCount = 0;
		hasExtraFunction = true;
		extraFunction = _func;
	}

	__device__ PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats)
	{
		scene = _scene;
		prob = _prob;
		materialCount = _mats.size();
		memcpy(extraMaterials, _mats.begin(), sizeof(Material) * materialCount);
		hasExtraFunction = false;
	}

	__device__ PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats, void (*_func)(int x, int y, uint32_t seed, PixelScene scene, MapConfig mCfg, SpawnableConfig sCfg, uint8_t* output, int& offset, int& sCount))
	{
		scene = _scene;
		prob = _prob;
		materialCount = _mats.size();
		memcpy(extraMaterials, _mats.begin(), sizeof(Material) * materialCount);
		hasExtraFunction = true;
		extraFunction = _func;
	}
};
struct PixelSceneList
{
	int count;
	float probSum;
	int extraHeightNeeded;
	PixelSceneData scenes[20];
};

struct EnemyData
{
	float prob;
	int minCount;
	int maxCount;
	Enemy enemy;
};
struct EnemyList
{
	int count;
	float probSum;
	EnemyData enemies[20];
};

__universal__
uint8_t readByte(uint8_t* ptr, int& offset)
{
	return ptr[offset++];
}

__universal__
void writeByte(uint8_t* ptr, int& offset, uint8_t b)
{
	ptr[offset++] = b;
}

__universal__
int readInt(uint8_t* ptr, int& offset)
{
	int tmp;
	memcpy(&tmp, ptr + offset, 4);
	offset += 4;
	return tmp;
}

__universal__
void writeInt(uint8_t* ptr, int& offset, int val)
{
	memcpy(ptr + offset, &val, 4);
	offset += 4;
}

__universal__
void incrInt(uint8_t* ptr)
{
	int offsetTmp = 0;
	int tmp = readInt(ptr, offsetTmp);
	offsetTmp = 0;
	writeInt(ptr, offsetTmp, tmp + 1);
}

__universal__
short readShort(uint8_t* ptr, int& offset)
{
	return (readByte(ptr, offset) | (readByte(ptr, offset) << 8));
}

__universal__
void writeShort(uint8_t* ptr, int& offset, short s)
{
	writeByte(ptr, offset, ((short)s) & 0xff);
	writeByte(ptr, offset, (((short)s) >> 8) & 0xff);
}

__device__ int readMisaligned(int* ptr2)
{
	uint8_t* ptr = (uint8_t*)ptr2;
	int offset = 0;
	return readInt(ptr, offset);
}

__universal__ Spawnable readMisalignedSpawnable(Spawnable* sPtr)
{
	uint8_t* bPtr = (uint8_t*)sPtr;
	Spawnable s;
	int offset = 0;
	s.x = readInt(bPtr, offset);
	s.y = readInt(bPtr, offset);
	s.sType = (SpawnableMetadata)readByte(bPtr, offset);
	s.count = readInt(bPtr, offset);
	return s;
}

__universal__ WandData readMisalignedWand(WandData* wPtr)
{
	WandData w = {};
	memcpy(&w, wPtr, 37);
	return w;
}
