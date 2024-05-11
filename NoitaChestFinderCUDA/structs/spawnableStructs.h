#pragma once
#include "../platforms/platform_compute_helpers.h"

#include "primitives.h"
#include "enums.h"

#include <cstring>
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
	int seed;
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
	const char* name;
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

struct WandLevel
{
	float prob;
	Item id;
	_universal constexpr WandLevel() : prob(), id() {}
	_universal constexpr WandLevel(float _p, Item _w) : prob(_p), id(_w) {}
};
struct BiomeWands
{
	int count;
	WandLevel levels[6];

	_universal constexpr BiomeWands() : count(), levels() {}
	_universal constexpr BiomeWands(int _c, std::initializer_list<WandLevel> list) : count(_c), levels() {
		for (int i = 0; i < list.size(); i++) levels[i] = list.begin()[i];
	}
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

struct PixelSceneSpawn
{
	PixelSceneSpawnType spawnType;
	short x;
	short y;
	_universal constexpr PixelSceneSpawn() : spawnType(), x(), y() {}
	_universal constexpr PixelSceneSpawn(PixelSceneSpawnType _t, short _x, short _y) : spawnType(_t), x(_x), y(_y) {}
};
struct PixelSceneData
{
	PixelScene scene;
	float prob;
	short materialCount;
	short spawnCount;
	Material materials[11];
	PixelSceneSpawn spawns[10];

	_universal constexpr PixelSceneData()
		: scene(PS_NONE), prob(0), materialCount(0), spawnCount(0), materials(), spawns() {}

	_universal constexpr PixelSceneData(PixelScene _scene, float _prob)
		: scene(_scene), prob(_prob), materialCount(0), spawnCount(0), materials(), spawns() {}

	_universal constexpr PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats)
		: scene(_scene), prob(_prob), materialCount(), spawnCount(0), materials(), spawns()
	{
		scene = _scene;
		prob = _prob;
		materialCount = _mats.size();
		for (int i = 0; i < materialCount; i++) materials[i] = _mats.begin()[i];
	}

	_universal constexpr PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<PixelSceneSpawn> _spawns)
		: scene(_scene), prob(_prob), materialCount(0), spawnCount(), materials(), spawns()
	{
		scene = _scene;
		prob = _prob;
		spawnCount = _spawns.size();
		for (int i = 0; i < spawnCount; i++) spawns[i] = _spawns.begin()[i];
	}

	_universal constexpr PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats, std::initializer_list<PixelSceneSpawn> _spawns)
		: scene(_scene), prob(_prob), materialCount(), spawnCount(), materials(), spawns()
	{
		scene = _scene;
		prob = _prob;
		materialCount = _mats.size();
		spawnCount = _spawns.size();
		for (int i = 0; i < materialCount; i++) materials[i] = _mats.begin()[i];
		for (int i = 0; i < spawnCount; i++) spawns[i] = _spawns.begin()[i];
	}
};
struct PixelSceneList
{
	int count;
	float probSum;
	PixelSceneData scenes[20];
	_universal constexpr PixelSceneList() : count(), probSum(), scenes() {}
	_universal constexpr PixelSceneList(int _c, std::initializer_list<PixelSceneData> list) : count(_c), probSum(), scenes()
	{
		float pSum = 0;
		for (int i = 0; i < list.size(); i++)
		{
			pSum += list.begin()[i].prob;
			scenes[i] = list.begin()[i];
		}
		probSum = pSum;
	}
};

struct EnemyData
{
	float prob;
	int minCount;
	int maxCount;
	Enemy enemy;
	_universal constexpr EnemyData() : prob(), minCount(), maxCount(), enemy() {}
	_universal constexpr EnemyData(float _p, int _min, int _max, Enemy _e) : prob(_p), minCount(_min), maxCount(_max), enemy(_e) {}
};
struct EnemyList
{
	int count;
	float probSum;
	EnemyData enemies[20];
	_universal constexpr EnemyList() : count(), probSum(), enemies() {}
	_universal constexpr EnemyList(int _c, std::initializer_list<EnemyData> list) : count(_c), probSum(), enemies()
	{
		float pSum = 0;
		for (int i = 0; i < list.size(); i++)
		{
			pSum += list.begin()[i].prob;
			enemies[i] = list.begin()[i];
		}
		probSum = pSum;
	}
};

_universal uint8_t readByte(uint8_t* ptr, int& offset)
{
	return ptr[offset++];
}

_universal void writeByte(uint8_t* ptr, int& offset, uint8_t b)
{
	ptr[offset++] = b;
}

_universal int readInt(uint8_t* ptr, int& offset)
{
	ptr += offset;
	offset += 4;
	return (ptr[3] << 24) | (ptr[2] << 16) | (ptr[1] << 8) | (ptr[0]);
}

_universal void writeInt(uint8_t* ptr, int& offset, int val)
{
	ptr += offset;
	ptr[0] = val;
	ptr[1] = val >> 8;
	ptr[2] = val >> 16;
	ptr[3] = val >> 24;
	offset += 4;
}

_universal void incrInt(uint8_t* ptr)
{
	int offsetTmp = 0;
	int tmp = readInt(ptr, offsetTmp);
	offsetTmp = 0;
	writeInt(ptr, offsetTmp, tmp + 1);
}

_universal short readShort(uint8_t* ptr, int& offset)
{
	return (readByte(ptr, offset) | (readByte(ptr, offset) << 8));
}

_universal void writeShort(uint8_t* ptr, int& offset, short s)
{
	writeByte(ptr, offset, ((short)s) & 0xff);
	writeByte(ptr, offset, (((short)s) >> 8) & 0xff);
}

_universal int readMisaligned(int* ptr2)
{
	uint8_t* ptr = (uint8_t*)ptr2;
	int offset = 0;
	return readInt(ptr, offset);
}

_universal Spawnable readMisalignedSpawnable(Spawnable* sPtr)
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

_universal WandData readMisalignedWand(WandData* wPtr)
{
	WandData w = {};
	memcpy(&w, wPtr, 37);
	return w;
}