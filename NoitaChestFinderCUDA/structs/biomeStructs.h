#pragma once
#include "../platforms/platform_compute_helpers.h"

#include "primitives.h"
#include "enums.h"
#include "spawnableStructs.h"

#include "../Configuration.h"

struct BiomeSector
{
	Biome b;

	int worldX;
	int worldY;
	int worldW;
	int worldH;
	uint32_t tiles_w;
	uint32_t tiles_h;
	uint32_t map_w;
	uint32_t map_h;
};

struct BiomeWangScope
{
	uint8_t* tileSet;
	BiomeSector bSec;
};

struct SpawnParams
{
	int seed;
	BiomeSector currentSector;
	SpawnableConfig sCfg;
	uint8_t* bytes;
	int& offset;
	int& sCount;

	void(*spawnSmallEnemies)(int x, int y, SpawnParams params);
	void(*spawnBigEnemies)(int x, int y, SpawnParams params);
	bool(*spawnItem)(int x, int y, SpawnParams params);
};

struct SpawnFunction
{
	uint32_t color;
	void(*func)(int, int, SpawnParams);

	_compute constexpr SpawnFunction() : color(0), func(NULL) {}
	_compute constexpr SpawnFunction(uint32_t _c, void(*_fn)(int, int, SpawnParams)) : color(_c), func(_fn) {}
};

struct BiomeSpawnFunctions
{
	int count;
	void(*setSharedFuncs)(SpawnParams& params);
	SpawnFunction funcs[10];

	_compute constexpr BiomeSpawnFunctions() : count(0), setSharedFuncs(NULL), funcs() {}
	_compute constexpr BiomeSpawnFunctions(int _c, void(*_fn)(SpawnParams& params), std::initializer_list<SpawnFunction> list) : count(_c), setSharedFuncs(_fn), funcs()
	{
		for (int i = 0; i < list.size(); i++) funcs[i] = list.begin()[i];
	}
};

_compute BiomeSpawnFunctions* AllSpawnFunctions[30];
_compute BiomeWands* AllWandLevels[30];