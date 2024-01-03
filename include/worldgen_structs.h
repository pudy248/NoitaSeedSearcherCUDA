#pragma once
#include "../platforms/platform_implementation.h"
#include "primitives.h"
#include "enums.h"
#include "search_structs.h"


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
	uint8_t wang_w;
	uint8_t wang_h;
};

typedef int16_t WangTileIndex;
typedef int16_t WangFuncIndex;

struct WangSpawn
{
	uint8_t x;
	uint8_t y;
	WangFuncIndex i;
};
constexpr int _WangTileMaxSpawns = 6;
struct WangTile
{
	char s[6];
	WangSpawn spawns[_WangTileMaxSpawns];
};

struct WangTileset
{
	char is_corner;
	int num_color[6];
	int short_side_len;
	int numH, maxH, numV, maxV;
	WangTile hTiles[72];
	WangTile vTiles[72];
	uint8_t hIndices[72 * 2];
	uint8_t vIndices[72 * 2];
};

struct WangConfig
{
	char is_corner;
	int short_side_len; // rectangles is 2n x n, n = short_side_len
	int num_color[6];   // see below diagram for meaning of the index to this;
	int num_vary_x;     // additional number of variations along x axis in the template
	int num_vary_y;     // additional number of variations along y axis in the template
	int corner_type_color_template[4][4];
};

struct BiomeWangScope
{
	WangTileset* tileSet;
	BiomeSector bSec;
};

struct SpawnParams
{
	int seed;
	BiomeWangScope* currentBiome;
	SpawnableConfig* sCfg;
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
	_compute constexpr BiomeSpawnFunctions(void(*_fn)(SpawnParams& params), std::initializer_list<SpawnFunction> list) : count(list.size()), setSharedFuncs(_fn), funcs()
	{
		for (int i = 0; i < list.size(); i++) funcs[i] = list.begin()[i];
	}
};

_data BiomeSpawnFunctions* AllSpawnFunctions[30];
_data BiomeWands* AllWandLevels[30];
