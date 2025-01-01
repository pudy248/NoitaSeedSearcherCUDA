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
constexpr int _WangTileMaxSpawns = 4;
struct WangTile
{
	bool should_block;
	char colors[6];
	WangSpawn spawns[_WangTileMaxSpawns];
};

struct WangTileset
{
	char is_corner;
	int num_vary[2];
	int num_color[6];
	int short_side_len;
	int widthH, heightH, widthV, heightV;
	WangTile hTiles[72];
	WangTile vTiles[72];
	uint16_t hIndices[64];
	uint16_t vIndices[64];
	uint8_t* tileData;
	uint32_t tdStride;
	_universal uint32_t h_tile_at(int tx, int ty, int xoff, int yoff) const;
	_universal uint32_t v_tile_at(int tx, int ty, int xoff, int yoff) const;
};

struct MainPathFill {
	bool active;
	int x1, x2;
};

struct BiomeWangScope
{
	WangTileset ts;
	BiomeSector bSec;
};

struct GeneratedBiome
{
	const BiomeWangScope& scope;
	WangTileIndex* indices;
	WangFuncIndex* funcs;
};

struct SpawnParams
{
	int seed;
	const BiomeWangScope& currentBiome;
	const SpawnableConfig& sCfg;
	MemSpan bytes;
	int& offset;
	int& sCount;

	void(*spawnSmallEnemies)(int x, int y, const SpawnParams& params);
	void(*spawnBigEnemies)(int x, int y, const SpawnParams& params);
	bool(*spawnItem)(int x, int y, const SpawnParams& params);
};

struct SpawnFunction
{
	uint32_t color;
	void(*func)(int, int, const SpawnParams&);

	_compute constexpr SpawnFunction() : color(0), func(NULL) {}
	_compute constexpr SpawnFunction(uint32_t _c, void(*_fn)(int, int, const SpawnParams&)) : color(_c), func(_fn) {}
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
