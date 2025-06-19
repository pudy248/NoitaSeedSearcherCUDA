#pragma once
#include "../platforms/platform_implementation.h"
#include "enums.h"
#include "primitives.h"
#include "search_structs.h"

struct BiomeMap {
	int w, h;
	Biome* map;
};
struct BiomeSector {
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

struct WangSpawn {
	uint8_t x;
	uint8_t y;
	WangFuncIndex i;
};
constexpr int _WangTileMaxSpawns = 6;
struct WangTile {
	bool should_block;
	char colors[6];
	WangSpawn spawns[_WangTileMaxSpawns];
};

struct WangTileset {
	char is_corner;
	int num_vary[2];
	int num_color[6];
	int max_colors;
	int short_side_len;
	int widthH, heightH, widthV, heightV;
	WangTile hTiles[72];
	WangTile vTiles[72];
	uint16_t hIndices[729];
	uint16_t vIndices[729];
	uint8_t* tileData;
	uint32_t tdStride;
	_universal uint32_t h_tile_at(int tx, int ty, int xoff, int yoff) const;
	_universal uint32_t v_tile_at(int tx, int ty, int xoff, int yoff) const;
};

struct MainPathFill {
	bool active;
	int x1, x2;
};

struct BiomeWangScope {
	BiomeMap map;
	WangTileset ts;
	BiomeSector bSec;
};

struct GeneratedBiome {
	const BiomeWangScope& scope;
	WangTileIndex* indices;
	WangFuncIndex* funcs;
};

struct SpawnParams {
	int seed;
	const BiomeWangScope& currentBiome;
	const SpawnableConfig& sCfg;
	MemSpan bytes;
	int& offset;
	int& sCount;
};

struct BiomeSpawnColors {
	int count;
	uint32_t colors[12];

	constexpr BiomeSpawnColors() = default;
	constexpr BiomeSpawnColors(std::initializer_list<uint32_t> list) : count(list.size()), colors() {
		Assert(list.size() <= 12, "Spawn function size overflow.");
		for (int i = 0; i < list.size(); i++)
			colors[i] = list.begin()[i];
	}
};

struct BiomeSpawnFunctions {
	int count;
	void (*init)(SpawnParams& params);
	void (*funcs[12])(int, int, const SpawnParams&);

	constexpr BiomeSpawnFunctions() = default;
	_compute constexpr BiomeSpawnFunctions(
		void (*_fn)(SpawnParams& params), std::initializer_list<void (*)(int, int, const SpawnParams&)> list)
		: count(list.size()), init(_fn), funcs() {
		Assert(list.size() <= 12, "Spawn function size overflow.");
		for (int i = 0; i < list.size(); i++)
			funcs[i] = list.begin()[i];
	}
};

struct PixelSceneSpawn {
	int i;
	short x;
	short y;
	constexpr PixelSceneSpawn() = default;
	_universal constexpr PixelSceneSpawn(int _t, short _x, short _y) : i(_t), x(_x), y(_y) {}
};
struct PixelSceneData {
	PixelScene scene;
	float prob;
	short materialCount;
	Material materials[16];
	short spawnCount;
	PixelSceneSpawn spawns[8];

	constexpr PixelSceneData() = default;
	_universal constexpr PixelSceneData(PixelScene _scene, float _prob)
		: scene(_scene), prob(_prob), materialCount(0), materials(), spawnCount(0), spawns() {}
	_universal constexpr PixelSceneData(
		PixelScene _scene, float _prob, std::initializer_list<Material> _mats)
		: scene(_scene), prob(_prob), materialCount(_mats.size()), materials(), spawnCount(0), spawns() {
		for (int i = 0; i < materialCount; i++)
			materials[i] = _mats.begin()[i];
	}
};
struct PixelSceneList {
	int count;
	float probSum;
	PixelSceneData scenes[20];
	constexpr PixelSceneList() = default;
	_universal constexpr PixelSceneList(std::initializer_list<PixelSceneData> list)
		: count(list.size()), probSum(), scenes() {
		Assert(list.size() <= 20, "Pixel scene list size overflow.");
		for (int i = 0; i < list.size(); i++) {
			probSum += list.begin()[i].prob;
			scenes[i] = list.begin()[i];
		}
	}
};
struct BiomePixelScenes {
	int count;
	PixelSceneList lists[10];
	constexpr BiomePixelScenes() = default;
	_universal constexpr BiomePixelScenes(std::initializer_list<PixelSceneList> list)
		: count(list.size())
		, lists() {
		Assert(list.size() <= 10, "Biome pixel scenes size overflow.");
		for (int i = 0; i < list.size(); i++) {
			lists[i] = list.begin()[i];
		}
	}
};

BiomeSpawnColors HostSpawnColors[B_BIOME_COUNT] = {};
BiomePixelScenes HostPixelSceneLists[B_LIQUIDCAVE + 1] = {};

_data BiomeWands AllWandLevels[B_BIOME_COUNT];
_data BiomeSpawnFunctions AllSpawnFunctions[B_BIOME_COUNT];
_data BiomePixelScenes AllPixelSceneLists[B_LIQUIDCAVE + 1];

_data constexpr int SpellCount = SPELL_CESSATION;
_data SpellTables spellTables = {};

constexpr uint32_t COLOR_PURPLE = 0x7f007fU;
constexpr uint32_t COLOR_BLACK = 0x000000U;
constexpr uint32_t COLOR_WHITE = 0xffffffU;
constexpr uint32_t COLOR_YELLOW = 0xffff00U;
constexpr uint32_t COLOR_COFFEE = 0xc0ffeeU;
constexpr uint32_t COLOR_HELL_GREEN = 0x8aff80U;

// No longer RGB! Stored as a palette. 1: B=0x42, 2: G=0x42, 3: G>0x10, 0 otherwise
_compute uint8_t* coalmine_overlay;
