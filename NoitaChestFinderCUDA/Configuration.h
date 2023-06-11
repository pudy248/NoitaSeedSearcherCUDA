#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"
#include "structs/enums.h"
#include "structs/staticPrecheckStructs.h"

#include <initializer_list>

struct GeneralConfig
{
	uint64_t requestedMemory;
	uint32_t startSeed;
	uint32_t endSeed;
	int seedBlockSize;
	int printInterval;
	int atomicGranularity;
	int passedGranularity;
};
struct MemSizeConfig
{
	uint64_t outputSize;
	uint64_t mapDataSize;
	uint64_t miscMemSize;
	uint64_t visitedMemSize;
	uint64_t spawnableMemSize;

	uint64_t threadMemTotal;
};

struct StartingCartConfig
{
	bool check;
	CartType cart;
};
struct StartingFlaskConfig
{
	bool check;
	Material flask;
};
struct StartingWandConfig
{
	bool check;
	Spell projectile;
	Spell bomb;
};
struct RainConfig
{
	bool check;
	Material rain;
};
struct AlchemyConfig
{
	bool check;
	AlchemyOrdering ordering;
	AlchemyRecipe LC;
	AlchemyRecipe AP;
};
struct BiomeModifierConfig
{
	bool check;
	BiomeModifier modifiers[9];
};

constexpr int maxFungalShifts = 4;
struct FungalShiftConfig
{
	bool check;
	FungalShift shifts[maxFungalShifts];
};

constexpr int maxPerkFilters = 24;
struct PerkConfig
{
	bool check;
	PerkInfo perks[maxPerkFilters];
	uint8_t perksPerMountain[7];
};

struct StaticPrecheckConfig
{
	StartingCartConfig cart;
	StartingFlaskConfig flask;
	StartingWandConfig wands;
	RainConfig rain;
	AlchemyConfig alchemy;
	BiomeModifierConfig biomes;
	FungalShiftConfig fungal;
	PerkConfig perks;
	bool precheckUpwarps;
};

struct MapConfig
{
	//Initialized manually
	int worldX;
	int worldY;
	int worldW;
	int worldH;

	int biomeIdx;

	//Auto-initialized
	int minX;
	int maxX;
	int minY;
	int maxY;

	uint32_t tiles_w;
	uint32_t tiles_h;
	uint32_t map_w;
	uint32_t map_h;

	//bool isNightmare;
	bool isCoalMine;
	int maxTries;
};

struct SpawnableConfig
{
	Vec2i pwCenter;
	Vec2i pwWidth;
	int minHMidx;
	int maxHMidx;
	bool greedCurse;

	bool pacifist;
	bool shopSpells;
	bool shopWands;

	bool eyeRooms;
	bool staticUpwarps;

	bool biomeChests;
	bool biomePedestals;
	bool biomeAltars;
	bool pixelScenes;
	bool enemies;
	bool hellShops;

	bool genPotions;
	bool genSpells;
	bool genWands;
};

#define FILTER_OR_COUNT 5
#define TOTAL_FILTER_COUNT 10
struct ItemFilter
{
	Item items[FILTER_OR_COUNT];
	int duplicates;

	ItemFilter()
	{
		memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
		duplicates = 0;
	}

	ItemFilter(std::initializer_list<Item> _items)
	{
		memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
		memcpy(items, _items.begin(), sizeof(Item) * _items.size());
		duplicates = 1;
	}

	ItemFilter(std::initializer_list<Item> _items, int _dupes)
	{
		memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
		memcpy(items, _items.begin(), sizeof(Item) * _items.size());
		duplicates = _dupes;
	}
};
struct MaterialFilter
{
	Material materials[FILTER_OR_COUNT];
	int duplicates;

	MaterialFilter()
	{
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		duplicates = 0;
	}

	MaterialFilter(std::initializer_list<Material> _materials)
	{
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
		duplicates = 1;
	}

	MaterialFilter(std::initializer_list<Material> _materials, int _dupes)
	{
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
		duplicates = _dupes;
	}
};
struct SpellFilter
{
	Spell spells[FILTER_OR_COUNT];
	int duplicates;
	bool asAlwaysCast;

	SpellFilter()
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		duplicates = 0;
		asAlwaysCast = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = 1;
		asAlwaysCast = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells, int _dupes)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = _dupes;
		asAlwaysCast = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = _dupes;
		asAlwaysCast = _asAlwaysCast;
	}
};
struct PixelSceneFilter
{
	PixelScene pixelScenes[FILTER_OR_COUNT];
	Material materials[FILTER_OR_COUNT];
	int duplicates;
	bool checkMats;

	PixelSceneFilter()
	{
		memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		duplicates = 0;
		checkMats = false;
	}

	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes)
	{
		memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
		duplicates = 1;
		checkMats = false;
	}

	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, int _dupes)
	{
		memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
		duplicates = _dupes;
		checkMats = false;
	}

	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials)
	{
		memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
		memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
		duplicates = 1;
		checkMats = true;
	}

	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials, int _dupes)
	{
		memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
		memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
		memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
		memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
		duplicates = _dupes;
		checkMats = true;
	}
};

struct FilterConfig
{
	bool aggregate;
	int itemFilterCount;
	ItemFilter itemFilters[TOTAL_FILTER_COUNT];
	int materialFilterCount;
	MaterialFilter materialFilters[TOTAL_FILTER_COUNT];
	int spellFilterCount;
	SpellFilter spellFilters[TOTAL_FILTER_COUNT];
	int pixelSceneFilterCount;
	PixelSceneFilter pixelSceneFilters[TOTAL_FILTER_COUNT];
	bool checkBigWands;
	int howBig;
	FilterConfig(bool _aggregate, int _itemFilterCount, ItemFilter* _itemFilters, int _materialFilterCount, MaterialFilter* _materialFilters, int _spellFilterCount, SpellFilter* _spellFilters, int _pixelSceneFilterCount, PixelSceneFilter* _pixelSceneFilters, bool _checkBigWands, int _howBig)
	{
		aggregate = _aggregate;
		itemFilterCount = _itemFilterCount;
		materialFilterCount = _materialFilterCount;
		spellFilterCount = _spellFilterCount;
		pixelSceneFilterCount = _pixelSceneFilterCount;
		checkBigWands = _checkBigWands;
		howBig = _howBig;
		memcpy(itemFilters, _itemFilters, sizeof(ItemFilter) * itemFilterCount);
		memcpy(materialFilters, _materialFilters, sizeof(MaterialFilter) * materialFilterCount);
		memcpy(spellFilters, _spellFilters, sizeof(SpellFilter) * spellFilterCount);
		memcpy(pixelSceneFilters, _pixelSceneFilters, sizeof(PixelSceneFilter) * pixelSceneFilterCount);
	}
};