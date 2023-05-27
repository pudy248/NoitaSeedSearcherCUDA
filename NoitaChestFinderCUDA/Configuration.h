#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"
#include "structs/enums.h"
#include "structs/staticPrecheckStructs.h"

struct GeneralConfig
{
	size_t requestedMemory;
	uint startSeed;
	uint endSeed;
	int seedBlockSize;
	int printInterval;
	int atomicGranularity;
	int passedGranularity;
};
struct MemSizeConfig
{
	size_t outputSize;
	size_t mapDataSize;
	size_t miscMemSize;
	size_t visitedMemSize;
	size_t bufferSize;
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
	byte perksPerMountain[7];
};

struct StaticPrecheckConfig
{
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
	uint tiles_w;
	uint tiles_h;
	uint map_w;
	uint map_h;
	int worldX;
	int worldY;

	int minX;
	int maxX;
	int minY;
	int maxY;

	bool isCoalMine;
	bool isNightmare;
	int biomeIdx;

	int maxTries;
};

struct SpawnableConfig
{
	int pwCenter;
	int pwWidth;
	int minHMidx;
	int maxHMidx;
	bool greedCurse;

	bool pacifist;
	bool shopSpells;
	bool shopWands;

	bool biomeChests;
	bool biomePedestals;
	bool biomeAltars;
	bool pixelScenes;

	bool genPotions;
	bool genSpells;
	bool genWands;
};

#define FILTER_OR_COUNT 10
#define TOTAL_FILTER_COUNT 10
struct ItemFilter
{
	Item items[FILTER_OR_COUNT];
	int duplicates;

	ItemFilter()
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) items[i] = ITEM_NONE;
		duplicates = 0;
	}

	ItemFilter(Item _items[FILTER_OR_COUNT])
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) items[i] = _items[i];
		duplicates = 1;
	}

	ItemFilter(Item _items[FILTER_OR_COUNT], int _dupes)
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) items[i] = _items[i];
		duplicates = _dupes;
	}
};
struct MaterialFilter
{
	Material materials[FILTER_OR_COUNT];
	int duplicates;

	MaterialFilter()
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) materials[i] = MATERIAL_NONE;
		duplicates = 0;
	}

	MaterialFilter(Material _materials[FILTER_OR_COUNT])
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) materials[i] = _materials[i];
		duplicates = 1;
	}

	MaterialFilter(Material _materials[FILTER_OR_COUNT], int _dupes)
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) materials[i] = _materials[i];
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
		for (int i = 0; i < FILTER_OR_COUNT; i++) spells[i] = SPELL_NONE;
		duplicates = 0;
		asAlwaysCast = false;
	}

	SpellFilter(Spell _spells[FILTER_OR_COUNT])
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) spells[i] = _spells[i];
		duplicates = 1;
		asAlwaysCast = false;
	}

	SpellFilter(Spell _spells[FILTER_OR_COUNT], int _dupes)
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) spells[i] = _spells[i];
		duplicates = _dupes;
		asAlwaysCast = false;
	}

	SpellFilter(Spell _spells[FILTER_OR_COUNT], int _dupes, bool _asAlwaysCast)
	{
		for (int i = 0; i < FILTER_OR_COUNT; i++) spells[i] = _spells[i];
		duplicates = _dupes;
		asAlwaysCast = _asAlwaysCast;
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
	bool checkBigWands;
	int howBig;
	FilterConfig(bool _aggregate, int _itemFilterCount, ItemFilter* _itemFilters, int _materialFilterCount, MaterialFilter* _materialFilters, int _spellFilterCount, SpellFilter* _spellFilters, bool _checkBigWands, int _howBig)
	{
		aggregate = _aggregate;
		itemFilterCount = _itemFilterCount;
		materialFilterCount = _materialFilterCount;
		spellFilterCount = _spellFilterCount;
		checkBigWands = _checkBigWands;
		howBig = _howBig;
		memcpy(itemFilters, _itemFilters, sizeof(ItemFilter) * itemFilterCount);
		memcpy(materialFilters, _materialFilters, sizeof(MaterialFilter) * materialFilterCount);
		memcpy(spellFilters, _spellFilters, sizeof(SpellFilter) * spellFilterCount);
	}
};