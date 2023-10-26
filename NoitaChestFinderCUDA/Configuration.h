#pragma once
#include "platforms/platform_compute_helpers.h"

#include "structs/staticPrecheckStructs.h"
#include "structs/spawnableStructs.h"
#include "structs/filterStructs.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

struct GeneralConfig
{
	uint32_t seedStart;
	uint32_t endSeed;
	uint32_t seedBlockSize;
	bool seedBlockOverride;
};
struct MemSizeConfig
{
	size_t memoryCap;

	size_t outputSize;
	size_t mapDataSize;
	size_t miscMemSize;
	size_t visitedMemSize;
	size_t spawnableMemSize;

	size_t threadMemTotal;
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

constexpr int maxFungalShifts = 12;
struct FungalShiftConfig
{
	bool check;
	FungalShift shifts[maxFungalShifts];
};

constexpr int maxPerkFilters = 12;
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
	bool biomePixelScenes;
	bool biomeEnemies;
	bool hellShops;

	bool genPotions;
	bool genSpells;
	bool genWands;
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

	FilterConfig() = default;
};

struct OutputConfig
{
	float printInterval;
	bool printProgressLog;
	bool printOutputToConsole;
};