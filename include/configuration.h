#pragma once
#include "primitives.h"
#include "search_structs.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

struct GeneralConfig
{
	uint32_t seedStart;
	uint32_t seedEnd;
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
	bool upwarp;
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
};

struct OutputConfig
{
	float printInterval;
	bool printProgressLog;
	bool printOutputToConsole;
};