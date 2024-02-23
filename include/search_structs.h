#pragma once
#include "../platforms/platform_implementation.h"
#include "enums.h"
#include <cstdint>
#include <initializer_list>

struct AlchemyRecipe
{
	Material mats[4] = { MATERIAL_NONE, MATERIAL_NONE, MATERIAL_NONE, MATERIAL_NONE };

	_universal AlchemyRecipe();
	_universal AlchemyRecipe(Material mat1, Material mat2, Material mat3);
	_universal bool Equals(AlchemyRecipe reference, AlchemyRecipe test, AlchemyOrdering ordered);
};

struct FungalShift
{
	ShiftSource from;
	ShiftDest to;
	bool fromFlask;
	bool toFlask;
	int minIdx;
	int maxIdx;
	_universal constexpr FungalShift();
	_universal FungalShift(ShiftSource _from, ShiftDest _to, int _minIdx, int _maxIdx);
};

struct PerkData
{
	bool stackable;
	bool stackable_rare;
	uint8_t stackable_max;
	uint8_t max_in_pool;
	bool not_default;
	uint8_t stackable_how_often_reappears;
};

struct PerkInfo
{
	Perk p;
	bool lottery;
	int minPosition;
	int maxPosition;
};

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
	_universal constexpr PixelSceneSpawn();
	_universal constexpr PixelSceneSpawn(PixelSceneSpawnType _t, short _x, short _y);
};
struct PixelSceneData
{
	PixelScene scene;
	float prob;
	short materialCount;
	short spawnCount;
	Material materials[11];
	PixelSceneSpawn spawns[10];

	_universal constexpr PixelSceneData();
	_universal constexpr PixelSceneData(PixelScene _scene, float _prob);
	_universal constexpr PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats);
	_universal constexpr PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<PixelSceneSpawn> _spawns);
	_universal constexpr PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats, std::initializer_list<PixelSceneSpawn> _spawns);
};
struct PixelSceneList
{
	int count;
	float probSum;
	PixelSceneData scenes[20];
	_universal constexpr PixelSceneList();
	_universal constexpr PixelSceneList(int _c, std::initializer_list<PixelSceneData> list);
};

/*
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
*/

#define FILTER_OR_COUNT 5
#define TOTAL_FILTER_COUNT 10
struct ItemFilter
{
	Item items[FILTER_OR_COUNT];
	int duplicates;

	ItemFilter();
	ItemFilter(std::initializer_list<Item> _items);
	ItemFilter(std::initializer_list<Item> _items, int _dupes);
};
struct MaterialFilter
{
	Material materials[FILTER_OR_COUNT];
	int duplicates;

	MaterialFilter();
	MaterialFilter(std::initializer_list<Material> _items);
	MaterialFilter(std::initializer_list<Material> _items, int _dupes);
};
struct SpellFilter
{
	Spell spells[FILTER_OR_COUNT];
	int duplicates;
	bool asAlwaysCast;
	bool consecutive;

	SpellFilter();
	SpellFilter(std::initializer_list<Spell> _spells);
	SpellFilter(std::initializer_list<Spell> _spells, int _dupes);
	SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast);
	SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast, bool _consecutive);
};
struct PixelSceneFilter
{
	PixelScene pixelScenes[FILTER_OR_COUNT];
	Material materials[FILTER_OR_COUNT];
	int duplicates;
	bool checkMats;

	PixelSceneFilter();
	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes);
	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, int _dupes);
	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials);
	PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials, int _dupes);
};
