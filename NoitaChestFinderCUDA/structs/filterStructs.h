#pragma once

#include "../platforms/platform_compute_helpers.h"

#include "staticPrecheckStructs.h"
#include "spawnableStructs.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

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
	bool consecutive;

	SpellFilter()
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		duplicates = 0;
		asAlwaysCast = false;
		consecutive = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = 1;
		asAlwaysCast = false;
		consecutive = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells, int _dupes)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = _dupes;
		asAlwaysCast = false;
		consecutive = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = _dupes;
		asAlwaysCast = _asAlwaysCast;
		consecutive = false;
	}

	SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast, bool _consecutive)
	{
		memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
		memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
		duplicates = _dupes;
		asAlwaysCast = _asAlwaysCast;
		consecutive = _consecutive;
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
