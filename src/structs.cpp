#include "../include/search_structs.h"
#include "../include/worldgen_structs.h"
#include "../platforms/platform_implementation.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

ItemFilter::ItemFilter() {
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	duplicates = 0;
}
ItemFilter::ItemFilter(std::initializer_list<Item> _items) {
	Assert(_items.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	memcpy(items, _items.begin(), sizeof(Item) * _items.size());
	duplicates = 1;
}
ItemFilter::ItemFilter(std::initializer_list<Item> _items, int _dupes) {
	Assert(_items.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	memcpy(items, _items.begin(), sizeof(Item) * _items.size());
	duplicates = _dupes;
}

MaterialFilter::MaterialFilter() {
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	duplicates = 0;
}
MaterialFilter::MaterialFilter(std::initializer_list<Material> _materials) {
	Assert(_materials.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = 1;
}
MaterialFilter::MaterialFilter(std::initializer_list<Material> _materials, int _dupes) {
	Assert(_materials.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = _dupes;
}

SpellFilter::SpellFilter() {
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	duplicates = 0;
	asAlwaysCast = false;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells) {
	Assert(_spells.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = 1;
	asAlwaysCast = false;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes) {
	Assert(_spells.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = false;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast) {
	Assert(_spells.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = _asAlwaysCast;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast, bool _consecutive) {
	Assert(_spells.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = _asAlwaysCast;
	perWand = _consecutive;
}

PixelSceneFilter::PixelSceneFilter() {
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	duplicates = 0;
	checkMats = false;
}
PixelSceneFilter::PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes) {
	Assert(_pixelScenes.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	duplicates = 1;
	checkMats = false;
}
PixelSceneFilter::PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, int _dupes) {
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	duplicates = _dupes;
	checkMats = false;
}
PixelSceneFilter::PixelSceneFilter(
	std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials) {
	Assert(_pixelScenes.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = 1;
	checkMats = true;
}
PixelSceneFilter::PixelSceneFilter(
	std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials, int _dupes) {
	Assert(_pixelScenes.size() <= FILTER_OR_COUNT, "Filter size overflow.");
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = _dupes;
	checkMats = true;
}

_universal bool AlchemyRecipe::Equals(AlchemyRecipe reference, AlchemyRecipe test) {
	if (reference.ordering == STRICT_ORDERED) {
		bool passed1 = reference.mats[0] == MATERIAL_NONE || reference.mats[0] == test.mats[0];
		bool passed2 = reference.mats[1] == MATERIAL_NONE || reference.mats[1] == test.mats[1];
		bool passed3 = reference.mats[2] == MATERIAL_NONE || reference.mats[2] == test.mats[2];

		return passed1 && passed2 && passed3;
	} else if (reference.ordering == ONLY_CONSUMED) {
		bool passed1 = reference.mats[0] == MATERIAL_NONE ||
					   (reference.mats[0] == test.mats[0] || reference.mats[0] == test.mats[2]);
		bool passed2 = reference.mats[1] == MATERIAL_NONE || (reference.mats[1] == test.mats[1]);
		bool passed3 = reference.mats[2] == MATERIAL_NONE ||
					   (reference.mats[2] == test.mats[0] || reference.mats[2] == test.mats[2]);

		return passed1 && passed2 && passed3;
	} else {
		bool passed1 = reference.mats[0] == MATERIAL_NONE ||
					   (reference.mats[0] == test.mats[0] || reference.mats[0] == test.mats[1] ||
						   reference.mats[0] == test.mats[2]);
		bool passed2 = reference.mats[1] == MATERIAL_NONE ||
					   (reference.mats[1] == test.mats[0] || reference.mats[1] == test.mats[1] ||
						   reference.mats[1] == test.mats[2]);
		bool passed3 = reference.mats[2] == MATERIAL_NONE ||
					   (reference.mats[2] == test.mats[0] || reference.mats[2] == test.mats[1] ||
						   reference.mats[2] == test.mats[2]);

		return passed1 && passed2 && passed3;
	}
}

_universal FungalShift::FungalShift()
	: from(SS_NONE), to(SD_NONE), fromFlask(false), toFlask(false), minIdx(0), maxIdx(0) {}
_universal FungalShift::FungalShift(ShiftSource _from, ShiftDest _to, int _minIdx, int _maxIdx) {
	if (_from == SS_FLASK) {
		from = SS_NONE;
		fromFlask = true;
	} else {
		from = _from;
		fromFlask = false;
	}
	if (_to == SD_FLASK) {
		to = SD_NONE;
		toFlask = true;
	} else {
		to = _to;
		toFlask = false;
	}
	minIdx = _minIdx;
	maxIdx = _maxIdx;
}
