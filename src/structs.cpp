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
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	memcpy(items, _items.begin(), sizeof(Item) * _items.size());
	duplicates = 1;
}
ItemFilter::ItemFilter(std::initializer_list<Item> _items, int _dupes) {
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	memcpy(items, _items.begin(), sizeof(Item) * _items.size());
	duplicates = _dupes;
}

MaterialFilter::MaterialFilter() {
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	duplicates = 0;
}
MaterialFilter::MaterialFilter(std::initializer_list<Material> _materials) {
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = 1;
}
MaterialFilter::MaterialFilter(std::initializer_list<Material> _materials, int _dupes) {
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
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = 1;
	asAlwaysCast = false;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes) {
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = false;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast) {
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = _asAlwaysCast;
	perWand = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast, bool _consecutive) {
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
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = 1;
	checkMats = true;
}
PixelSceneFilter::PixelSceneFilter(
	std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials, int _dupes) {
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = _dupes;
	checkMats = true;
}

_universal AlchemyRecipe::AlchemyRecipe() {}
_universal AlchemyRecipe::AlchemyRecipe(Material mat1, Material mat2, Material mat3) {
	mats[0] = mat1;
	mats[1] = mat2;
	mats[2] = mat3;
}
_universal bool AlchemyRecipe::Equals(AlchemyRecipe reference, AlchemyRecipe test, AlchemyOrdering ordered) {
	if (ordered == STRICT_ORDERED) {
		bool passed1 = reference.mats[0] == MATERIAL_NONE || reference.mats[0] == test.mats[0];
		bool passed2 = reference.mats[1] == MATERIAL_NONE || reference.mats[1] == test.mats[1];
		bool passed3 = reference.mats[2] == MATERIAL_NONE || reference.mats[2] == test.mats[2];

		return passed1 && passed2 && passed3;
	} else if (ordered == ONLY_CONSUMED) {
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

// Worldgen structs
constexpr BiomeSpawnColors::BiomeSpawnColors(std::initializer_list<uint32_t> list) : count(list.size()), colors() {
	for (int i = 0; i < list.size(); i++)
		colors[i] = list.begin()[i];
}

_compute consteval BiomeSpawnFunctions::BiomeSpawnFunctions(
	void (*_fn)(SpawnParams& params), std::initializer_list<void (*)(int, int, const SpawnParams&)> list)
	: count(list.size()), init(_fn), funcs() {
	for (int i = 0; i < list.size(); i++)
		funcs[i] = list.begin()[i];
}
_universal constexpr PixelSceneSpawn::PixelSceneSpawn(int _t, short _x, short _y) : i(_t), x(_x), y(_y) {}

_universal constexpr PixelSceneData::PixelSceneData(PixelScene _scene, float _prob, const char* _path)
	: scene(_scene), prob(_prob), path(_path), materialCount(0), materials(), spawnCount(0), spawns() {}
_universal constexpr PixelSceneData::PixelSceneData(
	PixelScene _scene, float _prob, const char* _path, std::initializer_list<Material> _mats)
	: scene(_scene), prob(_prob), path(_path), materialCount(_mats.size()), materials(), spawnCount(0), spawns() {
	for (int i = 0; i < materialCount; i++)
		materials[i] = _mats.begin()[i];
}

_universal constexpr PixelSceneList::PixelSceneList(std::initializer_list<PixelSceneData> list)
	: count(list.size()), probSum(), scenes() {
	for (int i = 0; i < list.size(); i++) {
		probSum += list.begin()[i].prob;
		scenes[i] = list.begin()[i];
	}
}

_universal constexpr BiomePixelScenes::BiomePixelScenes(std::initializer_list<PixelSceneList> list)
	: count(list.size()), lists(/*list.size() ? (PixelSceneList*)malloc(sizeof(PixelSceneList) * list.size()) : 0*/) {
	for (int i = 0; i < list.size(); i++) {
		lists[i] = list.begin()[i];
	}
}
