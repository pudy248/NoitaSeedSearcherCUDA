#include "../platforms/platform_implementation.h"
#include "../include/search_structs.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

ItemFilter::ItemFilter()
{
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	duplicates = 0;
}
ItemFilter::ItemFilter(std::initializer_list<Item> _items)
{
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	memcpy(items, _items.begin(), sizeof(Item) * _items.size());
	duplicates = 1;
}
ItemFilter::ItemFilter(std::initializer_list<Item> _items, int _dupes)
{
	memset(items, 0, sizeof(Item) * FILTER_OR_COUNT);
	memcpy(items, _items.begin(), sizeof(Item) * _items.size());
	duplicates = _dupes;
}


MaterialFilter::MaterialFilter()
{
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	duplicates = 0;
}
MaterialFilter::MaterialFilter(std::initializer_list<Material> _materials)
{
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = 1;
}
MaterialFilter::MaterialFilter(std::initializer_list<Material> _materials, int _dupes)
{
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = _dupes;
}


SpellFilter::SpellFilter()
{
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	duplicates = 0;
	asAlwaysCast = false;
	consecutive = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells)
{
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = 1;
	asAlwaysCast = false;
	consecutive = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes)
{
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = false;
	consecutive = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast)
{
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = _asAlwaysCast;
	consecutive = false;
}
SpellFilter::SpellFilter(std::initializer_list<Spell> _spells, int _dupes, bool _asAlwaysCast, bool _consecutive)
{
	memset(spells, 0, sizeof(Spell) * FILTER_OR_COUNT);
	memcpy(spells, _spells.begin(), sizeof(Spell) * _spells.size());
	duplicates = _dupes;
	asAlwaysCast = _asAlwaysCast;
	consecutive = _consecutive;
}


PixelSceneFilter::PixelSceneFilter()
{
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	duplicates = 0;
	checkMats = false;
}
PixelSceneFilter::PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes)
{
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	duplicates = 1;
	checkMats = false;
}
PixelSceneFilter::PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, int _dupes)
{
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	duplicates = _dupes;
	checkMats = false;
}
PixelSceneFilter::PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials)
{
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = 1;
	checkMats = true;
}
PixelSceneFilter::PixelSceneFilter(std::initializer_list<PixelScene> _pixelScenes, std::initializer_list<Material> _materials, int _dupes)
{
	memset(pixelScenes, 0, sizeof(PixelScene) * FILTER_OR_COUNT);
	memset(materials, 0, sizeof(Material) * FILTER_OR_COUNT);
	memcpy(pixelScenes, _pixelScenes.begin(), sizeof(PixelScene) * _pixelScenes.size());
	memcpy(materials, _materials.begin(), sizeof(Material) * _materials.size());
	duplicates = _dupes;
	checkMats = true;
}

_universal constexpr PixelSceneSpawn::PixelSceneSpawn() : spawnType(), x(), y() {}
_universal constexpr PixelSceneSpawn::PixelSceneSpawn(PixelSceneSpawnType _t, short _x, short _y) : spawnType(_t), x(_x), y(_y) {}

_universal constexpr PixelSceneData::PixelSceneData()
	: scene(PS_NONE), prob(0), materialCount(0), spawnCount(0), materials(), spawns()
{
}
_universal constexpr PixelSceneData::PixelSceneData(PixelScene _scene, float _prob)
	: scene(_scene), prob(_prob), materialCount(0), spawnCount(0), materials(), spawns()
{
}
_universal constexpr PixelSceneData::PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats)
	: scene(_scene), prob(_prob), materialCount(), spawnCount(0), materials(), spawns()
{
	scene = _scene;
	prob = _prob;
	materialCount = _mats.size();
	for (int i = 0; i < materialCount; i++) materials[i] = _mats.begin()[i];
}
_universal constexpr PixelSceneData::PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<PixelSceneSpawn> _spawns)
	: scene(_scene), prob(_prob), materialCount(0), spawnCount(), materials(), spawns()
{
	scene = _scene;
	prob = _prob;
	spawnCount = _spawns.size();
	for (int i = 0; i < spawnCount; i++) spawns[i] = _spawns.begin()[i];
}
_universal constexpr PixelSceneData::PixelSceneData(PixelScene _scene, float _prob, std::initializer_list<Material> _mats, std::initializer_list<PixelSceneSpawn> _spawns)
	: scene(_scene), prob(_prob), materialCount(), spawnCount(), materials(), spawns()
{
	scene = _scene;
	prob = _prob;
	materialCount = _mats.size();
	spawnCount = _spawns.size();
	for (int i = 0; i < materialCount; i++) materials[i] = _mats.begin()[i];
	for (int i = 0; i < spawnCount; i++) spawns[i] = _spawns.begin()[i];
}

_universal constexpr PixelSceneList::PixelSceneList() : count(), probSum(), scenes() {}
_universal constexpr PixelSceneList::PixelSceneList(int _c, std::initializer_list<PixelSceneData> list) : count(_c), probSum(), scenes()
{
	float pSum = 0;
	for (int i = 0; i < list.size(); i++)
	{
		pSum += list.begin()[i].prob;
		scenes[i] = list.begin()[i];
	}
	probSum = pSum;
}

_universal AlchemyRecipe::AlchemyRecipe() {}
_universal AlchemyRecipe::AlchemyRecipe(Material mat1, Material mat2, Material mat3)
{
	mats[0] = mat1;
	mats[1] = mat2;
	mats[2] = mat3;
}
_universal bool AlchemyRecipe::Equals(AlchemyRecipe reference, AlchemyRecipe test, AlchemyOrdering ordered)
{
	if (ordered == STRICT_ORDERED)
	{
		bool passed1 = reference.mats[0] == MATERIAL_NONE || reference.mats[0] == test.mats[0];
		bool passed2 = reference.mats[1] == MATERIAL_NONE || reference.mats[1] == test.mats[1];
		bool passed3 = reference.mats[2] == MATERIAL_NONE || reference.mats[2] == test.mats[2];

		return passed1 && passed2 && passed3;
	}
	else if (ordered == ONLY_CONSUMED)
	{
		bool passed1 = reference.mats[0] == MATERIAL_NONE || (reference.mats[0] == test.mats[0] || reference.mats[0] == test.mats[2]);
		bool passed2 = reference.mats[1] == MATERIAL_NONE || (reference.mats[1] == test.mats[1]);
		bool passed3 = reference.mats[2] == MATERIAL_NONE || (reference.mats[2] == test.mats[0] || reference.mats[2] == test.mats[2]);

		return passed1 && passed2 && passed3;
	}
	else
	{
		bool passed1 = reference.mats[0] == MATERIAL_NONE || (reference.mats[0] == test.mats[0] || reference.mats[0] == test.mats[1] || reference.mats[0] == test.mats[2]);
		bool passed2 = reference.mats[1] == MATERIAL_NONE || (reference.mats[1] == test.mats[0] || reference.mats[1] == test.mats[1] || reference.mats[1] == test.mats[2]);
		bool passed3 = reference.mats[2] == MATERIAL_NONE || (reference.mats[2] == test.mats[0] || reference.mats[2] == test.mats[1] || reference.mats[2] == test.mats[2]);

		return passed1 && passed2 && passed3;
	}
}

_universal constexpr FungalShift::FungalShift() : from(SS_NONE), to(SD_NONE), fromFlask(false), toFlask(false), minIdx(0), maxIdx(0) { }
_universal FungalShift::FungalShift(ShiftSource _from, ShiftDest _to, int _minIdx, int _maxIdx)
{
	if (_from == SS_FLASK) { from = SS_NONE; fromFlask = true; }
	else { from = _from; fromFlask = false; }
	if (_to == SD_FLASK) { to = SD_NONE; toFlask = true; }
	else { to = _to; toFlask = false; }
	minIdx = _minIdx;
	maxIdx = _maxIdx;
}
