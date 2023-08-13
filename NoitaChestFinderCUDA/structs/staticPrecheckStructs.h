#pragma once
#include "../platforms/platform_compute_helpers.h"

#include "primitives.h"
#include "enums.h"

struct AlchemyRecipe
{
	Material mats[4];

	_universal AlchemyRecipe() {}
	_universal AlchemyRecipe(Material mat1, Material mat2, Material mat3)
	{
		mats[0] = mat1;
		mats[1] = mat2;
		mats[2] = mat3;
	}

	_universal static bool Equals(AlchemyRecipe reference, AlchemyRecipe test, AlchemyOrdering ordered)
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
};

struct FungalShift
{
	ShiftSource from;
	ShiftDest to;
	bool fromFlask;
	bool toFlask;
	int minIdx;
	int maxIdx;
	_universal constexpr FungalShift()
		: from(SS_NONE), to(SD_NONE), fromFlask(false), toFlask(false), minIdx(0), maxIdx(0)
	{
	}
	_universal FungalShift(ShiftSource _from, ShiftDest _to, int _minIdx, int _maxIdx)
	{
		if (_from == SS_FLASK) { from = SS_NONE; fromFlask = true; }
		else { from = _from; fromFlask = false; }
		if (_to == SD_FLASK) { to = SD_NONE; toFlask = true; }
		else { to = _to; toFlask = false; }
		minIdx = _minIdx;
		maxIdx = _maxIdx;
	}
};

struct BiomeBlacklist
{
	Biome blacklist[5];
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