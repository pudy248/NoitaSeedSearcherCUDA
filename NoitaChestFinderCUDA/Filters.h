#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/datatypes.h"
#include "data/items.h"
#include "data/spells.h"

#include "WorldgenSearch.h"
#include "misc/wandgen.h"
#include "Output.h"

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

__device__ void PrintBytes(byte* ptr, int count)
{
	char buffer[2000];
	int offset = 0;
	_putstr_offset("[--", buffer, offset);
	for (int i = 0; i < count; i++)
	{
		_putstr_offset("0x", buffer, offset);
		_itoa_offset_zeroes(*(ptr + i), buffer, 16, 2, offset);
		if (i < count - 1)
			_putstr_offset(" ", buffer, offset);
	}
	_putstr_offset("--]", buffer, offset);
	buffer[offset] = '\0';
	printf("%s\n", buffer);
}

__device__ void ItemFilterPassed(Spawnable* s, ItemFilter f, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_MATERIAL || c == DATA_SPELL)
		{
			n += 2;
			continue;
		}

		else if (c == DATA_WAND)
		{
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			n += 37 + dat.spellCount * 3;
		}

		else
		{
			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (f.items[i] != ITEM_NONE && c == f.items[i])
				{
					foundCount++;
					break;
				}
			}
		}
	}
}

__device__ void MaterialFilterPassed(Spawnable* s, MaterialFilter mf, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_MATERIAL)
		{
			int offset = n + 1;
			Material m2 = (Material)readShort((byte*)(&s->contents), offset);


			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (mf.materials[i] != MATERIAL_NONE && m2 == mf.materials[i])
				{
					foundCount++;
					break;
				}
			}

			n += 2;
			continue;
		}

		else if (c == DATA_SPELL)
		{
			n += 2;
			continue;
		}

		else if (c == DATA_WAND)
		{
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			n += 37 + dat.spellCount * 3;
		}

	}
}

__device__ void SpellFilterPassed(uint seed, Spawnable* s, SpellFilter sf, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_SPELL && !sf.asAlwaysCast)
		{
			int offset = n + 1;
			Spell sp2 = (Spell)readShort((byte*)(&s->contents), offset);


			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (sf.spells[i] != SPELL_NONE && sp2 == sf.spells[i])
				{
					foundCount++;
					break;
				}
			}

			n += 2;
			continue;
		}

		else if (c == DATA_MATERIAL)
		{
			n += 2;
			continue;
		}

		if (c == DATA_WAND)
		{
			n += 35;
			if (sf.asAlwaysCast)
			{
				int offset = n + 1;
				Spell AC = (Spell)readShort((byte*)(&s->contents), offset);
				for (int i = 0; i < FILTER_OR_COUNT; i++)
				{
					if (sf.spells[i] != SPELL_NONE && AC == sf.spells[i])
					{
						foundCount++;
						break;
					}
				}
			}
			n += 2;
			continue;
		}
	}
}

__device__ bool WandFilterPassed(uint seed, Spawnable* s, int howBig)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];

		if (c == DATA_MATERIAL || c == DATA_SPELL)
		{
			n += 2;
			continue;
		}

		if (c == DATA_WAND)
		{
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			if (dat.capacity >= howBig) return true;
			n += 36 + dat.spellCount * 3;
			continue;
		}
	}
	return false;
}

__device__ bool SpawnablesPassed(SpawnableBlock b, FilterConfig cfg)
{
	int relevantSpawnableCount = 0;
	Spawnable* relevantSpawnables[10];

	if (cfg.aggregate)
	{
		int itemsPassed[TOTAL_FILTER_COUNT];
		int materialsPassed[TOTAL_FILTER_COUNT];
		int spellsPassed[TOTAL_FILTER_COUNT];

		for (int i = 0; i < cfg.itemFilterCount; i++) itemsPassed[i] = 0;
		for (int i = 0; i < cfg.materialFilterCount; i++) materialsPassed[i] = 0;
		for (int i = 0; i < cfg.spellFilterCount; i++) spellsPassed[i] = 0;

		for (int j = 0; j < b.count; j++)
		{
			Spawnable* s = b.spawnables[j];
			if (s == NULL) continue;
			bool added = false;

			for (int i = 0; i < cfg.itemFilterCount; i++)
			{
				int prevPassCount = itemsPassed[i];
				ItemFilterPassed(s, cfg.itemFilters[i], itemsPassed[i]);
				if (itemsPassed[i] > prevPassCount && !added)
				{
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < cfg.materialFilterCount; i++)
			{
				int prevPassCount = materialsPassed[i];
				MaterialFilterPassed(s, cfg.materialFilters[i], materialsPassed[i]);
				if (materialsPassed[i] > prevPassCount && !added)
				{
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < cfg.spellFilterCount; i++)
			{
				int prevPassCount = spellsPassed[i];
				SpellFilterPassed(b.seed, s, cfg.spellFilters[i], spellsPassed[i]);
				if (spellsPassed[i] > prevPassCount && !added)
				{
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}
		}

		bool failed = false;
		for (int i = 0; i < cfg.itemFilterCount; i++)
			if (itemsPassed[i] < cfg.itemFilters[i].duplicates)
				failed = true;

		for (int i = 0; i < cfg.materialFilterCount; i++)
			if (materialsPassed[i] < cfg.materialFilters[i].duplicates)
				failed = true;

		for (int i = 0; i < cfg.spellFilterCount; i++)
			if (spellsPassed[i] < cfg.spellFilters[i].duplicates)
				failed = true;

		if (failed)
		{
			return false;
		}
	}
	else
	{
		for (int j = 0; j < b.count; j++)
		{
			Spawnable* s = b.spawnables[j];
			if (s == NULL) continue;

			bool failed = false;
			for (int i = 0; i < cfg.itemFilterCount; i++)
			{
				int passCount = 0;
				ItemFilterPassed(s, cfg.itemFilters[i], passCount);
				if (passCount < cfg.itemFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			for (int i = 0; i < cfg.materialFilterCount; i++)
			{
				int passCount = 0;
				MaterialFilterPassed(s, cfg.materialFilters[i], passCount);
				if (passCount < cfg.materialFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			for (int i = 0; i < cfg.spellFilterCount; i++)
			{
				int passCount = 0;
				SpellFilterPassed(b.seed, s, cfg.spellFilters[i], passCount);
				if (passCount < cfg.spellFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			if (cfg.checkBigWands) {
				if (!WandFilterPassed(b.seed, s, cfg.howBig)) continue;
			}

			relevantSpawnables[relevantSpawnableCount++] = s;
		}

		if (relevantSpawnableCount == 0)
		{
			return false;
		}
	}

	PrintSpawnableBlock(b.seed, relevantSpawnables, relevantSpawnableCount);
	return true;
}
