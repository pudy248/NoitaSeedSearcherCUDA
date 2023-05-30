#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"
#include "structs/enums.h"
#include "structs/spawnableStructs.h"

#include "data/uiNames.h"

#include "Output.h"
#include "Configuration.h"

__device__ void PrintBytes(uint8_t* ptr, int count)
{
	char buffer[2000];
	int offset = 0;
	_putstr_offset("[--", buffer, offset);
	for (int i = 0; i < count; i++)
	{
		_putstr_offset("0x", buffer, offset);
		_itoa_offset_zeroes(*(ptr + i), 16, 2, buffer, offset);
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
			Material m2 = (Material)readShort((uint8_t*)(&s->contents), offset);


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

__device__ void SpellFilterPassed(uint32_t seed, Spawnable* s, SpellFilter sf, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_SPELL && !sf.asAlwaysCast)
		{
			int offset = n + 1;
			Spell sp2 = (Spell)readShort((uint8_t*)(&s->contents), offset);


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
				Spell AC = (Spell)readShort((uint8_t*)(&s->contents), offset);
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

__device__ bool WandFilterPassed(uint32_t seed, Spawnable* s, int howBig)
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

__device__ bool SpawnablesPassed(SpawnableBlock b, FilterConfig fCfg, uint8_t* output, bool write)
{
	int relevantSpawnableCount = 0;
	Spawnable* relevantSpawnables[50];

	if (fCfg.aggregate)
	{
		int itemsPassed[TOTAL_FILTER_COUNT];
		int materialsPassed[TOTAL_FILTER_COUNT];
		int spellsPassed[TOTAL_FILTER_COUNT];

		for (int i = 0; i < fCfg.itemFilterCount; i++) itemsPassed[i] = 0;
		for (int i = 0; i < fCfg.materialFilterCount; i++) materialsPassed[i] = 0;
		for (int i = 0; i < fCfg.spellFilterCount; i++) spellsPassed[i] = 0;

		for (int j = 0; j < b.count; j++)
		{
			Spawnable* s = b.spawnables[j];
			if (s == NULL) continue;
			bool added = false;

			for (int i = 0; i < fCfg.itemFilterCount; i++)
			{
				int prevPassCount = itemsPassed[i];
				ItemFilterPassed(s, fCfg.itemFilters[i], itemsPassed[i]);
				if (itemsPassed[i] > prevPassCount && !added)
				{
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < fCfg.materialFilterCount; i++)
			{
				int prevPassCount = materialsPassed[i];
				MaterialFilterPassed(s, fCfg.materialFilters[i], materialsPassed[i]);
				if (materialsPassed[i] > prevPassCount && !added)
				{
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < fCfg.spellFilterCount; i++)
			{
				int prevPassCount = spellsPassed[i];
				SpellFilterPassed(b.seed, s, fCfg.spellFilters[i], spellsPassed[i]);
				if (spellsPassed[i] > prevPassCount && !added)
				{
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}
		}

		bool failed = false;
		for (int i = 0; i < fCfg.itemFilterCount; i++)
			if (itemsPassed[i] < fCfg.itemFilters[i].duplicates)
				failed = true;

		for (int i = 0; i < fCfg.materialFilterCount; i++)
			if (materialsPassed[i] < fCfg.materialFilters[i].duplicates)
				failed = true;

		for (int i = 0; i < fCfg.spellFilterCount; i++)
			if (spellsPassed[i] < fCfg.spellFilters[i].duplicates)
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

			Spawnable sDat = readMisalignedSpawnable(s);

			bool failed = false;
			for (int i = 0; i < fCfg.itemFilterCount; i++)
			{
				int passCount = 0;
				ItemFilterPassed(s, fCfg.itemFilters[i], passCount);
				if (passCount < fCfg.itemFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			for (int i = 0; i < fCfg.materialFilterCount; i++)
			{
				int passCount = 0;
				MaterialFilterPassed(s, fCfg.materialFilters[i], passCount);
				if (passCount < fCfg.materialFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			for (int i = 0; i < fCfg.spellFilterCount; i++)
			{
				int passCount = 0;
				SpellFilterPassed(b.seed, s, fCfg.spellFilters[i], passCount);
				if (passCount < fCfg.spellFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			if (fCfg.checkBigWands) {
				if (!WandFilterPassed(b.seed, s, fCfg.howBig)) continue;
			}

			relevantSpawnables[relevantSpawnableCount++] = s;
		}

		if (relevantSpawnableCount == 0)
		{
			return false;
		}
	}

	if(write) WriteOutputBlock(output, b.seed, relevantSpawnables, relevantSpawnableCount);
	return true;
}
