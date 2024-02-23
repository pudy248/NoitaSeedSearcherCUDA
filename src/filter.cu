#include "../platforms/platform_implementation.h"
#include "../include/search_structs.h"
#include "../include/misc_funcs.h"
#include "../include/primitives.h"

#include <cstdio>
#include <cstdint>

_compute static void ItemFilterPassed(Spawnable* s, ItemFilter f, int& foundCount)
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
		if (c == DATA_PIXEL_SCENE)
		{
			n += 4;
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
			bool iFound = f.items[0] == ITEM_NONE;
			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (f.items[i] != ITEM_NONE && c == f.items[i])
				{
					iFound = true;
					break;
				}
			}
			if (iFound) foundCount++;
		}
	}
}
_compute static void MaterialFilterPassed(Spawnable* s, MaterialFilter mf, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_MATERIAL)
		{
			int offset = n + 1;
			Material m2 = (Material)readShort((uint8_t*)(&s->contents), offset);

			bool mPassed = mf.materials[0] == MATERIAL_NONE;
			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (mf.materials[i] != MATERIAL_NONE && m2 == mf.materials[i])
				{
					mPassed = true;
					break;
				}
			}
			if (mPassed) foundCount++;

			n += 2;
			continue;
		}

		else if (c == DATA_SPELL)
		{
			n += 2;
			continue;
		}
		if (c == DATA_PIXEL_SCENE)
		{
			n += 4;
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
_compute static void SpellFilterPassed(uint32_t seed, Spawnable* s, SpellFilter sf, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	int largestChain = 0;
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_SPELL && !sf.asAlwaysCast)
		{
			int offset = n + 1;
			Spell sp2 = (Spell)readShort((uint8_t*)(&s->contents), offset);

			bool foundOnThisSpell = sf.spells[0] == SPELL_NONE;
			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (sf.spells[i] != SPELL_NONE && sp2 == sf.spells[i])
				{
					foundOnThisSpell = true;
					break;
				}
			}
			if (foundOnThisSpell) foundCount++;
			else if (sf.consecutive)
			{
				largestChain = max(largestChain, foundCount);
				foundCount = 0;
			}

			n += 2;
			continue;
		}

		else if (c == DATA_MATERIAL)
		{
			n += 2;
			continue;
		}
		if (c == DATA_PIXEL_SCENE)
		{
			n += 4;
			continue;
		}

		if (c == DATA_WAND)
		{
			n += 35;
			if (sf.asAlwaysCast)
			{
				int offset = n + 1;
				Spell AC = (Spell)readShort((uint8_t*)(&s->contents), offset);
				bool foundOnThisSpell = false;
				for (int i = 0; i < FILTER_OR_COUNT; i++)
				{
					if (sf.spells[i] != SPELL_NONE && AC == sf.spells[i])
					{
						foundOnThisSpell = true;
						break;
					}
				}
				if (foundOnThisSpell) foundCount++;
				else if (sf.consecutive)
				{
					largestChain = max(largestChain, foundCount);
					foundCount = 0;
				}

			}
			n += 2;
			continue;
		}
	}
	if (sf.consecutive) foundCount = largestChain;
}
_compute static bool WandFilterPassed(uint32_t seed, Spawnable* s, int howBig)
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
		if (c == DATA_PIXEL_SCENE)
		{
			n += 4;
			continue;
		}

		if (c == DATA_WAND)
		{
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			if (dat.capacity >= howBig)
				return true;
			n += 36 + dat.spellCount * 3;
			continue;
		}
	}
	return false;
}
_compute static void PixelSceneFilterPassed(Spawnable* s, PixelSceneFilter psf, int& foundCount)
{
	int count = readMisaligned(&(s->count));
	for (int n = 0; n < count; n++)
	{
		Item c = (&s->contents)[n];
		if (c == DATA_PIXEL_SCENE)
		{
			int offset = n + 1;
			PixelScene ps = (PixelScene)readShort((uint8_t*)(&s->contents), offset);
			Material m = (Material)readShort((uint8_t*)(&s->contents), offset);

			bool psMatch = psf.pixelScenes[0] == PS_NONE;
			bool mMatch = !psf.checkMats;

			for (int i = 0; i < FILTER_OR_COUNT; i++)
			{
				if (psf.pixelScenes[i] != PS_NONE && ps == psf.pixelScenes[i])
					psMatch = true;
				if (psf.materials[i] != MATERIAL_NONE && m == psf.materials[i])
					mMatch = true;
			}
			if (psMatch && mMatch) foundCount++;

			n += 4;
			continue;
		}

		else if (c == DATA_SPELL || c == DATA_MATERIAL)
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

_compute bool SpawnablesPassed(SpawnableBlock b, FilterConfig fCfg, uint8_t* output, uint8_t* tmp, bool write)
{
	int relevantSpawnableCount = 0;
	MemoryArena localArena = { tmp, 0 };
	int* itemsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT);
	int* materialsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT);
	int* spellsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT);
	int* pixelScenesPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT);
	Spawnable** relevantSpawnables = (Spawnable**)ArenaAlloc(localArena, 0);

	if (fCfg.aggregate)
	{
		for (int i = 0; i < fCfg.itemFilterCount; i++) itemsPassed[i] = 0;
		for (int i = 0; i < fCfg.materialFilterCount; i++) materialsPassed[i] = 0;
		for (int i = 0; i < fCfg.spellFilterCount; i++) spellsPassed[i] = 0;
		for (int i = 0; i < fCfg.pixelSceneFilterCount; i++) pixelScenesPassed[i] = 0;

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

			for (int i = 0; i < fCfg.pixelSceneFilterCount; i++)
			{
				int prevPassCount = pixelScenesPassed[i];
				PixelSceneFilterPassed(s, fCfg.pixelSceneFilters[i], pixelScenesPassed[i]);
				if (pixelScenesPassed[i] > prevPassCount && !added)
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

		for (int i = 0; i < fCfg.pixelSceneFilterCount; i++)
			if (pixelScenesPassed[i] < fCfg.pixelSceneFilters[i].duplicates)
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

			bool failed = true;
			if (fCfg.upwarp) {
				if (sDat.x == 315 && sDat.y == 17) failed = false;
				if (sDat.x == 75 && sDat.y == 117) failed = false;
			}
			if (failed) continue;

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

			for (int i = 0; i < fCfg.pixelSceneFilterCount; i++)
			{
				int passCount = 0;
				PixelSceneFilterPassed(s, fCfg.pixelSceneFilters[i], passCount);
				if (passCount < fCfg.pixelSceneFilters[i].duplicates)
				{
					failed = true;
					break;
				}
			}
			if (failed) continue;

			if (fCfg.checkBigWands)
			{
				if (!WandFilterPassed(b.seed, s, fCfg.howBig)) continue;
			}

			relevantSpawnables[relevantSpawnableCount++] = s;
		}

		if (relevantSpawnableCount == 0 && (fCfg.itemFilterCount + fCfg.materialFilterCount + fCfg.spellFilterCount + fCfg.pixelSceneFilterCount + fCfg.checkBigWands) > 0)
		{
			return false;
		}
	}
#ifndef IMAGE_OUTPUT
	if (write) WriteOutputBlock(output, b.seed, relevantSpawnables, relevantSpawnableCount);
#endif
	return true;
}
