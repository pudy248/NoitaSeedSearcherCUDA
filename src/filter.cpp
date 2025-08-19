#include "../include/misc_funcs.h"
#include "../include/primitives.h"
#include "../include/search_structs.h"
#include "../platforms/platform_implementation.h"

#include <cstdint>
#include <cstdio>

_compute static void ItemFilterPassed(Spawnable* s, int count, ItemFilter f, int& foundCount) {
	for (int n = 0; n < count; n++) {
		Item c = (&s->contents)[n];
		if (c == DATA_MATERIAL || c == DATA_SPELL) {
			n += 2;
			continue;
		}
		if (c == DATA_PIXEL_SCENE) {
			n += 4;
			continue;
		} else if (c == DATA_WAND) {
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			n += 23 + dat.spellCount * 3;
		} else {
			bool iFound = f.items[0] == ITEM_NONE;
			for (int i = 0; i < FILTER_OR_COUNT; i++) {
				if (f.items[i] != ITEM_NONE && c == f.items[i]) {
					iFound = true;
					break;
				}
			}
			if (iFound)
				foundCount++;
		}
	}
}
_compute static void MaterialFilterPassed(Spawnable* s, int count, MaterialFilter mf, int& foundCount) {
	for (int n = 0; n < count; n++) {
		Item c = (&s->contents)[n];
		if (c == DATA_MATERIAL) {
			int offset = n + 1;
			Material m2 = (Material)readShort((uint8_t*)(&s->contents), offset);

			bool mPassed = mf.materials[0] == MATERIAL_NONE;
			for (int i = 0; i < FILTER_OR_COUNT; i++) {
				if (mf.materials[i] != MATERIAL_NONE && m2 == mf.materials[i]) {
					mPassed = true;
					break;
				}
			}
			if (mPassed)
				foundCount++;

			n += 2;
			continue;
		} else if (c == DATA_SPELL) {
			n += 2;
			continue;
		} else if (c == DATA_PIXEL_SCENE) {
			n += 4;
			continue;
		} else if (c == DATA_WAND) {
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			n += 23 + dat.spellCount * 3;
		} else if (c > TRUE_ORB) {
#ifdef DEVICE_LOGGING
			printf("Unrecognized byte in filter stream: %i at %i\n", c, n);
#endif
		}
	}
}
_compute static void SpellFilterPassed(uint32_t seed, Spawnable* s, int count, SpellFilter sf, int& foundCount) {
	for (int n = 0; n < count; n++) {
		Item c = (&s->contents)[n];
		if (c == DATA_SPELL) {
			if (!sf.asAlwaysCast && !sf.perWand) {
				int offset = n + 1;
				Spell sp2 = (Spell)readShort((uint8_t*)(&s->contents), offset);

				bool foundOnThisSpell = sf.spells[0] == SPELL_NONE;
				for (int i = 0; i < FILTER_OR_COUNT; i++) {
					if (sf.spells[i] != SPELL_NONE && sp2 == sf.spells[i]) {
						foundOnThisSpell = true;
						break;
					}
				}
				if (foundOnThisSpell)
					foundCount++;
			}
			n += 2;
			continue;
		} else if (c == DATA_MATERIAL) {
			n += 2;
			continue;
		} else if (c == DATA_PIXEL_SCENE) {
			n += 4;
			continue;
		} else if (c == DATA_WAND) {
			n += 20;
			int ctr = 0;
			int offset = n;
			uint8_t spellCount = readByte((uint8_t*)(&s->contents), offset);
			for (int j = -1; j < spellCount; j++) {
				offset++;
				Spell sp = (Spell)readShort((uint8_t*)(&s->contents), offset);
				bool foundOnThisSpell = false;
				for (int i = 0; i < FILTER_OR_COUNT; i++) {
					if (sf.spells[i] != SPELL_NONE && sp == sf.spells[i] && (j == -1 || !sf.asAlwaysCast)) {
						foundOnThisSpell = true;
						break;
					}
				}
				if (foundOnThisSpell)
					ctr++;
			}
			foundCount = max(ctr, foundCount);
			n = offset - 1;
			continue;
		} else if (c > TRUE_ORB) {
#ifdef DEVICE_LOGGING
			printf("Unrecognized byte in filter stream: %i at %i\n", c, n);
#endif
		}
	}
}
_compute static void WandStatFilterPassed(Spawnable* s, int count, WandStatFilter wsf, int& foundCount) {
	for (int n = 0; n < count; n++) {
		Item c = (&s->contents)[n];

		if (c == DATA_MATERIAL || c == DATA_SPELL) {
			n += 2;
			continue;
		}
		if (c == DATA_PIXEL_SCENE) {
			n += 4;
			continue;
		}
		if (c == DATA_WAND) {
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			float stat;	
			switch (wsf.stat) {
			case WandStat::SHUFFLE: stat = dat.shuffle; break;
			case WandStat::MULTICAST: stat = dat.multicast; break;
			case WandStat::CAST_DELAY: stat = dat.delay; break;
			case WandStat::RELOAD: stat = dat.reload; break;
			case WandStat::MANA: stat = dat.mana; break;
			case WandStat::REGEN: stat = dat.regen; break;
			case WandStat::CAPACITY: stat = floorf(dat.capacity); break;
			case WandStat::SPEED_MULT: stat = dat.speed; break;
			}
			bool passed = false;
			switch (wsf.comparison) {
			case 0: passed = stat < wsf.value; break;
			case 1: passed = stat <= wsf.value; break;
			case 2: passed = stat == wsf.value; break;
			case 3: passed = stat >= wsf.value; break;
			case 4: passed = stat > wsf.value; break;
			}
			if (passed)
				foundCount++;
			n += 22 + dat.spellCount * 3;
			continue;
		}
	}
}
_compute static void PixelSceneFilterPassed(Spawnable* s, int count, PixelSceneFilter psf, int& foundCount) {
	for (int n = 0; n < count; n++) {
		Item c = (&s->contents)[n];
		if (c == DATA_PIXEL_SCENE) {
			int offset = n + 1;
			PixelScene ps = (PixelScene)readShort((uint8_t*)(&s->contents), offset);
			Material m = (Material)readShort((uint8_t*)(&s->contents), offset);

			bool psMatch = psf.pixelScenes[0] == PS_NONE;
			bool mMatch = !psf.checkMats;

			for (int i = 0; i < FILTER_OR_COUNT; i++) {
				if (psf.pixelScenes[i] != PS_NONE && ps == psf.pixelScenes[i])
					psMatch = true;
				if (psf.materials[i] != MATERIAL_NONE && m == psf.materials[i])
					mMatch = true;
			}
			if (psMatch && mMatch)
				foundCount++;

			n += 4;
			continue;
		}

		else if (c == DATA_SPELL || c == DATA_MATERIAL) {
			n += 2;
			continue;
		}

		else if (c == DATA_WAND) {
			n++;
			WandData dat = readMisalignedWand((WandData*)(&s->contents + n));
			n += 24 + dat.spellCount * 3;
		}
	}
}

_compute bool SpawnablesPassed(
	const SpawnableBlock& b, const FilterConfig& fCfg, MemSpan output, MemSpan tmp, bool write, bool upwarps) {
	int relevantSpawnableCount = 0;
	MemoryArena localArena = {tmp.ptr, 0};

	Spawnable** relevantSpawnables = (Spawnable**)ArenaAlloc(localArena, 512).ptr;

	if (fCfg.aggregate) {
		int* itemsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT).ptr;
		int* materialsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT).ptr;
		int* spellsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT).ptr;
		int* pixelScenesPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT).ptr;
		int* wandStatsPassed = (int*)ArenaAlloc(localArena, 4 * TOTAL_FILTER_COUNT).ptr;

		for (int i = 0; i < fCfg.itemFilterCount; i++)
			itemsPassed[i] = 0;
		for (int i = 0; i < fCfg.materialFilterCount; i++)
			materialsPassed[i] = 0;
		for (int i = 0; i < fCfg.spellFilterCount; i++)
			spellsPassed[i] = 0;
		for (int i = 0; i < fCfg.pixelSceneFilterCount; i++)
			pixelScenesPassed[i] = 0;
		for (int i = 0; i < fCfg.wandStatFilterCount; i++)
			wandStatsPassed[i] = 0;

		for (int j = 0; j < b.count; j++) {
			Spawnable* s = b.spawnables[j];
			if (s == NULL)
				continue;

			Spawnable sDat = readMisalignedSpawnable(s);

			bool failed = upwarps;
			if (failed) {
				if (sDat.x == 315 && sDat.y == 17)
					failed = false;
				if (sDat.x == 75 && sDat.y == 117)
					failed = false;
			}
			if (failed)
				continue;
			bool added = false;

			for (int i = 0; i < fCfg.itemFilterCount; i++) {
				int prevPassCount = itemsPassed[i];
				ItemFilterPassed(s, sDat.count, fCfg.itemFilters[i], itemsPassed[i]);
				if (itemsPassed[i] > prevPassCount && !added) {
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < fCfg.materialFilterCount; i++) {
				int prevPassCount = materialsPassed[i];
				MaterialFilterPassed(s, sDat.count, fCfg.materialFilters[i], materialsPassed[i]);
				if (materialsPassed[i] > prevPassCount && !added) {
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < fCfg.spellFilterCount; i++) {
				int prevPassCount = spellsPassed[i];
				SpellFilterPassed(b.seed, s, sDat.count, fCfg.spellFilters[i], spellsPassed[i]);
				if (spellsPassed[i] > prevPassCount && !added) {
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < fCfg.pixelSceneFilterCount; i++) {
				int prevPassCount = pixelScenesPassed[i];
				PixelSceneFilterPassed(s, sDat.count, fCfg.pixelSceneFilters[i], pixelScenesPassed[i]);
				if (pixelScenesPassed[i] > prevPassCount && !added) {
					added = true;
					relevantSpawnables[relevantSpawnableCount++] = s;
				}
			}

			for (int i = 0; i < fCfg.wandStatFilterCount; i++) {
				int prevPassCount = wandStatsPassed[i];
				WandStatFilterPassed(s, sDat.count, fCfg.wandStatFilters[i], wandStatsPassed[i]);
				if (wandStatsPassed[i] > prevPassCount && !added) {
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

		for (int i = 0; i < fCfg.wandStatFilterCount; i++)
			if (wandStatsPassed[i] < fCfg.wandStatFilters[i].duplicates)
				failed = true;

		if (failed)
			return false;
	} else {
		for (int j = 0; j < b.count; j++) {
			Spawnable* s = b.spawnables[j];
			if (s == NULL)
				continue;

			Spawnable sDat = readMisalignedSpawnable(s);

			bool failed = upwarps;
			if (failed) {
				if (sDat.x == 315 && sDat.y == 17)
					failed = false;
				if (sDat.x == 75 && sDat.y == 117)
					failed = false;
			}
			if (failed)
				continue;

			for (int i = 0; i < fCfg.itemFilterCount; i++) {
				int passCount = 0;
				ItemFilterPassed(s, sDat.count, fCfg.itemFilters[i], passCount);
				if (passCount < fCfg.itemFilters[i].duplicates) {
					failed = true;
					break;
				}
			}
			if (failed)
				continue;

			for (int i = 0; i < fCfg.materialFilterCount; i++) {
				int passCount = 0;
				MaterialFilterPassed(s, sDat.count, fCfg.materialFilters[i], passCount);
				if (passCount < fCfg.materialFilters[i].duplicates) {
					failed = true;
					break;
				}
			}
			if (failed)
				continue;

			for (int i = 0; i < fCfg.spellFilterCount; i++) {
				int passCount = 0;
				SpellFilterPassed(b.seed, s, sDat.count, fCfg.spellFilters[i], passCount);
				if (passCount < fCfg.spellFilters[i].duplicates) {
					failed = true;
					break;
				}
			}
			if (failed)
				continue;

			for (int i = 0; i < fCfg.pixelSceneFilterCount; i++) {
				int passCount = 0;
				PixelSceneFilterPassed(s, sDat.count, fCfg.pixelSceneFilters[i], passCount);
				if (passCount < fCfg.pixelSceneFilters[i].duplicates) {
					failed = true;
					break;
				}
			}
			if (failed)
				continue;

			for (int i = 0; i < fCfg.wandStatFilterCount; i++) {
				int passCount = 0;
				WandStatFilterPassed(s, sDat.count, fCfg.wandStatFilters[i], passCount);
				if (passCount < fCfg.wandStatFilters[i].duplicates) {
					failed = true;
					break;
				}
			}
			if (failed)
				continue;

			relevantSpawnables[relevantSpawnableCount++] = s;
		}

		if (relevantSpawnableCount == 0 && (fCfg.itemFilterCount + fCfg.materialFilterCount + fCfg.spellFilterCount +
											   fCfg.pixelSceneFilterCount + fCfg.wandStatFilterCount) > 0)
			return false;
	}
#ifndef IMAGE_OUTPUT
	if (write)
		WriteOutputBlock(output, {b.seed, relevantSpawnableCount, relevantSpawnables});
#endif
	return true;
}
