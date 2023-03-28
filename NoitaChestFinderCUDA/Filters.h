#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/datatypes.h"
#include "data/items.h"
#include "data/spells.h"

#include "WorldgenSearch.h"
#include "misc/wandgen.h"

struct ItemFilter {
	Item item;
	bool isBlacklist;
	int duplicates;

	ItemFilter() {
		item = ITEM_NONE;
		duplicates = 0;
		isBlacklist = false;
	}

	ItemFilter(Item _item) {
		item = _item;
		duplicates = 1;
		isBlacklist = false;
	}

	ItemFilter(Item _item, int _dupes) {
		item = _item;
		duplicates = _dupes;
		isBlacklist = false;
	}

	ItemFilter(Item _item, int _dupes, bool _blacklist ) {
		item = _item;
		isBlacklist = _blacklist;
		duplicates = _dupes;
	}
};

struct FilterConfig {
	bool aggregate;
	int itemFilterCount;
	ItemFilter itemFilters[10];
	int materialFilterCount;
	Material materialFilters[10];
	int spellFilterCount;
	Spell spellFilters[10];
	bool checkBigWands;
	int howBig;
	FilterConfig(bool _aggregate, int _itemFilterCount, ItemFilter* _itemFilters, int _materialFilterCount, Material* _materialFilters, int _spellFilterCount, Spell* _spellFilters, bool _checkBigWands, int _howBig) {
		aggregate = _aggregate;
		itemFilterCount = _itemFilterCount;
		materialFilterCount = _materialFilterCount;
		spellFilterCount = _spellFilterCount;
		checkBigWands = _checkBigWands;
		howBig = _howBig;
		memcpy(itemFilters, _itemFilters, sizeof(ItemFilter) * itemFilterCount);
		memcpy(materialFilters, _materialFilters, sizeof(Material) * materialFilterCount);
		memcpy(spellFilters, _spellFilters, sizeof(Spell) * spellFilterCount);
	}
};

__device__ bool ItemFilterPassed(Spawnable s, ItemFilter f) {
	int foundCount = 0;
	for (int n = 0; n < s.count; n++) {
		if (s.contents[n] - DATA_MATERIAL < 2) {
			n += 2;
		}
		else if (s.contents[n] == DATA_WAND) {
			n += 0;//wand_size
		}
		else {
			//for (int i = 0; i < f.; i++) {
				if (s.contents[n] == f.item) {
					if (f.isBlacklist)
						return false;
					else
						foundCount++;
				}
			//}
		}
	}
	return foundCount >= f.duplicates;
}

__device__ bool MaterialFilterPassed(Spawnable s, Material m) {
	for (int n = 0; n < s.count; n++) {
		if (s.contents[n] == DATA_MATERIAL) {
			int offset = n + 1;
			Material m2 = (Material)readShort((byte*)s.contents, offset);
			if (m == m2) return true;
		}
	}
	return false;
}

__device__ bool SpellFilterPassed(Spawnable s, Spell sp) {
	for (int n = 0; n < s.count; n++) {
		if (s.contents[n] == DATA_SPELL) {
			int offset = n + 1;
			Spell sp2 = (Spell)readShort((byte*)s.contents, offset);
			if (sp == sp2) return true;
		}
	}
	return false;
}

__device__ bool WandFilterPassed(uint seed, Spawnable s, int howBig, bool print) {
	bool passed = false;
	for (int i = 0; i < s.count; i++) {
		if (s.contents[i] - DATA_MATERIAL < 2) {
			i += 2;
		}
		int rand_x = s.x;
		int rand_y = s.y;
		if (s.sType == TYPE_CHEST) {
			rand_x += 510;
			rand_y += 683;
		}
		Wand w;
		if (s.contents[i] == WAND_T6)
			w = GetWandWithLevel(seed, rand_x, rand_y, 6, false, false);
		else if (s.contents[i] == WAND_T6NS)
			w = GetWandWithLevel(seed, rand_x, rand_y, 6, true, false);
		else if (s.contents[i] == WAND_T6B)
			w = GetWandWithLevel(seed, rand_x, rand_y, 6, false, true);

		if (s.contents[i] == WAND_T5)
			w = GetWandWithLevel(seed, rand_x, rand_y, 5, false, false);
		else if (s.contents[i] == WAND_T5NS)
			w = GetWandWithLevel(seed, rand_x, rand_y, 5, true, false);
		else if (s.contents[i] == WAND_T5B)
			w = GetWandWithLevel(seed, rand_x, rand_y, 5, false, true);

		if (s.contents[i] == WAND_T4)
			w = GetWandWithLevel(seed, rand_x, rand_y, 4, false, false);
		else if (s.contents[i] == WAND_T4NS)
			w = GetWandWithLevel(seed, rand_x, rand_y, 4, true, false);
		else if (s.contents[i] == WAND_T4B)
			w = GetWandWithLevel(seed, rand_x, rand_y, 4, false, true);

		if (s.contents[i] == WAND_T3)
			w = GetWandWithLevel(seed, rand_x, rand_y, 3, false, false);
		else if (s.contents[i] == WAND_T3NS)
			w = GetWandWithLevel(seed, rand_x, rand_y, 3, true, false);
		else if (s.contents[i] == WAND_T3B)
			w = GetWandWithLevel(seed, rand_x, rand_y, 3, false, true);

		if (s.contents[i] == WAND_T2)
			w = GetWandWithLevel(seed, rand_x, rand_y, 2, false, false);
		else if (s.contents[i] == WAND_T2NS)
			w = GetWandWithLevel(seed, rand_x, rand_y, 2, true, false);
		else if (s.contents[i] == WAND_T2B)
			w = GetWandWithLevel(seed, rand_x, rand_y, 2, false, true);

		if (s.contents[i] == WAND_T1)
			w = GetWandWithLevel(seed, rand_x, rand_y, 1, false, false);
		else if (s.contents[i] == WAND_T1NS)
			w = GetWandWithLevel(seed, rand_x, rand_y, 1, true, false);
		else if (s.contents[i] == WAND_T1B)
			w = GetWandWithLevel(seed, rand_x, rand_y, 1, false, true);
		else continue;
		if (w.capacity >= howBig) {
			passed = true;
		}
		if (print) printf("WAND %i @ (%i %i) %i %i %i %i %i %i %i %f\n", seed, s.x, s.y, (int)w.capacity, w.multicast, w.delay, w.reload, w.mana, w.regen, w.spread, w.speed);
	}
	return passed;
}

__device__ bool SpawnablesPassed(SpawnableBlock b, FilterConfig cfg, bool print) {
	int relevantSpawnableCount = 0;
	Spawnable** relevantSpawnables = (Spawnable**)malloc(sizeof(Spawnable*) * b.count);

	if (cfg.aggregate) {
		bool* itemsPassed = (bool*)malloc(cfg.itemFilterCount);
		bool* materialsPassed = (bool*)malloc(cfg.materialFilterCount);
		bool* spellsPassed = (bool*)malloc(cfg.spellFilterCount);

		for (int i = 0; i < cfg.itemFilterCount; i++) itemsPassed[i] = false;
		for (int i = 0; i < cfg.materialFilterCount; i++) materialsPassed[i] = false;
		for (int i = 0; i < cfg.spellFilterCount; i++) spellsPassed[i] = false;

		for (int j = 0; j < b.count; j++) {
			Spawnable s = b.spawnables[j];

			for (int i = 0; i < cfg.itemFilterCount; i++) {
				if (itemsPassed[i]) continue;
				if (ItemFilterPassed(s, cfg.itemFilters[i])) {
					itemsPassed[i] = true;
					relevantSpawnables[relevantSpawnableCount++] = b.spawnables + j;
					break;
				}
			}

			for (int i = 0; i < cfg.materialFilterCount; i++) {
				if (materialsPassed[i]) continue;
				if (MaterialFilterPassed(s, cfg.materialFilters[i])) {
					materialsPassed[i] = true;
					relevantSpawnables[relevantSpawnableCount++] = b.spawnables + j;
					break;
				}
			}

			for (int i = 0; i < cfg.spellFilterCount; i++) {
				if (spellsPassed[i]) continue;
				if (SpellFilterPassed(s, cfg.spellFilters[i])) {
					spellsPassed[i] = true;
					relevantSpawnables[relevantSpawnableCount++] = b.spawnables + j;
					break;
				}
			}
		}

		for (int i = 0; i < cfg.itemFilterCount; i++) if (!itemsPassed[i]) {
			free(itemsPassed);
			free(materialsPassed);
			free(spellsPassed);
			free(relevantSpawnables);
			return false;
		}
		for (int i = 0; i < cfg.materialFilterCount; i++) if (!materialsPassed[i]) {
			free(itemsPassed);
			free(materialsPassed);
			free(spellsPassed);
			free(relevantSpawnables);
			return false;
		}
		for (int i = 0; i < cfg.spellFilterCount; i++) if (!spellsPassed[i]) {
			free(itemsPassed);
			free(materialsPassed);
			free(spellsPassed);
			free(relevantSpawnables);
			return false;
		}

		free(itemsPassed);
		free(materialsPassed);
		free(spellsPassed);
	}
	else {
		for (int j = 0; j < b.count; j++) {
			Spawnable s = b.spawnables[j];

			bool failed = false;
			for (int i = 0; i < cfg.itemFilterCount; i++) {
				if (!ItemFilterPassed(s, cfg.itemFilters[i])) {
					failed = true;
					break;
				}
			}
			if (failed) continue;

			for (int i = 0; i < cfg.materialFilterCount; i++) {
				if (!MaterialFilterPassed(s, cfg.materialFilters[i])) {
					failed = true;
					break;
				}
			}
			if (failed) continue;

			for (int i = 0; i < cfg.spellFilterCount; i++) {
				if (!SpellFilterPassed(s, cfg.spellFilters[i])) {
					failed = true;
					break;
				}
			}
			if (failed) continue;

			if (cfg.checkBigWands && !WandFilterPassed(b.seed, s, cfg.howBig, false)) continue;

			relevantSpawnables[relevantSpawnableCount++] = b.spawnables + j;
		}
	
		if (relevantSpawnableCount == 0) {
			free(relevantSpawnables);
			return false;
		}
	}

	if (print) {
		for (int i = 0; i < relevantSpawnableCount; i++) {
			Spawnable s = *relevantSpawnables[i];

			for (int n = 0; n < s.count; n++) {
				if (s.contents[n] == DATA_MATERIAL) {
					n += 2;
					continue;
				}
				if (s.contents[n] >= WAND_T1 && s.contents[n] <= WAND_T6B)
					WandFilterPassed(b.seed, s, cfg.howBig, true);
			}

			printf("%i @ (%i[%i], %i): %s, %i bytes: (", b.seed, s.x, roundRNGPos(s.x), s.y, SpawnableTypeNames[s.sType - TYPE_CHEST], s.count);

			for (int n = 0; n < s.count; n++) {
				if (s.contents[n] >= SAMPO)
					printf("%x ", s.contents[n]);
				else if (s.contents[n] == DATA_MATERIAL) {
					int offset = n + 1;
					short m = readShort((byte*)s.contents, offset);
					printf("%s ", MaterialNames[m]);
					n += 2;
					continue;
				}
				else {
					int idx = s.contents[n] - GOLD_NUGGETS;
					printf("%s ", ItemStrings[idx]);
				}
			}
			printf(")\n");
		}
	}

	free(relevantSpawnables);
	return true;
}
