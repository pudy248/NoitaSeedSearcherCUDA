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
	int itemFilterCount;
	ItemFilter itemFilters[10];
	int materialFilterCount;
	Material materialFilters[10];
	int spellFilterCount;
	Spell spellFilters[10];
	bool checkBigWands;
	int howBig;
	FilterConfig(int _itemFilterCount, ItemFilter* _itemFilters, int _materialFilterCount, Material* _materialFilters, int _spellFilterCount, Spell* _spellFilters, bool _checkBigWands, int _howBig) {
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
			byte* ptr = (byte*)s.contents + n + 1;
			Material m2 = (Material)readShort(&ptr);
			if (m == m2) return true;
		}
	}
	return false;
}

__device__ bool SpellFilterPassed(Spawnable s, Spell sp) {
	for (int n = 0; n < s.count; n++) {
		if (s.contents[n] == DATA_SPELL) {
			byte* ptr = (byte*)s.contents + n + 1;
			Spell sp2 = (Spell)readShort(&ptr);
			if (sp == sp2) return true;
		}
	}
	return false;
}

__device__ bool WandFilterPassed(uint seed, Spawnable s, int howBig) {
	bool passed = false;
	for (int i = 0; i < s.count; i++) {
		Wand w;
		if (s.contents[i] == WAND_T6)
			w = GetWandWithLevel(seed, s.x, s.y, 6, false, false);
		else if (s.contents[i] == WAND_T6NS)
			w = GetWandWithLevel(seed, s.x, s.y, 6, true, false);
		else if (s.contents[i] == WAND_T6B)
			w = GetWandWithLevel(seed, s.x, s.y, 6, false, true);
		else continue;
		if (w.capacity >= howBig) {
			passed = true;
			printf("WAND %i @ (%i %i) %i %i %i %i %i %i %i %f\n", seed, s.x, s.y, (int)w.capacity, w.multicast, w.delay, w.reload, w.mana, w.regen, w.spread, w.speed);
		}
	}
	return passed;
}

__device__ bool SpawnablePassed(uint seed, Spawnable s, FilterConfig cfg) {
	for (int i = 0; i < cfg.itemFilterCount; i++)
		if (!ItemFilterPassed(s, cfg.itemFilters[i])) return false;
	for (int i = 0; i < cfg.materialFilterCount; i++)
		if (!MaterialFilterPassed(s, cfg.materialFilters[i])) return false;
	for (int i = 0; i < cfg.spellFilterCount; i++)
		if (!SpellFilterPassed(s, cfg.spellFilters[i])) return false;
	if (cfg.checkBigWands && !WandFilterPassed(seed, s, cfg.howBig)) return false;
	return true;
}
