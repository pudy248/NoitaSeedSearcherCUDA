#pragma once

#include "../Compute.h"

#include "guiLayout.h"

void CreateConfigsAndDispatch()
{
	SearchGui* gui = sfmlState->gui;
	
	
	BiomeWangScope biomes[20];
	int biomeCount = 0;
	int maxMapArea = 0;
	if (gui->worldConfig.centerPanel.checkboxes[1].enabled) InstantiateBiome("wang_tiles/coalmine.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/coalmine_alt.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/excavationsite.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/fungicave.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/snowcave.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/snowcastle.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/rainforest.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/rainforest_open.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/vault.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/crypt.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/fungiforest.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/vault_frozen.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/liquidcave.png", biomes, biomeCount, maxMapArea);
	//InstantiateBiome("wang_tiles/meat.png", biomes, biomeCount, maxMapArea);
	config.biomeCount = biomeCount;
	memcpy(config.biomeScopes, biomes, sizeof(BiomeWangScope) * 20);

	config.memSizes = {
			40_GB,

	#ifdef DO_WORLDGEN
			(size_t)3 * maxMapArea + 4096,
	#else
			(size_t)512,
	#endif
			(size_t)3 * maxMapArea + 128,
			(size_t)4 * maxMapArea,
			(size_t)maxMapArea,
			(size_t)4096,
	};

	config.generalCfg = {
		(uint32_t)atoi(gui->searchConfig.config.inputs[0].text.str.toAnsiString().c_str()),
		(uint32_t)atoi(gui->searchConfig.config.inputs[1].text.str.toAnsiString().c_str()),
		(uint32_t)atoi(gui->searchConfig.config.inputs[2].text.str.toAnsiString().c_str()),
		false
	};
#ifdef REALTIME_SEEDS
	generalCfg.seedBlockSize = 1;
#endif
	config.spawnableCfg = { 
		{
			atoi(gui->worldConfig.leftPanel.inputs[0].text.str.toAnsiString().c_str()),
			atoi(gui->worldConfig.leftPanel.inputs[2].text.str.toAnsiString().c_str()),
		},
		{
			atoi(gui->worldConfig.leftPanel.inputs[1].text.str.toAnsiString().c_str()),
			atoi(gui->worldConfig.leftPanel.inputs[3].text.str.toAnsiString().c_str()),
		},
		gui->worldConfig.leftPanel.dropdowns[0].list.selectedElement,
		gui->worldConfig.leftPanel.dropdowns[1].list.selectedElement,

		gui->worldConfig.leftPanel.checkboxes[12].enabled, //greed
		gui->worldConfig.leftPanel.checkboxes[2].enabled, //pacifist
		gui->worldConfig.leftPanel.checkboxes[0].enabled, //shop spells
		gui->worldConfig.leftPanel.checkboxes[1].enabled, //shop wands
		gui->worldConfig.leftPanel.checkboxes[8].enabled, //eye rooms
		gui->worldConfig.leftPanel.checkboxes[13].enabled, //upwarp check
		gui->worldConfig.leftPanel.checkboxes[3].enabled, //biome chests
		gui->worldConfig.leftPanel.checkboxes[4].enabled, //biome pedestals
		gui->worldConfig.leftPanel.checkboxes[5].enabled, //biome altars
		gui->worldConfig.leftPanel.checkboxes[6].enabled, //biome pixelscenes
		false, //enemies
		false, //hell shops
		gui->worldConfig.leftPanel.checkboxes[11].enabled, //potion contents
		gui->worldConfig.leftPanel.checkboxes[10].enabled, //chest spells
		gui->worldConfig.leftPanel.checkboxes[9].enabled, //wand stats
	};

	config.filterCfg = FilterConfig(
		gui->worldConfig.leftPanel.checkboxes[14].enabled,
		gui->worldConfig.leftPanel.checkboxes[13].enabled,
		0, {}, 0, {}, 0, {}, 0, {},
		gui->worldConfig.leftPanel.checkboxes[15].enabled, 27);

	config.precheckCfg = {
		{false, CART_NONE},
		{false, MATERIAL_NONE},
		{false, SPELL_NONE, SPELL_NONE},
		{false, MATERIAL_NONE},
		{false, AlchemyOrdering::UNORDERED, {}, {}},
		{false, {}},
		{false, {}},
		{false, {}, {3, 3, 3, 3, 3, 3, 3}},
	};

	config.outputCfg = { 0.05f, false, false };

	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.x * 2 + 1;
	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.y * 2 + 1;
	config.memSizes.spawnableMemSize *= max(1, biomeCount);

	GuiDropdown* dd = &gui->staticConfig.leftPanel.dropdowns[0];
	
	if (dd->list.selectedElement < dd->list.numElements - 1) config.precheckCfg.cart = { true, (CartType)(dd->list.selectedElement + 1) };

	dd = &gui->staticConfig.leftPanel.dropdowns[1];
	if (dd->list.selectedElement < dd->list.numElements - 1) config.precheckCfg.flask = { true, _potionStarting[dd->list.selectedElement] };

	dd = &gui->staticConfig.leftPanel.dropdowns[2];
	if (dd->list.selectedElement < dd->list.numElements - 1) config.precheckCfg.rain = { true, _rainMaterials[dd->list.selectedElement] };

	dd = &gui->staticConfig.leftPanel.dropdowns[3];
	if (dd->list.selectedElement < dd->list.numElements - 1) config.precheckCfg.wands = { true, _startingProjectiles[dd->list.selectedElement] };

	dd = &gui->staticConfig.leftPanel.dropdowns[4];
	if (dd->list.selectedElement < dd->list.numElements - 1) config.precheckCfg.wands = { true, config.precheckCfg.wands.projectile, _startingBombs[dd->list.selectedElement] };
	
	bool anySelected = false;
	for (int i = 5; i < 11; i++) anySelected |= gui->staticConfig.leftPanel.dropdowns[i].list.selectedElement < gui->staticConfig.leftPanel.dropdowns[i].list.numElements - 1;

	if (anySelected)
	{
		config.precheckCfg.alchemy.check = true;
		for (int i = 0; i < 3; i++)
		{
			dd = &gui->staticConfig.leftPanel.dropdowns[i + 5];
			if (dd->list.selectedElement == dd->list.numElements - 1) continue;
			if (dd->list.selectedElement < alchemyLiquidCount) config.precheckCfg.alchemy.LC.mats[i] = _alchemyLiquids[dd->list.selectedElement];
			else config.precheckCfg.alchemy.LC.mats[i] = _alchemySolids[dd->list.selectedElement - 30];
		}
		for (int i = 0; i < 3; i++)
		{
			dd = &gui->staticConfig.leftPanel.dropdowns[i + 8];
			if (dd->list.selectedElement == dd->list.numElements - 1) continue;
			if (dd->list.selectedElement < alchemyLiquidCount) config.precheckCfg.alchemy.AP.mats[i] = _alchemyLiquids[dd->list.selectedElement];
			else config.precheckCfg.alchemy.AP.mats[i] = _alchemySolids[dd->list.selectedElement - 30];
		}
		config.precheckCfg.alchemy.ordering = (AlchemyOrdering)gui->staticConfig.leftPanel.dropdowns[11].list.selectedElement;
	}

	anySelected = false;
	for (int i = 12; i < 21; i++) anySelected |= gui->staticConfig.leftPanel.dropdowns[i].list.selectedElement < gui->staticConfig.leftPanel.dropdowns[i].list.numElements - 1;

	if (anySelected)
	{
		config.precheckCfg.biomes.check = true;
		for (int i = 0; i < 9; i++)
		{
			dd = &gui->staticConfig.leftPanel.dropdowns[i + 12];
			if (dd->list.selectedElement == dd->list.numElements - 1) continue;
			config.precheckCfg.biomes.modifiers[i] = _allBMLists[i][dd->list.selectedElement];
		}
	}

	if (gui->staticConfig.centerPanel.fungalShiftRowCount > 0)
	{
		config.precheckCfg.fungal.check = true;
		for (int i = 0; i < gui->staticConfig.centerPanel.fungalShiftRowCount; i++)
		{
			config.precheckCfg.fungal.shifts[i] = FungalShift(
				_fungalMaterialsFrom[gui->staticConfig.centerPanel.rows[i].source.list.selectedElement],
				_fungalMaterialsTo[gui->staticConfig.centerPanel.rows[i].dest.list.selectedElement],
				atoi(gui->staticConfig.centerPanel.rows[i].lowerBound.text.str.toAnsiString().c_str()),
				atoi(gui->staticConfig.centerPanel.rows[i].upperBound.text.str.toAnsiString().c_str()));
		}
		for (int i = gui->staticConfig.centerPanel.fungalShiftRowCount; i < maxFungalShifts; i++) config.precheckCfg.fungal.shifts[i] = FungalShift();
	}

	if (gui->staticConfig.rightPanel.perkRowCount > 0)
	{
		config.precheckCfg.perks.check = true;
		for (int i = 0; i < gui->staticConfig.rightPanel.perkRowCount; i++)
		{
			config.precheckCfg.perks.perks[i] = {
				(Perk)((gui->staticConfig.rightPanel.rows[i].perk.list.selectedElement + 1) % _perkCount),
				gui->staticConfig.rightPanel.rows[i].lotteryBox.enabled,
				atoi(gui->staticConfig.rightPanel.rows[i].lowerBound.text.str.toAnsiString().c_str()),
				atoi(gui->staticConfig.rightPanel.rows[i].upperBound.text.str.toAnsiString().c_str()) };
		}
		for (int i = gui->staticConfig.rightPanel.perkRowCount; i < maxPerkFilters; i++) config.precheckCfg.perks.perks[i] = { PERK_NONE, false, 0, 0 };
	}

	if (gui->worldConfig.rightPanel.itemRowCount > 0)
	{
		config.filterCfg.itemFilterCount = gui->worldConfig.rightPanel.itemRowCount;
		for (int i = 0; i < gui->worldConfig.rightPanel.itemRowCount; i++)
		{
			config.filterCfg.itemFilters[i] = {
				{_items[gui->worldConfig.rightPanel.rows[i].item.list.selectedElement]},
				atoi(gui->worldConfig.rightPanel.rows[i].count.text.str.toAnsiString().c_str())
			};
		}
		for (int i = gui->worldConfig.rightPanel.itemRowCount; i < TOTAL_FILTER_COUNT; i++) config.filterCfg.itemFilters[i] = { {}, 0 };
	}

	if (gui->filterConfig.leftPanel.matRowCount > 0)
	{
		config.filterCfg.materialFilterCount = gui->filterConfig.leftPanel.matRowCount;
		for (int i = 0; i < gui->filterConfig.leftPanel.matRowCount; i++)
		{
			config.filterCfg.materialFilters[i] = {
				{(Material)((gui->filterConfig.leftPanel.rows[i].material.list.selectedElement - 1 + _materialCount) % _materialCount)},
				atoi(gui->filterConfig.leftPanel.rows[i].count.text.str.toAnsiString().c_str())
			};
		}
		for (int i = gui->filterConfig.leftPanel.matRowCount; i < TOTAL_FILTER_COUNT; i++) config.filterCfg.materialFilters[i] = { {}, 0 };
	}

	if (gui->filterConfig.centerPanel.spellRowCount > 0)
	{
		config.filterCfg.spellFilterCount = gui->filterConfig.centerPanel.spellRowCount;
		for (int i = 0; i < gui->filterConfig.centerPanel.spellRowCount; i++)
		{
			config.filterCfg.spellFilters[i] = {
				{(Spell)((gui->filterConfig.centerPanel.rows[i].spell.list.selectedElement - 1 + _spellCount) % _spellCount)},
				atoi(gui->filterConfig.centerPanel.rows[i].count.text.str.toAnsiString().c_str())
			};
		}
		for (int i = gui->filterConfig.centerPanel.spellRowCount; i < TOTAL_FILTER_COUNT; i++) config.filterCfg.spellFilters[i] = { {}, 0 };
	}


	gui->searchConfig.progDat.progressPercent = 0;
	gui->searchConfig.progDat.elapsedMillis = 0;
	gui->searchConfig.progDat.searchedSeeds = 0;
	gui->searchConfig.progDat.validSeeds = 0;

	SearchMain((OutputProgressData&)gui->searchConfig.progDat, AppendOutput);

	gui->searchConfig.progDat.progressPercent = 1;
	gui->searchConfig.progDat.elapsedMillis = 0;
	gui->searchConfig.progDat.abort = false;
	gui->searchConfig.updateCtr = 0;
	gui->searchConfig.searchDone = true;
	return;
}