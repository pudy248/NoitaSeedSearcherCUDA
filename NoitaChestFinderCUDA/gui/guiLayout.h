#pragma once

#include "../structs/primitives.h"
#include "../misc/pngutils.h"
#include "guiPrimitives.h"
#include "guiIntermediates.h"

#include "../data/uiNames.h"
#include "../data/guiData.h"

#include <thread>
#include <mutex>
#include <vector>
#include <intrin.h>

struct TabSelector : GuiObject
{
	int selectedTab;
	BGTextRect staticConfig;
	BGTextRect worldConfig;
	BGTextRect filterConfig;
	BGTextRect searchConfig;
	BGTextRect output;

	TabSelector()
	{
		sf::Color textCol = sf::Color::White;
		sf::Color bgCol = sf::Color(20, 20, 20);
		selectedTab = 0;
		staticConfig = BGTextRect("static tab", sf::FloatRect(20, 20, 110, 50), 36, textCol, bgCol);
		worldConfig = BGTextRect("world tab", sf::FloatRect(140, 20, 110, 50), 36, textCol, bgCol);
		filterConfig = BGTextRect("filter tab", sf::FloatRect(260, 20, 110, 50), 36, textCol, bgCol);
		searchConfig = BGTextRect("search tab", sf::FloatRect(380, 20, 110, 50), 36, textCol, bgCol);
		output = BGTextRect("output tab", sf::FloatRect(500, 20, 110, 50), 36, textCol, bgCol);
	}

	void Render()
	{
		staticConfig.Render();
		worldConfig.Render();
		filterConfig.Render();
		searchConfig.Render();
		output.Render();
	}

	void CalculateColors()
	{
		auto setColor = [this](BGTextRect& tr, bool selected, bool hovering)
		{
			sf::Color baseBGCol = sf::Color(30, 30, 30);
			sf::Color selectedBGCol = sf::Color(60, 60, 60);
			sf::Color hoverBGCol = sf::Color(20, 20, 20);
			sf::Color selectedHoverBGCol = sf::Color(50, 50, 50);

			tr.bgColor = hovering ? (selected ? selectedHoverBGCol : hoverBGCol) : (selected ? selectedBGCol : baseBGCol);
		};

		setColor(staticConfig, selectedTab == 0, staticConfig.mRect.containedMouseLastFrame);
		setColor(worldConfig, selectedTab == 1, worldConfig.mRect.containedMouseLastFrame);
		setColor(filterConfig, selectedTab == 2, filterConfig.mRect.containedMouseLastFrame);
		setColor(searchConfig, selectedTab == 3, searchConfig.mRect.containedMouseLastFrame);
		setColor(output, selectedTab == 4, output.mRect.containedMouseLastFrame);

	}

	bool HandleClick(sf::Vector2f position)
	{
		if (staticConfig.mRect.rect.contains(position)) { selectedTab = 0; CalculateColors(); return true; }
		if (worldConfig.mRect.rect.contains(position)) { selectedTab = 1; CalculateColors(); return true; }
		if (filterConfig.mRect.rect.contains(position)) { selectedTab = 2; CalculateColors(); return true; }
		if (searchConfig.mRect.rect.contains(position)) { selectedTab = 3; CalculateColors(); return true; }
		if (output.mRect.rect.contains(position)) { selectedTab = 4; CalculateColors(); return true; }

		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{
		staticConfig.mRect.HandleMouse(position);
		worldConfig.mRect.HandleMouse(position);
		filterConfig.mRect.HandleMouse(position);
		searchConfig.mRect.HandleMouse(position);
		output.mRect.HandleMouse(position);

		CalculateColors();
	}
};

struct StaticConfigTab : GuiObject
{
	struct StaticConfigLeftPanel : GuiObject
	{
		sf::FloatRect rect = sf::FloatRect(20, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);

#define textCount 10
#define dropdownCount 21
		BGTextRect texts[textCount];
		GuiDropdown dropdowns[dropdownCount];

		StaticConfigLeftPanel()
		{
			int textIdx = 0;
			int ddIdx = 0;

			texts[textIdx++] = BGTextRect("Basic Filters", sf::FloatRect(330, 125, 0, 0), 64);
			texts[0].fontIdx = 1;
			texts[textIdx++] = BGTextRect("Starting Cart", sf::FloatRect(125, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Starting Flask", sf::FloatRect(330, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Rain", sf::FloatRect(535, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Starting Projectile", sf::FloatRect(175, 285, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Starting Bomb", sf::FloatRect(485, 285, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Lively Concoction", sf::FloatRect(330, 380, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Alchemic Precursor", sf::FloatRect(330, 455, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Ingredient ordering", sf::FloatRect(375, 530, 0, 0), 24);
			texts[textIdx++] = BGTextRect("Biome Modifiers", sf::FloatRect(330, 580, 0, 0), 48);

			int idx = 0;
			const char** cartTypes = (const char**)malloc(sizeof(const char*) * 4);
			cartTypes[idx++] = "Minecart";
			cartTypes[idx++] = "Wooden Cart";
			cartTypes[idx++] = "Skateboard";
			cartTypes[idx++] = "Any";
			dropdowns[ddIdx++] = GuiDropdown(4, cartTypes, sf::FloatRect(25, 220, 200, 800), 3);

			const char** startingFlaskNames = (const char**)malloc(sizeof(const char*) * _startingMaterialCount);
			for (int i = 0; i < _startingMaterialCount; i++)
				startingFlaskNames[i] = MaterialNames[_potionStarting[i]];
			dropdowns[ddIdx++] = GuiDropdown(_startingMaterialCount, startingFlaskNames, sf::FloatRect(230, 220, 200, 800), _startingMaterialCount - 1);

			const char** rainNames = (const char**)malloc(sizeof(const char*) * (_rainCount + 1));
			for (int i = 0; i < _rainCount; i++)
				rainNames[i] = MaterialNames[_rainMaterials[i]];
			rainNames[_rainCount] = MaterialNames[MATERIAL_NONE];
			dropdowns[ddIdx++] = GuiDropdown(_rainCount + 1, rainNames, sf::FloatRect(435, 220, 200, 800), _rainCount);

			const char** startingProjNames = (const char**)malloc(sizeof(const char*) * _startingProjectileCount);
			for (int i = 0; i < _startingProjectileCount; i++)
				startingProjNames[i] = SpellNames[_startingProjectiles[i]];
			dropdowns[ddIdx++] = GuiDropdown(_startingProjectileCount, startingProjNames, sf::FloatRect(25, 300, 302.5f, 800), _startingProjectileCount - 1);

			const char** startingBombNames = (const char**)malloc(sizeof(const char*) * _startingBombCount);
			for (int i = 0; i < _startingBombCount; i++)
				startingBombNames[i] = SpellNames[_startingBombs[i]];
			dropdowns[ddIdx++] = GuiDropdown(_startingBombCount, startingBombNames, sf::FloatRect(332.5f, 300, 302.5f, 800), _startingBombCount - 1);

			const char** alchemyIngredients = (const char**)malloc(sizeof(const char*) * (_alchemyLiquidCount + _alchemySolidCount + 1));
			for (int i = 0; i < _alchemyLiquidCount; i++)
				alchemyIngredients[i] = MaterialNames[_alchemyLiquids[i]];
			for (int i = 0; i < _alchemySolidCount; i++)
				alchemyIngredients[_alchemyLiquidCount + i] = MaterialNames[_alchemySolids[i]];
			alchemyIngredients[_alchemyLiquidCount + _alchemySolidCount] = MaterialNames[MATERIAL_NONE];

			//LC
			dropdowns[ddIdx++] = GuiDropdown(_alchemyLiquidCount + _alchemySolidCount + 1, alchemyIngredients, sf::FloatRect(25, 395, 200, 800), _alchemyLiquidCount + _alchemySolidCount);
			dropdowns[ddIdx++] = GuiDropdown(_alchemyLiquidCount + _alchemySolidCount + 1, alchemyIngredients, sf::FloatRect(230, 395, 200, 800), _alchemyLiquidCount + _alchemySolidCount);
			dropdowns[ddIdx++] = GuiDropdown(_alchemyLiquidCount + _alchemySolidCount + 1, alchemyIngredients, sf::FloatRect(435, 395, 200, 800), _alchemyLiquidCount + _alchemySolidCount);

			//AP
			dropdowns[ddIdx++] = GuiDropdown(_alchemyLiquidCount + _alchemySolidCount + 1, alchemyIngredients, sf::FloatRect(25, 470, 200, 800), _alchemyLiquidCount + _alchemySolidCount);
			dropdowns[ddIdx++] = GuiDropdown(_alchemyLiquidCount + _alchemySolidCount + 1, alchemyIngredients, sf::FloatRect(230, 470, 200, 800), _alchemyLiquidCount + _alchemySolidCount);
			dropdowns[ddIdx++] = GuiDropdown(_alchemyLiquidCount + _alchemySolidCount + 1, alchemyIngredients, sf::FloatRect(435, 470, 200, 800), _alchemyLiquidCount + _alchemySolidCount);

			idx = 0;
			const char** alchemyOrderings = (const char**)malloc(sizeof(const char*) * 3);
			alchemyOrderings[idx++] = "Exact Ordering";
			alchemyOrderings[idx++] = "Only Consumed";
			alchemyOrderings[idx++] = "Unordered";
			dropdowns[ddIdx++] = GuiDropdown(3, alchemyOrderings, sf::FloatRect(435, 515, 200, 800), 2);
			dropdowns[ddIdx - 1].SetEntryHeight(30);

			for (int biome = 0; biome < 9; biome++)
			{
				const char** modifierNames = (const char**)malloc(sizeof(const char*) * _allBMCounts[biome]);
				for (int i = 0; i < _allBMCounts[biome]; i++)
					modifierNames[i] = BiomeModifierNames[_allBMLists[biome][i]];
				dropdowns[ddIdx++] = GuiDropdown(_allBMCounts[biome], modifierNames, sf::FloatRect(232.5f, 605 + 50 * biome, 302.5f, 800), _allBMCounts[biome] - 1);
			}
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			for (int i = 0; i < textCount; i++) texts[i].Render();
			for(int i = 0; i < 9; i++)
				AlignedTextRect(BiomeNames[_allBMBiomes[i]], sf::FloatRect(30, 605 + 50 * i, 182.5f, 40), 24, sf::Color::White, 0, 2, 0).Render();

			for (int i = 0; i < dropdownCount; i++) dropdowns[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			for (int i = 0; i < dropdownCount; i++) if(dropdowns[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
#undef dropdownCount
#undef textCount
	};
	struct StaticConfigCenterPanel : GuiObject
	{
		struct FungalShiftRow : GuiObject
		{
			OutlinedRect bg;
			InputRect lowerBound;
			InputRect upperBound;
			GuiDropdown source;
			GuiDropdown dest;
			FungalShiftRow() = default;
			FungalShiftRow(float top, const char** sourceNames, const char** destNames)
			{
				bg = OutlinedRect(sf::FloatRect(660, top, 600, 50), 1, sf::Color(30, 30, 30), sf::Color(20, 20, 20));
				lowerBound = InputRect(sf::FloatRect(670, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "0", sf::Color(30, 30, 30));
				upperBound = InputRect(sf::FloatRect(820, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "20", sf::Color(30, 30, 30));
				source = GuiDropdown((int)_fungalMaterialsFromCount, sourceNames, sf::FloatRect(895, top + 5, 160, 600), _fungalMaterialsFromCount - 1);
				dest = GuiDropdown((int)_fungalMaterialsToCount, destNames, sf::FloatRect(1095, top + 5, 160, 600), _fungalMaterialsToCount - 1);
			}
			void Render()
			{
				bg.Render();
				lowerBound.Render();
				upperBound.Render();
				DrawTextCentered("< Shift # <", sf::Vector2f(775, bg.mRect.rect.top + 25), 24, sf::Color::White, 0);
				source.Render();
				dest.Render();
				DrawTextCentered("=>", sf::Vector2f(1075, bg.mRect.rect.top + 25), 36, sf::Color::White, 0);
			}
			bool HandleClick(sf::Vector2f position)
			{
				if (lowerBound.HandleClick(position)) return true;
				if (upperBound.HandleClick(position)) return true;
				if (source.HandleClick(position)) return true;
				if (dest.HandleClick(position)) return true;
				return false;
			}
			void HandleMouse(sf::Vector2f position)
			{

			}
		};

		sf::FloatRect rect = sf::FloatRect(650, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);
		BGTextRect title;
		BGTextRect addFilter;
		BGTextRect removeFilter;

		int fungalShiftRowCount = 0;
		FungalShiftRow rows[12];

		const char** sourceNames;
		const char** destNames;

		StaticConfigCenterPanel()
		{
			title = BGTextRect("Fungal Shifts", sf::FloatRect(960, 125, 0, 0), 64);
			title.fontIdx = 1;

			addFilter = BGTextRect("Add Filter", sf::FloatRect(660, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));
			removeFilter = BGTextRect("Remove Filter", sf::FloatRect(1060, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));

			sourceNames = (const char**)malloc(sizeof(const char*) * _fungalMaterialsFromCount);
			for (int i = 0; i < _fungalMaterialsFromCount; i++)
				if (_fungalMaterialsFrom[i] == SS_FLASK) sourceNames[i] = "Flask";
				else sourceNames[i] = MaterialNames[_fungalMaterialsFrom[i]];

			destNames = (const char**)malloc(sizeof(const char*) * _fungalMaterialsToCount);
			for (int i = 0; i < _fungalMaterialsToCount; i++)
				if (_fungalMaterialsTo[i] == SD_FLASK) destNames[i] = "Flask";
				else destNames[i] = MaterialNames[_fungalMaterialsTo[i]];
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			title.Render();
			addFilter.Render();
			removeFilter.Render();
			for(int i = 0; i < fungalShiftRowCount; i++) rows[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			if (addFilter.mRect.rect.contains(position))
			{
				if (fungalShiftRowCount < 12)
				{
					rows[fungalShiftRowCount] = FungalShiftRow(250 + 60 * fungalShiftRowCount, sourceNames, destNames);
					fungalShiftRowCount++;
				}
				return true;
			}
			if (removeFilter.mRect.rect.contains(position))
			{
				if(fungalShiftRowCount > 0) fungalShiftRowCount--;
				return true;
			}
			for (int i = 0; i < fungalShiftRowCount; i++)
				if (rows[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
	};
	struct StaticConfigRightPanel : GuiObject
	{
		struct PerkRow : GuiObject
		{
			OutlinedRect bg;
			InputRect lowerBound;
			InputRect upperBound;
			BGTextRect lotteryLabel;
			GuiCheckbox lotteryBox;
			GuiDropdown perk;
			PerkRow() = default;
			PerkRow(float top, const char** perkNames)
			{
				bg = OutlinedRect(sf::FloatRect(1290, top, 600, 50), 1, sf::Color(30, 30, 30), sf::Color(20, 20, 20));
				lowerBound = InputRect(sf::FloatRect(1300, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "0", sf::Color(30, 30, 30));
				upperBound = InputRect(sf::FloatRect(1450, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "100", sf::Color(30, 30, 30));
				lotteryLabel = BGTextRect("Lottery", sf::FloatRect(1515, top + 3, 50, 20), 18);
				lotteryBox = GuiCheckbox(sf::FloatRect(1530, top + 20, 20, 20), sf::Color(40, 40, 40), false);
				perk = GuiDropdown(_perkCount, perkNames, sf::FloatRect(1565, top + 5, 320, 600), _perkCount - 1);
			}
			void Render()
			{
				bg.Render();
				lowerBound.Render();
				upperBound.Render();
				DrawTextCentered("< Perk # <", sf::Vector2f(1405, bg.mRect.rect.top + 25), 24, sf::Color::White, 0);
				lotteryLabel.Render();
				lotteryBox.Render();
				perk.Render();
			}
			bool HandleClick(sf::Vector2f position)
			{
				if (lowerBound.HandleClick(position)) return true;
				if (upperBound.HandleClick(position)) return true;
				if (perk.HandleClick(position)) return true;
				if (lotteryBox.HandleClick(position)) return true;
				return false;
			}
			void HandleMouse(sf::Vector2f position)
			{

			}
		};

		sf::FloatRect rect = sf::FloatRect(1280, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);
		BGTextRect title;
		BGTextRect addFilter;
		BGTextRect removeFilter;

		int perkRowCount = 0;
		PerkRow rows[12];

		const char** perkNames;

		StaticConfigRightPanel()
		{
			title = BGTextRect("Perks", sf::FloatRect(1590, 125, 0, 0), 64);
			title.fontIdx = 1;

			addFilter = BGTextRect("Add Filter", sf::FloatRect(1290, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));
			removeFilter = BGTextRect("Remove Filter", sf::FloatRect(1690, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));

			perkNames = (const char**)malloc(sizeof(const char*) * _perkCount);
			for (int i = 0; i < _perkCount; i++)
				perkNames[i] = PerkNames[(i + _perkCount + 1) % _perkCount];
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			title.Render();
			addFilter.Render();
			removeFilter.Render();
			for (int i = 0; i < perkRowCount; i++) rows[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			if (addFilter.mRect.rect.contains(position))
			{
				if (perkRowCount < 12)
				{
					rows[perkRowCount] = PerkRow(250 + 60 * perkRowCount, perkNames);
					perkRowCount++;
				}
				return true;
			}
			if (removeFilter.mRect.rect.contains(position))
			{
				if (perkRowCount > 0) perkRowCount--;
				return true;
			}
			for (int i = 0; i < perkRowCount; i++)
				if (rows[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
	};

	StaticConfigLeftPanel leftPanel;
	StaticConfigCenterPanel centerPanel;
	StaticConfigRightPanel rightPanel;

	StaticConfigTab()
	{

	}

	void Render()
	{
		leftPanel.Render();
		centerPanel.Render();
		rightPanel.Render();
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (leftPanel.HandleClick(position)) return true;
		if (centerPanel.HandleClick(position)) return true;
		if (rightPanel.HandleClick(position)) return true;
		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

struct WorldConfigTab : GuiObject
{
	struct WorldConfigLeftPanel : GuiObject
	{
		sf::FloatRect rect = sf::FloatRect(20, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);

#define textCount 13
#define checkboxCount 12
#define dropdownCount 1
		BGTextRect texts[textCount];
		GuiCheckbox checkboxes[checkboxCount];
		GuiDropdown dropdowns[dropdownCount];

		WorldConfigLeftPanel()
		{
			int textIdx = 0;
			int chkboxIdx = 0;
			int ddIdx = 0;

			texts[textIdx++] = BGTextRect("Search Scope", sf::FloatRect(330, 125, 0, 0), 64);
			texts[0].fontIdx = 1;
			texts[textIdx++] = BGTextRect("Shop Spells", sf::FloatRect(125, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Shop Wands", sf::FloatRect(330, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Pacifist Chest", sf::FloatRect(535, 205, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 220, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 220, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 220, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Chests", sf::FloatRect(125, 275, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Item Pedestals", sf::FloatRect(330, 275, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Wand Altars", sf::FloatRect(535, 275, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 290, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 290, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 290, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Pixel Scenes", sf::FloatRect(125, 345, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Enemies", sf::FloatRect(330, 345, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Eye Rooms", sf::FloatRect(535, 345, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 360, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 360, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 360, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Wand Stats", sf::FloatRect(125, 415, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Spells", sf::FloatRect(330, 415, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Flask Contents", sf::FloatRect(535, 415, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 430, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 430, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 430, 30, 30), sf::Color(40, 40, 40), false);
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			for (int i = 0; i < textCount; i++) texts[i].Render();
			for (int i = 0; i < checkboxCount; i++) checkboxes[i].Render();
			//for (int i = 0; i < 9; i++)
			//	AlignedTextRect(BiomeNames[_allBMBiomes[i]], sf::FloatRect(30, 605 + 50 * i, 182.5f, 40), 24, sf::Color::White, 0, 2, 0).Render();

			//for (int i = 0; i < dropdownCount; i++) dropdowns[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			for (int i = 0; i < checkboxCount; i++) if (checkboxes[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
#undef dropdownCount
#undef checkboxCount
#undef textCount
	};
	struct WorldConfigCenterPanel : GuiObject
	{
		sf::FloatRect rect = sf::FloatRect(650, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);

#define textCount 22
#define checkboxCount 21
#define dropdownCount 1
		BGTextRect texts[textCount];
		GuiCheckbox checkboxes[checkboxCount];
		GuiDropdown dropdowns[dropdownCount];

		WorldConfigCenterPanel()
		{
			int textIdx = 0;
			int chkboxIdx = 0;
			int ddIdx = 0;

			texts[textIdx++] = BGTextRect("Biomes", sf::FloatRect(960, 125, 0, 0), 64);
			texts[0].fontIdx = 1;
			texts[textIdx++] = BGTextRect("Abandoned Lab", sf::FloatRect(755, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Mines", sf::FloatRect(960, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Sky", sf::FloatRect(1165, 205, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 220, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 220, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 220, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Fungal Caverns", sf::FloatRect(755, 275, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Coal Pits", sf::FloatRect(960, 275, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Pyramid", sf::FloatRect(1165, 275, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 290, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 290, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 290, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Magical Temple", sf::FloatRect(755, 345, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Snowy Depths", sf::FloatRect(960, 345, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Wandmart", sf::FloatRect(1165, 345, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 360, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 360, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 360, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Frozen Vault", sf::FloatRect(755, 415, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Hiisi Base", sf::FloatRect(960, 415, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Sandcave", sf::FloatRect(1165, 415, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 430, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 430, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 430, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Lukki Lair", sf::FloatRect(755, 485, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Jungle", sf::FloatRect(960, 485, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Tower", sf::FloatRect(1165, 485, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 500, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 500, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 500, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Snow Chasm", sf::FloatRect(755, 555, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Vault", sf::FloatRect(960, 555, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Power Plant", sf::FloatRect(1165, 555, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 570, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 570, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 570, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Hell", sf::FloatRect(755, 625, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Temple", sf::FloatRect(960, 625, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Wizard's Den", sf::FloatRect(1165, 625, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(740, 640, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(945, 640, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(1150, 640, 30, 30), sf::Color(40, 40, 40), false);
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			for (int i = 0; i < textCount; i++) texts[i].Render();
			for (int i = 0; i < checkboxCount; i++) checkboxes[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			for (int i = 0; i < checkboxCount; i++) if (checkboxes[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
#undef dropdownCount
#undef checkboxCount
#undef textCount
	};
	struct WorldConfigRightPanel : GuiObject
	{
		struct ItemRow : GuiObject
		{
			OutlinedRect bg;
			InputRect count;
			GuiDropdown perk;
			ItemRow() = default;
			ItemRow(float top, const char** perkNames)
			{
				bg = OutlinedRect(sf::FloatRect(1290, top, 600, 50), 1, sf::Color(30, 30, 30), sf::Color(20, 20, 20));
				count = InputRect(sf::FloatRect(1300, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "0", sf::Color(30, 30, 30));
				perk = GuiDropdown(_perkCount, perkNames, sf::FloatRect(1565, top + 5, 320, 600), _perkCount - 1);
			}
			void Render()
			{
				bg.Render();
				count.Render();
				perk.Render();
			}
			bool HandleClick(sf::Vector2f position)
			{
				if (count.HandleClick(position)) return true;
				if (perk.HandleClick(position)) return true;
				return false;
			}
			void HandleMouse(sf::Vector2f position)
			{

			}
		};

		sf::FloatRect rect = sf::FloatRect(1280, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);
		BGTextRect title;
		BGTextRect addFilter;
		BGTextRect removeFilter;

		int perkRowCount = 0;
		ItemRow rows[12];

		const char** perkNames;

		WorldConfigRightPanel()
		{
			title = BGTextRect("Item Filters", sf::FloatRect(1590, 125, 0, 0), 64);
			title.fontIdx = 1;

			addFilter = BGTextRect("Add Filter", sf::FloatRect(1290, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));
			removeFilter = BGTextRect("Remove Filter", sf::FloatRect(1690, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));

			perkNames = (const char**)malloc(sizeof(const char*) * _perkCount);
			for (int i = 0; i < _perkCount; i++)
				perkNames[i] = PerkNames[(i + _perkCount + 1) % _perkCount];
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			title.Render();
			addFilter.Render();
			removeFilter.Render();
			for (int i = 0; i < perkRowCount; i++) rows[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			if (addFilter.mRect.rect.contains(position))
			{
				if (perkRowCount < 12)
				{
					rows[perkRowCount] = ItemRow(250 + 60 * perkRowCount, perkNames);
					perkRowCount++;
				}
				return true;
			}
			if (removeFilter.mRect.rect.contains(position))
			{
				if (perkRowCount > 0) perkRowCount--;
				return true;
			}
			for (int i = 0; i < perkRowCount; i++)
				if (rows[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
	};

	WorldConfigLeftPanel leftPanel;
	WorldConfigCenterPanel centerPanel;
	WorldConfigRightPanel rightPanel;

	WorldConfigTab()
	{

	}

	void Render()
	{
		leftPanel.Render();
		centerPanel.Render();
		rightPanel.Render();
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (leftPanel.HandleClick(position)) return true;
		if (centerPanel.HandleClick(position)) return true;
		if (rightPanel.HandleClick(position)) return true;
		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

struct FilterConfigTab : GuiObject
{
	struct FilterConfigLeftPanel : GuiObject
	{
		sf::FloatRect rect = sf::FloatRect(20, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);

#define textCount 13
#define checkboxCount 12
#define dropdownCount 1
		BGTextRect texts[textCount];
		GuiCheckbox checkboxes[checkboxCount];
		GuiDropdown dropdowns[dropdownCount];

		FilterConfigLeftPanel()
		{
			int textIdx = 0;
			int chkboxIdx = 0;
			int ddIdx = 0;

			texts[textIdx++] = BGTextRect("Search Scope", sf::FloatRect(330, 125, 0, 0), 64);
			texts[0].fontIdx = 1;
			texts[textIdx++] = BGTextRect("Shop Spells", sf::FloatRect(125, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Shop Wands", sf::FloatRect(330, 205, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Pacifist Chest", sf::FloatRect(535, 205, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 220, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 220, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 220, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Chests", sf::FloatRect(125, 275, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Item Pedestals", sf::FloatRect(330, 275, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Wand Altars", sf::FloatRect(535, 275, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 290, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 290, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 290, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Pixel Scenes", sf::FloatRect(125, 345, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Enemies", sf::FloatRect(330, 345, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Eye Rooms", sf::FloatRect(535, 345, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 360, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 360, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 360, 30, 30), sf::Color(40, 40, 40), false);

			texts[textIdx++] = BGTextRect("Wand Stats", sf::FloatRect(125, 415, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Spells", sf::FloatRect(330, 415, 0, 0), 36);
			texts[textIdx++] = BGTextRect("Flask Contents", sf::FloatRect(535, 415, 0, 0), 36);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(110, 430, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(315, 430, 30, 30), sf::Color(40, 40, 40), false);
			checkboxes[chkboxIdx++] = GuiCheckbox(sf::FloatRect(520, 430, 30, 30), sf::Color(40, 40, 40), false);
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			for (int i = 0; i < textCount; i++) texts[i].Render();
			for (int i = 0; i < checkboxCount; i++) checkboxes[i].Render();
			//for (int i = 0; i < 9; i++)
			//	AlignedTextRect(BiomeNames[_allBMBiomes[i]], sf::FloatRect(30, 605 + 50 * i, 182.5f, 40), 24, sf::Color::White, 0, 2, 0).Render();

			//for (int i = 0; i < dropdownCount; i++) dropdowns[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			for (int i = 0; i < checkboxCount; i++) if (checkboxes[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
#undef dropdownCount
#undef textCount
	};
	struct FilterConfigCenterPanel : GuiObject
	{
		struct FungalShiftRow : GuiObject
		{
			OutlinedRect bg;
			InputRect lowerBound;
			InputRect upperBound;
			GuiDropdown source;
			GuiDropdown dest;
			FungalShiftRow() = default;
			FungalShiftRow(float top, const char** sourceNames, const char** destNames)
			{
				bg = OutlinedRect(sf::FloatRect(660, top, 600, 50), 1, sf::Color(30, 30, 30), sf::Color(20, 20, 20));
				lowerBound = InputRect(sf::FloatRect(670, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "0", sf::Color(30, 30, 30));
				upperBound = InputRect(sf::FloatRect(820, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "20", sf::Color(30, 30, 30));
				source = GuiDropdown((int)_fungalMaterialsFromCount, sourceNames, sf::FloatRect(895, top + 5, 160, 600), _fungalMaterialsFromCount - 1);
				dest = GuiDropdown((int)_fungalMaterialsToCount, destNames, sf::FloatRect(1095, top + 5, 160, 600), _fungalMaterialsToCount - 1);
			}
			void Render()
			{
				bg.Render();
				lowerBound.Render();
				upperBound.Render();
				DrawTextCentered("< Shift # <", sf::Vector2f(775, bg.mRect.rect.top + 25), 24, sf::Color::White, 0);
				source.Render();
				dest.Render();
				DrawTextCentered("=>", sf::Vector2f(1075, bg.mRect.rect.top + 25), 36, sf::Color::White, 0);
			}
			bool HandleClick(sf::Vector2f position)
			{
				if (lowerBound.HandleClick(position)) return true;
				if (upperBound.HandleClick(position)) return true;
				if (source.HandleClick(position)) return true;
				if (dest.HandleClick(position)) return true;
				return false;
			}
			void HandleMouse(sf::Vector2f position)
			{

			}
		};

		sf::FloatRect rect = sf::FloatRect(650, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);
		BGTextRect title;
		BGTextRect addFilter;
		BGTextRect removeFilter;

		int fungalShiftRowCount = 0;
		FungalShiftRow rows[12];

		const char** sourceNames;
		const char** destNames;

		FilterConfigCenterPanel()
		{
			title = BGTextRect("Fungal Shifts", sf::FloatRect(960, 125, 0, 0), 64);
			title.fontIdx = 1;

			addFilter = BGTextRect("Add Filter", sf::FloatRect(660, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));
			removeFilter = BGTextRect("Remove Filter", sf::FloatRect(1060, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));

			sourceNames = (const char**)malloc(sizeof(const char*) * _fungalMaterialsFromCount);
			for (int i = 0; i < _fungalMaterialsFromCount; i++)
				if (_fungalMaterialsFrom[i] == SS_FLASK) sourceNames[i] = "Flask";
				else sourceNames[i] = MaterialNames[_fungalMaterialsFrom[i]];

			destNames = (const char**)malloc(sizeof(const char*) * _fungalMaterialsToCount);
			for (int i = 0; i < _fungalMaterialsToCount; i++)
				if (_fungalMaterialsTo[i] == SD_FLASK) destNames[i] = "Flask";
				else destNames[i] = MaterialNames[_fungalMaterialsTo[i]];
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			title.Render();
			addFilter.Render();
			removeFilter.Render();
			for (int i = 0; i < fungalShiftRowCount; i++) rows[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			if (addFilter.mRect.rect.contains(position))
			{
				if (fungalShiftRowCount < 12)
				{
					rows[fungalShiftRowCount] = FungalShiftRow(250 + 60 * fungalShiftRowCount, sourceNames, destNames);
					fungalShiftRowCount++;
				}
				return true;
			}
			if (removeFilter.mRect.rect.contains(position))
			{
				if (fungalShiftRowCount > 0) fungalShiftRowCount--;
				return true;
			}
			for (int i = 0; i < fungalShiftRowCount; i++)
				if (rows[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
	};
	struct FilterConfigRightPanel : GuiObject
	{
		struct PerkRow : GuiObject
		{
			OutlinedRect bg;
			InputRect lowerBound;
			InputRect upperBound;
			BGTextRect lotteryLabel;
			ColorRect lotteryBox;
			bool lottery = false;
			GuiDropdown perk;
			PerkRow() = default;
			PerkRow(float top, const char** perkNames)
			{
				bg = OutlinedRect(sf::FloatRect(1290, top, 600, 50), 1, sf::Color(30, 30, 30), sf::Color(20, 20, 20));
				lowerBound = InputRect(sf::FloatRect(1300, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "0", sf::Color(30, 30, 30));
				upperBound = InputRect(sf::FloatRect(1450, top + 8, 60, 34), TextInput::CHARSET_Numeric, 2, "100", sf::Color(30, 30, 30));
				lotteryLabel = BGTextRect("Lottery", sf::FloatRect(1515, top + 3, 50, 20), 18);
				lotteryBox = ColorRect(sf::FloatRect(1530, top + 20, 20, 20), sf::Color(40, 40, 40));
				perk = GuiDropdown(_perkCount, perkNames, sf::FloatRect(1565, top + 5, 320, 600), _perkCount - 1);
			}
			void Render()
			{
				bg.Render();
				lowerBound.Render();
				upperBound.Render();
				DrawTextCentered("< Perk # <", sf::Vector2f(1405, bg.mRect.rect.top + 25), 24, sf::Color::White, 0);
				lotteryLabel.Render();
				lotteryBox.Render();
				if (lottery)
					BGTextRect("X", lotteryBox.mRect.rect).Render();
				perk.Render();
			}
			bool HandleClick(sf::Vector2f position)
			{
				if (lowerBound.HandleClick(position)) return true;
				if (upperBound.HandleClick(position)) return true;
				if (perk.HandleClick(position)) return true;
				if (lotteryBox.mRect.rect.contains(position))
				{
					lottery = !lottery;
					return true;
				}
				return false;
			}
			void HandleMouse(sf::Vector2f position)
			{

			}
		};

		sf::FloatRect rect = sf::FloatRect(1280, 180, 620, 880);
		sf::Color bgColor = sf::Color(20, 20, 20);
		BGTextRect title;
		BGTextRect addFilter;
		BGTextRect removeFilter;

		int perkRowCount = 0;
		PerkRow rows[12];

		const char** perkNames;

		FilterConfigRightPanel()
		{
			title = BGTextRect("Perks", sf::FloatRect(1590, 125, 0, 0), 64);
			title.fontIdx = 1;

			addFilter = BGTextRect("Add Filter", sf::FloatRect(1290, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));
			removeFilter = BGTextRect("Remove Filter", sf::FloatRect(1690, 190, 200, 50), 36, sf::Color::White, sf::Color(30, 30, 30));

			perkNames = (const char**)malloc(sizeof(const char*) * _perkCount);
			for (int i = 0; i < _perkCount; i++)
				perkNames[i] = PerkNames[(i + _perkCount + 1) % _perkCount];
		}

		void Render()
		{
			ColorRect(rect, bgColor).Render();
			title.Render();
			addFilter.Render();
			removeFilter.Render();
			for (int i = 0; i < perkRowCount; i++) rows[i].Render();
		}

		bool HandleClick(sf::Vector2f position)
		{
			if (addFilter.mRect.rect.contains(position))
			{
				if (perkRowCount < 12)
				{
					rows[perkRowCount] = PerkRow(250 + 60 * perkRowCount, perkNames);
					perkRowCount++;
				}
				return true;
			}
			if (removeFilter.mRect.rect.contains(position))
			{
				if (perkRowCount > 0) perkRowCount--;
				return true;
			}
			for (int i = 0; i < perkRowCount; i++)
				if (rows[i].HandleClick(position)) return true;
			return false;
		}
		void HandleMouse(sf::Vector2f position) {}
	};

	FilterConfigLeftPanel leftPanel;
	FilterConfigCenterPanel centerPanel;
	FilterConfigRightPanel rightPanel;

	FilterConfigTab()
	{

	}

	void Render()
	{
		leftPanel.Render();
		centerPanel.Render();
		rightPanel.Render();
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (leftPanel.HandleClick(position)) return true;
		if (centerPanel.HandleClick(position)) return true;
		if (rightPanel.HandleClick(position)) return true;
		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

void CreateConfigsAndDispatch();
struct SearchConfigTab : GuiObject
{
	struct CPUPanel : GuiObject
	{
		sf::FloatRect rect = sf::FloatRect(960, 90, 930, 390);
		ColorRect bg;
		BGTextRect deviceName;
		
		#ifdef BACKEND_CPU
		//GuiCheckbox selected;
		//InputRect numThreads;
		//InputRect maxMemory;
		#else
		BGTextRect _disabled;
		#endif

		void GetProcessorName(char* buffer)
		{
			int CPUInfo[4] = { -1 };
			__cpuid(CPUInfo, 0x80000000);
			unsigned int nExIds = CPUInfo[0];

			memset(buffer, 0, sizeof(0x40));

			// Get the information associated with each extended ID.
			for (int i = 0x80000000; i <= nExIds; ++i)
			{
				__cpuid(CPUInfo, i);
				// Interpret CPU brand string.
				if (i == 0x80000002)
					memcpy(buffer, CPUInfo, sizeof(CPUInfo));
				else if (i == 0x80000003)
					memcpy(buffer + 16, CPUInfo, sizeof(CPUInfo));
				else if (i == 0x80000004)
					memcpy(buffer + 32, CPUInfo, sizeof(CPUInfo));
			}
		}

		CPUPanel()
		{
			bg = ColorRect(rect, sf::Color(20, 20, 20));
			char* cpuName = (char*)malloc(64);
			GetProcessorName(cpuName);
			deviceName = BGTextRect(cpuName, sf::FloatRect(960, 90, 930, 50), 48);
			#ifdef BACKEND_CPU

			#else
			_disabled = BGTextRect("DISABLED", rect, 128);
			#endif
		}

		void Render()
		{
			bg.Render();
			deviceName.Render();
			
			#ifdef BACKEND_CPU

			#else
			_disabled.Render();
			#endif
		}

		void HandleMouse(sf::Vector2f position)
		{

		}

		bool HandleClick(sf::Vector2f position)
		{
			#ifdef BACKEND_CPU

			#endif
			return false;
		}
	};

	struct GPUPanel : GuiObject
	{
		sf::FloatRect rect = sf::FloatRect(960, 490, 930, 460);
		ColorRect bg;
		BGTextRect deviceName;
		
		#ifdef BACKEND_CUDA
		InputRect numMultiprocessors;
		InputRect maxMemory;
		#else
		BGTextRect _disabled;
		#endif

		void GetProcessorName(char* buffer)
		{
			#ifdef BACKEND_CUDA
			int devCount;
			cudaGetDeviceCount(&devCount);
			if (devCount == 0)
			{
				sprintf(buffer, "No CUDA-capable GPU found!");
			}

			int device;
			cudaGetDevice(&device);

			cudaDeviceProp properties;
			checkCudaErrors(cudaGetDeviceProperties_v2(&properties, 0));

			sprintf(buffer, "GPU %i: %s", device, properties.name);
			#else
			sprintf(buffer, "CUDA driver disabled.");
			#endif
		}

		GPUPanel()
		{
			bg = ColorRect(rect, sf::Color(20, 20, 20));
			char* gpuName = (char*)malloc(64);
			GetProcessorName(gpuName);
			deviceName = BGTextRect(gpuName, sf::FloatRect(960, 490, 930, 50), 48);
			#ifdef BACKEND_CUDA
			numMultiprocessors = InputRect(
				AlignedTextRect("", sf::FloatRect(970, 600, 400, 50), 48, sf::Color::White, 0, 1, 0),
				TextInput::CHARSET_Numeric, 4, "32");
			maxMemory = InputRect(
				AlignedTextRect("", sf::FloatRect(970, 660, 400, 50), 48, sf::Color::White, 0, 1, 0),
				TextInput::CHARSET_Numeric, 8, "4096");
			#else
			_disabled = BGTextRect("DISABLED", rect, 128);
			#endif
		}

		void Render()
		{
			bg.Render();
			deviceName.Render();
			#ifdef BACKEND_CUDA
			//numMultiprocessors.Render();
			//maxMemory.Render();
			#else
			_disabled.Render();
			#endif
		}

		void HandleMouse(sf::Vector2f position)
		{

		}

		bool HandleClick(sf::Vector2f position)
		{
			#ifdef BACKEND_CUDA
			//if (numMultiprocessors.HandleClick(position)) return true;
			//if (maxMemory.HandleClick(position)) return true;
			#endif
			return false;
		}
	};

	ColorRect listBG;
	CPUPanel cpuPanel;
	GPUPanel gpuPanel;
	BGTextRect searchButton;
	BGTextRect abortButton;
	std::thread* searchThread = NULL;
	float progressFraction = 0;
	int elapsedMillis = 0;
	int updateCtr = 0;
	int updateInterval = 1000;

	int cachedHours = 0;
	int cachedMinutes = 0;
	int cachedSeconds = 0;

	bool searchDone = false;
	bool abortToggle = false;

	ColorRect progressBarBG;

	SearchConfigTab()
	{
		listBG = ColorRect(sf::FloatRect(20, 90, 930, 860), sf::Color(20, 20, 20));
		cpuPanel = CPUPanel();
		gpuPanel = GPUPanel();
		searchButton = BGTextRect("Search!", sf::FloatRect(30, 860, 450, 80), 48, sf::Color::White, sf::Color(40, 40, 40));
		abortButton = BGTextRect("Cancel", sf::FloatRect(490, 860, 450, 80), 48, sf::Color::White, sf::Color(180, 60, 60));
		progressBarBG = ColorRect(sf::FloatRect(20, 960, 1870, 100), sf::Color(40, 40, 40));
	}

	void Render()
	{
		if (searchDone)
		{
			searchDone = false;
			searchThread = NULL;
		}

		listBG.Render();
		cpuPanel.Render();
		gpuPanel.Render();
		searchButton.Render();
		abortButton.Render();
		progressBarBG.Render();
		ColorRect(sf::FloatRect(progressBarBG.mRect.rect.left, progressBarBG.mRect.rect.top, progressBarBG.mRect.rect.width * progressFraction, progressBarBG.mRect.rect.height), 
			sf::Color(20, 20, 200)).Render();
		if (elapsedMillis > updateCtr * updateInterval)
		{
			updateCtr++;
			int predictedDurationMillis = (double)elapsedMillis * (double)(1 - progressFraction) / (double)progressFraction;
			int predictedSeconds = predictedDurationMillis / 1000ULL;
			int predictedMinutes = predictedSeconds / 60U;
			int predictedHours = predictedMinutes / 60;
			cachedHours = predictedHours;
			cachedMinutes = predictedMinutes % 60;
			cachedSeconds = predictedSeconds % 60;
		}
		char buffer[50];
		if(elapsedMillis > 0) sprintf(buffer, "%2.3f%% (%02ih %02im %02is remaining)", progressFraction * 100, cachedHours, cachedMinutes, cachedSeconds);
		else sprintf(buffer, "%2.3f%%", progressFraction * 100);
		AlignedTextRect(buffer, progressBarBG.mRect.rect, 36, sf::Color::White, 0, 1, 1).Render();
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (searchButton.mRect.rect.contains(position) && searchThread == NULL)
		{
			std::thread t = std::thread(CreateConfigsAndDispatch);
			searchThread = &t;
			t.detach();
			return true;
		}
		if (abortButton.mRect.rect.contains(position) && searchThread != NULL)
		{
			abortToggle = true;
			return true;
		}
		if (cpuPanel.HandleClick(position)) return true;
		if (gpuPanel.HandleClick(position)) return true;
		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

constexpr int MAX_OUTPUT_SEEDS = 10000;
struct OutputTab : GuiObject
{
	ColorRect listBG;
	ColorRect infoBG;
	BGTextRect seedCount;
	BGTextRect clearButton;
	GuiScrollList seedList;
	
	std::mutex* vectorLock;
	char** seeds;
	char** seedInfos;

	OutputTab()
	{
		vectorLock = new std::mutex;

		seeds = (char**)malloc(sizeof(char*) * MAX_OUTPUT_SEEDS);
		seedInfos = (char**)malloc(sizeof(char*) * MAX_OUTPUT_SEEDS);

		listBG = ColorRect(sf::FloatRect(20, 90, 620, 970), sf::Color(20, 20, 20));
		infoBG = ColorRect(sf::FloatRect(660, 90, 1240, 970), sf::Color(20, 20, 20));
		seedList = GuiScrollList(0, (const char**)seeds, sf::FloatRect(30, 160, 600, 890), 0);

		char* seedCountBuffer = (char*)malloc(40);
		strcpy(seedCountBuffer, "0 seeds");
		seedCount = BGTextRect(seedCountBuffer, sf::FloatRect(30, 100, 490, 50), 48, sf::Color::White, sf::Color(40, 40, 40));
		clearButton = BGTextRect("Clear", sf::FloatRect(530, 100, 100, 50), 36, sf::Color::White, sf::Color(180, 60, 60));
	}

	void Render()
	{
		listBG.Render();
		infoBG.Render();
		seedCount.Render();
		clearButton.Render();
		vectorLock->lock();
		seedList.Render();

		AlignedTextRect("Click me to copy text!", sf::FloatRect(680, 100, 1200, 50), 36, sf::Color::White, 0, 1, 1).Render();
		if (seedList.selectedElement < seedList.numElements)
		{
			AlignedTextRect(seedInfos[seedList.selectedElement], sf::FloatRect(680, 160, 1200, 880), 36, sf::Color::White, 0, 0, 0).Render();
		}

		vectorLock->unlock();
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (clearButton.mRect.rect.contains(position))
		{
			vectorLock->lock();
			for (int i = 0; i < seedList.numElements; i++)
			{
				free(seeds[i]); free(seedInfos[i]);
			}
			seedList.numElements = 0;
			seedList.scrollDistance = 0;
			strcpy((char*)seedCount.text, "0 seeds");
			vectorLock->unlock();
			return true;
		}
		if(seedList.HandleClick(position)) return true;
		if (infoBG.mRect.rect.contains(position) && seedList.selectedElement < seedList.numElements)
		{
			sf::Clipboard::setString(sf::String(seedInfos[seedList.selectedElement]));
		}
		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{
		seedList.HandleMouse(position);
	}
};

struct SearchGui : GuiObject
{
	BGTextRect close;

	TabSelector tabs;
	StaticConfigTab staticConfig;
	WorldConfigTab worldConfig;
	FilterConfigTab filterConfig;
	SearchConfigTab searchConfig;
	OutputTab output;

	SearchGui()
	{
		close = BGTextRect("X", sf::FloatRect(1880, 10, 30, 30), 36, sf::Color::White, sf::Color(180, 10, 10));

		tabs = TabSelector();
		staticConfig = StaticConfigTab();
		worldConfig = WorldConfigTab();
		filterConfig = FilterConfigTab();
		searchConfig = SearchConfigTab();
		output = OutputTab();
	}

	void Render()
	{
		close.Render();

		tabs.Render();
		if(tabs.selectedTab == 0)
			staticConfig.Render();
		else if (tabs.selectedTab == 1)
			worldConfig.Render();
		else if (tabs.selectedTab == 2)
			filterConfig.Render();
		else if (tabs.selectedTab == 3)
			searchConfig.Render();
		else
			output.Render();
	}

	bool HandleClick(sf::Vector2f clickPos)
	{
		if (close.mRect.rect.contains(clickPos)) exit(0);

		if (tabs.HandleClick(clickPos)) return true;
		if (tabs.selectedTab == 0)
			if (staticConfig.HandleClick(clickPos)) return true;
		if (tabs.selectedTab == 1)
			if (worldConfig.HandleClick(clickPos)) return true;
		if (tabs.selectedTab == 2)
			if (filterConfig.HandleClick(clickPos)) return true;
		if (tabs.selectedTab == 3)
			if (searchConfig.HandleClick(clickPos)) return true;
		if (tabs.selectedTab == 4)
			if (output.HandleClick(clickPos)) return true;

		return false;
	}

	void HandleMouse(sf::Vector2f clickPos)
	{
		tabs.HandleMouse(clickPos);
		if (tabs.selectedTab == 0)
			staticConfig.HandleMouse(clickPos);
		else if (tabs.selectedTab == 1)
			worldConfig.HandleMouse(clickPos);
		else if (tabs.selectedTab == 2)
			filterConfig.HandleMouse(clickPos);
		else if (tabs.selectedTab == 3)
			searchConfig.HandleMouse(clickPos);
		else
			output.HandleMouse(clickPos);
	}
};

void AppendOutput(char* seedNum, char* seedInfo)
{
	sfmlState->gui->output.vectorLock->lock();
	if (sfmlState->gui->output.seedList.numElements == MAX_OUTPUT_SEEDS)
	{
		sfmlState->gui->output.vectorLock->unlock();
		return;
	}
	sfmlState->gui->output.seeds[sfmlState->gui->output.seedList.numElements] = seedNum;
	sfmlState->gui->output.seedInfos[sfmlState->gui->output.seedList.numElements] = seedInfo;
	sfmlState->gui->output.seedList.numElements++;
	char buffer[40];
	sprintf(buffer, "%i seeds", sfmlState->gui->output.seedList.numElements);
	strcpy((char*)sfmlState->gui->output.seedCount.text, buffer);
	sfmlState->gui->output.vectorLock->unlock();
}