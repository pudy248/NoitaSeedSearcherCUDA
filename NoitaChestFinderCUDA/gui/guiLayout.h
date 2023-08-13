#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../misc/pngutils.h"
#include "guiPrimitives.h"
#include "guiIntermediates.h"

struct TabSelector : GuiObject
{
	int selectedTab;
	BGTextRect staticConfig;
	BGTextRect worldConfig;
	BGTextRect searchConfig;
	BGTextRect output;

	TabSelector()
	{
		sf::Color textCol = sf::Color::White;
		sf::Color bgCol = sf::Color(20, 20, 20);
		selectedTab = 0;
		staticConfig = BGTextRect("static tab", sf::FloatRect(20, 20, 110, 50), 36, textCol, bgCol);
		worldConfig = BGTextRect("world tab", sf::FloatRect(140, 20, 110, 50), 36, textCol, bgCol);
		searchConfig = BGTextRect("search tab", sf::FloatRect(260, 20, 110, 50), 36, textCol, bgCol);
		output = BGTextRect("output tab", sf::FloatRect(380, 20, 110, 50), 36, textCol, bgCol);
	}

	void Render()
	{
		staticConfig.Render();
		worldConfig.Render();
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
		setColor(searchConfig, selectedTab == 2, searchConfig.mRect.containedMouseLastFrame);
		setColor(output, selectedTab == 3, output.mRect.containedMouseLastFrame);

	}

	void HandleClick(sf::Vector2f position)
	{
		if (staticConfig.mRect.rect.contains(position)) selectedTab = 0;
		else if (worldConfig.mRect.rect.contains(position)) selectedTab = 1;
		else if (searchConfig.mRect.rect.contains(position)) selectedTab = 2;
		else if (output.mRect.rect.contains(position)) selectedTab = 3;

		CalculateColors();
	}

	void HandleMouse(sf::Vector2f position)
	{
		staticConfig.mRect.HandleMouse(position);
		worldConfig.mRect.HandleMouse(position);
		searchConfig.mRect.HandleMouse(position);
		output.mRect.HandleMouse(position);

		CalculateColors();
	}
};

struct StaticConfigTab : GuiObject
{
	ColorRect bg;

	StaticConfigTab()
	{
		bg = ColorRect(sf::FloatRect(20, 90, 1880, 970), sf::Color(30, 30, 30));
	}

	void Render()
	{
		bg.Render();
	}

	void HandleClick(sf::Vector2f position)
	{

	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

struct WorldConfigTab : GuiObject
{
	AlignedTextRect bg;
	InputRect inputText;

	WorldConfigTab()
	{
		bg = AlignedTextRect("WORLD CONFIG!", sf::FloatRect(100, 100, 500, 200), 128, sf::Color::White, 1, 0, 0);
		inputText = InputRect(sf::FloatRect(100, 400, 300, 100), TextInput::CHARSET_Full, 65535, "");
	}

	void Render()
	{
		bg.Render();
		DrawRectOutline(inputText.mRect.rect, 3, sf::Color(96, 0, 0), sf::Color::White);
		inputText.Render();
	}

	void HandleClick(sf::Vector2f position)
	{
		inputText.HandleClick(position);
	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

struct SearchConfigTab : GuiObject
{
	AlignedTextRect bg;
	GuiDropdown dropdown;

	SearchConfigTab()
	{
		bg = AlignedTextRect("SEARCH CONFIG!", sf::FloatRect(100, 100, 500, 200), 128, sf::Color::White, 1, 0, 0);

		dropdown.openRect = sf::FloatRect(100, 100, 500, 500);

		dropdown.numElements = 20;
		char buffer[4];
		char** elements = (char**)malloc(sizeof(char*) * dropdown.numElements);
		for (int i = 0; i < dropdown.numElements; i++)
		{
			char* element = (char*)malloc(20);
			strcpy(element, "ELEMENT ");
			strcpy(element + 8, itoa(i, buffer, 10));
			elements[i] = element;
		}

		dropdown.elements = (const char**)elements;
		dropdown.textSize = 36;
	}

	void Render()
	{
		//bg.Render();
		dropdown.Render();
	}

	void HandleClick(sf::Vector2f position)
	{
		dropdown.HandleClick(position);
	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

struct OutputTab : GuiObject
{
	AlignedTextRect bg;

	OutputTab()
	{
		bg = AlignedTextRect("OUTPUT!", sf::FloatRect(100, 100, 500, 200), 128, sf::Color::White, 1, 0, 0);
	}

	void Render()
	{
		bg.Render();
	}

	void HandleClick(sf::Vector2f position)
	{

	}

	void HandleMouse(sf::Vector2f position)
	{

	}
};

struct SearchGui : GuiObject
{
	TabSelector tabs;
	StaticConfigTab staticConfig;
	WorldConfigTab worldConfig;
	SearchConfigTab searchConfig;
	OutputTab output;

	SearchGui()
	{
		tabs = TabSelector();
		staticConfig = StaticConfigTab();
		worldConfig = WorldConfigTab();
		searchConfig = SearchConfigTab();
		output = OutputTab();
	}

	void Render()
	{
		tabs.Render();
		if(tabs.selectedTab == 0)
			staticConfig.Render();
		else if (tabs.selectedTab == 1)
			worldConfig.Render();
		else if (tabs.selectedTab == 2)
			searchConfig.Render();
		else
			output.Render();
	}

	void HandleClick(sf::Vector2f clickPos)
	{
		tabs.HandleClick(clickPos);
		if (tabs.selectedTab == 0)
			staticConfig.HandleClick(clickPos);
		else if (tabs.selectedTab == 1)
			worldConfig.HandleClick(clickPos);
		else if (tabs.selectedTab == 2)
			searchConfig.HandleClick(clickPos);
		else
			output.HandleClick(clickPos);
	}

	void HandleMouse(sf::Vector2f clickPos)
	{
		tabs.HandleMouse(clickPos);
		if (tabs.selectedTab == 0)
			staticConfig.HandleMouse(clickPos);
		else if (tabs.selectedTab == 1)
			worldConfig.HandleMouse(clickPos);
		else if (tabs.selectedTab == 2)
			searchConfig.HandleMouse(clickPos);
		else
			output.HandleMouse(clickPos);
	}
};
