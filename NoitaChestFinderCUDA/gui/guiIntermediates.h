#pragma once

#include "../structs/primitives.h"
#include "../misc/pngutils.h"
#include "guiPrimitives.h"

struct GuiObject
{
	GuiObject() = default;
	virtual void Render() = 0;
	virtual bool HandleClick(sf::Vector2f position) = 0;
	virtual void HandleMouse(sf::Vector2f position) = 0;
};

struct GuiCheckbox : GuiObject
{
	BGTextRect box;
	bool enabled;

	GuiCheckbox() = default;
	GuiCheckbox(sf::FloatRect rect, sf::Color bgColor, bool defaultVal)
	{
		box = BGTextRect("", rect, rect.height, sf::Color::White, bgColor);
		if (defaultVal) box.text = "X";
		enabled = defaultVal;
	}

	void Render()
	{
		box.Render();
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (!box.mRect.interactable) return false;
		if (!box.mRect.Captures(position)) return false;
		enabled = !enabled;
		if (enabled) box.text = "X"; else box.text = "";
		return true;
	}

	void HandleMouse(sf::Vector2f position) {}
};

struct GuiScrollList : GuiObject
{
	MouseRect mRect;
	float entryHeight = 40;
	float scrollbarWidth = 5;
	float textSize = 30;

	sf::Color backgroundColor = sf::Color(30, 30, 30);

	int numElements = 0;
	int selectedElement = 0;
	float scrollDistance = 0;
	const char** elements;

	GuiScrollList() = default;
	GuiScrollList(int _numElements, const char** _elements, sf::FloatRect _rect, int _selected)
	{
		mRect.rect = _rect;
		numElements = _numElements;
		elements = _elements;
		selectedElement = _selected;
	}

	void Render()
	{
		sf::View dropdownView;
		dropdownView.reset(sf::FloatRect(sf::Vector2f(), sf::Vector2f(mRect.rect.width, mRect.rect.height)));
		dropdownView.setViewport(sf::FloatRect(mRect.rect.left / 1920, mRect.rect.top / 1080,
			mRect.rect.width / 1920, mRect.rect.height / 1080));
		sfmlState->window.setView(dropdownView);

		float heightDiff = fmaxf(0, numElements * entryHeight - mRect.rect.height);
		float top = 0;
		for (int i = 0; i < numElements; i++)
		{
			BGTextRect element(elements[i], sf::FloatRect(0, top - scrollDistance, mRect.rect.width, entryHeight), 24, sf::Color::White, backgroundColor);
			element.Render();
			top += entryHeight;
		}

		//Scrollbar
		if (mRect.rect.height < numElements * entryHeight)
		{
			float scrollbarHeight = mRect.rect.height * (mRect.rect.height / (numElements * entryHeight));
			ColorRect(sf::FloatRect(mRect.rect.width - scrollbarWidth, (mRect.rect.height - scrollbarHeight) * scrollDistance / heightDiff, scrollbarWidth, scrollbarHeight), sf::Color(255, 255, 255, 40)).Render();
		}

		//Outlines
		OutlinedRect(sf::FloatRect(0, 0, mRect.rect.width, mRect.rect.height), 1, sf::Color(60, 60, 60)).Render();
		top = 0;
		for (int i = 0; i < numElements; i++)
		{
			if (i == selectedElement)
				OutlinedRect(sf::FloatRect(0, top - scrollDistance, mRect.rect.width, entryHeight), 1, sf::Color::White).Render();
			top += entryHeight;
		}

		sfmlState->window.setView(sf::View(sf::FloatRect(0, 0, 1920, 1080)));
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (!mRect.interactable) return false;
		if (mRect.Captures(position))
		{
			float heightDiff = fmaxf(0, numElements * entryHeight - mRect.rect.height);
			float top = 0;
			for (int i = 0; i < numElements; i++)
			{
				if (sf::FloatRect(mRect.rect.left, mRect.rect.top + top - scrollDistance, mRect.rect.width, entryHeight).contains(position))
				{
					sfmlState->selectedScrollList = NULL;
					selectedElement = i;
					return true;
				}
				top += entryHeight;
			}
		}
		return false;
	}

	void HandleMouse(sf::Vector2f position)
	{
		if (!mRect.interactable) return;
		if (mRect.Captures(position)) sfmlState->selectedScrollList = this;
	}
};

struct GuiDropdown : GuiObject
{
	sf::FloatRect openRect;
	GuiScrollList list;

	GuiDropdown() = default;
	GuiDropdown(int _numElements, const char** _elements, sf::FloatRect _rect, int _selected)
	{
		openRect = _rect;
		openRect.height = fminf(openRect.height, _numElements * list.entryHeight);
		sf::FloatRect clippedOpenRect = sf::FloatRect(openRect);
		if (openRect.top + openRect.height > 1080)
		{
			clippedOpenRect.top -= openRect.top + openRect.height - 1080;
		}
		list = GuiScrollList(_numElements, _elements, clippedOpenRect, _selected);
	}

	void SetEntryHeight(int _height)
	{
		list.entryHeight = _height;
		openRect.height = fminf(openRect.height, list.numElements * list.entryHeight);
		sf::FloatRect clippedOpenRect = sf::FloatRect(openRect);
		if (openRect.top + openRect.height > 1080)
		{
			clippedOpenRect.top -= openRect.top + openRect.height - 1080;
		}
		list.mRect.rect = clippedOpenRect;
	}

	void Render()
	{
		if (sfmlState->selectedScrollList != &list)
		{
			BGTextRect bg(list.elements[list.selectedElement], sf::FloatRect(openRect.left, openRect.top, openRect.width, list.entryHeight), list.textSize, sf::Color::White, list.backgroundColor);
			bg.Render();
		}
	}

	bool HandleClick(sf::Vector2f position)
	{
		if (!list.mRect.interactable) return false;
		if (sf::FloatRect(openRect.left, openRect.top, openRect.width, list.entryHeight).contains(position))
		{
			if (sfmlState->selectedScrollList != NULL)
			{
				if (!sfmlState->selectedScrollList->mRect.Captures(position))
				{
					sfmlState->selectedScrollList = &list;
					return true;
				}
			}
			else
			{
				sfmlState->selectedScrollList = &list;
				return true;
			}
		}
		if (sfmlState->selectedScrollList == &list)
		{
			return list.HandleClick(position);
		}
		return false;
	}

	void HandleMouse(sf::Vector2f position) {}
};