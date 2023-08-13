#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../misc/pngutils.h"
#include "guiPrimitives.h"

struct GuiObject
{
	GuiObject() = default;
	virtual void Render() = 0;
	virtual void HandleClick(sf::Vector2f position) = 0;
	virtual void HandleMouse(sf::Vector2f position) = 0;
};

struct GuiCheckbox : GuiObject
{
	ImageRect disabledImg;
	ImageRect enabledImg;
	bool enabled;

	GuiCheckbox(sf::FloatRect rect, RawImage _disabled, RawImage _enabled, bool defaultVal)
	{
		disabledImg = ImageRect(_disabled, rect);
		enabledImg = ImageRect(_enabled, rect);
		enabled = defaultVal;
	}

	void Render()
	{
		if (enabled) enabledImg.Render();
		else disabledImg.Render();
	}

	void HandleClick(sf::Vector2f position)
	{
		enabled = !enabled;
	}

	void HandleMouse(sf::Vector2f position) {}
};

struct GuiDropdown : GuiObject
{
	sf::FloatRect openRect;
	float entryHeight = 50;
	float textSize = 24;

	sf::Color backgroundColor = sf::Color(30, 30, 30);

	int numElements = 0;
	int selectedElement = 0;
	float scrollFraction = 0;
	const char** elements;

	bool opened = false;

	GuiDropdown() = default;

	void Render()
	{
		if (!opened)
		{
			BGTextRect bg(elements[selectedElement], sf::FloatRect(openRect.left, openRect.left, openRect.width, entryHeight), 24, sf::Color::White, backgroundColor);
			bg.Render();
		}
		else
		{
			sf::View dropdownView;
			dropdownView.reset(sf::FloatRect(sf::Vector2f(), sf::Vector2f(openRect.width, openRect.height)));
			dropdownView.setViewport(sf::FloatRect(openRect.left / sfmlState->videoMode.width, openRect.top / sfmlState->videoMode.height,
				openRect.width / sfmlState->videoMode.width, openRect.height / sfmlState->videoMode.height));
			sfmlState->window.setView(dropdownView);

			float top = 0;
			for (int i = 0; i < numElements; i++)
			{
				BGTextRect element(elements[i], sf::FloatRect(0, top, openRect.width, entryHeight), 24, sf::Color::White, backgroundColor);
				OutlinedRect outline(sf::FloatRect(1, top + 1, openRect.width - 2, entryHeight - 2), 1, sf::Color(100, 100, 100));
				element.Render();
				outline.Render();
				top += entryHeight;
			}

			sfmlState->window.setView(sfmlState->window.getDefaultView());
		}
	}

	void HandleClick(sf::Vector2f position)
	{
		if (!opened && sf::FloatRect(openRect.left, openRect.top, openRect.width, entryHeight).contains(position)) opened = !opened;
		else if (opened && openRect.contains(position))
		{
			float top = 0;
			for (int i = 0; i < numElements; i++)
			{
				if (sf::FloatRect(openRect.left, openRect.top + top, openRect.width, entryHeight).contains(position))
				{
					opened = !opened;
					selectedElement = i;
					break;
				}
				top += entryHeight;
			}
		}
	}

	void HandleMouse(sf::Vector2f position) {}
};