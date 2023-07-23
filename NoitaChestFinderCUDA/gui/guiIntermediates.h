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
	sf::FloatRect drawRect;
	float entryHeight;

	int numElements;
	const char** elements;
	bool opened;

	GuiDropdown() = default;

	void Render()
	{
		sf::View dropdownView;
		dropdownView.reset(sf::FloatRect(sf::Vector2f(), sf::Vector2f(drawRect.width, drawRect.height)));
		dropdownView.setViewport(sf::FloatRect(drawRect.left / sfmlState->videoMode.width, drawRect.top / sfmlState->videoMode.height,
			drawRect.width / sfmlState->videoMode.width, drawRect.height / sfmlState->videoMode.height));
		sfmlState->window.setView(dropdownView);

		BGTextRect rect = BGTextRect("hi :)", sf::FloatRect(0, 0, 100, 50));
		rect.Render();

		sfmlState->window.setView(sfmlState->window.getDefaultView());
	}

	void HandleClick(sf::Vector2f position)
	{
		if (!opened) opened = true;
		else
		{

		}
	}

	void HandleMouse(sf::Vector2f position) {}
};