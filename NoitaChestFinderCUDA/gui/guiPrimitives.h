#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../misc/pngutils.h"

#include <iostream>
#include <SFML/Graphics.hpp>

struct MouseRect
{
	sf::FloatRect rect;
	bool containedMouseLastFrame;

	int HandleMouse(sf::Vector2f mPos)
	{
		bool _last = containedMouseLastFrame;
		bool contains = rect.contains(mPos);
		containedMouseLastFrame = contains;

		if (contains && !_last) //mouse entered
			return 2;
		else if (_last && !contains) //mouse exited
			return -1;
		else if (contains) //mouse in
			return 1;
		else return 0; //mouse out
	}
};
struct TextInput
{
	//bits: 1 = numbers, 2 = lower, 4 = upper, 8 = everything
	enum Charset : uint8_t
	{
		CHARSET_Numeric = 1,
		CHARSET_AlphabeticalLower = 2,
		CHARSET_AlphabeticalUpper = 4,
		CHARSET_Full = 8
	} allowedChars;
	int maxLen;
	sf::String str = sf::String();

	void HandleEvent(sf::Event e)
	{
		if (e.type == sf::Event::TextEntered)
		{
			uint32_t character = e.text.unicode;

			if (character == 0x08)
			{
				if (str.getSize() > 0) str.erase(str.getSize() - 1);
				return;
			}
			if (character == 0x7f)
			{
				str = sf::String("");
				return;
			}

			bool isNum = 0x30 <= character && character <= 0x39;
			bool isLower = 0x61 <= character && character <= 0x7a;
			bool isUpper = 0x41 <= character && character <= 0x5a;

			bool allowed = (allowedChars & CHARSET_Full) > 0;
			allowed |= isNum && (allowedChars & CHARSET_Numeric) > 0;
			allowed |= isLower && (allowedChars & CHARSET_AlphabeticalLower) > 0;
			allowed |= isUpper && (allowedChars & CHARSET_AlphabeticalUpper) > 0;

			if (allowed && str.getSize() < maxLen)
				str += character;
		}
	}
};
struct SfmlSharedState
{
	sf::RenderWindow& window;
	sf::VideoMode videoMode;
	sf::Text* texts;

	MouseRect* clickedRect = NULL;
	TextInput* selectedText = NULL;

} *sfmlState;
struct GuiPrimitive
{
	MouseRect mRect;
	GuiPrimitive() = default;
	virtual void Render() = 0;
};

struct RawImage
{
	uint8_t* pixels;
	int w;
	int h;
};
RawImage LoadAndFormatImageRGB(const char* filepath)
{
	Vec2i dims = GetImageDimensions(filepath);
	uint8_t* tmp = (uint8_t*)malloc(3 * dims.x * dims.y);
	uint8_t* data = (uint8_t*)malloc(4 * dims.x * dims.y);
	ReadImage(filepath, tmp);
	for (int i = 0; i < dims.x * dims.y; i++)
	{
		data[4 * i] = 0xff;
		memcpy(data + 4 * i + 1, tmp + 3 * i, 3);
	}
	free(tmp);
	return { data, dims.x, dims.y };
}
RawImage LoadAndFormatImageRGBA(const char* filepath)
{
	Vec2i dims = GetImageDimensions(filepath);
	uint8_t* data = (uint8_t*)malloc(4 * dims.x * dims.y);
	ReadImageRGBA(filepath, data);
	return { data, dims.x, dims.y };
}

void DrawRect(sf::FloatRect rect, sf::Color color)
{
	sf::RectangleShape sprite(sf::Vector2f(rect.width, rect.height));
	sprite.setFillColor(color);
	sprite.setPosition(sf::Vector2f(rect.left, rect.top));
	sfmlState->window.draw(sprite);
}
void DrawRectOutline(sf::FloatRect rect, float borderThickness, sf::Color color, sf::Color borderColor)
{
	sf::RectangleShape sprite(sf::Vector2f(rect.width, rect.height));
	sprite.setOutlineThickness(borderThickness);
	sprite.setFillColor(color);
	sprite.setOutlineColor(borderColor);
	sprite.setPosition(sf::Vector2f(rect.left, rect.top));
	sfmlState->window.draw(sprite);
}

void DrawText(const char* text, sf::Vector2f position, uint32_t size, sf::Color color, int fontIdx)
{
	sfmlState->texts[fontIdx].setCharacterSize(size);
	sfmlState->texts[fontIdx].setFillColor(color);
	sfmlState->texts[fontIdx].setString(text);
	sfmlState->texts[fontIdx].setPosition(position);
	sfmlState->window.draw(sfmlState->texts[fontIdx]);
}
void DrawTextCentered(const char* text, sf::Vector2f position, uint32_t size, sf::Color color, int fontIdx)
{
	sfmlState->texts[fontIdx].setCharacterSize(size);
	sfmlState->texts[fontIdx].setFillColor(color);
	sfmlState->texts[fontIdx].setString(text);
	sfmlState->texts[fontIdx].setPosition(position);
	sf::FloatRect rect = sfmlState->texts[fontIdx].getGlobalBounds();
	sfmlState->texts[fontIdx].setPosition(sf::Vector2f(2 * position.x - rect.left - rect.width * 0.5f, 2 * position.y - rect.top - rect.height * 0.5f));
	sfmlState->window.draw(sfmlState->texts[fontIdx]);
}
void DrawTextAligned(const char* text, sf::FloatRect rect, uint32_t size, sf::Color color, int fontIdx, int hAlign, int vAlign)
{
	sfmlState->texts[fontIdx].setCharacterSize(size);
	sfmlState->texts[fontIdx].setFillColor(color);
	sfmlState->texts[fontIdx].setString(text);
	sfmlState->texts[fontIdx].setPosition(sf::Vector2f(rect.left, rect.top));
	sf::FloatRect bounds = sfmlState->texts[fontIdx].getGlobalBounds();
	int xPos = rect.left;
	int yPos = rect.top;
	if (hAlign == 2) xPos = rect.left + rect.width - bounds.width;
	else if (hAlign == 1) xPos = rect.left + (rect.width - bounds.width) * 0.5f;
	if (vAlign == 2) yPos = rect.top + rect.height - bounds.height;
	else if (vAlign == 1) yPos = rect.top + (rect.height - bounds.height) * 0.5f;
	sfmlState->window.draw(sfmlState->texts[fontIdx]);
}

void DrawTexture(sf::Texture tex, int w, int h, sf::FloatRect rect)
{
	float hScale = rect.width / w;
	float vScale = rect.height / h;
	sf::Sprite sprite;
	sprite.setPosition(rect.left, rect.top);
	sprite.setTexture(tex);
	sprite.setScale(hScale, vScale);
	sfmlState->window.draw(sprite);
}
void DrawImage(RawImage image, sf::FloatRect rect)
{
	sf::Texture generatedTex = sf::Texture();
	generatedTex.create(image.w, image.h);
	generatedTex.update((unsigned char*)image.pixels, image.w, image.h, 0, 0);
	DrawTexture(generatedTex, image.w, image.h, rect);
}

struct ColorRect : GuiPrimitive
{
	sf::Color color;
	ColorRect() = default;
	ColorRect(sf::FloatRect _rect, sf::Color _color)
	{
		mRect = { _rect, false };
		color = _color;
	}

	void Render()
	{
		DrawRect(mRect.rect, color);
	}
};
struct OutlinedRect : GuiPrimitive
{
	float outlineThickness;
	sf::Color color;
	sf::Color outlineColor;
	bool fill = false;
	OutlinedRect() = default;
	OutlinedRect(sf::FloatRect _rect, float _thickness, sf::Color _outlineColor)
	{
		mRect = { _rect, false };
		outlineThickness = _thickness;
		outlineColor = _outlineColor;
		fill = false;
	}
	OutlinedRect(sf::FloatRect _rect, float _thickness, sf::Color _outlineColor, sf::Color _fillColor)
	{
		mRect = { _rect, false };
		outlineThickness = _thickness;
		outlineColor = _outlineColor;
		color = _fillColor;
		fill = true;
	}

	void Render()
	{
		DrawRectOutline(mRect.rect, outlineThickness, fill ? color : sf::Color::Transparent, outlineColor);
	}
};
struct BGTextRect : GuiPrimitive
{
	const char* text;
	uint32_t textSize;
	sf::Color textColor;
	sf::Color bgColor;
	int fontIdx = 0;

	BGTextRect() = default;

	BGTextRect(const char* _text, sf::FloatRect _rect)
	{
		text = _text;
		mRect = { _rect, false };
		textSize = 24;
		textColor = sf::Color::White;
		bgColor = sf::Color::Transparent;
	}

	BGTextRect(const char* _text, sf::FloatRect _rect, uint32_t size)
	{
		text = _text;
		mRect = { _rect, false };
		textSize = size;
		textColor = sf::Color::White;
		bgColor = sf::Color::Transparent;
	}

	BGTextRect(const char* _text, sf::FloatRect _rect, uint32_t size, sf::Color textCol, sf::Color bgCol)
	{
		text = _text;
		mRect = { _rect, false };
		textSize = size;
		textColor = textCol;
		bgColor = bgCol;
	}

	void Render()
	{
		DrawRect(mRect.rect, bgColor);
		DrawTextCentered(text, sf::Vector2f(mRect.rect.left + mRect.rect.width * 0.5f, mRect.rect.top + mRect.rect.height * 0.5f), textSize, textColor, fontIdx);
	}
};
struct AlignedTextRect : GuiPrimitive
{
	const char* text;
	uint32_t textSize;
	sf::Color textColor;
	int fontIdx;
	int hTextAlignment;
	int vTextAlignment;


	AlignedTextRect() = default;

	AlignedTextRect(const char* _text, sf::FloatRect _rect, uint32_t size, sf::Color textCol, int _fontIdx, int hAlign, int vAlign)
	{
		text = _text;
		mRect = { _rect, false };
		textSize = size;
		textColor = textCol;
		fontIdx = _fontIdx;
		hTextAlignment = hAlign;
		vTextAlignment = vAlign;
	}

	void Render()
	{
		DrawTextAligned(text, mRect.rect, textSize, textColor, fontIdx, hTextAlignment, vTextAlignment);
	}
};
struct ImageRect : GuiPrimitive
{
	RawImage image;
	sf::Texture generatedTex;

	ImageRect() = default;
	ImageRect(RawImage _img, sf::FloatRect _rect)
	{
		image = _img;
		mRect = { _rect, false };
		generatedTex = sf::Texture();
		generatedTex.create(image.w, image.h);
		generatedTex.update((unsigned char*)image.pixels, image.w, image.h, 0, 0);
	}

	void Render()
	{
		DrawTexture(generatedTex, image.w, image.h, mRect.rect);
	}
};

struct InputRect : GuiPrimitive
{
	TextInput text;

	InputRect() = default;

	InputRect(sf::FloatRect _rect, uint8_t charset, int maxLen, const char* defaultText)
	{
		mRect = { _rect, false };
		text = { (TextInput::Charset)charset, maxLen, sf::String(defaultText) };
	}

	void Render()
	{
		DrawText(text.str.toAnsiString().c_str(), sf::Vector2f(mRect.rect.left, mRect.rect.top), 48, sf::Color::White, 0);
	}

	void HandleClick(sf::Vector2f pos)
	{
		if (mRect.rect.contains(pos)) sfmlState->selectedText = &text;
	}
};