#pragma once
#include <vector>
#include <SFML/Graphics.hpp>
#include "primitives.h"
#include "pngutils.h"

namespace gui {
	struct RawImage {
		uint8_t* pixels;
		int w;
		int h;
	};
	struct TextInput {
		//bits: 1 = numbers, 2 = lower, 4 = upper, 8 = everything
		enum Charset : uint8_t {
			CHARSET_Numeric = 1,
			CHARSET_AlphabeticalLower = 2,
			CHARSET_AlphabeticalUpper = 4,
			CHARSET_Full = 8
		} allowedChars;
		int maxLen = 10;
		sf::String str;

		void HandleEvent(sf::Event e) {
			if (e.type == sf::Event::TextEntered) {
				uint32_t character = e.text.unicode;

				if (character == 0x08) {
					if (str.getSize() > 0) str.erase(str.getSize() - 1);
					return;
				}
				if (character == 0x7f) {
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
	struct GuiObject {
		sf::FloatRect rect;
		GuiObject() = default;
		GuiObject(sf::FloatRect rect) : rect(rect) {}
		virtual void Render(const sf::FloatRect& window) = 0;
	};
	struct GuiInteractable : GuiObject {
		bool interactable = true;
		bool containedMouseLastFrame = false;
		GuiInteractable() = default;
		GuiInteractable(sf::FloatRect rect) : GuiObject(rect) {}
		virtual bool HandleMouse(sf::Vector2f pos) = 0;
		virtual bool HandleClick(sf::Vector2f pos) = 0;
	};
	struct SfmlSharedState {
		sf::RenderWindow& window;
		struct SearchGui* gui;
		sf::VideoMode videoMode;
		sf::Text* texts;
		GuiInteractable* clickedRect = NULL;
		TextInput* selectedText = NULL;
		struct ScrollList* selectedScrollList = NULL;

	} *sfmlState;

	sf::Vector2f TransformMouse(sf::Vector2f pos) {
		return sf::Vector2f(pos * (1920.0f / sfmlState->videoMode.width));
	}
	sf::FloatRect AddWindow(const sf::FloatRect& window, const sf::FloatRect& rect) {
		return sf::FloatRect(rect.left + window.left, rect.top + window.top, rect.width, rect.height);
	}
	sf::Vector2f AddWindow(const sf::FloatRect& window, const sf::Vector2f& point) {
		return sf::Vector2f(point.x + window.left, point.y + window.top);
	}
	sf::Vector2f SubWindow(const sf::FloatRect& window, const sf::Vector2f& point) {
		return sf::Vector2f(point.x - window.left, point.y - window.top);
	}
	int HandleInteraction(const sf::Vector2f& mPos, const sf::FloatRect& rect, bool& interactable, bool& containedLastFrame) {
		if (!interactable) return 0;
		bool _last = containedLastFrame;
		bool contains = rect.contains(mPos);
		containedLastFrame = contains;

		if (contains && !_last) //mouse entered
			return 2;
		else if (_last && !contains) //mouse exited
			return -1;
		else if (contains) //mouse in
			return 1;
		else return 0; //mouse out
	}

	RawImage LoadAndFormatImageRGB(const char* filepath) {
		Vec2i dims = GetImageDimensions(filepath);
		uint8_t* tmp = (uint8_t*)malloc(3 * dims.x * dims.y);
		uint8_t* data = (uint8_t*)malloc(4 * dims.x * dims.y);
		ReadImage(filepath, tmp);
		for (int i = 0; i < dims.x * dims.y; i++) {
			data[4 * i] = 0xff;
			memcpy(data + 4 * i + 1, tmp + 3 * i, 3);
		}
		free(tmp);
		return { data, dims.x, dims.y };
	}
	RawImage LoadAndFormatImageRGBA(const char* filepath) {
		Vec2i dims = GetImageDimensions(filepath);
		uint8_t* data = (uint8_t*)malloc(4 * dims.x * dims.y);
		ReadImageRGBA(filepath, data);
		return { data, dims.x, dims.y };
	}

	void DrawRect(const sf::FloatRect& rect, sf::Color color, sf::Color borderColor, float borderThickness) {
		sf::RectangleShape sprite(sf::Vector2f(rect.width - borderThickness * 2, rect.height - borderThickness * 2));
		sprite.setOutlineThickness(borderThickness);
		sprite.setFillColor(color);
		sprite.setOutlineColor(borderColor);
		sprite.setPosition(rect.left + borderThickness, rect.top + borderThickness);
		sfmlState->window.draw(sprite);
	}

	void DrawTextCentered(const sf::String& text, const sf::Vector2f& position, uint32_t size, sf::Color color, int fontIdx) {
		sfmlState->texts[fontIdx].setCharacterSize(size);
		sfmlState->texts[fontIdx].setFillColor(color);
		sfmlState->texts[fontIdx].setString(text);
		sfmlState->texts[fontIdx].setPosition(position);
		sf::FloatRect rect = sfmlState->texts[fontIdx].getGlobalBounds();
		sfmlState->texts[fontIdx].setPosition(2 * position.x - rect.left - rect.width * 0.5f, 2 * position.y - rect.top - rect.height * 0.5f);
		sfmlState->window.draw(sfmlState->texts[fontIdx]);
	}
	void DrawTextAligned(const sf::String& text, const sf::FloatRect& rect, uint32_t size, sf::Color color, int fontIdx, int hAlign, int vAlign) {
		sfmlState->texts[fontIdx].setCharacterSize(size);
		sfmlState->texts[fontIdx].setFillColor(color);
		sfmlState->texts[fontIdx].setString(text);
		sfmlState->texts[fontIdx].setPosition(rect.left, rect.top);
		sf::FloatRect bounds = sfmlState->texts[fontIdx].getGlobalBounds();
		int xPos = rect.left;
		int yPos = rect.top;
		if (hAlign == 2) xPos = rect.left + rect.width - bounds.width;
		else if (hAlign == 1) xPos = rect.left + (rect.width - bounds.width) * 0.5f;
		if (vAlign == 2) yPos = rect.top + rect.height - bounds.height;
		else if (vAlign == 1) yPos = rect.top + (rect.height - bounds.height) * 0.5f;
		sfmlState->texts[fontIdx].setPosition(xPos, yPos);
		sfmlState->window.draw(sfmlState->texts[fontIdx]);
	}

	void DrawTexture(const sf::Texture& tex, int w, int h, const sf::FloatRect& rect) {
		float hScale = rect.width / w;
		float vScale = rect.height / h;
		sf::Sprite sprite;
		sprite.setPosition(rect.left, rect.top);
		sprite.setTexture(tex);
		sprite.setScale(hScale, vScale);
		sfmlState->window.draw(sprite);
	}
	void DrawImage(RawImage image, sf::FloatRect rect) {
		sf::Texture generatedTex = sf::Texture();
		generatedTex.create(image.w, image.h);
		generatedTex.update((unsigned char*)image.pixels, image.w, image.h, 0, 0);
		DrawTexture(generatedTex, image.w, image.h, rect);
	}
	
	struct GenericRect : GuiObject {
		sf::Color color = sf::Color::Transparent;
		sf::Color outlineColor = sf::Color::Transparent;
		float outlineThickness = 0;
		GenericRect() = default;
		GenericRect(sf::FloatRect rect, sf::Color color = sf::Color::Transparent, sf::Color outlineColor = sf::Color::Transparent, float outlineThickness = 0)
			: GuiObject(rect), color(color), outlineColor(outlineColor), outlineThickness(outlineThickness) {}

		void Render(const sf::FloatRect& window) {
			DrawRect(AddWindow(window, rect), color, outlineColor, outlineThickness);
		}
	};
	struct GenericTextRect : GenericRect {
		sf::String text;
		uint32_t textSize = 24;
		sf::Color textColor = sf::Color::White;
		int hAlign = 1, vAlign = 1;
		int fontIndex = 0;

		GenericTextRect() = default;
		GenericTextRect(sf::FloatRect rect, sf::String text, sf::Color textColor = sf::Color::White, uint32_t textSize = 24, int hAlign = 1, int vAlign = 1, int fontIdx = 0) 
			: GenericRect(rect), text(text), textColor(textColor), textSize(textSize), hAlign(hAlign), vAlign(vAlign), fontIndex(fontIdx) {}

		GenericTextRect(GenericRect bg, sf::String text = "", sf::Color textColor = sf::Color::White, uint32_t textSize = 24, int hAlign = 1, int vAlign = 1, int fontIdx = 0)
			: GenericRect(bg), text(text), textColor(textColor), textSize(textSize), hAlign(hAlign), vAlign(vAlign), fontIndex(fontIdx) {}

		void Render(const sf::FloatRect& window) {
			DrawRect(AddWindow(window, rect), color, outlineColor, outlineThickness);
			DrawTextAligned(text, AddWindow(window, rect), textSize, textColor, fontIndex, hAlign, vAlign);
		}
	};

	struct ImageRect : GuiObject {
		RawImage image;
		sf::Texture generatedTex;

		ImageRect() = default;
		ImageRect(RawImage _img, sf::FloatRect _rect) {
			image = _img;
			rect = _rect;
			generatedTex = sf::Texture();
			generatedTex.create(image.w, image.h);
			generatedTex.update((unsigned char*)image.pixels, image.w, image.h, 0, 0);
		}

		void Render(const sf::FloatRect& window) {
			DrawTexture(generatedTex, image.w, image.h, AddWindow(window, rect));
		}
	};

	struct InputRect : GuiInteractable {
		GenericTextRect bg;
		TextInput textInput;

		InputRect() = default;
		InputRect(GenericTextRect bg, TextInput::Charset charset = TextInput::Charset::CHARSET_Full, int maxLen = 10)
			: GuiInteractable(bg.rect), bg(bg), textInput({ charset, maxLen, bg.text }) {}

		void Render(const sf::FloatRect& window) {
			if (sfmlState->selectedText == &textInput) bg.outlineThickness = 1; else bg.outlineThickness = 0;
			bg.text = textInput.str;
			bg.Render(window);
		}

		bool HandleClick(sf::Vector2f pos) {
			if (rect.contains(pos) && interactable) {
				sfmlState->selectedText = &textInput;
				return true;
			}
			return false;
		}
	};
	struct Checkbox : GuiInteractable {
		GenericTextRect box;
		bool enabled = false;

		Checkbox() = default;
		Checkbox(GenericRect bg, bool defaultVal = false) : GuiInteractable(bg.rect), box(bg), enabled(defaultVal) {
			if (enabled) box.text = "X";
			else box.text = "";
		}
		Checkbox(sf::FloatRect rect, sf::Color bg, bool defaultVal = false) : GuiInteractable(rect), box(GenericRect(rect, bg), ""), enabled(defaultVal) {
			if (enabled) box.text = "X";
		}

		void Render(const sf::FloatRect& window) {
			box.Render(window);
		}

		bool HandleClick(sf::Vector2f position) {
			if (!interactable) return false;
			if (!rect.contains(position)) return false;
			enabled = !enabled;
			if (enabled) box.text = "X";
			else box.text = "";
			return true;
		}

		bool HandleMouse(sf::Vector2f position) { return false; }
	};

	struct ScrollList : GuiInteractable {
		GenericTextRect templateEntry;
		float scrollbarWidth = 5;

		int selectedElement = 0;
		float scrollDistance = 0;
		std::vector<sf::String> elements;

		ScrollList() = default;
		ScrollList(sf::FloatRect rect, GenericTextRect templateEntry, std::vector<sf::String> elements = std::vector<sf::String>(), float scrollbarWidth = 5) :
			GuiInteractable(rect), templateEntry(templateEntry), elements(elements) {}

		void Render(const sf::FloatRect& window) {
			sf::FloatRect tmprect(AddWindow(window, rect));
			tmprect.height = fminf(tmprect.height, elements.size() * templateEntry.rect.height);
			float heightDiff = fmaxf(0, elements.size() * templateEntry.rect.height - tmprect.height);

			sf::View dropdownView;
			dropdownView.reset(sf::FloatRect(0, 0, tmprect.width, tmprect.height));
			dropdownView.setViewport(sf::FloatRect(tmprect.left / 1920, tmprect.top / 1080, tmprect.width / 1920, tmprect.height / 1080));
			sfmlState->window.setView(dropdownView);

			//Elements
			for (int i = 0; i < elements.size(); i++) {
				GenericTextRect copyRect(templateEntry);
				copyRect.text = elements[i];
				copyRect.rect.top += i * templateEntry.rect.height - scrollDistance;
				copyRect.Render(sf::FloatRect(0, 0, 0, 0));
			}

			//Scrollbar
			if (tmprect.height < elements.size() * templateEntry.rect.height) {
				float scrollbarHeight = tmprect.height * tmprect.height / (elements.size() * templateEntry.rect.height);
				GenericRect(
					sf::FloatRect(tmprect.width - scrollbarWidth, (tmprect.height - scrollbarHeight) * scrollDistance / heightDiff, scrollbarWidth, scrollbarHeight),
					sf::Color(255, 255, 255, 40)
				).Render(sf::FloatRect(0, 0, 0, 0));
			}

			//Outlines
			GenericRect(sf::FloatRect(0, 0, tmprect.width, tmprect.height), sf::Color::Transparent, sf::Color(60, 60, 60), 1).Render(sf::FloatRect(0, 0, 0, 0));
			GenericRect(
				sf::FloatRect(0, selectedElement * templateEntry.rect.height - scrollDistance, templateEntry.rect.width, templateEntry.rect.height),
				sf::Color::Transparent, sf::Color::White, 1
			).Render(sf::FloatRect(0, 0, 0, 0));

			sfmlState->window.setView(sf::View(sf::FloatRect(0, 0, 1920, 1080)));
		}

		bool HandleClick(sf::Vector2f position) {
			if (!interactable) return false;
			sf::FloatRect tmprect(rect);
			tmprect.height = fminf(tmprect.height, elements.size() * templateEntry.rect.height);
			if (tmprect.contains(position)) {
				float heightDiff = fmaxf(0, elements.size() * templateEntry.rect.height - rect.height);
				selectedElement = (position.y - tmprect.top + scrollDistance) / templateEntry.rect.height;
				return true;
			}
			return false;
		}

		bool HandleMouse(sf::Vector2f position) {
			if (!interactable) return false;
			sf::FloatRect tmprect(rect);
			tmprect.height = fminf(tmprect.height, elements.size() * templateEntry.rect.height);
			if (tmprect.contains(position)) {
				sfmlState->selectedScrollList = this;
				return true;
			}
			return false;
		}
	};

	/*struct GuiDropdown : GuiInteractable {
		sf::FloatRect openRect;
		ScrollList list;

		GuiDropdown() = default;
		GuiDropdown(int _numElements, const char** _elements, sf::FloatRect _rect, int _selected) {
			openRect = _rect;
			openRect.height = fminf(openRect.height, _numElements * list.entryHeight);
			sf::FloatRect clippedOpenRect = sf::FloatRect(openRect);
			if (openRect.top + openRect.height > 1080) {
				clippedOpenRect.top -= openRect.top + openRect.height - 1080;
			}
			list = GuiScrollList(_numElements, _elements, clippedOpenRect, _selected);
		}

		void SetEntryHeight(int _height) {
			list.entryHeight = _height;
			openRect.height = fminf(openRect.height, list.numElements * list.entryHeight);
			sf::FloatRect clippedOpenRect = sf::FloatRect(openRect);
			if (openRect.top + openRect.height > 1080) {
				clippedOpenRect.top -= openRect.top + openRect.height - 1080;
			}
			list.mRect.rect = clippedOpenRect;
		}

		void Render() {
			if (sfmlState->selectedScrollList != &list) {
				BGTextRect bg(list.elements[list.selectedElement], sf::FloatRect(openRect.left, openRect.top, openRect.width, list.entryHeight), list.textSize, sf::Color::White, list.backgroundColor);
				bg.Render();
			}
		}

		bool HandleClick(sf::Vector2f position) {
			if (!list.mRect.interactable) return false;
			if (sf::FloatRect(openRect.left, openRect.top, openRect.width, list.entryHeight).contains(position)) {
				if (sfmlState->selectedScrollList != NULL) {
					if (!sfmlState->selectedScrollList->rect.contains(position)) {
						sfmlState->selectedScrollList = &list;
						return true;
					}
				}
				else {
					sfmlState->selectedScrollList = &list;
					return true;
				}
			}
			if (sfmlState->selectedScrollList == &list) {
				return list.HandleClick(position);
			}
			return false;
		}

		bool HandleMouse(sf::Vector2f position) { return false; }
	};*/
}

