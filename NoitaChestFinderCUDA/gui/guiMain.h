#pragma once

#include "guiPrimitives.h"
#include "guiLayout.h"
#include "guiInterop.h"

#include <iostream>
#include <chrono>
#include <SFML/Graphics.hpp>

void SfmlMain()
{
	sf::RenderWindow window(sf::VideoMode::getDesktopMode(), "SFML Window", sf::Style::None, sf::ContextSettings(24U, 8U, 4U, 4U, 0U, sf::ContextSettings::Attribute::Default));
	window.setPosition(sf::Vector2i(0, 0));
	window.setView(sf::View(sf::FloatRect(0, 0, 1920, 1080)));
	window.setVerticalSyncEnabled(true);

	constexpr int numFonts = 3;
	sf::Font fonts[numFonts];
	fonts[0].loadFromFile("NoitaPixel.ttf");
	fonts[1].loadFromFile("NoitaBlackletter.ttf");
	fonts[2].loadFromFile("arial.ttf");
	sf::Text texts[numFonts];
	for (int i = 0; i < numFonts; i++)
	{
		texts[i].setFont(fonts[i]);
	}

	SfmlSharedState refs = { window, NULL, sf::VideoMode::getDesktopMode(), texts };
	sfmlState = &refs;

	printf("%ix%i\n", refs.videoMode.width, refs.videoMode.height);

	SearchGui gui = SearchGui();
	refs.gui = &gui;

	//fps averager
	constexpr auto fpsAverage = 100;
	int fpsIdx = 0;
	double frameTimes[fpsAverage];
	for (int i = 0; i < fpsAverage; i++) frameTimes[i] = 1.0f / fpsAverage;
	sf::Text fpsCounter;
	fpsCounter.setFont(fonts[2]);
	fpsCounter.setCharacterSize(18);
	fpsCounter.setFillColor(sf::Color(255, 255, 100));
	std::chrono::steady_clock::time_point lastTimepoint = std::chrono::steady_clock::now();


	while (window.isOpen())
	{
		std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
		std::chrono::nanoseconds sinceLast = time1 - lastTimepoint;
		lastTimepoint = time1;
		double frameSeconds = sinceLast.count() / 1'000'000'000.0;

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
			if (refs.selectedText != NULL) refs.selectedText->HandleEvent(event);
			if (event.type == sf::Event::MouseButtonPressed)
			{
				sf::Vector2f mPos = TransformMouse(sf::Vector2f(sf::Mouse::getPosition()));
				refs.selectedText = NULL;
				if (refs.selectedScrollList != NULL && !refs.selectedScrollList->rect.contains(mPos))
					refs.selectedScrollList = NULL;

				gui.HandleClick(mPos);
			}
			if (event.type == sf::Event::MouseMoved)
			{
				sf::Vector2f position = TransformMouse(sf::Vector2f(sf::Mouse::getPosition()));
				gui.HandleMouse(position);
			}
			if (event.type == sf::Event::MouseWheelScrolled && refs.selectedScrollList != NULL)
			{
				float scrollLength = fmaxf(refs.selectedScrollList->entryHeight * refs.selectedScrollList->numElements - refs.selectedScrollList->rect.height, 0);
				refs.selectedScrollList->scrollDistance = fminf(fmaxf(refs.selectedScrollList->scrollDistance - 200 * event.mouseWheelScroll.delta, 0), scrollLength);
			}
			if (refs.selectedScrollList != NULL && event.type == sf::Event::KeyPressed && (event.key.code == sf::Keyboard::Up || event.key.code == sf::Keyboard::Down))
			{
				if (event.key.code == sf::Keyboard::Up && refs.selectedScrollList->selectedElement > 0) refs.selectedScrollList->selectedElement--;
				if (event.key.code == sf::Keyboard::Down && refs.selectedScrollList->selectedElement < refs.selectedScrollList->numElements - 1) 
					refs.selectedScrollList->selectedElement++;
				refs.selectedScrollList->scrollDistance = fmaxf(fminf(refs.selectedScrollList->scrollDistance,
					refs.selectedScrollList->selectedElement * refs.selectedScrollList->entryHeight),
					(refs.selectedScrollList->selectedElement + 1) * refs.selectedScrollList->entryHeight - refs.selectedScrollList->rect.height);
			}
		}

		window.clear();
		gui.Render();
		if (refs.selectedScrollList != NULL) refs.selectedScrollList->Render();

		//FPS counter
		frameTimes[fpsIdx] = frameSeconds;
		fpsIdx = (fpsIdx + 1) % fpsAverage;
		double frameAverage = 0;
		for (int i = 0; i < fpsAverage; i++) frameAverage += frameTimes[i];
		frameAverage /= fpsAverage;
		double frameRate = 1.0f / frameAverage;
		char buffer[25];
		sprintf(buffer, "%.2f FPS", frameRate);
		fpsCounter.setString(buffer);
		window.draw(fpsCounter);

		window.display();
	}

}