#pragma once

#include "guiPrimitives.h"
#include "guiLayout.h"

#include <iostream>
#include <chrono>
#include <SFML/Graphics.hpp>

void SfmlMain()
{
	sf::RenderWindow window(sf::VideoMode::getDesktopMode(), "SFML Window", sf::Style::Fullscreen, sf::ContextSettings(24U, 8U, 4U, 4U, 0U, sf::ContextSettings::Attribute::Default));
	window.setPosition(sf::Vector2i(0, 0));
	window.setView(sf::View(sf::FloatRect(0, 0, 1920, 1080)));
	//window.setVerticalSyncEnabled(true);

	constexpr int numFonts = 2;
	sf::Font fonts[numFonts];
	fonts[0].loadFromFile("NoitaScript.ttf");
	fonts[1].loadFromFile("NoitaBlackletter.ttf");
	sf::Text texts[numFonts];
	for (int i = 0; i < numFonts; i++)
	{
		texts[i].setFont(fonts[i]);
	}

	SfmlSharedState refs = { window, sf::VideoMode::getDesktopMode(), texts };
	sfmlState = &refs;

	SearchGui gui = SearchGui();

	//fps averager
	constexpr auto fpsAverage = 100;
	int fpsIdx = 0;
	double frameTimes[fpsAverage];
	for (int i = 0; i < fpsAverage; i++) frameTimes[i] = 1.0f / fpsAverage;
	sf::Font arial;
	arial.loadFromFile("arial.ttf");
	sf::Text fpsCounter;
	fpsCounter.setFont(arial);
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
				refs.selectedText = NULL;
				sf::Vector2f position = sf::Vector2f(sf::Mouse::getPosition());
				gui.HandleClick(position);
			}
			if (event.type == sf::Event::MouseMoved)
			{
				sf::Vector2f position = sf::Vector2f(sf::Mouse::getPosition());
				gui.HandleMouse(position);
			}
		}

		window.clear();
		gui.Render();


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