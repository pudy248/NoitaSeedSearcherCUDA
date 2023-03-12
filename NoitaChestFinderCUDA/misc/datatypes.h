#pragma once

typedef unsigned char byte;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long int ulong;

struct ConfigOptions {
	//Search space
	uint batch;
	uint startSeed;
	uint endSeed;
	ushort pwCount;
	bool biomes[26];
	//ng+

	//Search scope
	bool searchChests;
	bool searchPedestals;
	bool searchWandAltars;
	bool searchPixelScenes;

	bool checkPotions;
	bool checkWands;
	bool checkCards;

	//More TBD
};


