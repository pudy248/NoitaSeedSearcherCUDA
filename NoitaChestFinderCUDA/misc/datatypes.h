#pragma once

typedef unsigned char byte;
typedef signed char sbyte;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long int ulong;

struct IntPair {
	int x;
	int y;
};

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


