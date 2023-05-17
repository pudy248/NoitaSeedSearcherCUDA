#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../misc/datatypes.h"
#include "../misc/wandgen.h"

__device__ const int temple_perk_x[] = {
	-32,
	-32,
	-32,
	-32,
	-32,
	-32,
	2560
};

__device__ const int temple_perk_y[] = {
	1410,
	2946,
	4994,
	6530,
	8578,
	10626,
	13181
};

constexpr int shopOffsetX = -299;
constexpr int shopOffsetY = -15;

__device__ Wand GetShopWand(NoitaRandom* random, int x, int y, int level)
{
	random->SetRandomSeed(x, y);
	bool shuffle = random->Random(0, 100) <= 50;
	return GetWandWithLevel(random->world_seed, x, y, level, shuffle, false);
}