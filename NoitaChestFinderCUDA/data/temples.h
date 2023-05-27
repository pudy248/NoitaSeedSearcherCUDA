#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ const int temple_x[] = {
	-32,
	-32,
	-32,
	-32,
	-32,
	-32,
	2560
};

__device__ const int temple_y[] = {
	1410,
	2946,
	4994,
	6530,
	8578,
	10626,
	13181
};

__device__ const int temple_tiers[] = {
	0,
	1,
	2,
	2,
	3,
	4,
	6
};

constexpr int shopOffsetX = -299;
constexpr int shopOffsetY = -15;

constexpr int chestOffsetX = -46;
constexpr int chestOffsetY = -39;