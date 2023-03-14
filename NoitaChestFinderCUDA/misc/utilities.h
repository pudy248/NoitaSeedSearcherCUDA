#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "datatypes.h"
#include "noita_random.h"

__device__ float random_next(float min, float max, NoitaRandom* random, IntPair* rnd) {
	float result = random->ProceduralRandomf(rnd->x, rnd->y, min, max);
	rnd->y += 1;
	return result;
}
__device__ int random_nexti(float min, float max, NoitaRandom* random, IntPair* rnd) {
	int result = random->ProceduralRandomi(rnd->x, rnd->y, min, max);
	rnd->y += 1;
	return result;
}

__device__ int pick_random_from_table_backwards(const float* probs, int length, NoitaRandom* random, IntPair* rnd) {
	for (int i = length - 1; i > 0; i--) {
		if (random_next(0, 1, random, rnd) <= probs[i]) return i;
	}
	return 0;
}

__device__ int pick_random_from_table_weighted(const float* probs, float sum, int length, NoitaRandom* random, IntPair* rnd) {
	float val = random_next(0, sum, random, rnd);
	for (int i = 0; i < length; i++) {
		if (val < probs[i]) return i;
		val -= probs[i];
	}
	return 0;
}

__device__ ulong createRGB(const byte r, const byte g, const byte b)
{
	return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
}


__device__ IntPair GetGlobalPos(int x, int y, int px, int py)
{
	//if (y == 14 && py > 400 && py < 600)
	//{
	//	py += 10;
	//}
	int gx = (int)(((x - 35) * 512) / 10) * 10 + px - 15;
	int gy = (int)(((y - 14) * 512) / 10) * 10 + py - 13;
	return { gx, gy };
}
