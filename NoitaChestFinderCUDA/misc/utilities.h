#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "datatypes.h"
#include "noita_random.h"

__device__ float random_next(float min, float max, NoitaRandom* random, IntPair* rnd)
{
	float result = random->ProceduralRandomf(rnd->x, rnd->y, min, max);
	rnd->y += 1;
	return result;
}
__device__ int random_nexti(float min, float max, NoitaRandom* random, IntPair* rnd)
{
	int result = random->ProceduralRandomi(rnd->x, rnd->y, min, max);
	rnd->y += 1;
	return result;
}

__device__ int pick_random_from_table_backwards(const float* probs, int length, NoitaRandom* random, IntPair* rnd)
{
	for (int i = length - 1; i > 0; i--)
	{
		if (random_next(0, 1, random, rnd) <= probs[i]) return i;
	}
	return 0;
}

__device__ int pick_random_from_table_weighted(const float* probs, float sum, int length, NoitaRandom* random, IntPair* rnd)
{
	float val = random_next(0, sum, random, rnd);
	for (int i = 0; i < length; i++)
	{
		if (val < probs[i]) return i;
		val -= probs[i];
	}
	return 0;
}

__device__ uint createRGB(const byte r, const byte g, const byte b)
{
	return (r << 16) | (g << 8) | b;
}


__device__ IntPair GetGlobalPos(const int x, const int y, const int px, int py)
{
	int gx = (int)(((x - 35) * 512) / 10) * 10 + px - 15;
	int gy = (int)(((y - 14) * 512) / 10) * 10 + py - 3;
	return { gx, gy };
}

__device__ int roundRNGPos(int num)
{
	if (-1000000 < num && num < 1000000) return num;
	else if (-10000000 < num && num < 10000000) return roundf(num / 10.0) * 10;
	else if (-100000000 < num && num < 100000000) return roundf(num / 100.0) * 100;
	return num;
}

__device__ void _itoa_offset(int num, char* buffer, int base, int& offset)
{
	char internal_buffer[11]; //ints can't be bigger than this!
	int i = 10;
	bool isNegative = false;

	if (num == 0)
	{
		buffer[offset++] = '0';
		return;
	}

	if (num < 0 && base == 10)
	{
		isNegative = true;
		num = -num;
	}

	while (num != 0)
	{
		int rem = num % base;

		internal_buffer[i--] = (rem > 9) ? (rem - 10) + 'a' : rem + '0';

		num = num / base;
	}

	if (isNegative)
		internal_buffer[i--] = '-';

	for (int j = i + 1; j < 11; j++)
		buffer[offset++] = internal_buffer[j];
}

__device__ void _putstr_offset(const char* str, char* buffer, int& offset)
{
	int i = 0;
	while (str[i] != '\0')
	{
		buffer[offset++] = str[i++];
	}
}