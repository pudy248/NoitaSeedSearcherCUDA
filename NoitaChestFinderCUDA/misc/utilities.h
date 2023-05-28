#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"

#include "noita_random.h"

__device__ float random_next(float min, float max, NollaPRNG& random, IntPair& rnd)
{
	random.SetRandomSeedInt(rnd.x, rnd.y);
	float result = min + ((max - min) * random.Next());
	rnd.y += 1;
	return result;
}
__device__ int random_nexti(float min, float max, NollaPRNG& random, IntPair& rnd)
{
	random.SetRandomSeedInt(rnd.x, rnd.y);
	int result = random.Random(min, max);
	rnd.y += 1;
	return result;
}

__device__ int pick_random_from_table_backwards(const float* probs, int length, NollaPRNG& random, IntPair& rnd)
{
	for (int i = length - 1; i > 0; i--)
	{
		if (random_next(0, 1, random, rnd) <= probs[i]) return i;
	}
	return 0;
}

__device__ int pick_random_from_table_weighted(const float* probs, float sum, int length, NollaPRNG& random, IntPair& rnd)
{
	float val = random_next(0, sum, random, rnd);
	for (int i = 0; i < length; i++)
	{
		if (val < probs[i]) return i;
		val -= probs[i];
	}
	return 0;
}

__device__ uint32_t createRGB(const uint8_t r, const uint8_t g, const uint8_t b)
{
	return (r << 16) | (g << 8) | b;
}

__device__ __host__ IntPair GetGlobalPos(const int x, const int y, const int px, int py)
{
	int gx = ((512 * x) / 10 - (512 * 35) / 10) * 10 + px - 5;
	int gy = ((512 * y) / 10 - (512 * 14) / 10) * 10 + py - 13;
	return { gx, gy };
}

__device__ int roundRNGPos(int num)
{
	if (-1000000 < num && num < 1000000) return num;
	else if (-10000000 < num && num < 10000000) return rintf(num / 10.0) * 10;
	else if (-100000000 < num && num < 100000000) return rintf(num / 100.0) * 100;
	return num;
}

__device__ __host__ void _itoa_offset(int num, int base, char* buffer, int& offset)
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

		internal_buffer[i--] = (rem > 9) ? (rem - 10) + 'A' : rem + '0';

		num = num / base;
	}

	if (isNegative)
		internal_buffer[i--] = '-';

	for (int j = i + 1; j < 11; j++)
		buffer[offset++] = internal_buffer[j];
}

__device__ __host__ void _itoa_offset_decimal(int num, int base, int fixedPoint, char* buffer, int& offset)
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

		internal_buffer[i--] = (rem > 9) ? (rem - 10) + 'A' : rem + '0';

		if (10 - i == fixedPoint) internal_buffer[i--] = '.';

		num = num / base;
	}

	if (isNegative)
		internal_buffer[i--] = '-';

	for (int j = i + 1; j < 11; j++)
		buffer[offset++] = internal_buffer[j];
}

__device__ __host__ void _itoa_offset_zeroes(int num, int base, int leadingZeroes, char* buffer, int& offset)
{
	char internal_buffer[11]; //ints can't be bigger than this!
	int i = 10;
	bool isNegative = false;

	if (num < 0 && base == 10)
	{
		isNegative = true;
		num = -num;
	}

	while (num != 0 || 10 - i < leadingZeroes)
	{
		int rem = num % base;

		internal_buffer[i--] = (rem > 9) ? (rem - 10) + 'A' : rem + '0';

		num = num / base;
	}

	if (isNegative)
		internal_buffer[i--] = '-';

	for (int j = i + 1; j < 11; j++)
		buffer[offset++] = internal_buffer[j];
}

__device__ __host__ void _putstr_offset(const char* str, char* buffer, int& offset)
{
	int i = 0;
	while (str[i] != '\0')
	{
		buffer[offset++] = str[i++];
	}
}