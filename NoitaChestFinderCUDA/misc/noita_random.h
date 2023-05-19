#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "datatypes.h"

#include <cmath>

// Thanks to kaliuresis!
// Check out his orb atlas repository: https://github.com/kaliuresis/noa
// #include <stdint.h>

using namespace std;

class WorldgenPRNG
{
public:
	double Seed;

	__host__ __device__
		WorldgenPRNG(double seed)
	{
		Seed = seed;
		Next();
	}

	__host__ __device__
		uint NextU()
	{
		Next();
		return (uint)((Seed * 4.656612875e-10) * 2147483645.0);
	}

	__host__ __device__
		double Next()
	{
		int v4 = (int)Seed * 0x41a7 + ((int)Seed / 0x1f31d) * -0x7fffffff;
		if (v4 < 0)
		{
			v4 += 0x7fffffff;
		}
		Seed = (double)v4;
		return Seed / 0x7fffffff;
	}
};

__device__ uint StaticRandom(WorldgenPRNG* prng)
{
	return prng->NextU();
}

class NollaPRNG
{
public:
	int randomCTR = 0;

	__host__ __device__
		NollaPRNG(uint worldSeed)
	{
		world_seed = worldSeed;
		Seed = worldSeed;
		Next();
	}

	uint world_seed = 0;
	int Seed;

	__host__ __device__
		ulong SetRandomSeedHelper(double r)
	{
		ulong e = *(ulong*)&r;

		if (((e >> 0x20 & 0x7fffffff) < 0x7ff00000) && (-9.223372036854776e+18 <= r) && (r < 9.223372036854776e+18))
		{
			e <<= 1;
			e >>= 1;
			double s = *(double*)&e;
			ulong i = 0;
			if (s != 0.0)
			{
				ulong f = (e & 0xfffffffffffff) | 0x0010000000000000;
				ulong g = 0x433 - (e >> 0x34);
				ulong h = f >> (int)g;

				uint j = ~(uint)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
				i = (ulong)j << 0x20 | j;
				i = ~i & h | f << (((int)s >> 0x34) - 0x433) & i;
				i = ~(~(uint)(r == s ? 1 : 0) + 1) & (~i + 1) | i & (~(uint)(r == s ? 1 : 0) + 1);
			}
			return i & 0xffffffff;
		}
		return 0;
	}

	__host__ __device__
		uint SetRandomSeedHelper2(uint a, uint b, uint ws)
	{
		uint uVar1;
		uint uVar2;
		uint uVar3;

		uVar2 = (a - b) - ws ^ ws >> 0xd;
		uVar1 = (b - uVar2) - ws ^ uVar2 << 8;
		uVar3 = (ws - uVar2) - uVar1 ^ uVar1 >> 0xd;
		uVar2 = (uVar2 - uVar1) - uVar3 ^ uVar3 >> 0xc;
		uVar1 = (uVar1 - uVar2) - uVar3 ^ uVar2 << 0x10;
		uVar3 = (uVar3 - uVar2) - uVar1 ^ uVar1 >> 5;
		uVar2 = (uVar2 - uVar1) - uVar3 ^ uVar3 >> 3;
		uVar1 = (uVar1 - uVar2) - uVar3 ^ uVar2 << 10;
		return (uVar3 - uVar2) - uVar1 ^ uVar1 >> 0xf;
	}

	uint H2(uint a, uint b, uint ws)
	{
		uint v3;
		uint v4;
		uint v5;
		int v6;
		uint v7;
		uint v8;
		int v9;

		v3 = (ws >> 13) ^ (b - a - ws);
		v4 = (v3 << 8) ^ (a - v3 - ws);
		v5 = (v4 >> 13) ^ (ws - v3 - v4);
		v6 = (int)((v5 >> 12) ^ (v3 - v4 - v5));
		v7 = (uint)(v6 << 16) ^ (uint)(v4 - v6 - v5);
		v8 = (v7 >> 5) ^ (uint)(v5 - v6 - v7);
		v9 = (int)((v8 >> 3) ^ (uint)(v6 - v7 - v8));
		return (((uint)(v9 << 10) ^ (uint)(v7 - v9 - v8)) >> 15) ^ (uint)(v8 - v9 - ((uint)(v9 << 10) ^ (uint)(v7 - v9 - v8)));
	}

	__host__ __device__
		void SetRandomSeed(double x, double y)
	{
		randomCTR = 0;

		uint ws = world_seed;
		uint a = ws ^ 0x93262e6f;
		uint b = a & 0xfff;
		uint c = (a >> 0xc) & 0xfff;

		double x_ = x + b;

		double y_ = y + c;

		double r = x_ * 134217727.0;
		ulong e = SetRandomSeedHelper(r);

		ulong _x = *(ulong*)&x_ & 0x7fffffffffffffff;
		ulong _y = *(ulong*)&y_ & 0x7fffffffffffffff;
		if (102400.0 <= *(double*)&_y || *(double*)&_x <= 1.0)
		{
			r = y_ * 134217727.0;
		}
		else
		{
			double y__ = y_ * 3483.328;
			double t = (double)e;
			y__ += t;
			y_ *= y__;
			r = y_;
		}

		ulong f = SetRandomSeedHelper(r);

		uint g = SetRandomSeedHelper2((uint)e, (uint)f, ws);

		//double s = g;
		//s /= 4294967295.0;
		//s *= 2147483639.0;
		//s += 1.0;
		//Seed = (int)s;

		//Kaliuresis bithackery!!! Nobody knows how it works. Equivalent to the above FP64 code.
		const uint diddle_table[17] = { 0, 4, 6, 25, 12, 39, 52, 9, 21, 64, 78, 92, 104, 118, 18, 32, 44 };
		constexpr uint magic_number = 252645135; //magic number is 1/(1-2*actual ratio)
		uint t = g;
		t = g + (g < 2147483648) + (g == 0);
		t -= g / magic_number;
		t += (g % magic_number < diddle_table[g / magic_number]) && (g < 0xc3c3c3c3 + 4 || g >= 0xc3c3c3c3 + 62);
		t = (t + (g > 0x80000000)) >> 1;
		t = (int)t + (g == 0xffffffff);
		Seed = t;

		Next();

		uint h = ws & 3;
		while (h > 0)
		{
			Next();
			h--;
		}
	}

	//__host__ __device__
	//	uint NextU()
	//{
	//	Next();
	//	return (uint)((Seed * 4.656612875e-10) * 2147483645.0);
	//}

	__host__ __device__
		float Next()
	{
		randomCTR++;
		int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
		if (v4 < 0)
		{
			v4 += 0x7fffffff;
		}
		Seed = v4;
		return (float)Seed / 0x7fffffff;
	}

	__host__ __device__
		int Random(int a, int b)
	{
		randomCTR++;
		int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
		if (v4 < 0)
		{
			v4 += 0x7fffffff;
		}
		Seed = v4;
		return a + (int)(((ulong)(b + 1 - a) * (ulong)Seed) >> 31);
	}

	//__host__ __device__
	//	int Random(int a, int b)
	//{
	//	return a + (int)((b + 1 - a) * (double)Next());
	//}

	__host__ __device__
		float ProceduralRandomf(double x, double y, float a, float b)
	{
		SetRandomSeed(x, y);
		return (a + ((b - a) * Next()));
	}

	__host__ __device__
		int ProceduralRandomi(double x, double y, int a, int b)
	{
		SetRandomSeed(x, y);
		return Random(a, b);
	}

	__host__ __device__
		float GetDistribution(float mean, float sharpness, float baseline)
	{
		int i = 0;
		do
		{
			float r1 = Next();
			float r2 = Next();
			float div = fabsf(r1 - mean);
			if (r2 < ((1.0 - div) * baseline))
			{
				return r1;
			}
			if (div < 0.5)
			{
				// double v11 = sin(((0.5f - mean) + r1) * M_PI);
				float v11 = sinf(((0.5f - mean) + r1) * 3.1415f);
				float v12 = powf(v11, sharpness);
				if (v12 > r2)
				{
					return r1;
				}
			}
			i++;
		} while (i < 100);
		return Next();
	}

	__host__ __device__
		int RandomDistribution(int min, int max, int mean, float sharpness)
	{
		if (sharpness == 0)
		{
			return Random(min, max);
		}

		float adjMean = (mean - min) / (float)(max - min);
		float v7 = GetDistribution(adjMean, sharpness, 0.005f); // Baseline is always this
		int d = (int)roundf((max - min) * v7);
		return min + d;
	}

	__host__ __device__
		int RandomDistribution(float min, float max, float mean, float sharpness)
	{
		return (int)RandomDistribution((int)min, (int)max, (int)mean, sharpness);
	}

	__host__ __device__
		float RandomDistributionf(float min, float max, float mean, float sharpness)
	{
		if (sharpness == 0.0)
		{
			float r = Next();
			return (r * (max - min)) + min;
		}
		float adjMean = (mean - min) / (max - min);
		return min + (max - min) * GetDistribution(adjMean, sharpness, 0.005f); // Baseline is always this
	}
};