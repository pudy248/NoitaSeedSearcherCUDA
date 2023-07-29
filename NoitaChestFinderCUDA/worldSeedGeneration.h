#pragma once

#include <cmath>

int SetWorldSeedHelper(double r)
{
	uint64_t e = *(uint64_t*)&r;

	if (((e >> 0x20 & 0x7fffffff) < 0x7ff00000) && (-9.223372036854776e+18 <= r) && (r < 9.223372036854776e+18))
	{
		e <<= 1;
		e >>= 1;
		double s = *(double*)&e;
		uint64_t i = 0;
		if (s != 0.0)
		{
			uint64_t f = (e & 0xfffffffffffff) | 0x0010000000000000;
			uint64_t g = 0x433 - (e >> 0x34);
			uint64_t h = f >> (int)g;

			uint32_t j = ~(uint32_t)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
			i = (uint64_t)j << 0x20 | j;
			i = ~i & h | f << (((int)s >> 0x34) - 0x433) & i;
			i = ~(~(uint32_t)(r == s ? 1 : 0) + 1) & (~i + 1) | i & (~(uint32_t)(r == s ? 1 : 0) + 1);
		}
		return i & 0xffffffff;
	}
	return 0;
}
int GenerateSeed(uint64_t ECX)
{
	if (ECX > 0x7fffffff)
		ECX >>= 1;
	double XMM0_D = (double)ECX;
	ECX >>= 0x1f;
	XMM0_D += ECX * 8;

	if (XMM0_D > 2147483647.0)
		XMM0_D *= 0.5;
	int ESI = (int)XMM0_D;

	ESI = ESI * 0x41a7 + (ESI / 0x1f31d) * -0x7fffffff;
	if (ESI < 0)
	{
		ESI += 0x7fffffff;
	}
	int i_ECX = ESI * 0x41a7 + (ESI / 0x1f31d) * -0x7fffffff;
	if (i_ECX < 0)
	{
		i_ECX += 0x7fffffff;
	}
	XMM0_D = i_ECX;
	XMM0_D = ((XMM0_D * 4.656612875e-10) * 2147483646.0);
	int out = SetWorldSeedHelper(XMM0_D);
	return out;
}

time_t NextOccurenceOf(uint32_t seed)
{
	for (time_t i = _time64(NULL) - 100000; i < UINT_MAX; i++)
	{
		//if (i % 100000000 == 0) printf("incr %lli\n", i);
		uint32_t cs = GenerateSeed(i);
		if (cs == seed) return i;
	}
	return 0;
}