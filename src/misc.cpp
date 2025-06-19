#include "../include/misc_funcs.h"
#include "../include/noita_random.h"
#include "../include/search_structs.h"
#include "../platforms/platform_implementation.h"
#include <cmath>
#include <cstdint>
#include <cstring>

_universal uint8_t readByte(const uint8_t* ptr, int& offset) { return ptr[offset++]; }
_universal void writeByte(MemSpan ptr, int& offset, uint8_t b) {
	ptr.is_safe("writeByte", "", offset);
	ptr.ptr[offset++] = b;
}
_universal int readInt(const uint8_t* ptr, int& offset) {
	int tmp;
	memcpy(&tmp, ptr + offset, 4);
	offset += 4;
	return tmp;
}
_universal void writeInt(MemSpan ptr, int& offset, int val) {
	ptr.is_safe("writeInt", "", offset);
	memcpy(ptr.ptr + offset, &val, 4);
	offset += 4;
}
_universal void incrInt(int* ptr) {
	int offsetTmp = 0;
	int tmp = readInt((const uint8_t*)ptr, offsetTmp);
	offsetTmp = 0;
	writeInt({(uint8_t*)ptr, 4}, offsetTmp, tmp + 1);
}
_universal short readShort(const uint8_t* ptr, int& offset) {
	return (readByte(ptr, offset) | (readByte(ptr, offset) << 8));
}
_universal void writeShort(MemSpan ptr, int& offset, short s) {
	writeByte(ptr, offset, ((short)s) & 0xff);
	writeByte(ptr, offset, (((short)s) >> 8) & 0xff);
}
_universal int readMisaligned(const int* ptr) {
	int offset = 0;
	return readInt((const uint8_t*)ptr, offset);
}
_universal Spawnable readMisalignedSpawnable(const Spawnable* sPtr) {
	const uint8_t* ptr = (const uint8_t*)sPtr;
	Spawnable s;
	int offset = 0;
	s.x = readInt(ptr, offset);
	s.y = readInt(ptr, offset);
	s.sType = (SpawnableMetadata)readByte(ptr, offset);
	s.count = readInt(ptr, offset);
	return s;
}
_universal WandData readMisalignedWand(const WandData* wPtr) {
	WandData w = {};
	cMemcpyU(&w, wPtr, 23);
	return w;
}

_universal WorldgenPRNG::WorldgenPRNG(uint32_t seed) {
	Seed = seed;
	Next();
}
_universal uint32_t WorldgenPRNG::NextU() {
	Next();
	return (uint32_t)((Seed * 4.656612875e-10) * 2147483645.0);
}
_universal void WorldgenPRNG::Next() {
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0) {
		v4 += 0x7fffffff;
	}
	Seed = v4;
}

_universal static uint64_t SetRandomSeedHelper(double r) {
	uint64_t e = *(uint64_t*)&r;
	e &= 0x7fffffffffffffff;

	int64_t c = (r < 0) ? -1 : 1;

	uint64_t f = (e & 0xfffffffffffff) | 0x0010000000000000;
	uint64_t g = 0x433 - (e >> 0x34);
	uint64_t h = f >> ((int)g & 63);

	uint32_t j = ~(uint32_t)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
	uint64_t a = (uint64_t)j << 0x20 | j;
	int64_t b = ((~a & h) | (f << 0xd & a)) * c;
	return b & 0xffffffff;
}
_universal static uint64_t SetRandomSeedHelperInt(int64_t r) {
	double dr = r;
	uint64_t e = *(uint64_t*)&dr;
	e &= 0x7fffffffffffffff;

	int64_t c = (r < 0) ? -1 : 1;

	uint64_t f = (e & 0xfffffffffffff) | 0x0010000000000000;
	uint64_t g = 0x433 - (e >> 0x34);
	uint64_t h = f >> ((int)g & 63);

	uint32_t j = ~(uint32_t)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
	uint64_t a = (uint64_t)j << 0x20 | j;
	int64_t b = ((~a & h) | (f << 0xd & a)) * c;
	return b & 0xffffffff;
}

_universal static uint32_t SetRandomSeedHelper2(uint32_t a, uint32_t b, uint32_t ws) {
	uint32_t uVar1;
	uint32_t uVar2;
	uint32_t uVar3;

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

_universal NollaPRNG::NollaPRNG(uint32_t worldSeed) {
	world_seed = worldSeed;
	Seed = worldSeed;
}
_universal _noinline void NollaPRNG::SetRandomSeed(double x, double y) {
	uint32_t ws = world_seed;
	uint32_t a = ws ^ 0x93262e6f;
	x += a & 0xfff;
	y += (a >> 12) & 0xfff;
	double r = x * 134217727.0;
	// Apparently equivalent?
	// Seems to be correct for the inputs that get generated anyway.
	uint32_t e = r ? (uint32_t)r : 2u; //SetRandomSeedHelper(r);
	if (102400.0 <= fabs(y) || fabs(x) <= 1.0)
		r = y * 134217727.0;
	else
		r = y * (y * 3483.328 + e);

	uint32_t g = SetRandomSeedHelper2(e, r ? (uint32_t)r : 2u, ws);

#ifndef __CUDA_ARCH__
	double s = g;
	s /= 4294967295.0;
	s *= 2147483639.0;
	s += 1.0;
	Seed = (int)s;
#else
	//Kaliuresis bithackery!!! Nobody knows how it works. Equivalent to the above FP64 code.
	const uint32_t diddle_table[17] = {0, 4, 6, 25, 12, 39, 52, 9, 21, 64, 78, 92, 104, 118, 18, 32, 44};
	constexpr uint32_t magic_number = 252645135; //magic number is 1/(1-2*actual ratio)
	uint32_t t = g + (g < 2147483648) + (g == 0);
	t -= g / magic_number;
	t += (g % magic_number < diddle_table[g / magic_number]) && (g < 0xc3c3c3c3 + 4 || g >= 0xc3c3c3c3 + 62);
	t = (t + (g > 0x80000000)) >> 1;
	t = (int)t + (g == 0xffffffff);
	Seed = t;
#endif

	Next();
	for (int h = ws & 3; h > 0; h--)
		Next();
}
_universal _noinline void NollaPRNG::SetRandomSeedInt(int x, int y) {
	uint32_t ws = world_seed;
	uint32_t a = ws ^ 0x93262e6f;
	x += a & 0xfff;
	y += (a >> 12) & 0xfff;
	uint64_t r = x * 134217727LLU;
	uint64_t e = r ? (uint32_t)r : 2u; //SetRandomSeedHelperInt(r);
	if (102400 <= abs(y) || abs(x) <= 1)
		r = y * 134217727;
	else
		r = (uint32_t)(y * (y * 3483.328 + e));
	uint32_t g = SetRandomSeedHelper2(e, r, ws);

#ifndef __CUDA_ARCH__
	double s = g;
	s /= 4294967295.0;
	s *= 2147483639.0;
	s += 1.0;
	Seed = (int)s;
#else
	//Kaliuresis bithackery!!! Nobody knows how it works. Equivalent to the above FP64 code.
	const uint32_t diddle_table[17] = {0, 4, 6, 25, 12, 39, 52, 9, 21, 64, 78, 92, 104, 118, 18, 32, 44};
	constexpr uint32_t magic_number = 252645135; //magic number is 1/(1-2*actual ratio)
	uint32_t t = g + (g < 2147483648) + (g == 0);
	t -= g / magic_number;
	t += (g % magic_number < diddle_table[g / magic_number]) && (g < 0xc3c3c3c3 + 4 || g >= 0xc3c3c3c3 + 62);
	t = (t + (g > 0x80000000)) >> 1;
	t = (int)t + (g == 0xffffffff);
	Seed = t;
#endif

	Next();
	for (int h = ws & 3; h > 0; h--)
		Next();
}
_universal float NollaPRNG::Next() {
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0) {
		v4 += 0x7fffffff;
	}
	Seed = v4;
	return (float)Seed / 0x7fffffff;
}
_universal void NollaPRNG::Prev() { Seed = ((uint64_t)Seed * 1407677000ull) % 0x7fffffffu; }
_universal double NollaPRNG::NextD() {
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0) {
		v4 += 0x7fffffff;
	}
	Seed = v4;
	return (double)Seed / 0x7fffffff;
}
_universal int NollaPRNG::Random(int a, int b) {
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0) {
		v4 += 0x7fffffff;
	}
	Seed = v4;
	//return a + (int)(((b - a + 1) * (uint64_t)(Seed)) >> 31);
	return a + (int)(((double)(b - a + 1) * (double)Seed * 4.656612875e-10));
	//return a + (int)(((float)(b + 1 - a) * (float)Seed * 4.656612875e-10f));
}
_universal int NollaPRNG::RandomD(int a, int b) {
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0) {
		v4 += 0x7fffffff;
	}
	Seed = v4;
	return a + (int)(((double)(b - a + 1) * (double)Seed * 4.656612875e-10));
}
_universal float NollaPRNG::ProceduralRandomf(double x, double y, float a, float b) {
	SetRandomSeed(x, y);
	return a + ((b - a) * Next());
}
_universal int NollaPRNG::ProceduralRandomi(double x, double y, int a, int b) {
	SetRandomSeed(x, y);
	return Random(a, b);
}
_universal _noinline float NollaPRNG::GetDistribution(float mean, float sharpness, float baseline) {
	int i = 0;
	do {
		float r1 = Next();
		float r2 = Next();
		float div = fabsf(r1 - mean);
		if (r2 < ((1.0f - div) * baseline)) {
			return r1;
		}
		if (div < 0.5f) {
			// double v11 = sin(((0.5f - mean) + r1) * M_PI);
			float v11 = sinf(((0.5f - mean) + r1) * 3.14159265f);
			float v12 = powf(v11, sharpness);
			if (v12 > r2) {
				return r1;
			}
		}
		i++;
	} while (i < 100);
	return Next();
}
_universal int NollaPRNG::RandomDistribution(int min, int max, int mean, float sharpness) {
	if (sharpness == 0) {
		return Random(min, max);
	}

	float adjMean = (mean - min) / (float)(max - min);
	float v7 = GetDistribution(adjMean, sharpness, 0.005f); // Baseline is always this
	int d = (int)rintf((max - min) * v7);
	return min + d;
}
_universal int NollaPRNG::RandomDistribution(float min, float max, float mean, float sharpness) {
	return (int)RandomDistribution((int)min, (int)max, (int)mean, sharpness);
}
_universal float NollaPRNG::RandomDistributionf(float min, float max, float mean, float sharpness) {
	if (sharpness == 0.0f) {
		float r = Next();
		return (r * (max - min)) + min;
	}
	float adjMean = (mean - min) / (max - min);
	return min + (max - min) * GetDistribution(adjMean, sharpness, 0.005f); // Baseline is always this
}

_compute float random_next(float min, float max, NollaPRNG& random, Vec2i& rnd) {
	random.SetRandomSeedInt(rnd.x, rnd.y);
	float result = min + ((max - min) * random.Next());
	rnd.y += 1;
	return result;
}
_compute int random_nexti(float min, float max, NollaPRNG& random, Vec2i& rnd) {
	random.SetRandomSeedInt(rnd.x, rnd.y);
	int result = random.Random(min, max);
	rnd.y += 1;
	return result;
}
_compute int pick_random_from_table_backwards(const float* probs, int length, NollaPRNG& random, Vec2i& rnd) {
	for (int i = length - 1; i > 0; i--) {
		if (random_next(0, 1, random, rnd) <= probs[i])
			return i;
	}
	return 0;
}
_compute int pick_random_from_table_weighted(const float* probs, float sum, int length, NollaPRNG& random, Vec2i& rnd) {
	float val = random_next(0, sum, random, rnd);
	for (int i = 0; i < length; i++) {
		if (val < probs[i])
			return i;
		val -= probs[i];
	}
	return 0;
}

int pick_world_seed(uint64_t time) {
	if (time > 0x7fffffff)
		time >>= 1;
	double r = (double)time;
	time >>= 0x1f;
	r += time * 8;

	if (r > 2147483647.0)
		r *= 0.5;

	int Seed = (int)r;
	for (int i = 0; i < 2; i++) {
		Seed = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
		if (Seed < 0) {
			Seed += 0x7fffffff;
		}
	}
	r = Seed;

	r = ((r * 4.656612875e-10) * 2147483646.0);
	int out = SetRandomSeedHelper(r);
	return out;
}

_compute MemSpan ArenaAlloc(MemoryArena& arena, uint32_t size) {
	uint8_t* ptr = arena.ptr + arena.offset;
	arena.offset += size;
	return {ptr, size};
}
_compute MemSpan ArenaAlloc(MemoryArena& arena, uint32_t size, uint32_t alignmentWidth) {
	uint64_t alignedAddr = ((uint64_t)arena.ptr + arena.offset + alignmentWidth - 1) & ~((uint64_t)alignmentWidth - 1);
	arena.offset = alignedAddr - (uint64_t)arena.ptr + size;
	return {(uint8_t*)alignedAddr, size};
}
_compute void ArenaSetOffset(MemoryArena& arena, uint8_t* endPointer) {
	uint64_t offset = endPointer - arena.ptr;
	arena.offset = offset;
}

_universal uint32_t createRGB(const uint8_t r, const uint8_t g, const uint8_t b) { return (r << 16) | (g << 8) | b; }
_universal int GetWidthFromPix(int a, int b) { return ((b * 512) / 10 - (a * 512) / 10); }
_universal Vec2i GetGlobalPos(const int x, const int y, const int px, int py) {
	int gx = ((512 * x - 512 * 35 - 9 * (x < 35)) / 10) * 10 + px - 5;
	int gy = ((512 * y - 512 * 14 + 9 * (y > 14)) / 10) * 10 + py - 13;
	return {gx, gy};
}
_universal Vec2i GetLocalPos(const int gx, int gy) {
	int x = (((gx + 5) / 10) * 10 + (512 * 35)) / 512;
	int y = (((gy + 13) / 10) * 10 + (512 * 14)) / 512;
	return {x, y};
}
_compute int roundRNGPos(int num) {
	if (-1000000 < num && num < 1000000)
		return num;
	else if (-10000000 < num && num < 10000000)
		return int(num / 10.0f) * 10;
	else if (-100000000 < num && num < 100000000)
		return int(num / 100.0f) * 100;
	return num;
}