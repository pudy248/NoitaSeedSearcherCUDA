#include "../platforms/platform_implementation.h"
#include "../include/misc_funcs.h"
#include "../include/search_structs.h"
#include "../include/noita_random.h"
#include <cstdint>
#include <cstring>
#include <cmath>

_universal uint8_t readByte(uint8_t* ptr, int& offset)
{
	return ptr[offset++];
}
_universal void writeByte(uint8_t* ptr, int& offset, uint8_t b)
{
	ptr[offset++] = b;
}
_universal int readInt(uint8_t* ptr, int& offset) {
	ptr += offset;
	offset += 4;
	return (ptr[3] << 24) | (ptr[2] << 16) | (ptr[1] << 8) | (ptr[0]);
}

_universal void writeInt(uint8_t* ptr, int& offset, int val) {
	ptr += offset;
	offset += 4;
	ptr[0] = val;
	ptr[1] = val >> 8;
	ptr[2] = val >> 16;
	ptr[3] = val >> 24;
}
_universal void incrInt(uint8_t* ptr)
{
	int offsetTmp = 0;
	int tmp = readInt(ptr, offsetTmp);
	offsetTmp = 0;
	writeInt(ptr, offsetTmp, tmp + 1);
}
_universal short readShort(uint8_t* ptr, int& offset)
{
	return (readByte(ptr, offset) | (readByte(ptr, offset) << 8));
}
_universal void writeShort(uint8_t* ptr, int& offset, short s)
{
	writeByte(ptr, offset, ((short)s) & 0xff);
	writeByte(ptr, offset, (((short)s) >> 8) & 0xff);
}
_compute int readMisaligned(int* ptr2)
{
	uint8_t* ptr = (uint8_t*)ptr2;
	int offset = 0;
	return readInt(ptr, offset);
}
_compute Spawnable readMisalignedSpawnable(Spawnable* sPtr)
{
	uint8_t* bPtr = (uint8_t*)sPtr;
	Spawnable s;
	int offset = 0;
	s.x = readInt(bPtr, offset);
	s.y = readInt(bPtr, offset);
	s.sType = (SpawnableMetadata)readByte(bPtr, offset);
	s.count = readInt(bPtr, offset);
	return s;
}
_compute WandData readMisalignedWand(WandData* wPtr)
{
	WandData w = {};
	cMemcpyU(&w, wPtr, 37);
	return w;
}

_universal WorldgenPRNG::WorldgenPRNG(double seed)
{
	Seed = seed;
	Next();
}
_universal uint32_t WorldgenPRNG::NextU()
{
	Next();
	return (uint32_t)((Seed * 4.656612875e-10) * 2147483645.0);
}
_universal double WorldgenPRNG::Next()
{
	int v4 = (int)Seed * 0x41a7 + ((int)Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0)
	{
		v4 += 0x7fffffff;
	}
	Seed = (double)v4;
	return Seed / 0x7fffffff;
}

_universal static uint64_t SetRandomSeedHelper(double r)
{
	uint64_t e = *(uint64_t*)&r;
	e &= 0x7fffffffffffffff;

	int64_t c = (r < 0) ? -1 : 1;

	uint64_t f = (e & 0xfffffffffffff) | 0x0010000000000000;
	uint64_t g = 0x433 - (e >> 0x34);
	uint64_t h = f >> (int)g;

	uint32_t j = ~(uint32_t)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
	uint64_t a = (uint64_t)j << 0x20 | j;
	int64_t b = ((~a & h) | (f << 0xd & a)) * c;
	return b & 0xffffffff;
}
_universal static uint64_t SetRandomSeedHelperInt(int64_t r)
{
	double dr = r;
	uint64_t e = *(uint64_t*)&dr;
	e &= 0x7fffffffffffffff;

	int64_t c = (r < 0) ? -1 : 1;

	uint64_t f = (e & 0xfffffffffffff) | 0x0010000000000000;
	uint64_t g = 0x433 - (e >> 0x34);
	uint64_t h = f >> (int)g;

	uint32_t j = ~(uint32_t)(0x433 < (((e >> 0x20) & 0xffffffff) >> 0x14) ? 1 : 0) + 1;
	uint64_t a = (uint64_t)j << 0x20 | j;
	int64_t b = ((~a & h) | (f << 0xd & a)) * c;
	return b & 0xffffffff;
}
_universal static uint32_t SetRandomSeedHelper2(uint32_t a, uint32_t b, uint32_t ws)
{
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

_universal NollaPRNG::NollaPRNG(uint32_t worldSeed)
{
	world_seed = worldSeed;
	Seed = worldSeed;
}
_universal _noinline void NollaPRNG::SetRandomSeed(double x, double y)
{
	uint32_t ws = world_seed;
	uint32_t a = ws ^ 0x93262e6f;
	uint32_t b = a & 0xfff;
	uint32_t c = (a >> 0xc) & 0xfff;

	double x_ = x + b;

	double y_ = y + c;

	double r = x_ * 134217727.0;
	// Apparently equivalent?
	// Seems to be correct for the inputs that get generated anyway.
	uint32_t e = (uint32_t)(int64_t)r; //SetRandomSeedHelper(r);
	// Debug, remove later
	if (SetRandomSeedHelper(r) != (uint32_t)(int64_t)r) printf("e %lli : %lli (%f)\n", SetRandomSeedHelper(r), (uint32_t)(int64_t)r, r);


	uint64_t _x = *(uint64_t*)&x_ & 0x7fffffffffffffff;
	uint64_t _y = *(uint64_t*)&y_ & 0x7fffffffffffffff;
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

	uint32_t f = (uint32_t)(int64_t)r; //SetRandomSeedHelper(r);
	//if (SetRandomSeedHelper(r) != (uint32_t)(int64_t)r) printf("f %lli : %lli (%f)\n", SetRandomSeedHelper(r), (uint32_t)(int64_t)r, r);

	uint32_t g = SetRandomSeedHelper2((uint32_t)e, (uint32_t)f, ws);

	//double s = g;
	//s /= 4294967295.0;
	//s *= 2147483639.0;
	//s += 1.0;
	//Seed = (int)s;

	//Kaliuresis bithackery!!! Nobody knows how it works. Equivalent to the above FP64 code.
	const uint32_t diddle_table[17] = { 0, 4, 6, 25, 12, 39, 52, 9, 21, 64, 78, 92, 104, 118, 18, 32, 44 };
	constexpr uint32_t magic_number = 252645135; //magic number is 1/(1-2*actual ratio)
	uint32_t t = g + (g < 2147483648) + (g == 0);
	t -= g / magic_number;
	t += (g % magic_number < diddle_table[g / magic_number]) && (g < 0xc3c3c3c3 + 4 || g >= 0xc3c3c3c3 + 62);
	t = (t + (g > 0x80000000)) >> 1;
	t = (int)t + (g == 0xffffffff);
	Seed = t;

	Next();

	uint32_t h = ws & 3;
	while (h > 0)
	{
		Next();
		h--;
	}
}
_universal _noinline void NollaPRNG::SetRandomSeedInt(int x, int y)
{
	uint32_t ws = world_seed;
	uint32_t a = ws ^ 0x93262e6f;
	int b = a & 0xfff;
	int c = (a >> 0xc) & 0xfff;

	int x_ = x + b;

	int y_ = y + c;

	long long r = x_ * 134217727LLU;
	uint64_t e = SetRandomSeedHelperInt(r);

	int _x = abs(x_);
	int _y = abs(y_);
	if (102400 <= _y || _x <= 1)
	{
		r = y_ * 134217727LLU;
	}
	else
	{
		double y__ = y_ * 3483.328;
		double t = (double)e;
		y__ += t;
		r = y_ * y__;
	}

	uint64_t f = SetRandomSeedHelperInt(r);

	uint32_t g = SetRandomSeedHelper2((uint32_t)e, (uint32_t)f, ws);

	//double s = g;
	//s /= 4294967295.0;
	//s *= 2147483639.0;
	//s += 1.0;
	//Seed = (int)s;

	//Kaliuresis bithackery!!! Nobody knows how it works. Equivalent to the above FP64 code.
	const uint32_t diddle_table[17] = { 0, 4, 6, 25, 12, 39, 52, 9, 21, 64, 78, 92, 104, 118, 18, 32, 44 };
	constexpr uint32_t magic_number = 252645135; //magic number is 1/(1-2*actual ratio)
	uint32_t t = g + (g < 2147483648) + (g == 0);
	t -= g / magic_number;
	t += (g % magic_number < diddle_table[g / magic_number]) && (g < 0xc3c3c3c3 + 4 || g >= 0xc3c3c3c3 + 62);
	t = (t + (g > 0x80000000)) >> 1;
	t = (int)t + (g == 0xffffffff);
	Seed = t;

	Next();

	uint32_t h = ws & 3;
	while (h > 0)
	{
		Next();
		h--;
	}
}
_universal float NollaPRNG::Next()
{
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0)
	{
		v4 += 0x7fffffff;
	}
	Seed = v4;
	return (float)Seed / 0x7fffffff;
}
_universal double NollaPRNG::NextD()
{
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0)
	{
		v4 += 0x7fffffff;
	}
	Seed = v4;
	return (double)Seed / 0x7fffffff;
}
_universal int NollaPRNG::Random(int a, int b)
{
	int v4 = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
	if (v4 < 0)
	{
		v4 += 0x7fffffff;
	}
	Seed = v4;
	return a + (int)(((uint64_t)(b + 1 - a) * (uint64_t)Seed) >> 31);
}
_universal float NollaPRNG::ProceduralRandomf(double x, double y, float a, float b)
{
	SetRandomSeed(x, y);
	return a + ((b - a) * Next());
}
_universal int NollaPRNG::ProceduralRandomi(double x, double y, int a, int b)
{
	SetRandomSeed(x, y);
	return Random(a, b);
}
_universal _noinline float NollaPRNG::GetDistribution(float mean, float sharpness, float baseline)
{
	int i = 0;
	do
	{
		float r1 = Next();
		float r2 = Next();
		float div = fabsf(r1 - mean);
		if (r2 < ((1.0f - div) * baseline))
		{
			return r1;
		}
		if (div < 0.5f)
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
_universal int NollaPRNG::RandomDistribution(int min, int max, int mean, float sharpness)
{
	if (sharpness == 0)
	{
		return Random(min, max);
	}

	float adjMean = (mean - min) / (float)(max - min);
	float v7 = GetDistribution(adjMean, sharpness, 0.005f); // Baseline is always this
	int d = (int)rintf((max - min) * v7);
	return min + d;
}
_universal int NollaPRNG::RandomDistribution(float min, float max, float mean, float sharpness)
{
	return (int)RandomDistribution((int)min, (int)max, (int)mean, sharpness);
}
_universal float NollaPRNG::RandomDistributionf(float min, float max, float mean, float sharpness)
{
	if (sharpness == 0.0)
	{
		float r = Next();
		return (r * (max - min)) + min;
	}
	float adjMean = (mean - min) / (max - min);
	return min + (max - min) * GetDistribution(adjMean, sharpness, 0.005f); // Baseline is always this
}

_compute float random_next(float min, float max, NollaPRNG& random, Vec2i& rnd)
{
	random.SetRandomSeedInt(rnd.x, rnd.y);
	float result = min + ((max - min) * random.Next());
	rnd.y += 1;
	return result;
}
_compute int random_nexti(float min, float max, NollaPRNG& random, Vec2i& rnd)
{
	random.SetRandomSeedInt(rnd.x, rnd.y);
	int result = random.Random(min, max);
	rnd.y += 1;
	return result;
}
_compute int pick_random_from_table_backwards(const float* probs, int length, NollaPRNG& random, Vec2i& rnd)
{
	for (int i = length - 1; i > 0; i--)
	{
		if (random_next(0, 1, random, rnd) <= probs[i]) return i;
	}
	return 0;
}
_compute int pick_random_from_table_weighted(const float* probs, float sum, int length, NollaPRNG& random, Vec2i& rnd)
{
	float val = random_next(0, sum, random, rnd);
	for (int i = 0; i < length; i++)
	{
		if (val < probs[i]) return i;
		val -= probs[i];
	}
	return 0;
}

int pick_world_seed(uint64_t time)
{
	if (time > 0x7fffffff)
		time >>= 1;
	double r = (double)time;
	time >>= 0x1f;
	r += time * 8;

	if (r > 2147483647.0)
		r *= 0.5;

	int Seed = (int)r;
	for (int i = 0; i < 2; i++)
	{
		Seed = Seed * 0x41a7 + (Seed / 0x1f31d) * -0x7fffffff;
		if (Seed < 0)
		{
			Seed += 0x7fffffff;
		}
	}
	r = Seed;
	
	r = ((r * 4.656612875e-10) * 2147483646.0);
	int out = SetRandomSeedHelper(r);
	return out;
}

_compute uint8_t* ArenaAlloc(MemoryArena& arena, uint64_t size)
{
	uint8_t* ptr = arena.ptr + arena.offset;
	arena.offset += size;
	return ptr;
}
_compute uint8_t* ArenaAlloc(MemoryArena& arena, uint64_t size, uint64_t alignmentWidth)
{
	uint8_t* ptr = arena.ptr + arena.offset;
	uint64_t ptrAddr = (uint64_t)ptr;
	int alignment = ptrAddr % alignmentWidth;
	arena.offset += alignmentWidth - alignment;
	uint8_t* alignedPtr = arena.ptr + arena.offset;
	arena.offset += size;
	return alignedPtr;
}
_compute void ArenaSetOffset(MemoryArena& arena, uint8_t* endPointer)
{
	uint64_t offset = endPointer - arena.ptr;
	arena.offset = offset;
}

_universal uint32_t createRGB(const uint8_t r, const uint8_t g, const uint8_t b)
{
	return (r << 16) | (g << 8) | b;
}
_universal int GetWidthFromPix(int a, int b)
{
	return ((b * 512) / 10 - (a * 512) / 10);
}
_universal Vec2i GetGlobalPos(const int x, const int y, const int px, int py)
{
	int gx = ((512 * x) / 10 - (512 * 35) / 10) * 10 + px - 5;
	int gy = ((512 * y) / 10 - (512 * 14) / 10) * 10 + py - 13;
	return { gx, gy };
}
_universal Vec2i GetLocalPos(const int gx, int gy)
{
	int x = (((gx + 5) / 10) * 10 + (512 * 35)) / 512;
	int y = (((gy + 13) / 10) * 10 + (512 * 14)) / 512;
	return { x, y };
}
_compute int roundRNGPos(int num)
{
	if (-1000000 < num && num < 1000000) return num;
	else if (-10000000 < num && num < 10000000) return rintf(num / 10.0) * 10;
	else if (-100000000 < num && num < 100000000) return rintf(num / 100.0) * 100;
	return num;
}

_universal void _itoa_offset(int num, int base, char* buffer, int& offset)
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
_universal void _itoa_offset_decimal(int num, int base, int fixedPoint, char* buffer, int& offset)
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

	while (num != 0 || 10 - i < (fixedPoint + 2))
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
_universal void _itoa_offset_zeroes(int num, int base, int leadingZeroes, char* buffer, int& offset)
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
_universal void _putstr_offset(const char* str, char* buffer, int& offset)
{
	int i = 0;
	while (str[i] != '\0')
	{
		buffer[offset++] = str[i++];
	}
}
