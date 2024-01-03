#pragma once
#include "../platforms/platform_implementation.h"
#include "primitives.h"
#include <cstdint>

// Thanks to kaliuresis!
// Check out his orb atlas repository: https://github.com/kaliuresis/noa

class WorldgenPRNG
{
public:
	double Seed;
	_universal WorldgenPRNG(double seed);
	_universal uint32_t NextU();
	_universal double Next();
};

class NollaPRNG
{
public:
	uint32_t world_seed;
	int Seed;

	_universal NollaPRNG(uint32_t worldSeed);
	_universal uint64_t SetRandomSeedHelper(double r);
	_universal uint64_t SetRandomSeedHelperInt(int64_t r);
	_universal uint32_t SetRandomSeedHelper2(uint32_t a, uint32_t b, uint32_t ws);
	_universal _noinline void SetRandomSeed(double x, double y);
	_universal _noinline void SetRandomSeedInt(int x, int y);
	_universal float Next();
	_universal double NextD();
	_universal int Random(int a, int b);
	_universal float ProceduralRandomf(double x, double y, float a, float b);
	_universal int ProceduralRandomi(double x, double y, int a, int b);
	_universal _noinline float GetDistribution(float mean, float sharpness, float baseline);
	_universal int RandomDistribution(int min, int max, int mean, float sharpness);
	_universal int RandomDistribution(float min, float max, float mean, float sharpness);
	_universal float RandomDistributionf(float min, float max, float mean, float sharpness);
};

_compute float random_next(float min, float max, NollaPRNG& random, Vec2i& rnd);
_compute int random_nexti(float min, float max, NollaPRNG& random, Vec2i& rnd);
_compute int pick_random_from_table_backwards(const float* probs, int length, NollaPRNG& random, Vec2i& rnd);
_compute int pick_random_from_table_weighted(const float* probs, float sum, int length, NollaPRNG& random, Vec2i& rnd);
