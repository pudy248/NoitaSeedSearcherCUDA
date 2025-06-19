#pragma once
#include "../platforms/platform_implementation.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace DEBUG {
enum DEBUG {
	LOG_VERBOSE = 1,
	LOG_BACKEND = 2,
	LOG_BIOME_SECTORS = 4,
	LOG_SPAWN_PIXELS = 8,
	LOG_SEED_BLOCKS = 16,
	LOG_CLI_PARSING = 32,
	SINGLE_THREAD = 64,
	ALL = 31,
};
}
int DEBUG_FLAGS = 0;
int DEBUG_SEED_BLOCK_OVERRIDE = 0;
int DEBUG_DISPATCH_RATE_OVERRIDE = 0;
bool QUIET = false;

template <typename A, typename B>
_universal constexpr auto max(A a, B b) {
	return (a > b) ? a : b;
}
template <typename A, typename B>
_universal constexpr auto min(A a, B b) {
	return (a > b) ? b : a;
}
template <typename A, typename B>
_universal constexpr auto mod(A a, B b) {
	return ((a % b) + b) % b;
}

_universal constexpr void Assert(bool condition, const char* msg) {
	if (!condition) {
		printf("%s\n", msg);
#ifndef __CUDA_ARCH__
		exit(-1);
#endif
	}
}

struct Vec2i {
	int x;
	int y;

	_universal Vec2i() {
		x = -1;
		y = -1;
	}

	_universal Vec2i(int _x, int _y) {
		x = _x;
		y = _y;
	}

	_universal Vec2i operator+(Vec2i other) { return {x + other.x, y + other.y}; }

	_universal Vec2i operator*(int scalar) { return {x * scalar, y * scalar}; }
};

struct MemSpan {
	uint8_t* ptr;
	uint32_t sz;
#ifdef NDEBUG
	constexpr static bool check = true;
#else
	constexpr static bool check = true;
#endif
	_universal constexpr bool is_safe(const char* func, const char* memCategory, int idx, int elem_size = 1) const {
		if constexpr (!check)
			return true;
#ifdef DEVICE_LOGGING
		if (idx < 0 || idx * elem_size >= sz) {
			printf("%s(): Ran out of %s space (%i vs. %i)\n", func, memCategory, idx * elem_size, sz);
		}
#endif
		return idx < 0 || idx * elem_size < sz;
	}
};

uint64_t operator""_KB(unsigned long long x) { return x * 1024; }
uint64_t operator""_MB(unsigned long long x) { return x * 1024 * 1024; }
uint64_t operator""_GB(unsigned long long x) { return x * 1024 * 1024 * 1024; }