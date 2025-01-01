#pragma once
#include "../platforms/platform_implementation.h"
#include <cstdint>
#include <cstdio>

template <typename A, typename B>
_universal constexpr auto max(A a, B b) { return (a > b) ? a : b; }
template <typename A, typename B>
_universal constexpr auto min(A a, B b) { return (a > b) ? b : a; }

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
	
	_universal Vec2i operator+(Vec2i other) {
		return { x + other.x, y + other.y };
	}
	
	_universal Vec2i operator*(int scalar) {
		return { x * scalar, y * scalar };
	}
};

struct MemSpan {
	uint8_t* ptr;
	uint64_t sz;
#ifdef NDEBUG
	constexpr static bool check = true;
#else
	constexpr static bool check = true;
#endif
	_universal constexpr bool is_safe(int idx, int elem_size = 1) const {
		if constexpr (!check) return true;
		if (idx < 0 || idx * elem_size >= sz) {
			printf("ERR");
		}
		return idx < 0 || idx * elem_size < sz;
	}
};

uint64_t operator""_KB(unsigned long long x)
{
	return x * 1024;
}
uint64_t operator""_MB(unsigned long long x)
{
	return x * 1024 * 1024;
}
uint64_t operator""_GB(unsigned long long x)
{
	return x * 1024 
		* 1024 * 1024;
}