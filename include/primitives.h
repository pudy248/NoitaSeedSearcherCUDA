#pragma once
#include "../platforms/platform_implementation.h"
#include <cstdint>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

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

uint64_t operator""_KB(uint64_t x)
{
	return x * 1024;
}
uint64_t operator""_MB(uint64_t x)
{
	return x * 1024 * 1024;
}
uint64_t operator""_GB(uint64_t x)
{
	return x * 1024 
		* 1024 * 1024;
}