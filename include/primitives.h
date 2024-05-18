#pragma once
#include "../platforms/platform_implementation.h"
#include <cstdint>
#include <cmath>

//#define max(a, b) ((a) > (b) ? (a) : (b))
//#define min(a, b) ((a) < (b) ? (a) : (b))

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

unsigned long long operator""_KB(unsigned long long x)
{
	return x * 1024;
}
unsigned long long operator""_MB(unsigned long long x)
{
	return x * 1024 * 1024;
}
unsigned long long operator""_GB(unsigned long long x)
{
	return x * 1024 
		* 1024 * 1024;
}