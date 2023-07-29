#pragma once

#define __universal__ __host__ __device__

#include <cstdint>
struct Vec2i {
	int x;
	int y;

	__universal__ Vec2i() {
		x = -1;
		y = -1;
	}

	__universal__ Vec2i(int _x, int _y) {
		x = _x;
		y = _y;
	}
	
	__universal__ Vec2i operator+(Vec2i other) {
		return { x + other.x, y + other.y };
	}
	
	__universal__ Vec2i operator*(int scalar) {
		return { x * scalar, y * scalar };
	}
};

uint64_t operator""_MB(uint64_t x)
{
	return x * 1024 * 1024;
}

uint64_t operator""_GB(uint64_t x)
{
	return x * 1024 
		* 1024 * 1024;
}
//putting these here because primitives are included like everywhere
__universal__ int min(int a, int b);
__universal__ int max(int a, int b);