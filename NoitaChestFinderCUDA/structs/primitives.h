#pragma once

struct IntPair {
	int x;
	int y;

	__host__ __device__ IntPair() {
		x = -1;
		y = -1;
	}

	__host__ __device__ IntPair(int _x, int _y) {
		x = _x;
		y = _y;
	}
	
	__host__ __device__ IntPair operator+(IntPair other) {
		return { x + other.x, y + other.y };
	}
	
	__host__ __device__ IntPair operator*(int scalar) {
		return { x * scalar, y * scalar };
	}
};

uint64_t operator""_MB(uint64_t x)
{
	return x * 1024 * 1024;
}

uint64_t operator""_GB(uint64_t x)
{
	return x * 1024 * 1024 * 1024;
}