#pragma once

typedef unsigned char byte;
typedef signed char sbyte;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long int ulong;

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
