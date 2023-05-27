#pragma once

#ifdef _MSC_VER //already defined for GCC?
typedef unsigned char byte;
#endif
typedef signed char sbyte;
typedef unsigned short ushort;
typedef unsigned int uint;
#ifdef _MSC_VER //already defined for GCC?
typedef unsigned long long int ulong;
#endif

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

size_t operator""_MB(size_t x)
{
	return x * 1024 * 1024;
}

size_t operator""_GB(size_t x)
{
	return x * 1024 * 1024 * 1024;
}