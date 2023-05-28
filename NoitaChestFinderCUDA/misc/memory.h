#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"

#include <iostream>

struct MemoryArena
{
	uint8_t* ptr;
	uint64_t offset;
};

__device__ uint8_t* ArenaAlloc(MemoryArena& arena, uint64_t size)
{
	uint8_t* ptr = arena.ptr + arena.offset;
	arena.offset += size;
	return ptr;
}

__device__ uint8_t* ArenaAlloc(MemoryArena& arena, uint64_t size, uint64_t alignmentWidth)
{
	uint8_t* ptr = arena.ptr + arena.offset;
	uint64_t ptrAddr = (uint64_t)ptr;
	int alignment = ptrAddr % alignmentWidth;
	arena.offset += alignmentWidth - alignment;
	uint8_t* alignedPtr = arena.ptr + arena.offset;
	arena.offset += size;
	return alignedPtr;
}

__device__ void ArenaSetOffset(MemoryArena& arena, uint8_t* endPointer)
{
	uint64_t offset = endPointer - arena.ptr;
	arena.offset = offset;
}