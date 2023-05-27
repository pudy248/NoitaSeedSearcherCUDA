#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"

#include <iostream>

struct MemoryArena
{
	byte* ptr;
	size_t offset;
};

__device__ byte* ArenaAlloc(MemoryArena& arena, size_t size)
{
	byte* ptr = arena.ptr + arena.offset;
	arena.offset += size;
	return ptr;
}

__device__ byte* ArenaAlloc(MemoryArena& arena, size_t size, size_t alignmentWidth)
{
	byte* ptr = arena.ptr + arena.offset;
	size_t ptrAddr = (size_t)ptr;
	int alignment = ptrAddr % alignmentWidth;
	arena.offset += alignmentWidth - alignment;
	byte* alignedPtr = arena.ptr + arena.offset;
	arena.offset += size;
	return alignedPtr;
}

__device__ void ArenaSetOffset(MemoryArena& arena, byte* endPointer)
{
	size_t offset = endPointer - arena.ptr;
	arena.offset = offset;
}