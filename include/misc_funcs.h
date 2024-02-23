#pragma once
#include "../platforms/platform_implementation.h"
#include "../include/enums.h"
#include "../include/search_structs.h"
#include <cstdint>

_universal uint8_t readByte(uint8_t* ptr, int& offset);
_universal void writeByte(uint8_t* ptr, int& offset, uint8_t b);
_universal int readInt(uint8_t* ptr, int& offset);
_universal void writeInt(uint8_t* ptr, int& offset, int val);
_universal void incrInt(uint8_t* ptr);
_universal short readShort(uint8_t* ptr, int& offset);
_universal void writeShort(uint8_t* ptr, int& offset, short s);
_compute int readMisaligned(int* ptr2);
_compute Spawnable readMisalignedSpawnable(Spawnable* sPtr);
_compute WandData readMisalignedWand(WandData* wPtr);

struct MemoryArena
{
	uint8_t* ptr;
	uint64_t offset;
};
_compute uint8_t* ArenaAlloc(MemoryArena& arena, uint64_t size);
_compute uint8_t* ArenaAlloc(MemoryArena& arena, uint64_t size, uint64_t alignmentWidth);
_compute void ArenaSetOffset(MemoryArena& arena, uint8_t* endPointer);

_universal uint32_t createRGB(const uint8_t r, const uint8_t g, const uint8_t b);
_universal int GetWidthFromPix(int a, int b);
_universal Vec2i GetGlobalPos(const int x, const int y, const int px, int py);
_universal Vec2i GetLocalPos(const int gx, int gy);
_compute int roundRNGPos(int num);

_universal void _itoa_offset(int num, int base, char* buffer, int& offset);
_universal void _itoa_offset_decimal(int num, int base, int fixedPoint, char* buffer, int& offset);
_universal void _itoa_offset_zeroes(int num, int base, int leadingZeroes, char* buffer, int& offset);
_universal void _putstr_offset(const char* str, char* buffer, int& offset);

constexpr uint32_t COLOR_PURPLE = 0x7f007fU;
constexpr uint32_t COLOR_BLACK = 0x000000U;
constexpr uint32_t COLOR_WHITE = 0xffffffU;
constexpr uint32_t COLOR_YELLOW = 0xffff00U;
constexpr uint32_t COLOR_COFFEE = 0xc0ffeeU;
constexpr uint32_t COLOR_FROZEN_VAULT_MINT = 0xcff7c8;
constexpr uint32_t COLOR_HELL_GREEN = 0x8aff80;

_compute uint8_t* coalmine_overlay;
