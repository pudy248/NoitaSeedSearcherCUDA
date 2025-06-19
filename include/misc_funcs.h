#pragma once
#include "../include/enums.h"
#include "../include/search_structs.h"
#include "../platforms/platform_implementation.h"
#include <cstdint>

_universal uint8_t readByte(const uint8_t* ptr, int& offset);
_universal void writeByte(MemSpan ptr, int& offset, uint8_t b);
_universal int readInt(const uint8_t* ptr, int& offset);
_universal void writeInt(MemSpan ptr, int& offset, int val);
_universal void incrInt(int* ptr);
_universal short readShort(const uint8_t* ptr, int& offset);
_universal void writeShort(MemSpan ptr, int& offset, short s);
_universal int readMisaligned(const int* ptr);
_universal Spawnable readMisalignedSpawnable(const Spawnable* sPtr);
_universal WandData readMisalignedWand(const WandData* wPtr);

struct MemoryArena {
	uint8_t* ptr;
	uint32_t offset;
};
_compute MemSpan ArenaAlloc(MemoryArena& arena, uint32_t size);
_compute MemSpan ArenaAlloc(MemoryArena& arena, uint32_t size, uint32_t alignmentWidth);
_compute void ArenaSetOffset(MemoryArena& arena, uint8_t* endPointer);

template <int coalmine_mode = 1, bool skip_fill = false>
_compute uint32_t get_pixel(const GeneratedBiome& s, const MainPathFill& f, int x, int y, int ssl, const uint8_t* rcoal_overlay);
_universal uint32_t createRGB(const uint8_t r, const uint8_t g, const uint8_t b);
_universal int GetWidthFromPix(int a, int b);
_universal Vec2i GetGlobalPos(const int x, const int y, const int px, int py);
_universal Vec2i GetLocalPos(const int gx, int gy);
_compute int roundRNGPos(int num);

struct BiomeChunk {
	Biome b;
	int x, y, w, h;
};
struct BiomeMapChunks {
	int w, h;
	Biome* map;
	std::vector<BiomeChunk> chunks;
};
BiomeMapChunks load_biome_map(const char* path, int ngplus);

_compute void SetBiomeData();
void SetBiomePixelScenes();
void UploadBiomeData();
WangTileset stbhw_build_tileset_from_image(uint8_t* data, int biome, int stride, int w, int h);
void InstantiateBiomes(BiomeWangScope** ss, int& biomeCount, int& maxMapArea, BiomeMapChunks c, std::vector<Biome>& b);
