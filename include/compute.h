#pragma once
#include "../platforms/platform_implementation.h"
#include "../include/primitives.h"
#include "../include/worldgen_structs.h"
#include "../include/noita_random.h"
#undef max
#include <cstdio>
#include <vector>

_compute void CopySpawnFuncs();
WangTileset stbhw_build_tileset_from_image(uint8_t* data, BiomeSpawnFunctions** funcs, int stride, int w, int h);
void InstantiateBiome(const char* path, BiomeWangScope** ss, int& bC, int& mA);

_compute bool PrecheckSeed(uint32_t seed, StaticPrecheckConfig c);
_compute int stbhw_generate_image(WangTileIndex* output, const BiomeWangScope& scope, int w, int h, WorldgenPRNG& prng);
_compute bool isValid(const GeneratedBiome& s, MemSpan stackMemArea, MemSpan visited);
_compute GeneratedBiome GenerateMap(uint32_t worldSeed, const BiomeWangScope& scope, MemSpan output, MemSpan res, MemSpan visited, MemSpan miscMem);

_compute void spawnHeart(int x, int y, const SpawnParams& params);
_compute void spawnChest(int x, int y, const SpawnParams& params);
_compute void spawnPotion(int x, int y, const SpawnParams& params);
_compute void spawnWand(int x, int y, const SpawnParams& params);
_compute static void LoadPixelScene(int x, int y, PixelSceneList list, const SpawnParams& params);

_compute Spell GetRandomAction(uint32_t seed, double x, double y, int level, int offset);
_compute Spell GetRandomActionWithType(uint32_t seed, double x, double y, int level, ActionType type, int offset);
_compute _noinline Wand GetWandWithLevel(uint32_t seed, double x, double y, int level, bool nonshuffle, bool better);

_compute void CheckSpawnables(const GeneratedBiome& s, SpawnParams& params);
_compute void CheckMountains(const SpawnParams& params);
_compute void CheckEyeRooms(const SpawnParams& params);
_compute SpawnableBlock ParseSpawnableBlock(const uint8_t* block, MemSpan spawnables_block, const SpawnableConfig& sCfg, int seed, int sCount);
_compute bool SpawnablesPassed(const SpawnableBlock& b, const FilterConfig& fCfg, MemSpan output, MemSpan tmp, bool write);
_compute void WriteOutputBlock(MemSpan output, const SpawnableBlock& b);

void PrintOutputBlock(const uint8_t* output, FILE* outputFile, OutputConfig outputCfg, void(*appendOutput)(char*, char*));

struct OutputProgressData
{
	float progressPercent;
	int elapsedMillis;
	int searchedSeeds;
	int validSeeds;

	volatile bool abort = false;
};
_compute SpanRet PLATFORM_API::EvaluateSpan(SearchConfig config, SpanParams span, void* threadMemBlock, void* outputPtr);
Vec2i OutputLoop(FILE* outputFile, time_t startTime, OutputProgressData& progress, void(*appendOutput)(char*, char*));
void SearchMain(OutputProgressData& progress, void(*appendOutput)(char*, char*));