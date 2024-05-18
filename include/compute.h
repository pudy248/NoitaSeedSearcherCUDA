#pragma once
#include "../platforms/platform_implementation.h"
#include "../include/primitives.h"
#include "../include/worldgen_structs.h"
#include "../include/noita_random.h"
#undef max
#include <cstdio>
#include <vector>


_compute void CopySpawnFuncs();
int stbhw_build_tileset_from_image(uint8_t* data, WangTileset* tileSet, BiomeSpawnFunctions** funcs, int stride, int w, int h);
void blockOutRooms(uint8_t* map, int map_w, int map_h, uint32_t targetColor);
void InstantiateBiome(const char* path, BiomeWangScope* ss, int& bC, int& mA);

_compute bool PrecheckSeed(uint32_t seed, StaticPrecheckConfig c);
_compute int stbhw_generate_image(WangTileIndex* output, WangTileset* tileSet, int w, int h, WorldgenPRNG* prng);
_compute void GenerateMap(uint32_t worldSeed, BiomeWangScope scope, uint8_t* output, uint8_t* res, uint8_t* visited, uint8_t* miscMem);

_compute void spawnHeart(int x, int y, SpawnParams params);
_compute void spawnChest(int x, int y, SpawnParams params);
_compute void spawnPotion(int x, int y, SpawnParams params);
_compute void spawnWand(int x, int y, SpawnParams params);
_compute static void LoadPixelScene(int x, int y, PixelSceneList list, SpawnParams params);

_compute Spell GetRandomAction(uint32_t seed, double x, double y, int level, int offset);
_compute Spell GetRandomActionWithType(uint32_t seed, double x, double y, int level, ActionType type, int offset);
_compute _noinline Wand GetWandWithLevel(uint32_t seed, double x, double y, int level, bool nonshuffle, bool better);

_compute void CheckSpawnables(WangFuncIndex* idxs, WangTileset* tileSet, SpawnParams params, int maxMemory);
_compute void CheckMountains(int seed, SpawnableConfig* sCfg, uint8_t* bytes, int& offset, int& sCount);
_compute void CheckEyeRooms(int seed, SpawnableConfig* sCfg, uint8_t* bytes, int& offset, int& sCount);
_compute void CheckNightmareSpawnWands(int seed, SpawnableConfig* sCfg, uint8_t* bytes, int& offset, int& sCount);
_compute SpawnableBlock ParseSpawnableBlock(uint8_t* bytes, uint8_t* putSpawnablesHere, SpawnableConfig sCfg, int maxMemory);
_compute bool SpawnablesPassed(SpawnableBlock b, FilterConfig fCfg, uint8_t* output, uint8_t* tmp, bool write);
_compute void WriteOutputBlock(uint8_t* output, int seed, Spawnable** spawnables, int sCount);

void PrintOutputBlock(uint8_t* output, int time[2], FILE* outputFile, OutputConfig outputCfg, void(*appendOutput)(char*, char*));

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