#pragma once
#include "../platforms/platform_implementation.h"
#include "../include/primitives.h"
#include "../include/worldgen_structs.h"
#include "../include/noita_random.h"
#include <cstdio>
#include <vector>

struct OutputProgressData {
	float progressPercent;
	int elapsedMillis;
	int searchedSeeds;
	int validSeeds;

	volatile bool abort = false;
};

_compute void SetBiomeData();
void SetBiomePixelScenes();
void UploadBiomeData();
WangTileset stbhw_build_tileset_from_image(uint8_t* data, int biome, int stride, int w, int h);
void InstantiateBiome(int biome, BiomeWangScope** ss, int& bC, int& mA);

_compute bool PrecheckSeed(uint32_t seed, StaticPrecheckConfig c);
_compute int stbhw_generate_image(WangTileIndex* output, const BiomeWangScope& scope, int w, int h, WorldgenPRNG& prng);
_compute bool isValid(const GeneratedBiome& s, MemSpan stackMemArea, MemSpan visited);
_compute GeneratedBiome GenerateMap(uint32_t worldSeed, const BiomeWangScope& scope, MemSpan output, MemSpan res, MemSpan visited, MemSpan miscMem);

_compute _noinline static void CheckNormalChestLoot(int x, int y, bool hasMimicSign, const SpawnParams& params);
_compute _noinline static void CheckGreatChestLoot(int x, int y, bool hasMimicSign, const SpawnParams& params);
_compute _noinline static void CheckItemPedestalLoot(int x, int y, const SpawnParams& params);
_compute _noinline static void CheckUtilityBoxLoot(int x, int y, const SpawnParams& params);

_compute void spawnHeart(int x, int y, const SpawnParams& params);
_compute void spawnChest(int x, int y, const SpawnParams& params);
_compute void spawnPotion(int x, int y, const SpawnParams& params);
_compute void spawnWand(int x, int y, const SpawnParams& params);
_compute void LoadPixelScene(int x, int y, const PixelSceneList& list, const SpawnParams& params);

_compute Spell GetRandomAction(uint32_t seed, double x, double y, int level, int offset);
_compute Spell GetRandomActionWithType(uint32_t seed, double x, double y, int level, ActionType type, int offset);
_compute _noinline Wand GetWandWithLevel(uint32_t seed, double x, double y, int level, bool nonshuffle, bool better);

_compute void CheckSpawnables(const GeneratedBiome& s, SpawnParams& params);
_compute void CheckMountains(const SpawnParams& params);
_compute void CheckEyeRooms(const SpawnParams& params);
_compute void CheckNightmareSpawnWands(const SpawnParams& params);
_compute SpawnableBlock ParseSpawnableBlock(const uint8_t* block, MemSpan spawnables_block, const SpawnableConfig& sCfg, int seed, int sCount);
_compute bool SpawnablesPassed(const SpawnableBlock& b, const FilterConfig& fCfg, MemSpan output, MemSpan tmp, bool write, bool upwarps);
_compute void WriteOutputBlock(MemSpan output, const SpawnableBlock& b);

void PrintOutputBlock(uint8_t* output, int time[2], FILE* outputFile, OutputConfig outputCfg, void(*appendOutput)(char*, char*));
_compute SpanRet PLATFORM_API::EvaluateSpan(SearchConfig config, SpanParams span, void* threadMemBlock, void* outputPtr);
Vec2i OutputLoop(FILE* outputFile, time_t startTime, OutputProgressData& progress, void(*appendOutput)(char*, char*));
void SearchMain(OutputProgressData& progress, void(*appendOutput)(char*, char*));