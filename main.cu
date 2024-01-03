#include "platforms/platform_implementation.h"
#include "platforms/platform_api.h"
using namespace API_INTERNAL;
#include "src/platform_implementation_src.cu"

//#include "gui/guiMain.h"
#include "include/configuration.h"
#include "include/compute.h"

#include "src/structs.cu"
#include "src/misc.cu"
#include "src/compute.cu"
#include "src/precheck.cu"
#include "src/hbwang.cu"
#include "src/biome_impl.cu"
#include "src/worldgen.cu"
#include "src/wandgen.cu"
#include "src/search.cu"
#include "src/filter.cu"
#include "src/output.cu"
#define PNG_IMPL
#include "include/pngutils.h"

#include <chrono>

OutputProgressData d;
int seedCtr = 0;

void appendOutput(char* s, char* c)
{
	printf("%i: %s (checked %i)\n", d.elapsedMillis, s, d.searchedSeeds);
}

int main()
{
	BiomeWangScope biomes[20];
	int biomeCount = 0;
	int maxMapArea = 0;
	InstantiateBiome("resources/wang_tiles/coalmine.png", biomes, biomeCount, maxMapArea);
	config.biomeCount = biomeCount;
	memcpy(config.biomeScopes, biomes, sizeof(BiomeWangScope) * 20);

	config.memSizes = {
			40_GB,

	#ifdef DO_WORLDGEN
			(size_t)maxMapArea * 4 + 512,
	#else
			(size_t)512,
	#endif
			(size_t)maxMapArea * 4,
			(size_t)maxMapArea * 4,
			(size_t)maxMapArea * 4,
			(size_t)512,
	};

	config.generalCfg = { 1, INT_MAX, 1, false };
#ifdef REALTIME_SEEDS
	generalCfg.seedBlockSize = 1;
#endif
	config.spawnableCfg = {
		{0, 0}, {0, 0}, 0, 0,

		false, //greed
		false, //pacifist
		false, //shop spells
		false, //shop wands
		false, //eye rooms
		false, //upwarp check
		true, //biome chests
		false, //biome pedestals
		false, //biome altars
		false, //biome pixelscenes
		false, //enemies
		false, //hell shops
		false, //potion contents
		false, //chest spells
		false, //wand stats
	};

	config.filterCfg = {
		false, 1, { ItemFilter({SAMPO}) }, 0, {}, 0, {}, 0, {}, false, 27
	};

	config.precheckCfg = {
		{false, CART_NONE},
		{false, MATERIAL_NONE},
		{false, SPELL_NONE, SPELL_NONE},
		{false, MATERIAL_NONE},
		{false, AlchemyOrdering::UNORDERED, {}, {}},
		{false, {}},
		{false, {}},
		{false, {}, {3, 3, 3, 3, 3, 3, 3}},
	};

	config.outputCfg = { 0.05f, false, false };

	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.x * 2 + 1;
	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.y * 2 + 1;
	config.memSizes.spawnableMemSize *= max(1, biomeCount);

	SearchMain(d, appendOutput);

	//SfmlMain();
	return 0;
}
