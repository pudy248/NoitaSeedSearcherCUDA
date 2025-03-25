#include "platforms/platform_implementation.h"
#include "platforms/platform_api.h"
using namespace API_INTERNAL;
#include "src/platform_implementation_src.cpp"

//#include "gui/guiMain.h"
#include "include/configuration.h"
#include "include/compute.h"
#include "include/wak.h"

#include <atomic>
std::atomic<uint64_t> globalChestCounter = 0;
std::atomic<uint64_t> globalHeartCounter = 0;
std::atomic<uint64_t> globalItemCounter = 0;
std::atomic<uint64_t> globalWandCounter = 0;

#include "src/structs.cpp"
#include "src/misc.cpp"
#include "src/compute.cpp"
#include "src/precheck.cpp"
#include "src/hbwang.cpp"
#include "src/biome_impl.cpp"
#include "src/worldgen.cpp"
#include "src/pathfinding.cpp"
#include "src/wandgen.cpp"
#include "src/search.cpp"
#include "src/filter.cpp"
#include "src/output.cpp"
#include "src/wak.cpp"
#define PNG_IMPL
#include "include/pngutils.h"

#include <chrono>
#include <filesystem>

OutputProgressData d;

void appendOutput(char* s, char* c)
{
#ifdef SPAWNABLE_OUTPUT
	//printf("%i (checked %i): %s", d.elapsedMillis, d.searchedSeeds, c);
#else
	//printf("%i: %s (checked %i)\n", d.elapsedMillis, s, d.searchedSeeds);
#endif
}

#if 0
namespace HELPERS
{
	static void GenerateSpellData()
	{
		printf("_data const static bool spellSpawnableInChests[] = {\n");
		for (int j = 0; j < SpellCount; j++)
		{
			bool passed = false;
			for (int t = 0; t < 11; t++)
			{
				if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL || allSpells[j].s == SPELL_SEA_SWAMP)
				{
					passed = true;
					break;
				}
			}
			printf(passed ? "true" : "false");
			printf(",\n");
		}
		printf("};\n");

		printf("_data const static bool spellSpawnableInBoxes[] = {\n");
		for (int j = 0; j < SpellCount; j++)
		{
			bool passed = false;
			if (allSpells[j].type == MODIFIER || allSpells[j].type == UTILITY)
			{
				for (int t = 0; t < 11; t++)
				{
					if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL || allSpells[j].s == SPELL_SEA_SWAMP)
					{
						passed = true;
						break;
					}
				}
			}
			printf(passed ? "true" : "false");
			printf(",\n");
		}
		printf("};\n");

		int counters2[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
		double sums[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
		for (int t = 0; t < 11; t++)
		{
			printf("_data const static SpellProb spellProbs_%i[] = {\n", t);
			for (int j = 0; j < SpellCount; j++)
			{
				if (allSpells[j].spawn_probabilities[t] > 0)
				{
					counters2[t]++;
					sums[t] += allSpells[j].spawn_probabilities[t];
					printf("{%f,SPELL_%s},\n", sums[t], allSpells[j].name);
				}
			}
			printf("};\n");
		}

		printf("_data const static int spellTierCounts[] = {\n");
		for (int t = 0; t < 11; t++)
		{
			printf("%i,\n", counters2[t]);
		}
		printf("};\n");

		printf("_data const static float spellTierSums[] = {\n");
		for (int t = 0; t < 11; t++)
		{
			printf("%f,\n", sums[t]);
		}
		printf("};\n\n");


		for (int tier = 0; tier < 11; tier++)
		{
			int counters[8] = { 0,0,0,0,0,0,0,0 };
			for (int t = 0; t < 8; t++)
			{
				for (int j = 0; j < SpellCount; j++)
				{
					if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
					{
						counters[t]++;
					}
				}
			}
			for (int t = 0; t < 8; t++)
			{
				if (counters[t] > 0)
				{
					double sum = 0;
					printf("_data const static SpellProb spellProbs_%i_T%i[] = {\n", tier, t);
					for (int j = 0; j < SpellCount; j++)
					{
						if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
						{
							sum += allSpells[j].spawn_probabilities[tier];
							printf("{%f,SPELL_%s},\n", sum, allSpells[j].name);
						}
					}
					printf("};\n");
				}
			}
			printf("_data const static SpellProb* spellProbs_%i_Types[] = {\n", tier);
			for (int t = 0; t < 8; t++)
			{
				if (counters[t] > 0)
					printf("spellProbs_%i_T%i,\n", tier, t);
				else
					printf("NULL,\n");
			}
			printf("};\n");

			printf("_data const static int spellProbs_%i_Counts[] = {\n", tier);
			for (int t = 0; t < 8; t++)
			{
				printf("%i,\n", counters[t]);
			}
			printf("};\n\n");

			printf("_data const static int spellProbs_%i_Counts[] = {\n", tier);
			for (int t = 0; t < 8; t++)
			{
				printf("%i,\n", counters[t]);
			}
			printf("};\n\n");
		}
	}

#if 0
	constexpr uint64_t MAX_CNT = 100000000;
	__device__ int counter = 0;
	__global__ void CountForEach(int n) {
		uint64_t start = blockDim.x * blockIdx.x + threadIdx.x;
		uint64_t stride = gridDim.x * blockDim.x;
		for (uint64_t i = start; i < MAX_CNT; i += stride) {
			Wand w = GetWandWithLevelGivenSeed(i, n, false);
			if (w.alwaysCast.s == SPELL_REGENERATION_FIELD)
				dAtomicAdd(&counter, 1);
		}
	}

	static void HCountForEach() {
		for (int i = 1; i <= 6; i++) {
			int hCtr;
			CountForEach << <30, 64 >> > (i);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpyFromSymbol(&hCtr, counter, 4));
			printf("%i %f\n", hCtr, (double)hCtr / MAX_CNT);
		}
	}
#endif
}
#endif

int main()
{
	read_wak(find_wak().c_str());

	InitializePlatform();
	HSetBiomeData();

	int biomeCount = 0;
	int maxMapArea = 0;
	
	InstantiateBiome(B_CRYPT, config.biomeScopes, biomeCount, maxMapArea);

	config.biomeCount = biomeCount;

	config.memSizes = {
			.memoryCap = 40_GB,

#ifdef IMAGE_OUTPUT
			.outputSize = (size_t)maxMapArea * 3 + 512, // output
#else
			.outputSize = (size_t)512,
#endif
			.mapDataSize = (size_t)maxMapArea * 2,
			.miscMemSize = (size_t)maxMapArea * 2,
			.visitedMemSize = (size_t)maxMapArea + 256,
			.spawnableMemSize = (size_t)12288,
	};
	config.generalCfg = {
#ifdef SEEDS_AS_TRIES
		.seedStart = 1,
		.seedEnd = 100,
#else
		.seedStart = 1,
		.seedEnd = INT_MAX,
#endif
#ifdef REALTIME_SEEDS
		.seedBlockSize = 1;
		.seedBlockOverride = true
#else
		.seedBlockSize = biomeCount ? 1u : 256u, 
		.seedBlockOverride = false
#endif
	};

	// Runs before generation, once
	config.precheckCfg = {
		.cart = {false, CART_NONE},
		.flask = {false, GOLD}, // flask
		.wands = {false, SPELL_NONE, SPELL_NONE},
		.rain = {false, MATERIAL_NONE},
		.alchemy = {false, AlchemyOrdering::UNORDERED, {MUD, WATER, SOIL}, {MUD, WATER, SOIL}},
		.biomes = {false, {}},
		.fungal = {false, {FungalShift(SS_STEAM, SD_FLASK, 0, 1)}},
		.perks = {false, {
			{PERK_ANGRY_GHOST, false, 0, 3},
		}, {PERK_EDIT_WANDS_EVERYWHERE, PERK_INVISIBILITY}, { 3, 3, 3, 3, 3, 3, 3 }},
		.precheckUpwarps = false,
	};

	// Runs once per biome
	config.spawnableCfg = {
		.pwCenter = {0, 0},
		.pwWidth = {0, 0},
		.minHMidx = 0,
		.maxHMidx = 0,
		.greedCurse = false,
		.pacifist = false,
		.shopSpells = false,
		.shopWands = false,
		.eyeRooms = false,
		.biomeChests = false,
		.biomePedestals = false,
		.biomeAltars = true,
		.biomePixelSceneIndexing = true,
		.biomePixelSceneSearch = false,
		.biomeEnemies = false,
		.hellShops = false,
		.nightmare = false,
		.genPotions = false,
		.genSpells = false,
		.genWands = false,
	};

	// Runs after all biomes generated, once
	config.filterCfg = {
		.aggregate = false,
		.itemFilterCount = 0,
		.itemFilters = {ItemFilter({SAMPO, TRUE_ORB}, 1)},
		.materialFilterCount = 0,
		.materialFilters = {},
		.spellFilterCount = 1,
		.spellFilters = {SpellFilter({SPELL_NUKE_GIGA})},
		.pixelSceneFilterCount = 0,
		.pixelSceneFilters = {PixelSceneFilter({PS_CRYPT_POLYMORPHROOM}, 100)},
		.wandStats = false,
		.wandStatThreshold = 27,
	};

	config.outputCfg = {
		.outputMode = 0, 
		.printInterval = 5.f,
		.printProgressLog = true, 
		.printOutputToConsole = true,
		.printOutputToFile = true
	};

	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.x * 2 + 1;
	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.y * 2 + 1;
	config.memSizes.spawnableMemSize *= max(1, biomeCount);

	AllocateComputeMemory();
	SearchMain(d, appendOutput);
	FreeComputeMemory();
	
	DestroyPlatform();

	int chunkCount = 26;
	printf("C %f\n", (double)globalChestCounter.load() / 100000 / chunkCount);
	printf("H %f\n", (double)globalHeartCounter.load() / 100000 / chunkCount);
	printf("I %f\n", (double)globalItemCounter.load() / 100000 / chunkCount);
	printf("W %f\n", (double)globalWandCounter.load() / 100000 / chunkCount);

	//SfmlMain();
	return 0;
}