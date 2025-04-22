#include "platforms/platform_implementation.h"

#include "platforms/platform_api.h"
using namespace API_INTERNAL;
#include "src/platform_implementation_src.cpp"

//#include "gui/guiMain.h"
#include "include/compute.h"
#include "include/configuration.h"
#include "include/wak.h"

#ifdef DEBUG_ATOMIC_COUNTERS
#include <atomic>
std::atomic<uint64_t> globalChestCounter = 0;
std::atomic<uint64_t> globalHeartCounter = 0;
std::atomic<uint64_t> globalItemCounter = 0;
std::atomic<uint64_t> globalWandCounter = 0;
#endif

#include "src/biome_impl.cpp"
#include "src/cli.cpp"
#include "src/compute.cpp"
#include "src/filter.cpp"
#include "src/hbwang.cpp"
#include "src/misc.cpp"
#include "src/output.cpp"
#include "src/pathfinding.cpp"
#include "src/precheck.cpp"
#include "src/search.cpp"
#include "src/structs.cpp"
#include "src/wak.cpp"
#include "src/wandgen.cpp"
#include "src/worldgen.cpp"
#define PNG_IMPL
#include "include/pngutils.h"

#include <chrono>
#include <filesystem>

OutputProgressData d;

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

#if 1
	constexpr uint64_t MAX_CNT = INT_MAX;
	__device__ int counter = 0;
	__global__ void CountForEach() {
		uint64_t start = blockDim.x * blockIdx.x + threadIdx.x;
		uint64_t stride = gridDim.x * blockDim.x;
		for (uint64_t i = start; i < MAX_CNT; i += stride) {
			Wand w = GetWandWithLevelGivenSeed(i, 10, false);
			if (w.capacity >= 45 && w.alwaysCast.s == SPELL_NONE) {
				printf("%i %f %i\n", i, w.capacity, w.multicast);
				dAtomicAdd(&counter, 1);
			}
		}
	}

	static void HCountForEach() {
		int hCtr;
		CountForEach << <30, 64 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpyFromSymbol(&hCtr, counter, 4));
		printf("%i %f\n", hCtr, (double)hCtr / MAX_CNT);
	}
#else
	constexpr uint64_t MAX_CNT = 10000000;
	std::atomic<int> counter = 0;
	void CountForEach(int idx) {
		uint64_t start = idx;
		uint64_t stride = std::thread::hardware_concurrency();
		for (uint64_t i = start; i < MAX_CNT; i += stride) {
			Wand w = GetWandWithLevelGivenSeed(i, 1, false);
			int add_manas = 0;
			for (int i = 0; i < w.spellCount; i++)
				if (w.spells[i].s == SPELL_MANA_REDUCE)
					add_manas++;
			if (add_manas >= 28) {
				//printf("%i %f %i\n", i, w.capacity, w.multicast);
				counter++;
			}
		}
	}

	static void HCountForEach() {
		{
			std::vector<std::jthread> vec;
			for (int i = 0; i < std::thread::hardware_concurrency(); i++)
				vec.emplace_back(CountForEach, i);
		}
		printf("%i %f\n", counter.load(), (double)counter.load() / MAX_CNT);
	}
#endif
}
#endif

void cli_main(int argc, char** argv);

int main(int argc, char** argv) {
	//HCountForEach();
	//return 0;

	read_wak(find_wak().c_str());

	InitializePlatform();
	HSetBiomeData();

	int biomeCount = 0;
	int maxMapArea = 0;

	config.generalCfg = {
#ifdef SEEDS_AS_TRIES
		.seedStart = 1,
		.seedEnd = 100,
#else
		.seedStart = 1,
		.seedEnd = INT_MAX - 1,
#endif
		.seedBlockSize = 1,
		.seedBlockOverride = false,
		.priority = 0,
	};

	// Runs before generation, once
	config.precheckCfg = {
		.cart = {false, CART_NONE},
		.flask = {false, GOLD},
		.wands = {false, SPELL_NONE, SPELL_NONE},
		.rain = {false, MATERIAL_NONE},
		.alchemy = {false, {}, {}},
		.biomes = {false, {}},
		.fungal = {false, {}},
		.perks = {false, {}, {}, {3, 3, 3, 3, 3, 3, 3}},
		.precheckUpwarps = false,
	};

	// Runs once per biome
	config.spawnableCfg = {
		.pwCenter = {0, 0},
		.pwWidth = {0, 0},
		.minHMidx = 0,
		.maxHMidx = 6,
		.greedCurse = false,
		.pacifist = false,
		.shopSpells = false,
		.shopWands = false,
		.eyeRooms = false,
		.biomeChests = false,
		.biomePedestals = false,
		.biomeAltars = false,
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
		.itemFilters = {},
		.materialFilterCount = 0,
		.materialFilters = {},
		.spellFilterCount = 0,
		.spellFilters = {},
		.pixelSceneFilterCount = 0,
		.pixelSceneFilters = {},
		.wandStats = false,
		.wandStatThreshold = 27,
	};

	config.outputCfg = {
		.outputMode = 1,
		.outputFile = "output.txt",
		.printInterval = 15.f,
		.countPassesOnly = false,
		.printProgressLog = true,
		.printOutputToConsole = true,
		.printOutputToFile = false,
	};

	cli_main(argc, argv);
	for (int i = 0; i < biome_list.size(); i++)
		InstantiateBiome(biome_list[i], config.biomeScopes, biomeCount, maxMapArea);

	config.biomeCount = biomeCount;
	config.generalCfg.seedBlockSize = biomeCount ? (WorkerAppetite > 100 ? 1u : 512u) :
												   (WorkerAppetite > 100 ? 256u : 16384u);

	config.memSizes = {
		.memoryCap = 40_GB,

	#ifdef IMAGE_OUTPUT
		.outputSize = (size_t)maxMapArea * 3 + 512, // output
	#else
			.outputSize = (size_t)4096,
	#endif
		.mapDataSize = (size_t)maxMapArea * 2,
		.miscMemSize = (size_t)maxMapArea * 2,
		.visitedMemSize = (size_t)maxMapArea + 256,
		.spawnableMemSize = (size_t)maxMapArea / 4,
	};

	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.x * 2 + 1;
	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.y * 2 + 1;
	config.memSizes.spawnableMemSize *= max(1, biomeCount);

	AllocateComputeMemory();
	SearchMain(d, nullptr);
	FreeComputeMemory();

	DestroyPlatform();

	//SfmlMain();
	return 0;
}