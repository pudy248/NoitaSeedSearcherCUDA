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
#include "src/biome_map.cpp"
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

#include <array>
#include <chrono>
#include <filesystem>

OutputProgressData d;

static void GenerateSpellData() {
	SpellTables tbl;
	std::array<bool, SpellCount> spellSpawnableInChests = {};
	for (int j = 0; j < SpellCount; j++) {
		for (int t = 0; t < 11; t++) {
			if (HTables::spells[j].spawn_probabilities[t] > 0 || HTables::spells[j].s == SPELL_SUMMON_PORTAL ||
				HTables::spells[j].s == SPELL_SEA_SWAMP) {
				spellSpawnableInChests[j] = true;
				break;
			}
		}
	}
	tbl.spellSpawnableInChests =
		(const bool*)UploadToDevice(spellSpawnableInChests.data(), sizeof(spellSpawnableInChests));

	std::array<bool, SpellCount> spellSpawnableInBoxes = {};
	for (int j = 0; j < SpellCount; j++) {
		if (HTables::spells[j].type == MODIFIER || HTables::spells[j].type == UTILITY) {
			for (int t = 0; t < 11; t++) {
				if (HTables::spells[j].spawn_probabilities[t] > 0 || HTables::spells[j].s == SPELL_SUMMON_PORTAL ||
					HTables::spells[j].s == SPELL_SEA_SWAMP) {
					spellSpawnableInBoxes[j] = true;
					break;
				}
			}
		}
	}
	tbl.spellSpawnableInBoxes =
		(const bool*)UploadToDevice(spellSpawnableInBoxes.data(), sizeof(spellSpawnableInBoxes));

	for (int t = 0; t < 11; t++) {
		std::array<SpellProb, SpellCount> spellProbs_n = {};
		int n = 0;
		for (int j = 0; j < SpellCount; j++) {
			if (HTables::spells[j].spawn_probabilities[t] > 0) {
				tbl.spellTierCounts[t]++;
				tbl.spellTierSums[t] += HTables::spells[j].spawn_probabilities[t];
				spellProbs_n[n++] = {tbl.spellTierSums[t], HTables::spells[j].s};
			}
		}
		tbl.allSpellProbs[t] = (const SpellProb*)UploadToDevice(spellProbs_n.data(), sizeof(SpellProb) * n);
	}

	for (int tier = 0; tier < 11; tier++) {
		for (int t = 0; t < 8; t++) {
			for (int j = 0; j < SpellCount; j++) {
				if ((int)HTables::spells[j].type == t && HTables::spells[j].spawn_probabilities[tier] > 0) {
					tbl.spellProbs_Counts[tier][t]++;
				}
			}
		}
		for (int t = 0; t < 8; t++) {
			std::array<SpellProb, SpellCount> spellProbs_t_n = {};
			int n = 0;
			if (tbl.spellProbs_Counts[tier][t] > 0) {
				double sum = 0;
				for (int j = 0; j < SpellCount; j++) {
					if ((int)HTables::spells[j].type == t && HTables::spells[j].spawn_probabilities[tier] > 0) {
						sum += HTables::spells[j].spawn_probabilities[tier];
						spellProbs_t_n[n++] = {sum, HTables::spells[j].s};
					}
				}
				tbl.spellProbs_Sums[tier][t] = sum;
				tbl.spellProbs_Types[tier][t] =
					(const SpellProb*)UploadToDevice(spellProbs_t_n.data(), sizeof(SpellProb) * n);
			}
		}
	}
	HSetSpellData(&tbl);
	//for (int i = 0; i < spellTables.spellTierCounts[10]; i++) {
	//	printf("%s %f\n", SpellNames[spellTables.allSpellProbs[10][i].s], spellTables.allSpellProbs[10][i].p);
	//}
	//for (int i = 0; i < 10; i++)
	//	printf("%s\n", SpellNames[GetRandomAction(1502229, 2367, 13166, 10, i)]);
}

#if 0
namespace HELPERS
{

#if 0
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
	constexpr uint64_t MAX_CNT = 2147483647;
	std::atomic<int> counter = 0;
	void CountForEach(int idx) {
		uint64_t start = idx;
		uint64_t stride = std::thread::hardware_concurrency();
		for (uint64_t i = start; i < MAX_CNT; i += stride) {
			uint8_t chest[1000];
			BiomeWangScope sc = {};
			SpawnableConfig sCfg = {};
			int o = 0, scount = 0;
			CheckNormalChestLoot(0, 0, false, SpawnParams{(int)i, sc, sCfg, {chest, 1000}, o, scount});
			
			if (NollaPRNG(i).Random(1, 10000) == 1)
				counter++;
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
	//HELPERS::HCountForEach();
	//return 0;

	read_wak(find_wak().c_str());

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
		.wandStatFilterCount = 0,
		.wandStatFilters = {},
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

	InitializePlatform();
	if (DEBUG_DISPATCH_RATE_OVERRIDE)
		SetTargetDispatchRate(DEBUG_DISPATCH_RATE_OVERRIDE);
	HSetBiomeData();
	GenerateSpellData();

	BiomeMapChunks map = load_biome_map("data/biome_impl/biome_map.png", 0);
	int biomeCount = 0;
	int maxMapArea = 0;
	InstantiateBiomes(config.biomeScopes, biomeCount, maxMapArea, map, biome_list);

	config.biomeCount = biomeCount;
	if (!config.generalCfg.seedBlockOverride)
		config.generalCfg.seedBlockSize =
			DEBUG_SEED_BLOCK_OVERRIDE ?
				DEBUG_SEED_BLOCK_OVERRIDE :
				(biomeCount ? (WorkerAppetite > 100 ? 1u : 32u) : (WorkerAppetite > 100 ? 256u : 16384u));

	config.memSizes = {
		.memoryCap = 40_GB,

#ifdef IMAGE_OUTPUT
		.outputSize = (size_t)maxMapArea * 3 + 512, // output
#else
		.outputSize = (size_t)8192,
#endif
		.mapDataSize = (size_t)maxMapArea * 2,
		.miscMemSize = (size_t)maxMapArea * 2,
		.visitedMemSize = (size_t)maxMapArea + 512,
		.spawnableMemSize = max((size_t)maxMapArea / 4, 8192u),
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