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

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <vector>
#include <cstdlib>

#include <array>
#include <chrono>
#include <filesystem>

OutputProgressData d;

// --- Unlock gating (config and helpers) --------------------------------------
bool g_enable_unlock_gating = false;
bool g_allow_manual_unlocks = false; // kept for compatibility, not commonly needed

// Preferred inputs
std::string g_flags_dir;             // e.g., C:\\Users\\<name>\\AppData\\LocalLow\\Nolla_Games_Noita\\save00\\persistent\\flags
std::string g_locked_csv_path;       // Optional explicit path to UNLOCKZ.csv (export of UNLOCKZ.xlsx)

// Derived at runtime (internal)
static std::unordered_set<std::string> g_unlocked_flags; // names from files present in flags dir
static std::unordered_map<std::string, std::string> g_spell_required_flag; // spell ID -> required flag (from CSV)
static std::unordered_set<std::string> g_manual_unlock_ids; // from CSV (optional)
static std::vector<bool> g_spell_unlock_mask; // index by [0..SpellCount-1]

static inline std::string trim(const std::string& s) {
	const char* ws = " \t\r\n";
	size_t b = s.find_first_not_of(ws);
	if (b == std::string::npos) return "";
	size_t e = s.find_last_not_of(ws);
	return s.substr(b, e - b + 1);
}
static inline bool extract_quoted(const std::string& line, std::string& out) {
	size_t a = line.find('"');
	if (a == std::string::npos) return false;
	size_t b = line.find('"', a + 1);
	if (b == std::string::npos) return false;
	out = line.substr(a + 1, b - a - 1);
	return true;
}

// Fallback: parse gun_actions.lua if no CSV mapping is provided
static void ParseGunActionsLuaForUnlocks(const std::string& dataPath) {
	g_spell_required_flag.clear();
	g_manual_unlock_ids.clear();
	std::string p = dataPath + "/scripts/gun/gun_actions.lua";
	std::ifstream f(p);
	if (!f) return; // silently skip if not available
	std::string line;
	std::string cur_id;
	bool waiting_flag_value = false;

	while (std::getline(f, line)) {
		if (line.find("id") != std::string::npos && line.find('"') != std::string::npos) {
			std::string id;
			if (extract_quoted(line, id)) {
				cur_id = id;
				waiting_flag_value = false;
			}
			continue;
		}
		if (line.find("spawn_requires_flag") != std::string::npos) {
			std::string val;
			if (extract_quoted(line, val)) {
				if (!cur_id.empty()) g_spell_required_flag[cur_id] = val;
				waiting_flag_value = false;
			} else {
				waiting_flag_value = true;
			}
			continue;
		}
		if (waiting_flag_value) {
			std::string val;
			if (extract_quoted(line, val)) {
				if (!cur_id.empty()) g_spell_required_flag[cur_id] = val;
				waiting_flag_value = false;
			}
		}
		if (line.find("spawn_manual_unlock") != std::string::npos) {
			if (line.find("true") != std::string::npos) {
				if (!cur_id.empty()) g_manual_unlock_ids.insert(cur_id);
			}
			continue;
		}
	}
}


static void LoadFlagsFromDirectory(const std::string& dir) {
	g_unlocked_flags.clear();
	if (dir.empty()) return;
	std::error_code ec;
	for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
		if (ec) break;
		if (!entry.is_regular_file()) continue;
		std::string name = entry.path().filename().string();
		g_unlocked_flags.insert(name);
	}
}

static void LoadLockedMappingFromCSVContent(std::istream& in) {
	g_spell_required_flag.clear();
	g_manual_unlock_ids.clear();
	std::string line;
	while (std::getline(in, line)) {
		if (line.empty()) continue;
		if (line[0] == '#') continue;
		auto comma1 = line.find(',');
		if (comma1 == std::string::npos) continue;
		std::string id = trim(line.substr(0, comma1));
		std::string rest = line.substr(comma1 + 1);
		auto comma2 = rest.find(',');
		std::string flag = trim(comma2 == std::string::npos ? rest : rest.substr(0, comma2));
		if (!id.empty() && !flag.empty()) {
			g_spell_required_flag[id] = flag;
		}
		if (comma2 != std::string::npos) {
			std::string manual = trim(rest.substr(comma2 + 1));
			if (!manual.empty() && (manual == "1" || manual == "true" || manual == "TRUE"))
				g_manual_unlock_ids.insert(id);
		}
	}
}
static void LoadLockedMappingFromCSV(const std::string& csvPath) {
	if (csvPath.empty()) return;
	std::ifstream f(csvPath);
	if (!f) return;
	LoadLockedMappingFromCSVContent(f);
}

static std::string DefaultFlagsDir() {
	const char* up = std::getenv("USERPROFILE");
	if (!up) return std::string();
	std::string base(up);
	return base + "\\AppData\\LocalLow\\Nolla_Games_Noita\\save00\\persistent\\flags";
}
static std::string DefaultLockedCsvInCwd() {
	std::string p = std::filesystem::current_path().string() + std::string("\\UNLOCKZ.csv");
	if (std::filesystem::exists(p)) return p;
	return std::string();
}

static void BuildSpellUnlockMask() {
	g_spell_unlock_mask.assign(SpellCount, true);
	if (!g_enable_unlock_gating) return;

	// Default directories if not set
	if (g_flags_dir.empty()) g_flags_dir = DefaultFlagsDir();

	// Load current save flags
	LoadFlagsFromDirectory(g_flags_dir);

	// Load locked mapping from explicit path, else from UNLOCKZ.csv in CWD
	bool loaded_mapping = false;
	if (!g_locked_csv_path.empty() && std::filesystem::exists(g_locked_csv_path)) {
		LoadLockedMappingFromCSV(g_locked_csv_path);
		loaded_mapping = !g_spell_required_flag.empty();
	}
	if (!loaded_mapping) {
		std::string cwdCsv = DefaultLockedCsvInCwd();
		if (!cwdCsv.empty()) {
			LoadLockedMappingFromCSV(cwdCsv);
			g_locked_csv_path = cwdCsv;
			loaded_mapping = !g_spell_required_flag.empty();
		}
	}

	for (int j = 0; j < SpellCount; ++j) {
		const char* id = HTables::spells[j].name; // IDs like "BLACK_HOLE_GIGA"
		bool allowed = true; // default unlocked
		auto it = g_spell_required_flag.find(id);
		if (it != g_spell_required_flag.end()) {
			// This spell is gated; require the corresponding flag file to exist
			allowed = g_unlocked_flags.find(it->second) != g_unlocked_flags.end();
		}
		// Optional manual gates
		if (allowed && g_manual_unlock_ids.find(id) != g_manual_unlock_ids.end()) {
			allowed = g_allow_manual_unlocks;
		}
		g_spell_unlock_mask[j] = allowed;
	}
}

static inline bool spell_unlocked_idx(int j) {
	return (j >= 0 && j < SpellCount) && (!g_enable_unlock_gating || (j < (int)g_spell_unlock_mask.size() && g_spell_unlock_mask[j]));
}

static void GenerateSpellData() {
	SpellTables tbl;
	std::array<bool, SpellCount> spellSpawnableInChests = {};
	for (int j = 0; j < SpellCount; j++) {
		for (int t = 0; t < 11; t++) {
			if ((HTables::spells[j].spawn_probabilities[t] > 0 || HTables::spells[j].s == SPELL_SUMMON_PORTAL ||
				 HTables::spells[j].s == SPELL_SEA_SWAMP) && spell_unlocked_idx(j)) {
				spellSpawnableInChests[j] = true;
				break;
			}
		}
	}
	tbl.spellSpawnableInChests =
		(const bool*)UploadToDevice(spellSpawnableInChests.data(), sizeof(spellSpawnableInChests));

	std::array<bool, SpellCount> spellSpawnableInBoxes = {};
	for (int j = 0; j < SpellCount; j++) {
		if ((HTables::spells[j].type == MODIFIER || HTables::spells[j].type == UTILITY) && spell_unlocked_idx(j)) {
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
			if (HTables::spells[j].spawn_probabilities[t] > 0 && spell_unlocked_idx(j)) {
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
				if ((int)HTables::spells[j].type == t && HTables::spells[j].spawn_probabilities[tier] > 0 && spell_unlocked_idx(j)) {
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
					if ((int)HTables::spells[j].type == t && HTables::spells[j].spawn_probabilities[tier] > 0 && spell_unlocked_idx(j)) {
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
	std::filesystem::current_path() = argv[0];
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

	// Build unlock mask (if enabled via CLI) before generating spell tables
	BuildSpellUnlockMask();

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