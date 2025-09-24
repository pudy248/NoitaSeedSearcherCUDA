#include "../data/host_tables.h"
#include "../data/ids.h"
#include "../platforms/platform_api.h"

#include <charconv>
#include <ranges>
#include <string_view>
#include <vector>
#include <string>
#include <unordered_set>
#include <cstdio>

int g_argc;
char** g_argv;

struct cmd {
	const char* long_name;
	const char* short_name;
	int (*fn)(int);
	const char* desc;
};

int cmd_help(int);
int cmd_cart(int);
int cmd_flask(int);
// int cmd_starting_spells(int);
int cmd_rain(int);
int cmd_alchemy(int);
int cmd_biome_mods(int);
int cmd_fungal(int);
int cmd_perks(int);
int cmd_hms(int);
int cmd_pws(int);
int cmd_greed(int);
int cmd_pacifist(int);
int cmd_shop_spells(int);
int cmd_shop_wands(int);
int cmd_eye_rooms(int);
int cmd_biome_chests(int);
int cmd_biome_pedestals(int);
int cmd_biome_altars(int);
int cmd_biome_pixel_scene_indexing(int);
int cmd_biome_pixel_scene_search(int);
int cmd_hell_shops(int);
int cmd_nightmare(int);
int cmd_gen_potions(int);
int cmd_gen_spells(int);
int cmd_gen_wands(int);
int cmd_biomes(int);
int cmd_upwarps(int);
int cmd_aggregate(int);
int cmd_filter_items(int);
int cmd_filter_materials(int);
int cmd_filter_spells(int);
int cmd_filter_pixel_scenes(int);
int cmd_filter_wands(int);
int cmd_count_passed(int);
int cmd_print_to_console(int);
int cmd_print_to_file(int);
int cmd_output_file(int);
int cmd_logging_interval(int);
int cmd_output_mode(int);
int cmd_start_seed(int);
int cmd_end_seed(int);
int cmd_this_seed(int);
int cmd_seeds_from_file(int);
int cmd_priority(int);
int cmd_exact(int);
int cmd_quiet(int);
int cmd_verbose(int);
int cmd_debug(int);
// Unlock gating commands (new preferred flow)
int cmd_flags_dir(int);
int cmd_locked_spells_csv(int);
int cmd_allow_manual_unlocks(int);
int cmd_enable_unlock_gating(int);

static const cmd commands[] = {
	{"--help", "-h", cmd_help, "Display this menu."},
	{"--cart", nullptr, cmd_cart, "Select a starting cart."},
{"--flags-dir", nullptr, cmd_flags_dir, "Path to save00/persistent/flags directory (where achievement flags are files)."},
	{"--locked-spells-csv", nullptr, cmd_locked_spells_csv, "CSV mapping of locked spell IDs to required flags (export UNLOCKZ.xlsx to CSV)."},
	{"--allow-manual-unlocks", nullptr, cmd_allow_manual_unlocks, "Treat spawn_manual_unlock spells as unlocked."},
	{"--enable-unlock-gating", "-eug", cmd_enable_unlock_gating, "Enable gating of spells requiring flags; config is persisted."},
	{"--rain", "-r", cmd_rain, "Select an initial rain."},
	{"--starting-flask", "-sf", cmd_flask, "Select a starting flask material."},
	{"--alchemy", "-al", cmd_alchemy,
		"Select LC/AP materials. Takes a pair of reactions of the format {mat1,mat2,mat3[,ordering]} for LC and AP. Ex. \"-al {mud,water,soil,unordered} {any,water,any,only_consumed}\""},
	{"--biome-mods", "-bm", cmd_biome_mods,
		"Select biome modifiers. Takes a list of modifiers of the format {biome,modifier}. Ex. \"-bm {coalmine,extremely_lucrative}\""},
	{"--fungal", nullptr, cmd_fungal,
		"Filter fungal shifts. Takes a list of shifts of the format {from,to[,min-index,max-index]}. Ex. \"--fungal {gold,cheese,0,0} {steam,flask}\""},
	{"--perks", nullptr, cmd_perks,
		"Filter the perk deck. Takes a list of perks of the format {perk[,min-index,max-index][,lottery-safe]}. Ex. \"--perks {edit_wands_everywhere,0,2} {perks_lottery,0,2,true}\""},
	{"--holy-mountains", "-hm", cmd_hms,
		"Select which holy mountains to index, of the format {min-index,max-index}. Only has an effect for pacifist chests and shop items, and does not require biome generation. Ex. \"-hm {0,2}\" for the first 3."},
	{"--parallels", "-pw", cmd_pws,
		"Select how many parallel worlds to index, of the format {width[,height]} [{center_x,center_y}]. Ex. \"-pw {1,1}\" for a single parallel world in every direction, a total of 9 copies."},
	{"--greed", nullptr, cmd_greed, "Generate with curse of greed enabled."},
	{"--pacifist", "-pc", cmd_pacifist, "Generate pacifist chests."},
	{"--shop-spells", "-ss", cmd_shop_spells, "Generate HM spells."},
	{"--shop-wands", "-sw", cmd_shop_wands,
		"Generate HM wands. --gen-spells is required for spells on these wands as well."},
	{"--eye-rooms", "-e", cmd_eye_rooms, "Generate eye room spells."},
	{"--biome-chests", "-c", cmd_biome_chests, "Generate biome chest (and heart) spawns."},
	{"--biome-pedestals", "-i", cmd_biome_pedestals, "Generate biome item pedestal spawns."},
	{"--biome-altars", "-w", cmd_biome_altars, "Generate biome wand altar spawns."},
	{"--no-pixel-scene-indexing", nullptr, cmd_biome_pixel_scene_indexing,
		"Disable indexing spawns inside of pixel scenes. Otherwise on by default."},
	{"--pixel-scenes", "-ps", cmd_biome_pixel_scene_search,
		"Generate pixel scene objects (for filtering puzzles, etc.). Not required for other objects inside pixel scenes to spawn."},
	{"--t10-shops", nullptr, cmd_hell_shops, "Deprecated. Generate sky/hell shop items."},
	{"--nightmare", "-n", cmd_nightmare, "Deprecated. Generate a nightmare world."},
	{"--gen-potions", "-gp", cmd_gen_potions,
		"Generate potion contents instead of using generic items like 'potion_secret'."},
	{"--gen-spells", "-gs", cmd_gen_spells,
		"Generate spells instead of using generic items like 'random_spell'. Wand stats must also be generated for spells on wands."},
	{"--gen-wands", "-gw", cmd_gen_wands,
		"Generate wands instead of using generic items like 'wand_t6ns'. Spells must also be generated for spells on wands."},
	{"--biomes", "-b", cmd_biomes, "Select which biomes to generate. Ex. \"-b coalmine excavationsite crypt\"."},
	{"--upwarps", "-u", cmd_upwarps, "Only check upwarped chests. Much faster than full biome generation."},
	{"--aggregate", "-a", cmd_aggregate,
		"Aggregate filter checks between all loaded objects instead of requiring every filter to pass on a single object."},
	{"--filter-items", "-fi", cmd_filter_items,
		"Define item filters of the formats {item[,count]} or {{item1,or-item2[,...]}[,count]}. Ex. \"-fi {{sampo,true_orb}} {bomb,3}\""},
	{"--filter-materials", "-fm", cmd_filter_materials,
		"Define potion material filters of the same format as item filters."},
	{"--filter-spells", "-fs", cmd_filter_spells, "Define spell filters of the format {{spell1,or-spell2[,...]}[,count][,as-always-cast]}. Ex. \"-fs {regeneration_field,1,true}\""},
	{"--filter-pixel-scenes", "-fp", cmd_filter_pixel_scenes,
		"Define pixel scene filters of the same format as item filters, or {scene,{mat1,mat2[,...]}[,count]} for pixel scenes like oil tanks which contain materials."},
	{"--filter-wands", "-fw", cmd_filter_wands, "Define wand stat filters of the format {stat,value[,comparison][,count]}."},
	{"--start-seed", nullptr, cmd_start_seed, "Set first seed to search."},
	{"--end-seed", nullptr, cmd_end_seed, "Set last seed to search."},
	{"--this-seed", "-s", cmd_this_seed, "Search only one seed. Overrides start-seed and end-seed."},
	{"--seeds-from-file", "-if", cmd_seeds_from_file, "Filepath to load a list of specific seeds to search. Useful in conjunction with \"-of -om seed\" to composite searches that can't be done in one go."},
	{"--count-passed", "-cp", cmd_count_passed,
		"Do not record which seeds passed, only how many. Useful for gathering statistics where keeping track of specific seeds is an unnecessary slowdown."},
	{"--no-print-to-console", "-nc", cmd_print_to_console, "Don't print seeds to standard output."},
	{"--print-to-file", "-of", cmd_print_to_file, "Print seeds to an output file."},
	{"--output-file", "-o", cmd_output_file, "Specify output filename."},
	{"--logging-interval", "-li", cmd_logging_interval, "Set logging interval for progress updates, or 0 to disable."},
	{"--output-mode", "-om", cmd_output_mode,
		"Set output mode. 'image' only works when specifically compiled to output images."},
	{"--priority", nullptr, cmd_priority, "Set thread priority. Has no effect on non-CPU backends."},
	{"--exact", nullptr, cmd_exact, "Disable automatic inference of command-line arguments."},
	{"--quiet", "-q", cmd_quiet, "Disable most command-line output."},
	{"--verbose", "-v", cmd_verbose, "Enable verbose logging."},
	{"--debug", nullptr, cmd_debug, "Enable debugging flags."},
};
constexpr int num_commands = sizeof(commands) / sizeof(cmd);
static bool cmd_passed[num_commands] = {};

static bool was_cmd_passed(const char* name) {
	for (int j = 0; j < num_commands; j++) {
		if (!strcmp(name, commands[j].long_name) || (commands[j].short_name && !strcmp(name, commands[j].short_name)))
			return true;
	}
	return false;
}

static void check_argc(int i, int expected) {
	if (i > g_argc - expected) {
		fprintf(stderr, "%s expected at least %i arguments but recieved %i.\n", g_argv[i - 1], expected, g_argc - i);
		std::exit(-1);
	}
}
static void check_range(int i, int min, int max) {
	if (i < min || i > max) {
		fprintf(stderr, "Integer parameter %i not in the valid range [%i,%i].\n", i, min, max);
		std::exit(-1);
	}
}

static int to_int(const std::string_view s) {
	int result;
	auto err = std::from_chars(s.data(), s.data() + s.size(), result);
	if (err.ec != std::errc{} || err.ptr != s.data() + s.size()) {
		fprintf(stderr, "Value '%.*s' could not be converted to a number.\n", (int)s.size(), s.data());
		std::exit(-1);
	}
	return result;
}

template <std::size_t N, std::size_t N2>
static int list_to_id(std::string_view s, const char* (&lists)[N2][N]) {
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N2; j++)
			if (s == lists[j][i])
				return i;
	fprintf(stderr, "Invalid ID '%.*s' in parameter list. Did you mean:\n", (int)s.size(), s.data());
	for (int i = 0; i < N; i++) {
		bool passed = false;
		for (int j = 0; j < N2; j++)
			passed |= std::string_view(lists[j][i]).starts_with(s);
		if (!passed)
			continue;
		fprintf(stderr, "  ");
		for (int j = 0; j < N2; j++)
			fprintf(stderr, "'%s'%s", lists[j][i], j == N2 - 1 ? "\n" : ", ");
	}
	fprintf(stderr, "All valid options:\n");
	for (int i = 0; i < N; i++) {
		fprintf(stderr, "  ");
		for (int j = 0; j < N2; j++)
			fprintf(stderr, "'%s'%s", lists[j][i], j == N2 - 1 ? "\n" : ", ");
	}
	std::exit(-1);
}
template <std::size_t N, std::size_t N2, typename T, std::size_t M>
static Vec2i list_to_id(std::string_view s, const char* (&lists)[N2][N], T (&subset)[M]) {
	int idx = -1;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N2; j++) {
			if (s == lists[j][i]) {
				idx = i;
				break;
			}
		}
	}
	for (int i = 0; i < M; i++)
		if (idx == subset[i])
			return {idx, i};
	fprintf(stderr, "Invalid ID '%.*s' in parameter list. Did you mean:\n", (int)s.size(), s.data());
	for (int i = 0; i < M; i++) {
		bool passed = false;
		for (int j = 0; j < N2; j++)
			passed |= std::string_view(lists[j][subset[i]]).starts_with(s);
		if (!passed)
			continue;
		fprintf(stderr, "  ");
		for (int j = 0; j < N2; j++)
			fprintf(stderr, "'%s'%s", lists[j][subset[i]], j == N2 - 1 ? "\n" : ", ");
	}
	fprintf(stderr, "All valid options:\n");
	for (int i = 0; i < M; i++) {
		fprintf(stderr, "  ");
		for (int j = 0; j < N2; j++)
			fprintf(stderr, "'%s'%s", lists[j][subset[i]], j == N2 - 1 ? "\n" : ", ");
	}
	std::exit(-1);
}
static std::vector<std::string_view> decompose_inner(const char* str) {
	if (!strchr(str, '}')) {
		fprintf(stderr, "Unterminated composite object '%s'.\n", str);
		std::exit(-1);
	}
	std::vector<std::string_view> out;
	const char* cur_str = str + 1;
	if (DEBUG_FLAGS & DEBUG::LOG_CLI_PARSING)
		printf("input: %s\n", str);
	int len = strlen(cur_str);
	while (len > 1) {
		if (*cur_str == '{') {
			int frag_len = strcspn(cur_str, "}");
			out.emplace_back(cur_str, frag_len + 1);
			if (DEBUG_FLAGS & DEBUG::LOG_CLI_PARSING)
				printf("fragment: %.*s\n", frag_len + 1, cur_str);
			len -= frag_len + 1;
			cur_str += frag_len + 1;
			if (!strchr(str, '}')) {
				fprintf(stderr, "Unterminated composite object '%s'.\n", str);
				std::exit(-1);
			}
		} else {
			int frag_len = strcspn(cur_str, ",}");
			out.emplace_back(cur_str, frag_len);
			len -= frag_len + 1;
			cur_str += frag_len + 1;
		}
	}
	return out;
}
static void decompose_allowed_inner(
	const char* str, std::vector<std::string_view> sv, std::initializer_list<int> allowed_lengths) {
	bool passed = false;
	for (int i = 0; i < allowed_lengths.size(); i++)
		passed |= sv.size() == allowed_lengths.begin()[i];
	if (!passed) {
		fprintf(stderr, "Invalid entry count %i in composite object '%s'. Expected {", (int)sv.size(), str);
		for (int i = 0; i < allowed_lengths.size(); i++)
			fprintf(stderr, "%i%s", allowed_lengths.begin()[i], i == allowed_lengths.size() - 1 ? "}\n" : ",");
		std::exit(-1);
	}
}
static std::vector<std::string_view> decompose(const char* str) {
	if (str[0] != '{') {
		fprintf(stderr, "Composite object '%s' does not start with '{'.\n", str);
		std::exit(-1);
	}
	return decompose_inner(str);
}
static std::vector<std::string_view> decompose(const char* str, std::initializer_list<int> allowed_lengths) {
	auto ret = decompose(str);
	decompose_allowed_inner(str, ret, allowed_lengths);
	return ret;
}
static std::vector<std::string_view> maybe_decompose(const char* str) {
	if (str[0] != '{') {
		if (DEBUG_FLAGS & DEBUG::LOG_CLI_PARSING)
			printf("not a compisite: %s\n", str);
		std::vector<std::string_view> out;
		out.emplace_back(str);
		return out;
	}
	return decompose_inner(str);
}
static std::vector<std::string_view> maybe_decompose(const char* str, std::initializer_list<int> allowed_lengths) {
	auto ret = maybe_decompose(str);
	decompose_allowed_inner(str, ret, allowed_lengths);
	return ret;
}

int cmd_help(int idx) {
	printf(
		"Note: braces may be omitted around composite objects with only a single element, i.e. '{gold}' is the same as 'gold'.\n");
	printf("Possible flags are:\n");
	for (int i = 0; i < num_commands; i++) {
		if (commands[i].short_name)
			printf("%26s  %*s(%s)  %s\n", commands[i].long_name, 3 - (int)strlen(commands[i].short_name), "",
				commands[i].short_name, commands[i].desc);
		else
			printf("%26s         %s\n", commands[i].long_name, commands[i].desc);
	}
	std::exit(-1);
}

int cmd_quiet(int i) {
	QUIET = true;
	config.outputCfg.printProgressLog = false;
	config.outputCfg.printOutputToConsole = false;
	config.outputCfg.printInterval = 0;
	return i;
}
static bool AUTOMATIC_FLAGS = true;
int cmd_exact(int i) {
	AUTOMATIC_FLAGS = false;
	return i;
}

// --- Unlock gating implementation hooks (variables declared in main.cu) ---
extern bool g_enable_unlock_gating;
extern bool g_allow_manual_unlocks;
extern std::string g_flags_dir;
extern std::string g_locked_csv_path;

int cmd_flags_dir(int i) {
	check_argc(i, 1);
	g_flags_dir = std::string(g_argv[i++]);
	return i;
}
int cmd_locked_spells_csv(int i) {
	check_argc(i, 1);
	g_locked_csv_path = std::string(g_argv[i++]);
	return i;
}
int cmd_allow_manual_unlocks(int i) {
	g_allow_manual_unlocks = true;
	return i;
}
int cmd_enable_unlock_gating(int i) {
	g_enable_unlock_gating = true;
	return i;
}

int cmd_cart(int i) {
	check_argc(i, 1);
	int idx = list_to_id(g_argv[i++], IDs::carts);
	config.precheckCfg.cart = {idx != 0, (CartType)idx};
	return i;
}
int cmd_flask(int i) {
	check_argc(i, 1);
	int idx = list_to_id(g_argv[i++], IDs::materials, HTables::starting_flasks).x;
	config.precheckCfg.flask = {idx != 0, (Material)idx};
	return i;
}
int cmd_rain(int i) {
	check_argc(i, 1);
	int idx = list_to_id(g_argv[i++], IDs::materials, HTables::rain_materials).x;
	config.precheckCfg.rain = {idx != 0, (Material)idx};
	return i;
}

int cmd_biome_mods(int i) {
	check_argc(i, 1);
	config.precheckCfg.biomes.check = true;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = decompose(g_argv[i++], {2});
		Vec2i idx1 = list_to_id(composite[0], IDs::biomes, HTables::bm_biomes);
		int idx2 = list_to_id(composite[1], IDs::biome_modifiers, HTables::bm_lists[idx1.y]).x;
		if (config.precheckCfg.biomes.modifiers[idx1.y] != BM_NONE) {
			fprintf(stderr, "Conflicting biome modifiers specified for biome %s.\n", IDs::biomes[0][idx1.x]);
			exit(-1);
		}
		config.precheckCfg.biomes.modifiers[idx1.y] = (BiomeModifier)idx2;
	}
	return i;
}
int cmd_alchemy(int i) {
	check_argc(i, 2);
	config.precheckCfg.alchemy.check = true;
	{
		auto composite = decompose(g_argv[i++], {3, 4});
		int idx1 = list_to_id(composite[0], IDs::materials, HTables::alchemy_materials).x;
		int idx2 = list_to_id(composite[1], IDs::materials, HTables::alchemy_materials).x;
		int idx3 = list_to_id(composite[2], IDs::materials, HTables::alchemy_materials).x;
		int order = composite.size() > 3 ? list_to_id(composite[3], IDs::alchemy_orderings) :
										   AlchemyOrdering::UNORDERED;
		config.precheckCfg.alchemy.LC = {(Material)idx1, (Material)idx2, (Material)idx3};
	}
	{
		auto composite = decompose(g_argv[i++], {3, 4});
		int idx1 = list_to_id(composite[0], IDs::materials, HTables::alchemy_materials).x;
		int idx2 = list_to_id(composite[1], IDs::materials, HTables::alchemy_materials).x;
		int idx3 = list_to_id(composite[2], IDs::materials, HTables::alchemy_materials).x;
		int order = composite.size() > 3 ? list_to_id(composite[3], IDs::alchemy_orderings) :
										   AlchemyOrdering::UNORDERED;
		config.precheckCfg.alchemy.AP = {(Material)idx1, (Material)idx2, (Material)idx3};
	}
	return i;
}
int cmd_fungal(int i) {
	check_argc(i, 1);
	config.precheckCfg.fungal.check = true;
	int j = 0;
	for (; i < g_argc && g_argv[i][0] == '{';) {
		auto composite = decompose(g_argv[i++], {2, 4});
		int idx1 = list_to_id(composite[0], IDs::materials, HTables::fungal_from).x;
		int idx2 = list_to_id(composite[1], IDs::materials, HTables::fungal_to).x;
		int start = composite.size() > 2 ? to_int(composite[2]) : 0;
		int end = composite.size() > 2 ? to_int(composite[3]) : 20;
		check_range(start, 0, 20);
		check_range(end, 0, 20);
		config.precheckCfg.fungal.shifts[j++] = {(ShiftSource)idx1, (ShiftDest)idx2, start, end};
	}
	return i;
}
int cmd_perks(int i) {
	check_argc(i, 1);
	config.precheckCfg.perks.check = true;
	int j = 0;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = maybe_decompose(g_argv[i++], {1, 3, 4});
		int perk = list_to_id(composite[0], IDs::perks);
		int start = composite.size() > 1 ? to_int(composite[1]) : 0;
		int end = composite.size() > 1 ? to_int(composite[2]) : 2;
		int is_lottery = composite.size() > 3 ? list_to_id(composite[3], IDs::booleans) : 0;
		config.precheckCfg.perks.perks[j++] = {(Perk)perk, (bool)is_lottery, start, end};
	}
	return i;
}

int cmd_hms(int i) {
	check_argc(i, 1);
	auto composite = decompose(g_argv[i++], {2});
	config.spawnableCfg.minHMidx = to_int(composite[0]);
	config.spawnableCfg.maxHMidx = to_int(composite[1]);
	check_range(config.spawnableCfg.minHMidx, 0, 6);
	check_range(config.spawnableCfg.maxHMidx, 0, 6);
	return i;
}
int cmd_pws(int i) {
	check_argc(i, 1);
	auto composite = maybe_decompose(g_argv[i++], {1, 2});
	config.spawnableCfg.pwWidth = {to_int(composite[0]), composite.size() > 1 ? to_int(composite[1]) : 0};
	if (i < g_argc && g_argv[i][0] == '{') {
		auto composite = decompose(g_argv[i++], {2});
		config.spawnableCfg.pwCenter = {to_int(composite[0]), to_int(composite[1])};
	}
	return i;
}
int cmd_greed(int i) {
	config.spawnableCfg.greedCurse = true;
	return i;
}
int cmd_pacifist(int i) {
	config.spawnableCfg.pacifist = true;
	return i;
}
int cmd_shop_spells(int i) {
	config.spawnableCfg.shopSpells = true;
	return i;
}
int cmd_shop_wands(int i) {
	config.spawnableCfg.shopWands = true;
	return i;
}
int cmd_eye_rooms(int i) {
	config.spawnableCfg.eyeRooms = true;
	return i;
}
int cmd_biome_chests(int i) {
	config.spawnableCfg.biomeChests = true;
	return i;
}
int cmd_biome_pedestals(int i) {
	config.spawnableCfg.biomePedestals = true;
	return i;
}
int cmd_biome_altars(int i) {
	config.spawnableCfg.biomeAltars = true;
	return i;
}
int cmd_biome_pixel_scene_indexing(int i) {
	config.spawnableCfg.biomePixelSceneIndexing = false;
	return i;
}
int cmd_biome_pixel_scene_search(int i) {
	config.spawnableCfg.biomePixelSceneSearch = true;
	return i;
}
int cmd_hell_shops(int i) {
	config.spawnableCfg.hellShops = true;
	return i;
}
int cmd_nightmare(int i) {
	config.spawnableCfg.nightmare = true;
	return i;
}
int cmd_gen_potions(int i) {
	config.spawnableCfg.genPotions = true;
	return i;
}
int cmd_gen_spells(int i) {
	config.spawnableCfg.genSpells = true;
	return i;
}
int cmd_gen_wands(int i) {
	config.spawnableCfg.genWands = true;
	return i;
}
int cmd_upwarps(int i) {
	config.precheckCfg.precheckUpwarps = true;
	return i;
}
int cmd_aggregate(int i) {
	config.filterCfg.aggregate = true;
	return i;
}

std::vector<Biome> biome_list;
int cmd_biomes(int i) {
	check_argc(i, 1);
	for (; i < g_argc && g_argv[i][0] != '-';) {
		int n = list_to_id(g_argv[i++], IDs::biomes);
		if (n == 0) {
			biome_list.clear();
			for (int j = 1; j < 23; j++)
				biome_list.push_back((Biome)j);
		} else if (n == 23) {
			biome_list.push_back(B_COALMINE);
			biome_list.push_back(B_EXCAVATIONSITE);
			biome_list.push_back(B_SNOWCAVE);
			biome_list.push_back(B_SNOWCASTLE);
			biome_list.push_back(B_RAINFOREST);
			biome_list.push_back(B_RAINFOREST_OPEN);
			biome_list.push_back(B_VAULT);
			biome_list.push_back(B_CRYPT);
		} else {
			if (std::ranges::find(biome_list, (Biome)n) != biome_list.end()) {
				fprintf(stderr, "Biome %s specified more than once.\n", IDs::biomes[0][n]);
				exit(-1);
			}
			biome_list.push_back((Biome)n);
		}
	}
	return i;
}
int cmd_filter_items(int i) {
	check_argc(i, 1);
	int j = config.filterCfg.itemFilterCount;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = maybe_decompose(g_argv[i++], {1, 2});
		std::string tmp(composite[0]);
		auto inner = maybe_decompose(tmp.c_str());
		for (int k = 0; k < inner.size(); k++)
			config.filterCfg.itemFilters[j].items[k] = (Item)list_to_id(inner[k], IDs::items);
		config.filterCfg.itemFilters[j++].duplicates = composite.size() > 1 ? to_int(composite[1]) : 1;
	}
	config.filterCfg.itemFilterCount = j;
	return i;
}
int cmd_filter_materials(int i) {
	check_argc(i, 1);
	int j = config.filterCfg.materialFilterCount;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = maybe_decompose(g_argv[i++], {1, 2});
		std::string tmp(composite[0]);
		auto inner = maybe_decompose(tmp.c_str());
		for (int k = 0; k < inner.size(); k++)
			config.filterCfg.materialFilters[j].materials[k] = (Material)list_to_id(inner[k], IDs::materials);
		config.filterCfg.materialFilters[j++].duplicates = composite.size() > 1 ? to_int(composite[1]) : 1;
	}
	config.filterCfg.materialFilterCount = j;
	return i;
}
int cmd_filter_spells(int i) {
	check_argc(i, 1);
	int j = config.filterCfg.spellFilterCount;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = maybe_decompose(g_argv[i++], {1, 2, 3});
		std::string tmp(composite[0]);
		auto inner = maybe_decompose(tmp.c_str());
		for (int k = 0; k < inner.size(); k++)
			config.filterCfg.spellFilters[j].spells[k] = (Spell)list_to_id(inner[k], IDs::spells);
		config.filterCfg.spellFilters[j].asAlwaysCast = composite.size() > 2 ? list_to_id(composite[2], IDs::booleans) : false;
		config.filterCfg.spellFilters[j].perWand = false;
		config.filterCfg.spellFilters[j++].duplicates = composite.size() > 1 ? to_int(composite[1]) : 1;
	}
	config.filterCfg.spellFilterCount = j;
	return i;
}
int cmd_filter_pixel_scenes(int i) {
	check_argc(i, 1);
	int j = config.filterCfg.pixelSceneFilterCount;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = maybe_decompose(g_argv[i++], {1, 2, 3});
		std::string tmp(composite[0]);
		auto inner1 = maybe_decompose(tmp.c_str());
		std::string tmp2(composite.size() > 1 ? composite[1] : "");
		auto inner2 = composite.size() > 1 && composite[1][0] == '{' ? decompose(tmp2.c_str()) :
																	   std::vector<std::string_view>();
		for (int k = 0; k < inner1.size(); k++)
			config.filterCfg.pixelSceneFilters[j].pixelScenes[k] = (PixelScene)list_to_id(inner1[k], IDs::pixel_scenes);
		for (int k = 0; k < inner2.size(); k++)
			config.filterCfg.pixelSceneFilters[j].materials[k] = (Material)list_to_id(inner2[k], IDs::materials);
		config.filterCfg.pixelSceneFilters[j].checkMats = inner2.size();
		config.filterCfg.pixelSceneFilters[j++].duplicates = composite.size() > 2 ? to_int(composite[2]) :
															 composite.size() > 1 && !inner2.size() ?
																					to_int(composite[1]) :
																					1;
	}
	config.filterCfg.pixelSceneFilterCount = j;
	return i;
}
int cmd_filter_wands(int i) {
	check_argc(i, 1);
	int j = config.filterCfg.wandStatFilterCount;
	for (; i < g_argc && g_argv[i][0] != '-';) {
		auto composite = decompose(g_argv[i++], {2, 3, 4});
		config.filterCfg.wandStatFilters[j].stat = (WandStat)list_to_id(composite[0], IDs::wand_stats);
		config.filterCfg.wandStatFilters[j].value = to_int(composite[1]);
		config.filterCfg.wandStatFilters[j].comparison =
			composite.size() > 2 ? list_to_id(composite[2], IDs::comparisons) : 3;
		config.filterCfg.wandStatFilters[j++].duplicates = composite.size() > 3 ? to_int(composite[3]) : 1;
	}
	config.filterCfg.wandStatFilterCount = j;
	return i;
}
int cmd_count_passed(int i) {
	config.outputCfg.countPassesOnly = true;
	return i;
}
int cmd_print_to_console(int i) {
	config.outputCfg.printOutputToConsole = false;
	return i;
}
int cmd_print_to_file(int i) {
	config.outputCfg.printOutputToFile = true;
	return i;
}
int cmd_output_file(int i) {
	check_argc(i, 1);
	config.outputCfg.printOutputToFile = true;
	config.outputCfg.outputFile = g_argv[i++];
	return i;
}
int cmd_logging_interval(int i) {
	check_argc(i, 1);
	config.outputCfg.printInterval = to_int(g_argv[i++]);
	if (!config.outputCfg.printInterval)
		config.outputCfg.printProgressLog = false;
	return i;
}
int cmd_output_mode(int i) {
	check_argc(i, 1);
	config.outputCfg.outputMode = list_to_id(g_argv[i++], IDs::output_modes);
	return i;
}
int cmd_start_seed(int i) {
	check_argc(i, 1);
	config.generalCfg.seedStart = to_int(g_argv[i++]);
	return i;
}
int cmd_end_seed(int i) {
	check_argc(i, 1);
	config.generalCfg.seedEnd = to_int(g_argv[i++]);
	return i;
}
int cmd_this_seed(int i) {
	check_argc(i, 1);
	config.generalCfg.seedStart = to_int(g_argv[i++]);
	config.generalCfg.seedEnd = config.generalCfg.seedStart;
	return i;
}
std::vector<uint32_t> seed_list;
int cmd_seeds_from_file(int i) { check_argc(i, 1);
	std::ifstream f(g_argv[i++]);
	if (!f.good()) {
		fprintf(stderr, "Error opening seed list file.\n");
		std::exit(-1);
	}
	while (!f.eof()) {
		uint32_t n;
		f >> n;
		seed_list.emplace_back(n);
	}
	config.generalCfg.seedStart = 0;
	config.generalCfg.seedEnd = seed_list.size() - 1;
	config.generalCfg.seedBlockOverride = true;
	return i;
}

int cmd_priority(int i) {
	check_argc(i, 1);
	config.generalCfg.priority = list_to_id(g_argv[i++], IDs::priorities);
	return i;
}
int cmd_verbose(int i) {
	DEBUG_FLAGS |= DEBUG::LOG_VERBOSE;
	return i;
}
int cmd_debug(int i) {
	check_argc(i, 1);
	for (; i < g_argc && g_argv[i][0] != '-';) {
		int flag = list_to_id(g_argv[i++], IDs::debug_flags);
		if (flag >= 9) {
			check_argc(i, 1);
			int value = to_int(g_argv[i++]);
			switch (flag) {
			case 9:
				DEBUG_SEED_BLOCK_OVERRIDE = value;
				config.generalCfg.seedBlockOverride = true;
				break;
			case 10: DEBUG_DISPATCH_RATE_OVERRIDE = value; break;
			}
		}
		else if (!flag)
			DEBUG_FLAGS |= DEBUG::ALL;
		else
			DEBUG_FLAGS |= (1 << (flag - 1));
	}
	return i;
}

void cli_main(int argc, char** argv) {
	g_argc = argc;
	g_argv = argv;
	if (argc == 1) {
		fprintf(stderr, "Not enough arguments.");
		cmd_help(0);
		return;
	}
	for (int i = 1; i < argc;) {
		for (int j = 0; j < num_commands; j++) {
			if (!strcmp(argv[i], commands[j].long_name) ||
				(commands[j].short_name && !strcmp(argv[i], commands[j].short_name))) {
				cmd_passed[j] = true;
				i = commands[j].fn(++i);
				goto end;
			}
		}
		fprintf(stderr, "Unrecognized flag '%s'.\n", argv[i]);
		cmd_help(0);
		return;
end:
		continue;
	}

	bool has_biomes = !!biome_list.size();
	bool has_items = !!config.filterCfg.itemFilterCount;
	bool has_materials = !!config.filterCfg.materialFilterCount;
	bool has_spells = !!config.filterCfg.spellFilterCount;
	bool has_pixel_scenes = !!config.filterCfg.pixelSceneFilterCount;

	Item item_pedestal_only = ITEM_NONE;
	Item item_chest_only = ITEM_NONE;
	Item item_coalmine_only = ITEM_NONE;
	bool item_duplicates = config.filterCfg.itemFilterCount > 1;
	Spell spell_shop_only = SPELL_NONE;

	for (int i = 0; i < config.filterCfg.itemFilterCount; i++) {
		for (int j = 0; j < FILTER_OR_COUNT; j++) {
			uint8_t type = HTables::item_sources[config.filterCfg.itemFilters[i].items[j]];
			if (type == 0 && !item_chest_only)
				item_chest_only = config.filterCfg.itemFilters[i].items[j];
			if (type == 1 && !item_pedestal_only)
				item_pedestal_only = config.filterCfg.itemFilters[i].items[j];
			if (config.filterCfg.itemFilters[i].items[j] == SAMPO ||
				config.filterCfg.itemFilters[i].items[j] == TRUE_ORB)
				item_coalmine_only = config.filterCfg.itemFilters[i].items[j];
			if (config.filterCfg.itemFilters[i].duplicates > 1)
				item_duplicates = true;
		}
	}
	for (int i = 0; i < config.filterCfg.spellFilterCount; i++) {
		for (int j = 0; j < FILTER_OR_COUNT; j++) {
			ActionType type = HTables::spells[config.filterCfg.spellFilters[i].spells[j]].type;
			if (type >= ActionType::MATERIAL || type == ActionType::STATIC_PROJECTILE)
				spell_shop_only = config.filterCfg.spellFilters[i].spells[j];
		}
	}

	if (AUTOMATIC_FLAGS) {
	}
}