#include "../data/host_tables.h"
#include "../data/ids.h"
#include "../platforms/platform_api.h"

#include <charconv>
#include <ranges>
#include <string_view>
#include <vector>

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
// int cmd_filter_wands(int);
int cmd_count_passed(int);
int cmd_print_to_console(int);
int cmd_print_to_file(int);
int cmd_output_file(int);
int cmd_logging_interval(int);
int cmd_output_mode(int);
int cmd_start_seed(int);
int cmd_end_seed(int);
int cmd_priority(int);

const cmd commands[] = {
	{"--help", "-h", cmd_help, "Display this menu."},
	{"--cart", nullptr, cmd_cart, "Select a starting cart."},
	{"--rain", nullptr, cmd_rain, "Select an initial rain."},
	{"--starting-flask", "-fl", cmd_flask, "Select a starting flask material."},
	{"--alchemy", "-al", cmd_alchemy,
		"Select LC/AP materials. Takes a pair of reactions of the format {mat1,mat2,mat3[,ordering]} for LC and AP. Ex. \"--alchemy {mud,water,soil,unordered} {any,water,any,only_consumed}\""},
	{"--biome-mods", nullptr, nullptr, "Unimplemented in the command-line interface."},
	{"--fungal", "-fg", cmd_fungal,
		"Filter fungal shifts. Takes a list of shifts of the format {from,to[,min-index,max-index]}. Ex. \"-fs {gold,cheese,0,0} {steam,flask}\""},
	{"--perks", nullptr, cmd_perks,
		"Filter the perk deck. Takes a list of perks of the format {perk[,min-index,max-index][,lottery-safe]}. Ex. \"--perks {edit_wands_everywhere,0,2} {perks_lottery,0,2,true}\""},
	{"--holy-mountains", "-hm", cmd_hms,
		"Select which holy mountains to index, of the format {min-index,max-index}. Only has an effect for pacifist chests and shop items, and does not require biome generation. Ex. \"-hm {0,2}\" for the first 3."},
	{"--parallels", "-pw", cmd_pws,
		"Select how many parallel worlds to index, of the format {width[,height]} [{center_x,center_y}]. Ex. \"-pw {1,1}\" for a single parallel world in every direction, a total of 9 copies."},
	{"--greed", nullptr, cmd_greed, "Generate with curse of greed enabled."},
	{"--pacifist", nullptr, cmd_pacifist, "Generate pacifist chests."},
	{"--shop-spells", nullptr, cmd_shop_spells, "Generate HM spells."},
	{"--shop-wands", nullptr, cmd_shop_wands, "Generate HM wands. --gen-spells is required for spells on these wands as well."},
	{"--eye-rooms", nullptr, cmd_eye_rooms, "Generate eye room spells."},
	{"--biome-chests", "-gc", cmd_biome_chests, "Generate biome chest (and heart) spawns."},
	{"--biome-pedestals", "-gp", cmd_biome_pedestals, "Generate biome item pedestal spawns."},
	{"--biome-altars", "-gw", cmd_biome_altars, "Generate biome wand altar spawns."},
	{"--no-pixel-scene-indexing", nullptr, cmd_biome_pixel_scene_indexing, "Disable indexing spawns inside of pixel scenes. Otherwise on by default."},
	{"--pixel-scenes", "-ps", cmd_biome_pixel_scene_search,
		"Generate pixel scene objects (for filtering puzzles, etc.). Not required for other objects inside pixel scenes to spawn."},
	{"--t10-shops", nullptr, cmd_hell_shops, "Deprecated. Generate sky/hell shop items."},
	{"--nightmare", nullptr, cmd_nightmare, "Deprecated. Generate a nightmare world."},
	{"--gen-potions", "-p", cmd_gen_potions, "Generate potion contents instead of using generic items like 'potion_secret'."},
	{"--gen-spells", "-s", cmd_gen_spells,
		"Generate spells instead of using generic items like 'random_spell'. Wand stats must also be generated for spells on wands."},
	{"--gen-wands", "-w", cmd_gen_wands,
		"Generate wands instead of using generic items like 'wand_t6ns'. Spells must also be generated for spells on wands."},
	{"--biomes", "-b", cmd_biomes, "Select which biomes to generate. Ex. \"-b coalmine excavationsite crypt\"."},
	{"--upwarps", "-u", cmd_upwarps, "Only check upwarped chests. Much faster than full biome generation."},
	{"--aggregate", "-a", cmd_aggregate, "Aggregate filter checks between all loaded objects instead of requiring every filter to pass on a single object."},
	{"--filter-items", "-fi", cmd_filter_items,
		"Define item filters of the formats {item[,count]} or {{item1,or-item2[,...]}[,count]}. Ex. \"-fi {{sampo,true_orb}} {bomb,3}\""},
	{"--filter-materials", "-fm", cmd_filter_materials,
		"Define potion material filters of the same format as item filters."},
	{"--filter-spells", "-fs", cmd_filter_spells, "Define spell filters of the same format as item filters."},
	{"--filter-pixel-scenes", "-fp", cmd_filter_pixel_scenes,
		"Define pixel scene filters of the same format as item filters, or {scene,{mat1,mat2[,...]}[,count]} for pixel scenes like oil tanks which contain materials."},
	{"--start-seed", nullptr, cmd_start_seed, "Set first seed to search."},
	{"--end-seed", nullptr, cmd_end_seed, "Set last seed to search."},
	{"--count-passed", "-cp", cmd_count_passed, "Do not record which seeds passed, only how many. Useful for gathering statistics where keeping track of specific seeds is an unnecessary slowdown."},
	{"--print-to-console", "-oc", cmd_print_to_console, "Print seeds to standard output."},
	{"--print-to-file", "-of", cmd_print_to_file, "Print seeds to an output file (Default: output.txt)."},
	{"--output-file", "-o", cmd_output_file, "Specify output filename."},
	{"--logging-interval", "-li", cmd_logging_interval, "Set logging interval for progress updates, or 0 to disable."},
	{"--output-mode", nullptr, cmd_output_mode,
		"Set output mode. 'image' only works when specifically compiled to output images."},
	{"--priority", nullptr, cmd_priority,
		"Set thread priority. Has no effect on non-CPU backends."},

};
constexpr int num_commands = sizeof(commands) / sizeof(cmd);

void check_argc(int i, int expected) {
	if (i >= g_argc - expected)
		printf("Unexpectedly ran out of arguments.\n");
}

int to_int(const std::string_view s) {
	int result;
	auto err = std::from_chars(s.data(), s.data() + s.size(), result);
	if (err.ec != std::errc{} || err.ptr != s.data() + s.size()) {
		printf("Value '%.*s' could not be converted to a number.\n", (int)s.size(), s.data());
		std::abort();
	}
	return result;
}

template <std::size_t N>
constexpr int list_to_id(std::string_view s, const char* (&list)[N]) {
	for (int i = 0; i < N; i++)
		if (s == list[i])
			return i;
	printf("Invalid ID '%.*s' in parameter list. Valid options are:\n", (int)s.size(), s.data());
	for (int i = 0; i < N; i++)
		printf("  '%s'\n", list[i]);
	std::abort();
}
template <std::size_t N, typename T, std::size_t M>
constexpr int list_to_id(std::string_view s, const char* (&list)[N], T (&subset)[M]) {
	int idx = -1;
	for (int i = 0; i < N; i++) {
		if (s == list[i]) {
			idx = i;
			break;
		}
	}
	for (int i = 0; i < M; i++)
		if (idx == subset[i])
			return idx;
	printf("Invalid ID '%.*s' in parameter list. Valid options are:\n", (int)s.size(), s.data());
	for (int i = 0; i < M; i++)
		printf("  '%s'\n", list[subset[i]]);
	std::abort();
}
std::vector<std::string_view> decompose(const char* str) {
	if (str[0] != '{') {
		printf("Composite object '%s' does not start with {.\n", str);
		std::abort();
	}
	if (!strchr(str, '}')) {
		printf("Unterminated composite object '%s'.\n", str);
		std::abort();
	}
	std::vector<std::string_view> out;
	const char* cur_str = str + 1;
	int len = strlen(cur_str);
	while (len > 1) {
		if (*cur_str == '{') {
			int frag_len = strcspn(cur_str, "}");
			out.emplace_back(cur_str, frag_len + 1);
			len -= frag_len + 1;
			cur_str += frag_len + 1;
			if (!strchr(str, '}')) {
				printf("Unterminated composite object '%s'.\n", str);
				std::abort();
			}
		} else {
			int frag_len = strcspn(cur_str, ",}");
			//printf("%i %i '%.*s'\n", len, frag_len, frag_len, cur_str);
			out.emplace_back(cur_str, frag_len);
			len -= frag_len + 1;
			cur_str += frag_len + 1;
		}
	}
	return out;
}
std::vector<std::string_view> decompose(const char* str, std::initializer_list<int> allowed_lengths) {
	auto ret = decompose(str);
	bool passed = false;
	for (int i = 0; i < allowed_lengths.size(); i++)
		if (ret.size() == allowed_lengths.begin()[i])
			passed = true;
	if (!passed) {
		printf("Invalid entry count %i in composite object '%s'.\n", (int)ret.size(), str);
		std::abort();
	}
	return ret;
}
std::vector<std::string_view> maybe_decompose(const char* str) {
	std::vector<std::string_view> out;
	if (str[0] != '{') {
		//printf("'%s' is not a composite.\n", str);
		out.emplace_back(str);
		return out;
	}
	if (!strchr(str, '}')) {
		printf("Unterminated composite object '%s'.\n", str);
		std::abort();
	}
	const char* cur_str = str + 1;
	int len = strlen(cur_str);
	while (len > 1) {
		if (*cur_str == '{') {
			int frag_len = strcspn(cur_str, "}");
			out.emplace_back(cur_str, frag_len + 1);
			len -= frag_len + 1;
			cur_str += frag_len + 1;
			if (!strchr(str, '}')) {
				printf("Unterminated composite object '%s'.\n", str);
				std::abort();
			}
		} else {
			int frag_len = strcspn(cur_str, ",}");
			//printf("%i %i '%.*s'\n", len, frag_len, frag_len, cur_str);
			out.emplace_back(cur_str, frag_len);
			len -= frag_len + 1;
			cur_str += frag_len + 1;
		}
	}
	return out;
}

int cmd_help(int idx) {
	printf("Possible flags are:\n");
	for (int i = 0; i < num_commands; i++) {
		if (commands[i].short_name)
			printf("% 28s  %*s(%s)  %s\n", commands[i].long_name, 4 - strlen(commands[i].short_name), "",
				commands[i].short_name, commands[i].desc);
		else
			printf("% 28s          %s\n", commands[i].long_name, commands[i].desc);
	}
	std::abort();
}
int cmd_cart(int i) {
	check_argc(i, 1);
	int idx = list_to_id(g_argv[i + 1], IDs::carts);
	config.precheckCfg.cart = {idx != 0, (CartType)idx};
	return i + 1;
}
int cmd_flask(int i) {
	check_argc(i, 1);
	int idx = list_to_id(g_argv[i + 1], IDs::materials, HTables::starting_flasks);
	config.precheckCfg.flask = {idx != 0, (Material)idx};
	return i + 1;
}
int cmd_rain(int i) {
	check_argc(i, 1);
	int idx = list_to_id(g_argv[i + 1], IDs::materials, HTables::rain_materials);
	config.precheckCfg.rain = {idx != 0, (Material)idx};
	return i + 1;
}
int cmd_alchemy(int i) {
	check_argc(i, 2);
	config.precheckCfg.alchemy.check = true;
	{
		auto composite = decompose(g_argv[++i], {3, 4});
		int idx1 = list_to_id(composite[0], IDs::materials, HTables::alchemy_materials);
		int idx2 = list_to_id(composite[1], IDs::materials, HTables::alchemy_materials);
		int idx3 = list_to_id(composite[2], IDs::materials, HTables::alchemy_materials);
		int order = composite.size() > 3 ? list_to_id(composite[3], IDs::alchemy_orderings) :
										   AlchemyOrdering::UNORDERED;
		config.precheckCfg.alchemy.LC = {(Material)idx1, (Material)idx2, (Material)idx3};
	}
	{
		auto composite = decompose(g_argv[++i], {3, 4});
		int idx1 = list_to_id(composite[0], IDs::materials, HTables::alchemy_materials);
		int idx2 = list_to_id(composite[1], IDs::materials, HTables::alchemy_materials);
		int idx3 = list_to_id(composite[2], IDs::materials, HTables::alchemy_materials);
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
	++i;
	for (; i < g_argc && g_argv[i][0] == '{'; i++) {
		auto composite = decompose(g_argv[i], {2, 4});
		int idx1 = list_to_id(composite[0], IDs::materials, HTables::fungal_from);
		int idx2 = list_to_id(composite[1], IDs::materials, HTables::fungal_to);
		int start = composite.size() > 2 ? to_int(composite[2]) : 0;
		int end = composite.size() > 2 ? to_int(composite[3]) : 20;
		config.precheckCfg.fungal.shifts[j++] = {(ShiftSource)idx1, (ShiftDest)idx2, start, end};
	}
	return i - 1;
}
int cmd_perks(int i) {
	check_argc(i, 1);
	config.precheckCfg.perks.check = true;
	int j = 0;
	++i;
	for (; i < g_argc && g_argv[i][0] == '{'; i++) {
		auto composite = decompose(g_argv[i], {1, 3, 4});
		int perk = list_to_id(composite[0], IDs::perks);
		int start = composite.size() > 1 ? to_int(composite[1]) : 0;
		int end = composite.size() > 1 ? to_int(composite[2]) : -1;
		int is_lottery = composite.size() > 3 ? list_to_id(composite[3], IDs::booleans) : 0;
		config.precheckCfg.perks.perks[j++] = {(Perk)perk, (bool)is_lottery, start, end};
	}
	return i - 1;
}

int cmd_hms(int i) {
	check_argc(i, 1);
	auto composite = decompose(g_argv[i + 1], {2});
	config.spawnableCfg.minHMidx = to_int(composite[0]);
	config.spawnableCfg.minHMidx = to_int(composite[1]);
	return i + 1;
}
int cmd_pws(int i) {
	check_argc(i, 1);
	++i;
	auto composite = decompose(g_argv[i], {1, 2});
	config.spawnableCfg.pwWidth = {to_int(composite[0]), composite.size() > 1 ? to_int(composite[1]) : 0};
	if (i + 1 < g_argc && g_argv[i + 1][0] == '{') {
		++i;
		auto composite = decompose(g_argv[i], {2});
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

std::vector<int> biome_list;
int cmd_biomes(int i) {
	check_argc(i, 1);
	++i;
	for (; i < g_argc && g_argv[i][0] != '-'; i++)
		biome_list.emplace_back(list_to_id(g_argv[i], IDs::biomes) + 1);
	return i - 1;
}
int cmd_filter_items(int i) {
	check_argc(i, 1);
	config.precheckCfg.fungal.check = true;
	int j = 0;
	++i;
	for (; i < g_argc && g_argv[i][0] == '{'; i++) {
		auto composite = decompose(g_argv[i], {1,2});
		std::string tmp(composite[0]);
		auto inner = maybe_decompose(tmp.c_str());
		for (int k = 0; k < inner.size(); k++)
			config.filterCfg.itemFilters[j].items[k] = (Item)list_to_id(inner[k], IDs::items);
		config.filterCfg.itemFilters[j++].duplicates = composite.size() > 1 ? to_int(composite[1]) : 1;
	}
	config.filterCfg.itemFilterCount = j;
	return i - 1;
}
int cmd_filter_materials(int i) {
	check_argc(i, 1);
	config.precheckCfg.fungal.check = true;
	int j = 0;
	++i;
	for (; i < g_argc && g_argv[i][0] == '{'; i++) {
		auto composite = decompose(g_argv[i], {1, 2});
		std::string tmp(composite[0]);
		auto inner = maybe_decompose(tmp.c_str());
		for (int k = 0; k < inner.size(); k++)
			config.filterCfg.materialFilters[j].materials[k] = (Material)list_to_id(inner[i], IDs::materials);
		config.filterCfg.materialFilters[j++].duplicates = composite.size() > 1 ? to_int(composite[1]) : 1;
	}
	config.filterCfg.materialFilterCount = j;
	return i - 1;
}
int cmd_filter_spells(int i) {
	check_argc(i, 1);
	config.precheckCfg.fungal.check = true;
	int j = 0;
	++i;
	for (; i < g_argc && g_argv[i][0] == '{'; i++) {
		auto composite = decompose(g_argv[i], {1, 2});
		std::string tmp(composite[0]);
		auto inner = maybe_decompose(tmp.c_str());
		for (int k = 0; k < inner.size(); k++)
			config.filterCfg.spellFilters[j].spells[k] = (Spell)list_to_id(inner[k], IDs::spells);
		config.filterCfg.spellFilters[j].asAlwaysCast = false;
		config.filterCfg.spellFilters[j].perWand = false;
		config.filterCfg.spellFilters[j++].duplicates = composite.size() > 1 ? to_int(composite[1]) : 1;
	}
	config.filterCfg.spellFilterCount = j;
	return i - 1;
}
int cmd_filter_pixel_scenes(int i) {
	check_argc(i, 1);
	config.precheckCfg.fungal.check = true;
	int j = 0;
	++i;
	for (; i < g_argc && g_argv[i][0] == '{'; i++) {
		auto composite = decompose(g_argv[i], {1, 2, 3});
		std::string tmp(composite[0]);
		auto inner1 = maybe_decompose(tmp.c_str());
		std::string tmp2(composite.size() > 1 ? composite[1] : "");
		auto inner2 = composite.size() > 1 && composite[1][0] == '{' ? decompose(tmp2.c_str()) : std::vector<std::string_view>();
		for (int k = 0; k < inner1.size(); k++)
			config.filterCfg.pixelSceneFilters[j].pixelScenes[k] = (PixelScene)list_to_id(inner1[k], IDs::pixel_scenes);
		for (int k = 0; k < inner2.size(); k++)
			config.filterCfg.pixelSceneFilters[j].materials[k] = (Material)list_to_id(inner2[k], IDs::materials);
		config.filterCfg.pixelSceneFilters[j].checkMats = inner2.size();
		config.filterCfg.pixelSceneFilters[j++].duplicates =
			composite.size() > 2 ? to_int(composite[2]) : composite.size() > 1 && !inner2.size() ? to_int(composite[1]) : 1;
	}
	config.filterCfg.pixelSceneFilterCount = j;
	return i - 1;
}
int cmd_count_passed(int i) {
	config.outputCfg.countPassesOnly = true;
	return i;
}
int cmd_print_to_console(int i) {
	config.outputCfg.printOutputToConsole = true;
	return i;
}
int cmd_print_to_file(int i) {
	config.outputCfg.printOutputToFile = true;
	return i;
}
int cmd_output_file(int i) {
	check_argc(i, 1);
	config.outputCfg.printOutputToFile = true;
	config.outputCfg.outputFile = g_argv[i + 1];
	return i + 1;
}
int cmd_logging_interval(int i) {
	check_argc(i, 1);
	config.outputCfg.printInterval = to_int(g_argv[i + 1]);
	if (!config.outputCfg.printInterval)
		config.outputCfg.printProgressLog = false;
	return i + 1;
}

int cmd_output_mode(int i) {
	check_argc(i, 1);
	config.outputCfg.outputMode = list_to_id(g_argv[i + 1], IDs::output_modes);
	return i + 1;
}
int cmd_start_seed(int i) {
	check_argc(i, 1);
	config.generalCfg.seedStart = to_int(g_argv[i + 1]);
	return i + 1;
}
int cmd_end_seed(int i) {
	check_argc(i, 1);
	config.generalCfg.seedEnd = to_int(g_argv[i + 1]);
	return i + 1;
}
int cmd_priority(int i) {
	check_argc(i, 1);
	config.generalCfg.priority = list_to_id(g_argv[i + 1], IDs::priorities);
	return i + 1;
}

void cli_main(int argc, char** argv) {
	g_argc = argc;
	g_argv = argv;
	if (argc == 1) {
		printf("Not enough arguments.");
		cmd_help(0);
		return;
	}
	for (int i = 1; i < argc; i++) {
		for (int j = 0; j < num_commands; j++) {
			if (!strcmp(argv[i], commands[j].long_name) ||
				(commands[j].short_name && !strcmp(argv[i], commands[j].short_name))) {
				i = commands[j].fn(i);
				goto end;
			}
		}
		printf("Unrecognized flag '%s'.\n", argv[i]);
		cmd_help(0);
		return;
end:
		continue;
	}
}