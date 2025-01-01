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
#include "src/pathfinding.cu"
#include "src/wandgen.cu"
#include "src/search.cu"
#include "src/filter.cu"
#include "src/output.cu"
#define PNG_IMPL
#include "include/pngutils.h"

#include <chrono>

OutputProgressData d;

void appendOutput(char* s, char* c)
{
#ifdef SPAWNABLE_OUTPUT
	printf("%i (checked %i): %s", d.elapsedMillis, d.searchedSeeds, c);
#else
	//printf("%i: %s (checked %i)\n", d.elapsedMillis, s, d.searchedSeeds);
#endif
}

namespace DATA_SCRIPTS
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
		}
	}
}

			printf("_data const static int spellProbs_%i_Counts[] = {\n", tier);
			for (int t = 0; t < 8; t++)
			{
				printf("%i,\n", counters[t]);
			}
			printf("};\n\n");
		}
	}
}
int main()
{
	int biomeCount = 0;
	int maxMapArea = 0;
	//InstantiateBiome("resources/wang_tiles/coalmine.png", config.biomeScopes, biomeCount, maxMapArea);
	config.biomeCount = biomeCount;

	config.memSizes = {
			40_GB,

	#ifdef DO_WORLDGEN
			(size_t)maxMapArea * 3 + 512, // output
	#else
			(size_t)512,
	#endif
			(size_t)maxMapArea * 2, // map
			(size_t)maxMapArea * 2, // misc
			(size_t)maxMapArea + 256, // visited
			(size_t)2048, // spawnables
	};
	config.generalCfg = { 1, INT_MAX, biomeCount ? 1u : 256u, false };
#ifdef SEEDS_AS_TRIES
	config.generalCfg.seedStart = 1;
	config.generalCfg.endSeed = 100;
#endif
#ifdef REALTIME_SEEDS
	config.generalCfg.seedBlockSize = 1;
#endif
	config.spawnableCfg = {
		{0, 0}, {0, 0}, 0, 0,
		false, //greed
		false, //pacifist
		false, //shop spells
		false, //shop wands
		false, //eye rooms
		false, //upwarp check
		false, //biome chests
		false, //biome pedestals
		false, //biome altars
		false, //biome pixelscenes
		false, //enemies
		false, //hell shops
		false, //nightmare
		false, //potion contents
		false, //chest spells
		false, //wand stats
	};

	config.filterCfg = {
		false, 0, {}, 0, {}, 0, {}, 0, {}, false, 27
	};

	config.precheckCfg = {
		{false, CART_NONE},
		{true, GOLD}, // flask
		{false, SPELL_NONE, SPELL_NONE},
		{false, MATERIAL_NONE},
		{false, AlchemyOrdering::UNORDERED, {MUD, WATER, SOIL}, {MUD, WATER, SOIL}},
		{false, {}},
		{false, {FungalShift(SS_DIAMOND, SD_FLASK, 0, 1), FungalShift(SS_FLASK, SD_DIAMOND, 1, 2)}},
		{false, {
			{PERK_ANGRY_GHOST, false, 0, 3},
		}, {PERK_EDIT_WANDS_EVERYWHERE, PERK_INVISIBILITY}, { 3, 3, 3, 3, 3, 3, 3 }},
	};

	config.outputCfg = { 1.f, true, false };

	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.x * 2 + 1;
	config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.y * 2 + 1;
	config.memSizes.spawnableMemSize *= max(1, biomeCount);

	SearchMain(d, appendOutput);

	//SfmlMain();
	return 0;
}