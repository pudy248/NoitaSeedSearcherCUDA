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
int main()
{
	BiomeWangScope biomes[20];
	int biomeCount = 0;
	int maxMapArea = 0;
	//InstantiateBiome("resources/wang_tiles/coalmine.png", biomes, biomeCount, maxMapArea);
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

	config.generalCfg = { 1, INT_MAX, 65536, false };
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
		false, //potion contents
		false, //chest spells
		false, //wand stats
	};

	config.filterCfg = {
		false, false, 0, { ItemFilter({SAMPO}) }, 0, {}, 0, {}, 0, {}, false, 27
	};

	config.precheckCfg = {
		{false, CART_NONE},
		{true, GOLD},
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