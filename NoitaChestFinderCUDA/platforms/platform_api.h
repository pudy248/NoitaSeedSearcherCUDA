//////////////////////////////////////////////////////////////////////////////////////////////////////
// This file contains API functions to control how platform-specific functions are interfaced with. //
// Unlike platform.h, this file should be treated as read-only and no implementation is needed.     //
//////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "platform_compute_helpers.h"
#include "../Configuration.h"
#include "../Worldgen.h"

struct SearchConfig
{
	MemSizeConfig memSizes;
	GeneralConfig generalCfg;
	StaticPrecheckConfig precheckCfg;
	SpawnableConfig spawnableCfg;
	FilterConfig filterCfg;
	OutputConfig outputCfg;

	int biomeCount;
	BiomeWangScope biomeScopes[20];

	SearchConfig() = default;
};

//These are internal functions and variables and should not be accessed by platform-specific code.
namespace API_INTERNAL
{

	int WorkerAppetite = 1;
	int NumWorkers = 1;
	int DispatchRate = 1;
	SearchConfig config;

	void CreateConfig(int maxMapArea, int biomeCount)
	{
		config.memSizes = {
			40_GB,

	#ifdef DO_WORLDGEN
			(size_t)3 * maxMapArea + 4096,
	#else
			(size_t)512,
	#endif
			(size_t)3 * maxMapArea + 128,
			(size_t)4 * maxMapArea,
			(size_t)maxMapArea,
			(size_t)4096,
		};

		config.generalCfg = { 1, INT_MAX, 1, false };
#ifdef REALTIME_SEEDS
		generalCfg.seedBlockSize = 1;
#endif
		config.spawnableCfg = { {0, 0}, {0, 0}, 0, 7,
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
			false, //wand contents
		};

		ItemFilter iFilters[] = { ItemFilter({SAMPO}, 1), ItemFilter({MIMIC_SIGN}) };
		MaterialFilter mFilters[] = { MaterialFilter({WATER}, 2) };
		SpellFilter sFilters[] = {
			SpellFilter({SPELL_LUMINOUS_DRILL, SPELL_LASER_LUMINOUS_DRILL, SPELL_BLACK_HOLE, SPELL_BLACK_HOLE_DEATH_TRIGGER}, 1),
			SpellFilter({SPELL_LIGHT_BULLET, SPELL_LIGHT_BULLET_TRIGGER, SPELL_SPITTER}),
			SpellFilter({SPELL_CURSE_WITHER_PROJECTILE}) };
		PixelSceneFilter psFilters[] = { PixelSceneFilter({PS_NONE}, {MAGIC_LIQUID_HP_REGENERATION}) };

		config.filterCfg = FilterConfig(false, 1, iFilters, 0, mFilters, 0, sFilters, 0, psFilters, false, 10);

		config.precheckCfg = {
			{false, SKATEBOARD},
			{false, GOLD},
			{false, SPELL_LIGHT_BULLET, SPELL_DYNAMITE},
			{false, WATER},
			{false, AlchemyOrdering::ONLY_CONSUMED, {MATERIAL_NONE, MATERIAL_NONE, MATERIAL_NONE}, {MUD, WATER, SOIL}},
			{false, {BM_GOLD_VEIN_SUPER, BM_NONE}},
			{false, {FungalShift(SS_GOLD, SD_FLASK, 0, 1)}},
			{false, //Example: Searches for perk lottery + extra perk in first HM, then any 4 perks in 2nd HM as long as they all are lottery picks
				{{PERK_PERKS_LOTTERY, true, 0, 3}, {PERK_EXTRA_PERK, true, 0, 3}, {PERK_NONE, true, 3, 4}, {PERK_NONE, true, 4, 5}, {PERK_NONE, true, 5, 6}, {PERK_NONE, true, 6, 7}},
				{3, 4, 4, 4, 4, 4, 4} //Also, XX_NONE is considered to be equal to everything for like 90% of calculations
			},
		};
		config.outputCfg = { 10, false };

		config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.x * 2 + 1;
		config.memSizes.spawnableMemSize *= config.spawnableCfg.pwWidth.y * 2 + 1;
		config.memSizes.spawnableMemSize *= biomeCount;
	}
}

namespace PLATFORM_API
{
	//This struct contains all information which is specific to a single batch, which can include temporary 
	// thread handles, seed information, memory allocations, and other information. 
	struct SpanParams
	{
		//Actual seed span to be searched.
		int seedStart;
		int seedCount;
	};
	//Return value from span evaluation which should be passed back to the API.
	struct SpanRet
	{
		int seedStart;
		int seedCount;
		bool seedFound;
		int leftoverSeeds;

		//This should point to the host-accessible memory which the output of EvaluateSpan
		void* outputPtr;
	};

	//Sets how many spans are dispatched (and how many return structs are expected to be recieved) by each job.
	void SetWorkerAppetite(int appetite)
	{
		API_INTERNAL::WorkerAppetite = appetite;
	}
	//Sets how many workers should concurrently receive jobs.
	void SetWorkerCount(int count)
	{
		API_INTERNAL::NumWorkers = count;
	}
	void SetTargetDispatchRate(int dispatchesPerSecond)
	{
		API_INTERNAL::DispatchRate = max(1, dispatchesPerSecond / 20);
	}
	//Span evaluation will want to know what to look for, this will provide the relevant information.
	SearchConfig GetSearchConfig()
	{
		return API_INTERNAL::config;
	}
	//Minimum working memory each span should receive in SpanParams.threadMemBlock
	size_t GetMinimumSpanMemory()
	{
		SearchConfig config = GetSearchConfig();
#ifdef DO_WORLDGEN
		uint64_t minMemoryPerThread = config.memSizes.outputSize + config.memSizes.mapDataSize + config.memSizes.miscMemSize + config.memSizes.visitedMemSize + config.memSizes.spawnableMemSize;
#else
		uint64_t minMemoryPerThread = config.memSizes.outputSize + config.memSizes.spawnableMemSize;
#endif
		config.memSizes.threadMemTotal = minMemoryPerThread;
		return minMemoryPerThread;
	}
	//Minimum space needed to hold output binary data.
	size_t GetMinimumOutputMemory()
	{
		return GetSearchConfig().memSizes.outputSize;
	}

	//This is where the magic happens. This function will handle all of the actual seed search stuff. Everything outside
	// of it is just scaffolding for I/O and etc.
	//threadMemBlock should be a block of memory of at minimum GetMinimumSpanMemory() bytes.
	//outputPtr should be a block of memory of at minimum GetMinimumOutputMemory() bytes. The output which is copied to
	// this pointer should be provided to the host via a host-accessible pointer in SpanRet.outputPtr.
	_compute SpanRet EvaluateSpan(SearchConfig config, SpanParams span, void* threadMemBlock, void* outputPtr);
}
using namespace PLATFORM_API;