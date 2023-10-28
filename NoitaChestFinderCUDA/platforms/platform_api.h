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
	//Sets how many worker jobs should be dispatched per second, on average.
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