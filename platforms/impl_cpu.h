#pragma once
#include "platform.h"
#include <thread>
#include <intrin.h>

#include "../include/pngutils.h"
#include "../include/compute.h"
#include "../include/misc_funcs.h"

#include <iostream>

int NumThreads;
int memIdxCtr = 0;
struct HostPointers
{
	uint8_t* arena;
	uint8_t* output;
} hostPtrs;

//platform.h impl
struct Worker
{
	int memIdx;
	std::thread thread;
	bool returned;
	SpanRet ret;
};

void GetProcessorName(char* buffer)
{
	int CPUInfo[4] = { -1 };
	__cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];

	memset(buffer, 0, sizeof(0x40));

	// Get the information associated with each extended ID.
	for (int i = 0x80000000; i <= nExIds; ++i)
	{
		__cpuid(CPUInfo, i);
		// Interpret CPU brand string.
		if (i == 0x80000002)
			memcpy(buffer, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(buffer + 16, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(buffer + 32, CPUInfo, sizeof(CPUInfo));
	}
}

void InitializePlatform()
{
	NumThreads = std::thread::hardware_concurrency();
	char buffer[0x40];
	GetProcessorName(buffer);
	printf("Running with CPU backend using %s\n", buffer, NumThreads);
	memIdxCtr = 0;
}
void DestroyPlatform()
{

}

void AllocateComputeMemory()
{
	SearchConfig config = GetSearchConfig();

	SetWorkerCount(NumThreads);
	SetWorkerAppetite(1);
	SetTargetDispatchRate(5);
	printf("Creating %i threads\n", NumThreads);

	hostPtrs.arena = (uint8_t*)malloc(GetMinimumSpanMemory() * NumThreads);
	hostPtrs.output = (uint8_t*)malloc(GetMinimumOutputMemory() * NumThreads);

	coalmine_overlay = (uint8_t*)malloc(3 * 256 * 103);
	ReadImage("resources/wang_tiles/coalmine_overlay.png", coalmine_overlay);

	printf("Allocated %lliKB of host memory\n", ((GetMinimumSpanMemory() + GetMinimumOutputMemory()) * NumThreads) / 1_KB);
}
void FreeComputeMemory()
{
	free(hostPtrs.arena);
	free(hostPtrs.output);
	free(coalmine_overlay);
}

Worker CreateWorker()
{
	Worker w;
	w.memIdx = memIdxCtr++;
	w.returned = false;
	return w;
}
void DestroyWorker(Worker& worker)
{
	if (worker.thread.joinable()) worker.thread.join();
}

void ThreadMain(SpanParams params, Worker* worker)
{
	worker->ret = EvaluateSpan(GetSearchConfig(), params, hostPtrs.arena + GetMinimumSpanMemory() * worker->memIdx, hostPtrs.output + GetMinimumOutputMemory() * worker->memIdx);
	worker->ret.outputPtr = hostPtrs.output + GetMinimumOutputMemory() * worker->memIdx;
	worker->returned = true;
}

void DispatchJob(Worker& worker, SpanParams* spans)
{
	std::thread t = std::thread(ThreadMain, spans[0], &worker);
	worker.thread = std::move(t);
}
bool QueryWorker(Worker& worker)
{
	return worker.returned;
}
SpanRet* SubmitJob(Worker& worker)
{
	worker.returned = false;
	worker.thread.join();
	return &worker.ret;
}
void AbortJob(Worker& worker)
{
	worker.thread.join();
}

void* UploadToDevice(void* hostMem, size_t size)
{
	void* ptr = malloc(size);
	memcpy(ptr, hostMem, size);
	return ptr;
}
void CopyToDevice(void* dPtr, void* hPtr, size_t size)
{
	memcpy(dPtr, hPtr, size);
}