#pragma once
#include "platform.h"
#include <thread>
#include <cstdlib>
#ifdef WIN32
#define NOMINMAX
#include "Windows.h"
#endif
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include "../include/pngutils.h"
#include "../include/compute.h"
#include "../include/misc_funcs.h"
#include "../include/wak.h"


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
	memset(buffer, 0, sizeof(0x40));
#ifdef _MSC_VER
	int CPUInfo[4] = { -1 };
	__cpuid(CPUInfo, 0x80000002);
	memcpy(buffer, CPUInfo, sizeof(CPUInfo));
	__cpuid(CPUInfo, 0x80000003);
	memcpy(buffer + 16, CPUInfo, sizeof(CPUInfo));
	__cpuid(CPUInfo, 0x80000004);
	memcpy(buffer + 32, CPUInfo, sizeof(CPUInfo));
#else
	int eax, ebx, ecx, edx;
	__cpuid(0x80000002, eax, ebx, ecx, edx);
	memcpy(buffer, &eax, 4);
	memcpy(buffer + 4, &ebx, 4);
	memcpy(buffer + 8, &ecx, 4);
	memcpy(buffer + 12, &edx, 4);
	__cpuid(0x80000003, eax, ebx, ecx, edx);
	memcpy(buffer + 16, &eax, 4);
	memcpy(buffer + 20, &ebx, 4);
	memcpy(buffer + 24, &ecx, 4);
	memcpy(buffer + 28, &edx, 4);
	__cpuid(0x80000004, eax, ebx, ecx, edx);
	memcpy(buffer + 32, &eax, 4);
	memcpy(buffer + 36, &ebx, 4);
	memcpy(buffer + 40, &ecx, 4);
	memcpy(buffer + 44, &edx, 4);
#endif
}

void InitializePlatform()
{
#ifdef SINGLE_THREAD
	NumThreads = 1;
#else
	NumThreads = std::thread::hardware_concurrency();
#endif
	char buffer[0x40];
	GetProcessorName(buffer);
	printf("Running with CPU backend using %s\n", buffer);
	memIdxCtr = 0;
}
void DestroyPlatform()
{

}

void AllocateComputeMemory()
{
	//SearchConfig config = GetSearchConfig();

	SetWorkerCount(NumThreads);
	SetWorkerAppetite(1);
	SetTargetDispatchRate(5);
	printf("Creating %i threads\n", NumThreads);

	hostPtrs.arena = (uint8_t*)malloc(GetMinimumSpanMemory() * NumThreads);
	hostPtrs.output = (uint8_t*)malloc(GetMinimumOutputMemory() * NumThreads);

	coalmine_overlay = (uint8_t*)malloc(3 * 256 * 103);
	ReadBufferImage((uint8_t*)get_wak_file("data/wang_tiles/extra_layers/coalmine.png").c_str(), coalmine_overlay, false);

	printf("Allocated %lluKB of host memory\n", ((GetMinimumSpanMemory() + GetMinimumOutputMemory()) * NumThreads) / 1_KB);
}
void FreeComputeMemory()
{
	free(hostPtrs.arena);
	free(hostPtrs.output);
	free(coalmine_overlay);
}

Worker* CreateWorker()
{
	Worker* w = new Worker;
	w->memIdx = memIdxCtr++;
	w->returned = false;
	return w;
}
void DestroyWorker(Worker& worker)
{
	if (worker.thread.joinable()) worker.thread.join();
}
void ThreadMain(SpanParams params, Worker* worker)
{
	// These should be not taking up all of your system's resources :)
#ifdef WIN32
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#else
	int policy;
	sched_param params;
	pthread_getschedparam(pthread_self(), &policy, &params);
	params.sched_priority = sched_get_priority_min(policy);
	pthread_setschedparam(pthread_self(), policy, &params);
#endif
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

void* UploadToDevice(const void* hMem, size_t size) {
	void* dMem = malloc(size);
	memcpy(dMem, hMem, size);
	return dMem;
}
void HSetBiomeData() {
	SetBiomeData();
	SetBiomePixelScenes();
}
void HSetBiomeData2(BiomePixelScenes* l) {
	memcpy(AllPixelSceneLists, l , sizeof(HostPixelSceneLists));
}