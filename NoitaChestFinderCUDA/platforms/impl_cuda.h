#pragma once
#include "platform.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../misc/pngutils.h"

#include <iostream>

//Copy/pasted standard CUDA error handler
static const char* _cudaGetErrorEnum(cudaError_t error)
{
	return cudaGetErrorName(error);
}
template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char* errorMessage, const char* file,
	const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)
inline void __printLastCudaError(const char* errorMessage, const char* file,
	const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
	}
}

//CUDA-specific kernel config structs
constexpr int MAXBLOCKS = 8;
int BLOCKDIV = 30;
int BLOCKSIZE = 64 * BLOCKDIV;
int NumBlocks;
int memIdxCtr = 0;

//struct KernelIO
//{
//	cudaEvent_t terminated;
//	SpanParams params[BLOCKSIZE];
//	SpanRet ret[BLOCKSIZE];
//};
#define KIO_size BLOCKSIZE * (sizeof(SpanParams) + sizeof(SpanRet))
#define KIO_idx(ptr, idx) ((uint8_t*)((uint64_t)ptr + KIO_size * idx))
#define KIO_params(ptr) ((SpanParams*)((uint64_t)ptr + 0))
#define KIO_ret(ptr) ((SpanRet*)((uint64_t)ptr + BLOCKSIZE * (sizeof(SpanParams))))

struct ComputePointers
{
	volatile int* __restrict__ numActiveThreads;
	uint8_t* __restrict__ dArena;
	uint8_t* __restrict__ uOutput;
	uint8_t* __restrict__ uIO;
} computePtrs;
struct HostPointers
{
	volatile int* numActiveThreads;
	uint8_t* uOutput;
	uint8_t* hIO;
} hostPtrs;

//platform.h impl
struct Worker
{
	int memIdx;
	cudaStream_t stream;
	cudaEvent_t event;
};

//CUDA doesn't actually require much instantiation. Most of this is just debug info and error handling.
void InitializePlatform()
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	if (devCount == 0)
	{
		fprintf(stderr, "No CUDA-capable devices detected! Ensure that the CUDA drivers are installed and that your GPU has compute capability >=2.0");
		exit(EXIT_FAILURE);
	}

	int device;
	cudaGetDevice(&device);

	cudaDeviceProp properties;
	checkCudaErrors(cudaGetDeviceProperties_v2(&properties, 0));
	BLOCKDIV = properties.multiProcessorCount;
	BLOCKSIZE = 64 * BLOCKDIV;

	printf("Running with CUDA backend using device %i: %s.\n", device, properties.name);
	
	cudaSetDeviceFlags(cudaDeviceMapHost);

	memIdxCtr = 0;
}
void DestroyPlatform()
{
	//Optional but good practice.
	checkCudaErrors(cudaDeviceReset());
}

void AllocateComputeMemory()
{
	SearchConfig config = GetSearchConfig();

	//Determine how many workers we have space for.
	uint64_t freeMem;
	uint64_t physicalMem;
	cudaMemGetInfo(&freeMem, &physicalMem);
	printf("Memory free: %lli of %lli bytes\n", freeMem, physicalMem);
	freeMem *= 0.9f; //leave a bit of extra
	freeMem = min(freeMem, config.memSizes.memoryCap);

	size_t memPerThread = GetMinimumSpanMemory() + GetMinimumOutputMemory();
	//printf("Each thread requires %lli bytes of block memory\n", memPerThread);

	int numThreads = min((uint64_t)(config.generalCfg.endSeed - config.generalCfg.seedStart), freeMem / memPerThread);
	int numBlocks = numThreads / BLOCKSIZE;
	NumBlocks = max(min(MAXBLOCKS, numBlocks - numBlocks % 1), 1);
	config.generalCfg.seedBlockSize = min((uint32_t)config.generalCfg.seedBlockSize, (config.generalCfg.endSeed - config.generalCfg.seedStart) / (NumBlocks * BLOCKSIZE) + 1);
	
	SetWorkerCount(NumBlocks);
	SetWorkerAppetite(BLOCKSIZE);
	SetTargetDispatchRate(20);
	printf("Creating %ix%i threads\n", NumBlocks, BLOCKSIZE);

	//Do the actual allocation.
	size_t totalMemory = GetMinimumSpanMemory() * NumBlocks * BLOCKSIZE;
	size_t outputSize = GetMinimumOutputMemory() * NumBlocks * BLOCKSIZE;

	//Allocate all of the memories
	checkCudaErrors(cudaHostAlloc(&hostPtrs.numActiveThreads, 4, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&hostPtrs.uOutput, outputSize, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc(&hostPtrs.hIO, KIO_size * NumBlocks, cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&computePtrs.numActiveThreads, (void*)hostPtrs.numActiveThreads, 0));
	checkCudaErrors(cudaMalloc((void**)&computePtrs.dArena, totalMemory));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&computePtrs.uOutput, (void*)hostPtrs.uOutput, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&computePtrs.uIO, (void*)hostPtrs.hIO, 0));

	//Initialize the ones that need initializing
	*hostPtrs.numActiveThreads = 0;
	memset(hostPtrs.hIO, 0, KIO_size * NumBlocks);

	//Generate coalmine overlay, which is entirely separate for some reason.
	uint8_t* dOverlayMem; //It's probably fine to forget this pointer since we can copy it back from the coalmine_overlay global.
	checkCudaErrors(cudaMalloc((void**)&dOverlayMem, 3 * 256 * 103));
	uint8_t* hPtr = (uint8_t*)malloc(3 * 256 * 103);
	ReadImage("wang_tiles/coalmine_overlay.png", hPtr);
	checkCudaErrors(cudaMemcpy(dOverlayMem, hPtr, 3 * 256 * 103, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(coalmine_overlay, &dOverlayMem, sizeof(void*), 0));
	free(hPtr);

	printf("Allocated %lliMB of host and %lliMB of device memory\n", (2 * outputSize + KIO_size * NumBlocks) / 1_MB, (totalMemory + outputSize + KIO_size * NumBlocks) / 1_MB);
}
void FreeComputeMemory()
{
	//TODO fix the fact that we leak literally everything, I don't want to write this function right now.
	//Device reset fixes all woes.
}

Worker CreateWorker()
{
	Worker w;
	w.memIdx = memIdxCtr++;
	checkCudaErrors(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
	return w;
}
void DestroyWorker(Worker& worker)
{
	if(worker.stream != NULL) checkCudaErrors(cudaStreamDestroy(worker.stream));
}

__global__ void DispatchBlock(ComputePointers dPointers, size_t arenaPitch, SearchConfig config, int memIdx, int BLOCKSIZE)
{
	//dAtomicAdd((int*)dPointers.numActiveThreads, 1);
	uint32_t hwIdx = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t* ioPtr = KIO_idx(dPointers.uIO, memIdx);
	uint8_t* threadMemBlock = dPointers.dArena + arenaPitch * (memIdx * BLOCKSIZE + hwIdx);
	uint8_t* outputPtr = dPointers.uOutput + config.memSizes.outputSize * (memIdx * BLOCKSIZE + hwIdx);
	SpanRet ret = EvaluateSpan(config, KIO_params(ioPtr)[hwIdx], threadMemBlock, outputPtr);
	memcpy(&KIO_ret(ioPtr)[hwIdx], &ret, sizeof(SpanRet));
	//dAtomicAdd((int*)dPointers.numActiveThreads, -1);
}

void DispatchJob(Worker& worker, SpanParams* spans)
{
	memcpy(KIO_params(KIO_idx(hostPtrs.hIO, worker.memIdx)), spans, sizeof(SpanParams) * BLOCKSIZE);
	cudaEventCreateWithFlags(&worker.event, cudaEventDisableTiming);
	DispatchBlock<<<BLOCKDIV, BLOCKSIZE/BLOCKDIV, 0, worker.stream>>>(computePtrs, GetMinimumSpanMemory(), GetSearchConfig(), worker.memIdx, BLOCKSIZE);
	cudaEventRecord(worker.event, worker.stream);
}
bool QueryWorker(Worker& worker)
{
	cudaError e = cudaEventQuery(worker.event);
	if (e == cudaSuccess) return true;
	else if (e == cudaErrorNotReady) return false;
	else checkCudaErrors(e);
}
SpanRet* SubmitJob(Worker& worker)
{
	checkCudaErrors(cudaStreamSynchronize(worker.stream));
	for (int i = 0; i < BLOCKSIZE; i++)
		KIO_ret(KIO_idx(hostPtrs.hIO, worker.memIdx))[i].outputPtr = hostPtrs.uOutput + (worker.memIdx * BLOCKSIZE + i) * GetMinimumOutputMemory();
	return KIO_ret(KIO_idx(hostPtrs.hIO, worker.memIdx));
}
void AbortJob(Worker& worker)
{
	checkCudaErrors(cudaStreamDestroy(worker.stream));
	worker.stream = NULL;
}

__global__ void buildTS(uint8_t* dTileData, uint8_t* dTileSet, int tiles_w, int tiles_h)
{
	MemoryArena arena = { dTileSet, 0 };
	stbhw_build_tileset_from_image(dTileData, arena, tiles_w * 3, tiles_w, tiles_h);
}

uint8_t* BuildTileset(uint8_t* data, int w, int h)
{
	uint64_t tileDataSize = 3 * w * h;

	uint8_t* dTileData;
	uint8_t* dTileSet;

	checkCudaErrors(cudaMalloc(&dTileData, tileDataSize));
	checkCudaErrors(cudaMalloc(&dTileSet, tileDataSize));
	checkCudaErrors(cudaMemcpy(dTileData, data, tileDataSize, cudaMemcpyHostToDevice));
	buildTS << <1, 1 >> > (dTileData, dTileSet, w, h);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dTileData));

	return dTileSet;
}