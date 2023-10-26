#pragma once
#include "platform_compute_helpers.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

_compute void dAtomicAdd(int* ptr, int val)
{
	atomicAdd(ptr, val);
}
#define __cMemoryGranularity uint32_t
_compute void cMemcpy(void* dest, void* source, size_t size)
{
	size_t sizeDiv = size / sizeof(__cMemoryGranularity);
	for (int i = 0; i < sizeDiv; i++) ((__cMemoryGranularity*)dest)[i] = ((__cMemoryGranularity*)source)[i];
	for(int i = sizeDiv * sizeof(__cMemoryGranularity); i < size; i++) ((uint8_t*)dest)[i] = ((uint8_t*)source)[i];
	//memcpy(dest, source, size);
	//cudaMemcpyAsync(dest, source, size, cudaMemcpyDeviceToDevice);
};
_compute void cMemset(void* dest, int val, size_t size)
{
	size_t sizeDiv = size / sizeof(__cMemoryGranularity);
	for (int i = 0; i < sizeDiv; i++) ((__cMemoryGranularity*)dest)[i] = val;
	for (int i = sizeDiv * sizeof(__cMemoryGranularity); i < size; i++) ((uint8_t*)dest)[i] = val;
	//memset(dest, val, size);
	//cudaMemsetAsync(dest, val, size);
};
_compute void threadSync()
{
	__syncthreads();
}