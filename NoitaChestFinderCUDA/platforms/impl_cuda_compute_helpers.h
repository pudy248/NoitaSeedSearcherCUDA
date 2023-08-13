#pragma once
#include "platform_compute_helpers.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

_compute void dAtomicAdd(int* ptr, int val)
{
	atomicAdd(ptr, val);
}
_compute void cMemcpy(void* dest, void* source, size_t size)
{
	memcpy(dest, source, size);
	//cudaMemcpyAsync(dest, source, size, cudaMemcpyDeviceToDevice);
};
_compute void threadSync()
{
	__syncthreads();
}