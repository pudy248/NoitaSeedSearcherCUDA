#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char* errorMessage, const char* file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
	}
}
