#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define rainCount 4

__device__ __constant__ const char* rainNames[] = {
	"water",
	"blood",
	"acid",
	"slime"
};

#define rainProbSum 0.0513f
__device__ __constant__ const float rainProbs[] = {
	0.05f,
	0.001f,
	0.0002f,
	0.0001f
};