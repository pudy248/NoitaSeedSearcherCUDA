#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/enums.h"

constexpr int rainCount = 4;

__device__ const Material rainMaterials[] = {
	Material::WATER,
	Material::BLOOD,
	Material::ACID,
	Material::SLIME,
};

constexpr float rainProbSum = 0.0513f;
__device__ const float rainProbs[] = {
	0.05f,
	0.001f,
	0.0002f,
	0.0001f,
};