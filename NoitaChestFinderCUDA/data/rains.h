#pragma once
#include "../platforms/platform_compute_helpers.h"

#include "../structs/enums.h"

_data constexpr int rainCount = 4;
_data const Material rainMaterials[rainCount] = {
	Material::WATER,
	Material::BLOOD,
	Material::ACID,
	Material::SLIME,
};

_data constexpr float rainProbSum = 0.0513f;
_data const float rainProbs[rainCount] = {
	0.05f,
	0.001f,
	0.0002f,
	0.0001f,
};