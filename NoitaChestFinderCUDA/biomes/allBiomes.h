#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../biomes/coalmine.h"
#include "../biomes/coalmine_alt.h"
//todo excavationsite
#include "../biomes/liquidcave.h"

__device__ void SetFunctionPointerSetterFunctionPointerArrayPointers()
{
	BiomeFnPtrs[B_COALMINE] = FUNCS_COALMINE::SetFunctionPointers;
	BiomeFnPtrs[B_COALMINE_ALT] = FUNCS_COALMINE_ALT::SetFunctionPointers;

	BiomeFnPtrs[B_LIQUIDCAVE] = FUNCS_LIQUIDCAVE::SetFunctionPointers;
}

__host__ void SetBiomeData(BiomeData* ptr)
{
	ptr[B_COALMINE] = DAT_COALMINE;
	ptr[B_COALMINE_ALT] = DAT_COALMINE_ALT;

	ptr[B_LIQUIDCAVE] = DAT_LIQUIDCAVE;
}