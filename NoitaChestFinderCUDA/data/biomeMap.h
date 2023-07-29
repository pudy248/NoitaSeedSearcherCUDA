#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/enums.h"

__device__ const Biome biomeMap[70 * 48] = { B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_THE_SKY,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_PYRAMID,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_PYRAMID,
		B_PYRAMID,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_LIQUIDCAVE,
		B_LIQUIDCAVE,
		B_LIQUIDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_COALMINE,
		B_COALMINE,
		B_COALMINE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_PYRAMID,
		B_PYRAMID,
		B_PYRAMID,
		B_PYRAMID,
		B_PYRAMID,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_LIQUIDCAVE,
		B_LIQUIDCAVE,
		B_LIQUIDCAVE,
		B_LIQUIDCAVE,
		B_LIQUIDCAVE,
		B_NONE,
		B_COALMINE_ALT,
		B_COALMINE_ALT,
		B_COALMINE,
		B_COALMINE,
		B_COALMINE,
		B_COALMINE,
		B_COALMINE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGICAVE,
		B_FUNGICAVE,
		B_FUNGICAVE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_FUNGIFOREST,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGICAVE,
		B_FUNGICAVE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_EXCAVATIONSITE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_NONE,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_VAULT_FROZEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_SNOWCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_9,
		B_SOLID_WALL_TOWER_9,
		B_SOLID_WALL_TOWER_9,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_SANDCAVE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_8,
		B_SOLID_WALL_TOWER_8,
		B_SOLID_WALL_TOWER_8,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_FUNGIFOREST,
		B_SANDCAVE,
		B_FUNGIFOREST,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_SNOWCASTLE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_7,
		B_SOLID_WALL_TOWER_7,
		B_SOLID_WALL_TOWER_7,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_WIZARDCAVE,
		B_RAINFOREST_DARK,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_6,
		B_SOLID_WALL_TOWER_6,
		B_SOLID_WALL_TOWER_6,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_RAINFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_5,
		B_SOLID_WALL_TOWER_5,
		B_SOLID_WALL_TOWER_5,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_OPEN,
		B_RAINFOREST,
		B_RAINFOREST_OPEN,
		B_RAINFOREST,
		B_FUNGICAVE,
		B_RAINFOREST,
		B_RAINFOREST_OPEN,
		B_RAINFOREST,
		B_RAINFOREST_OPEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_4,
		B_SOLID_WALL_TOWER_4,
		B_SOLID_WALL_TOWER_4,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_NONE,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_RAINFOREST_OPEN,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_3,
		B_SOLID_WALL_TOWER_3,
		B_SOLID_WALL_TOWER_3,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_SANDCAVE,
		B_SANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_2,
		B_SOLID_WALL_TOWER_2,
		B_SOLID_WALL_TOWER_2,
		B_NONE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_FUNGICAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_SOLID_WALL_TOWER_1,
		B_SOLID_WALL_TOWER_1,
		B_SOLID_WALL_TOWER_1,
		B_NONE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_RAINFOREST_DARK,
		B_NONE,
		B_NONE,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_RAINFOREST_DARK,
		B_NONE,
		B_NONE,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_VAULT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_CRYPT,
		B_NONE,
		B_NONE,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_NONE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_ROBOBASE,
		B_ROBOBASE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_CRYPT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_WANDCAVE,
		B_NONE,
		B_WIZARDCAVE,
		B_FUNGIFOREST,
		B_FUNGIFOREST,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_FUNGIFOREST,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_WIZARDCAVE,
		B_NONE,
		B_WIZARDCAVE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_MEAT,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_THE_END,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_ROBOBASE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_THE_END,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE,
		B_NONE, 
};