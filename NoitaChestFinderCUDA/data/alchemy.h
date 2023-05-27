#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/enums.h"

constexpr int alchemyLiquidCount = 30;
__device__ const Material alchemyLiquids[] = {
	Material::ACID,
	Material::ALCOHOL,
	Material::BLOOD,
	Material::BLOOD_FUNGI,
	Material::BLOOD_WORM,
	Material::CEMENT,
	Material::LAVA,
	Material::MAGIC_LIQUID_BERSERK,
	Material::MAGIC_LIQUID_CHARM,
	Material::MAGIC_LIQUID_FASTER_LEVITATION,
	Material::MAGIC_LIQUID_FASTER_LEVITATION_AND_MOVEMENT,
	Material::MAGIC_LIQUID_INVISIBILITY,
	Material::MAGIC_LIQUID_MANA_REGENERATION,
	Material::MAGIC_LIQUID_MOVEMENT_FASTER,
	Material::MAGIC_LIQUID_PROTECTION_ALL,
	Material::MAGIC_LIQUID_TELEPORTATION,
	Material::MAGIC_LIQUID_UNSTABLE_POLYMORPH,
	Material::MAGIC_LIQUID_UNSTABLE_TELEPORTATION,
	Material::MAGIC_LIQUID_WORM_ATTRACTOR,
	Material::MATERIAL_CONFUSION,
	Material::MUD,
	Material::OIL,
	Material::POISON,
	Material::RADIOACTIVE_LIQUID,
	Material::SWAMP,
	Material::URINE,
	Material::WATER,
	Material::WATER_ICE,
	Material::WATER_SWAMP,
	Material::MAGIC_LIQUID_RANDOM_POLYMORPH
};

constexpr int alchemySolidCount = 18;
__device__ const Material alchemySolids[] = {
	Material::BONE,
	Material::BRASS,
	Material::COAL,
	Material::COPPER,
	Material::DIAMOND,
	Material::FUNGI,
	Material::GOLD,
	Material::GRASS,
	Material::GUNPOWDER,
	Material::GUNPOWDER_EXPLOSIVE,
	Material::ROTTEN_MEAT,
	Material::SAND,
	Material::SILVER,
	Material::SLIME,
	Material::SNOW,
	Material::SOIL,
	Material::WAX,
	Material::HONEY
};