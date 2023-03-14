#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define biomeModifierCount 22

enum BiomeModifier : byte {
	MODIFIER_NONE,
	MOIST,
	FOG_OF_WAR_REAPPEARS,
	HIGH_GRAVITY,
	LOW_GRAVITY,
	CONDUCTIVE,
	HOT,
	GOLD_VEIN,
	GOLD_VEIN_SUPER,
	PLANT_INFESTED,
	FURNISHED,
	BOOBY_TRAPPED,
	PERFORATED,
	SPOOKY,
	GRAVITY_FIELDS,
	FUNGAL,
	FLOODED,
	GAS_FLOODED,
	SHIELDED,
	PROTECTION_FIELDS,
	OMINOUS,
	INVISIBILITY,
	WORMY
};

__device__ __constant__ const char* biomeModifierNames[] = {
	"MOIST", 
	"FOG_OF_WAR_REAPPEARS", 
	"HIGH_GRAVITY", 
	"LOW_GRAVITY", 
	"CONDUCTIVE", 
	"HOT", 
	"GOLD_VEIN", 
	"GOLD_VEIN_SUPER", 
	"PLANT_INFESTED", 
	"FURNISHED", 
	"BOOBY_TRAPPED", 
	"PERFORATED", 
	"SPOOKY", 
	"GRAVITY_FIELDS", 
	"FUNGAL", 
	"FLOODED", 
	"GAS_FLOODED", 
	"SHIELDED", 
	"PROTECTION_FIELDS", 
	"OMINOUS", 
	"INVISIBILITY", 
	"WORMY"
};

#define biomeModifierProbSum 9.71025f
__device__ __constant__ const float biomeModifierProbs[] = {
	0.7f,
	1,
	0.5f,
	0.5f,
	0.2f,
	0.6f,
	0.01f,
	0.00025f,
	1,
	0.5f,
	0.75f,
	0.75f,
	0.5f,
	0.3f,
	0.5f,
	0.75f,
	0.5f,
	0.1f,
	0.2f,
	0.2f,
	0.1f,
	0.05f
};