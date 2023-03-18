#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define biomeModifierCount 22

enum Biome : byte {
	B_NONE,
	B_COALMINE,
	B_COALMINE_ALT,
	B_EXCAVATIONSITE,
	B_FUNGICAVE,
	B_SNOWCAVE,
	B_SNOWCASTLE,
	B_RAINFOREST,
	B_VAULT,
	B_CRYPT,
};

static const char* biomeNames[] = {
	"NONE",
	"COALMINE",
	"COALMINE_ALT",
	"EXCAVATIONSITE",
	"FUNGICAVE",
	"SNOWCAVE",
	"SNOWCASTLE",
	"RAINFOREST",
	"VAULT",
	"CRYPT"
};


enum BiomeModifier : byte {
	BM_NONE,
	BM_MOIST,
	BM_FOG_OF_WAR_REAPPEARS,
	BM_HIGH_GRAVITY,
	BM_LOW_GRAVITY,
	BM_CONDUCTIVE,
	BM_HOT,
	BM_GOLD_VEIN,
	BM_GOLD_VEIN_SUPER,
	BM_PLANT_INFESTED,
	BM_FURNISHED,
	BM_BOOBY_TRAPPED,
	BM_PERFORATED,
	BM_SPOOKY,
	BM_GRAVITY_FIELDS,
	BM_FUNGAL,
	BM_FLOODED,
	BM_GAS_FLOODED,
	BM_SHIELDED,
	BM_PROTECTION_FIELDS,
	BM_OMINOUS,
	BM_INVISIBILITY,
	BM_WORMY
};

static const char* biomeModifierNames[] = {
	"NONE",
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

struct BiomeBlacklist {
	Biome blacklist[5];
};