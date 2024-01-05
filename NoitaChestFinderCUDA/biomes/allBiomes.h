#pragma once
#include "../platforms/platform_compute_helpers.h"

#include "../biomes/coalmine.h"
//#include "../biomes/coalmine_alt.h"
//todo excavationsite
//#include "../biomes/liquidcave.h"

_compute void SetSpawnFuncsFromGlobals()
{
	AllSpawnFunctions[B_COALMINE] = &DAT_COALMINE;
	AllWandLevels[B_COALMINE] = &FUNCS_COALMINE::wandLevels;


}