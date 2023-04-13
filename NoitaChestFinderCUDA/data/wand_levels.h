#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../misc/datatypes.h"

struct wandLevel {
	float prob;
	Item id;
};
struct BiomeWands {
	int count;
	wandLevel levels[6];
};

__device__ __constant__ const static BiomeWands wandLevels[] = {
{ //coalmine
	2,
	{{17, UNKNOWN_WAND},
	{1.9, WAND_T1}}
},
{ //coalmine_alt
	2,
	{{17, UNKNOWN_WAND},
	{1.9, WAND_T1}}
},
{ //excavationsite
	3,
	{{2, WAND_T1NS},
	{2, WAND_T2},
	{2, WAND_T2B}}
},
{ //fungicave
	2,
	{{5, WAND_T2NS},
	{5, WAND_T1NS}}
},
{ //snowcave
	3,
	{{5, WAND_T2},
	{5, WAND_T2B},
	{5, WAND_T2NS}}
},
{ //snowcastle
	3,
	{{5, WAND_T3},
	{5, WAND_T3B},
	{5, WAND_T3NS}}
},
{ //rainforest
	5,
	{{5, WAND_T4},
	{3, WAND_T5},
	{3, WAND_T2NS},
	{3, WAND_T3NS},
	{3, WAND_T4B}}
},
{ //rainforest_open
	5,
	{{5, WAND_T4},
	{3, WAND_T5},
	{3, WAND_T2NS},
	{3, WAND_T3NS},
	{3, WAND_T4B}}
},
{ //rainforest_dark
	5,
	{{5, WAND_T4},
	{3, WAND_T5},
	{3, WAND_T3NS},
	{3, WAND_T4NS},
	{5, WAND_T5B}}
},
{ //vault
	4,
	{{5, WAND_T5},
	{5, WAND_T5B},
	{3, WAND_T3NS},
	{2, WAND_T4NS}}
},
{ //crypt
	4,
	{{5, WAND_T6},
	{5, WAND_T6B},
	{3, WAND_T5NS},
	{2, WAND_T6NS}}
},
{ //wandcave
	0
},
{ //vault_frozen
	3,
	{{5, WAND_T5},
	{3, WAND_T3NS},
	{2, WAND_T4NS}}
},
{ //wizardcave
	4,
	{{5, WAND_T6},
	{5, WAND_T6B},
	{3, WAND_T5NS},
	{2, WAND_T6NS}}
},
{ //sandcave
	2,
	{{5, WAND_T4},
	{5, WAND_T2NS}}
},
{ //fungiforest
	3,
	{{5, WAND_T3NS},
	{5, WAND_T4NS},
	{5, WAND_T5B}}
},
{ //solid_wall_tower_1
	0
},
{ //solid_wall_tower_2
	0
},
{ //solid_wall_tower_3
	0
},
{ //solid_wall_tower_4
	0
},
{ //solid_wall_tower_5
	0
},
{ //solid_wall_tower_6
	0
},
{ //solid_wall_tower_7
	0
},
{ //solid_wall_tower_8
	0
},
{ //solid_wall_tower_9
	0
},
{ //robobase
	4,
	{{5, WAND_T5},
	{5, WAND_T5B},
	{3, WAND_T3NS},
	{2, WAND_T4NS}}
}};