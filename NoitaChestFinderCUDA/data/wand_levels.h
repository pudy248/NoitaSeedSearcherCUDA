#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef unsigned char byte;

struct wandLevel {
	float prob;
	byte id;
};
struct BiomeWands {
	int count;
	wandLevel levels[6];
};

__device__ const static BiomeWands wandLevels[] = {
{ //coalmine
	2,
	{{17, 71},
	{1.9, 32}}
},
{ //coalmine_alt
	2,
	{{17, 71},
	{1.9, 32}}
},
{ //excavationsite
	3,
	{{2, 33},
	{2, 34},
	{2, 66}}
},
{ //fungicave
	2,
	{{5, 35},
	{5, 33}}
},
{ //snowcave
	3,
	{{5, 34},
	{5, 66},
	{5, 35}}
},
{ //snowcastle
	3,
	{{5, 36},
	{5, 67},
	{5, 37}}
},
{ //rainforest
	5,
	{{5, 38},
	{3, 40},
	{3, 35},
	{3, 37},
	{3, 68}}
},
{ //rainforest_open
	5,
	{{5, 38},
	{3, 40},
	{3, 35},
	{3, 37},
	{3, 68}}
},
{ //rainforest_dark
	5,
	{{5, 38},
	{3, 40},
	{3, 37},
	{3, 39},
	{5, 69}}
},
{ //vault
	4,
	{{5, 40},
	{5, 69},
	{3, 37},
	{2, 39}}
},
{ //crypt
	4,
	{{5, 42},
	{5, 70},
	{3, 41},
	{2, 43}}
},
{ //wandcave
	0
},
{ //vault_frozen
	3,
	{{5, 40},
	{3, 37},
	{2, 39}}
},
{ //wizardcave
	4,
	{{5, 42},
	{5, 70},
	{3, 41},
	{2, 43}}
},
{ //sandcave
	2,
	{{5, 38},
	{5, 35}}
},
{ //fungiforest
	3,
	{{5, 37},
	{5, 39},
	{5, 69}}
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
	{{5, 40},
	{5, 69},
	{3, 37},
	{2, 39}}
}};