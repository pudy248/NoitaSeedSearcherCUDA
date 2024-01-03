#pragma once
#include "../platforms/platform_implementation.h"
#include "../include/enums.h"
#include "../include/search_structs.h"
#include "../include/noita_random.h"

/*
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
},
{ //meat
	4,
	{{5, WAND_T5},
	{5, WAND_T6},
	{5, WAND_T4NS},
	{5, WAND_T5NS}}
} 
};*/

_compute bool wandCheck_fungicave(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r > 0.06;
}
_compute bool wandCheck_snowcave(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r < 0.45;
}
_compute bool wandCheck_snowcastle(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r > 0.2;
}
_compute bool wandCheck_rainforest(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r > 0.27;
}
_compute bool wandCheck_vault(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r < 0.93;
}
_compute bool wandCheck_crypt(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r > 0.38;
}
_compute bool wandCheck_sandcave(NollaPRNG& random, int x, int y)
{
	float r = random.ProceduralRandomf(x - 11.631, y + 10.2257, 0, 1);
	return r < 0.94;
}

_compute bool(*wandChecks[])(NollaPRNG&, int, int) = {
	wandCheck_fungicave,
	wandCheck_snowcave,
	wandCheck_snowcastle,
	wandCheck_rainforest,
	wandCheck_rainforest,
	wandCheck_rainforest,
	wandCheck_vault,
	wandCheck_crypt,
	wandCheck_crypt,
	wandCheck_vault,
	wandCheck_crypt,
	wandCheck_sandcave,
	wandCheck_fungicave
};