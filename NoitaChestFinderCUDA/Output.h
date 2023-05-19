#pragma once

#include "misc/datatypes.h"
#include "data/items.h"
#include "data/spells.h"

#include "WorldgenSearch.h"
#include "misc/wandgen.h"

struct OutputBlock
{
	int seed;
	int sCount;
	Spawnable* spawnableRefs;
	byte data;
};

__device__ OutputBlock* PackOutputData()
{

}