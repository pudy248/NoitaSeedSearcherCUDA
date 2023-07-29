 #pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"

#include "misc/noita_random.h"
#include "misc/stb_hbwang.h"
#include "misc/worldgen_helpers.h"
#include "misc/pathfinding.h"

#include "Configuration.h"

struct BiomeWangScope
{
	uint8_t* tileSet;
	BiomeSector bSec;
};

__global__ void buildTS(uint8_t* dTileData, uint8_t* dTileSet, int tiles_w, int tiles_h)
{
	MemoryArena arena = { dTileSet, 0 };
	stbhw_build_tileset_from_image(dTileData, arena, tiles_w * 3, tiles_w, tiles_h);
}

__device__ uint8_t* GenerateMap(uint32_t worldSeed, BiomeWangScope scope, uint8_t* output, uint8_t* res, uint8_t* visited, uint8_t* miscMem)
{
	constexpr int MAX_TRIES = 100;

	WorldgenPRNG rng = GetRNG(worldSeed, scope.bSec.map_w);
	//if (scope.bSec.isNightmare) rng.Next();
	int tries = 0;
	bool has_path = false;

	uint8_t* map = res + 4 * 3 * scope.bSec.map_w;
	while (!has_path)
	{
		if (tries >= MAX_TRIES) break;
		WorldgenPRNG rng2 = WorldgenPRNG(rng.NextU());

		stbhw_generate_image(res, (stbhw_tileset*)scope.tileSet, scope.bSec.map_w * 3, scope.bSec.map_w, scope.bSec.map_h + 4, StaticRandom, &rng2);

		if (scope.bSec.b == B_COALMINE)
		{
			doCoalMineHax(map, scope.bSec.map_w, scope.bSec.map_h);
		}

		has_path = isValid(map, miscMem, visited, scope.bSec.map_w, scope.bSec.map_h, scope.bSec.worldX, scope.bSec.worldY, scope.bSec.b == B_COALMINE);
		tries++;
	}
	//if (!has_path) memset(map, 0, 3 * scope.bSec.map_w * scope.bSec.map_h);

#ifdef IMAGE_OUTPUT
	memcpy(output + 4, &scope.bSec.map_w, 4);
	memcpy(output + 8, &scope.bSec.map_h, 4);
	memcpy(output + 12, map, 3 * scope.bSec.map_w * scope.bSec.map_h);
#endif
}
