#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/datatypes.h"
#include "misc/noita_random.h"
#include "misc/stb_hbwang.h"
#include "misc/worldgen_helpers.h"
#include "misc/pathfinding.h"

struct WorldgenConfig {
	uint tiles_w;
	uint tiles_h;
	uint map_w;
	uint map_h;
	int worldX;
	int worldY;
	bool isCoalMine;

	int maxTries;
};

__global__ void buildTS(byte* data, int tiles_w, int tiles_h)
{
	stbhw_build_tileset_from_image(data, tiles_w * 3, tiles_w, tiles_h);
}
__global__ void freeTS()
{
	stbhw_free_tileset();
}

__device__ byte* GenerateMap(uint worldSeed, byte* output, byte* res, byte* visited, byte* miscMem, WorldgenConfig c, int idx) {
	NollaPrng rng = GetRNG(worldSeed, c.map_w);
	int tries = 0;
	bool has_path = false;

	byte* map = res + 4 * 3 * c.map_w;
	while (!has_path) {
		if (tries >= c.maxTries) break;
		NollaPrng rng2 = NollaPrng(rng.NextU());

		stbhw_generate_image(res, c.map_w * 3, c.map_w, c.map_h + 4, StaticRandom, &rng2);

		if (c.isCoalMine) {
			doCoalMineHax(map, c.map_w, c.map_h);
		}
		blockOutRooms(map, c.map_w, c.map_h, COLOR_WHITE);

		has_path = isValid(map, miscMem, visited, c.map_w, c.map_h, c.worldX, c.worldY, c.isCoalMine);
		tries++;
	}
	if (!has_path) memset(map, 0, 3 * c.map_w * c.map_h);
	//printf("idx %i: took %i tries, map is %s\n", idx, tries, has_path ? "valid" : "invalid");
	//memcpy(output, map, 3 * c.map_w * c.map_h);
}
