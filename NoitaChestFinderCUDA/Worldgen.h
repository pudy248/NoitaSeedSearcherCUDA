 #pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "misc/datatypes.h"
#include "misc/noita_random.h"
#include "misc/stb_hbwang.h"
#include "misc/worldgen_helpers.h"
#include "misc/pathfinding.h"

struct WorldgenConfig
{
	uint tiles_w;
	uint tiles_h;
	uint map_w;
	uint map_h;
	int worldX;
	int worldY;
	bool isCoalMine;
	bool isNightmare;

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

__device__ byte* GenerateMap(uint worldSeed, byte* output, byte* res, byte* visited, byte* miscMem, WorldgenConfig c, int idx)
{
	WorldgenPRNG rng = GetRNG(worldSeed, c.map_w);
	if (c.isNightmare) rng.Next();
	int tries = 0;
	bool has_path = false;

	byte* map = res + 4 * 3 * c.map_w;
	while (!has_path)
	{
		if (tries >= c.maxTries - 1 + worldSeed) break;
		WorldgenPRNG rng2 = WorldgenPRNG(rng.NextU());

		stbhw_generate_image(res, c.map_w * 3, c.map_w, c.map_h + 4, StaticRandom, &rng2);

		blockOutRooms(map, c.map_w, c.map_h, COLOR_WHITE);
		if (c.isCoalMine)
		{
			doCoalMineHax(map, c.map_w, c.map_h);
		}

		has_path = isValid(map, miscMem, visited, c.map_w, c.map_h, c.worldX, c.worldY, c.isCoalMine);
		tries++;
	}
	//if (!has_path) memset(map, 0, 3 * c.map_w * c.map_h);

	/*constexpr auto center_x = 400;
	constexpr auto center_y = 200;

	if(threadIdx.x == 0 && blockIdx.x == 0) printf("%i %i\n", GetGlobalPos(c.worldX, c.worldY, center_x * 10, center_y * 10).x, GetGlobalPos(c.worldX, c.worldY, center_x * 10, center_y * 10).y);

	setPixelColor(map, c.map_w, center_x, center_y - 1, 0x0000FF);
	setPixelColor(map, c.map_w, center_x - 1, center_y, 0x0000FF);
	setPixelColor(map, c.map_w, center_x, center_y, 0x0000FF);
	setPixelColor(map, c.map_w, center_x + 1, center_y, 0x0000FF);
	setPixelColor(map, c.map_w, center_x, center_y + 1, 0x0000FF);

	memcpy(output, map, 3 * c.map_w * c.map_h);*/
}
