 #pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"

#include "misc/noita_random.h"
#include "misc/stb_hbwang.h"
#include "misc/worldgen_helpers.h"
#include "misc/pathfinding.h"

#include "Configuration.h"

__global__ void buildTS(byte* dTileData, byte* dTileSet, int tiles_w, int tiles_h)
{
	MemoryArena arena = { dTileSet, 0 };
	stbhw_build_tileset_from_image(dTileData, arena, tiles_w * 3, tiles_w, tiles_h);
}

__device__ byte* GenerateMap(uint worldSeed, byte* tileSet, byte* output, byte* res, byte* visited, byte* miscMem, MapConfig c, int idx)
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

		stbhw_generate_image(res, (stbhw_tileset*)tileSet, c.map_w * 3, c.map_w, c.map_h + 4, StaticRandom, &rng2);

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
