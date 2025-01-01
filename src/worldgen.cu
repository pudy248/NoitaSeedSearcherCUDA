#include "../platforms/platform_implementation.h"

#include "../include/worldgen_structs.h"
#include "../include/noita_random.h"
#include "../include/compute.h"
#include "../include/misc_funcs.h"


#define BIOME_PATH_FIND_WORLD_POS_MIN_X 159
#define BIOME_PATH_FIND_WORLD_POS_MAX_X 223
#define WORLD_OFFSET_X 35

_universal static WorldgenPRNG GetRNG(uint32_t world_seed, int map_w)
{
	WorldgenPRNG rng = WorldgenPRNG(world_seed);

	int iters = map_w + world_seed + 11 * (map_w / -11) - 12 * (world_seed / 12);

	if (iters > 0)
	{
		do
		{
			rng.Next();
			iters -= 1;
		} while (iters != 0);
	}
	return rng;
}
_universal static uint32_t getPos(const uint32_t w, const uint32_t s, const uint32_t x, const uint32_t y)
{
	return s * (w * y + x);
}
_universal static uint32_t getPixelColor(const uint8_t* map, uint32_t pos)
{
	return createRGB(map[pos], map[pos + 1], map[pos + 2]);
}
_universal static uint32_t getPixelColor(const uint8_t* map, const uint32_t w, const uint32_t x, const uint32_t y)
{
	uint32_t pos = getPos(w, 3, x, y);
	return getPixelColor(map, pos);
}
_universal static void setPixelColor(uint8_t* map, uint32_t pos, uint32_t color)
{
	uint8_t r = ((color >> 16) & 0xff);
	uint8_t g = ((color >> 8) & 0xff);
	uint8_t b = ((color) & 0xff);
	map[pos] = r;
	map[pos + 1] = g;
	map[pos + 2] = b;
}
_universal static void setPixelColor(uint8_t* map, uint32_t w, uint32_t x, uint32_t y, uint32_t color)
{
	uint32_t pos = getPos(w, 3, x, y);
	setPixelColor(map, pos, color);
}
_universal static void fill(uint8_t* map, int w, int x1, int x2, int y1, int y2, uint32_t color)
{
	for (int x = x1; x <= x2; x++)
	{
		for (int y = y1; y <= y2; y++)
		{
			setPixelColor(map, w, x, y, color);
		}
	}
}
_compute GeneratedBiome GenerateMap(uint32_t worldSeed, const BiomeWangScope& scope, MemSpan output, MemSpan res, MemSpan visited, MemSpan miscMem)
{
#ifdef SEEDS_AS_TRIES
	int MAX_TRIES = worldSeed;
	worldSeed = SEEDS_AS_TRIES;
#else
	constexpr int MAX_TRIES = 100;
#endif

	WorldgenPRNG rng = GetRNG(worldSeed, scope.bSec.map_w);
	//if (scope.bSec.isNightmare) rng.Next();

	WangTileIndex* idxs = (WangTileIndex*)res.ptr;
	GeneratedBiome b = { scope, idxs, 0 };

	int tries = 0;
	while (tries < MAX_TRIES)
	{
		tries++;
		WorldgenPRNG rng2 = WorldgenPRNG(rng.NextU());
		stbhw_generate_image(idxs, scope, scope.bSec.map_w, scope.bSec.map_h + 4, rng2);

		if (0) {
			printf("SEED %i, TRY %i\n", worldSeed, tries);
			for (int y = 0; y < scope.bSec.wang_h; y++) {
				for (int x = 0; x < scope.bSec.wang_w; x++) {
					WangTileIndex tile = idxs[y * scope.bSec.wang_w + x];
					printf(tile & 0x8000 ? "%s-- " : "%s%02i ", tile & 0x4000 ? "V" : "H", tile & 0x3fff);
				}
				printf("\n");
			}
			printf("\n");
		}
#ifndef SEEDS_AS_TRIES
		if (isValid(b, miscMem, visited))
			break;
#else
		isValid(b, miscMem, visited);
#endif
	}
	//printf("%i: %i\n", worldSeed, tries);
	//if (!has_path) memset(map, 0, 3 * scope.bSec.map_w * scope.bSec.map_h);

#ifdef IMAGE_OUTPUT
	if (!output.is_safe(3 * scope.bSec.map_w * scope.bSec.map_h + 11))
		printf("GenerateMap(): Ran out of image output space.\n");
	memcpy(output.ptr + 4, &scope.bSec.map_w, 4);
	memcpy(output.ptr + 8, &scope.bSec.map_h, 4);
	uint8_t* img = output.ptr + 12;
	for (int y = 0; y < scope.bSec.map_h; y++) {
		for (int x = 0; x < scope.bSec.map_w; x++) {
			uint32_t c = get_pixel<true, false>(b, {}, x, y, scope.ts.short_side_len);
			if (visited.ptr[y * scope.bSec.map_w + x] == 2 && !false)
				c = 0xff00ffU;
			int i = (y * scope.bSec.map_w + x) * 3;
			img[i + 0] = (c >> 16) & 0xff;
			img[i + 1] = (c >> 8) & 0xff;
			img[i + 2] = (c >> 0) & 0xff;
		}
	}
#endif

	return b;
}
