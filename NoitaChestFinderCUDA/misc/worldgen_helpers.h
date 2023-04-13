#pragma once

#include "../data/coalhax.h"
#include "noita_random.h"
#include "stb_hbwang.h"
#include "datatypes.h"

constexpr uint COLOR_PURPLE = 0x7f007fU;
constexpr uint COLOR_BLACK = 0x000000U;
constexpr uint COLOR_WHITE = 0xffffffU;
constexpr uint COLOR_YELLOW = 0xffff00U;
constexpr uint COLOR_COFFEE = 0xc0ffeeU;
constexpr uint COLOR_FROZEN_VAULT_MINT = 0xcff7c8;
constexpr uint COLOR_HELL_GREEN = 0x8aff80;

#define BIOME_PATH_FIND_WORLD_POS_MIN_X 159
#define BIOME_PATH_FIND_WORLD_POS_MAX_X 223
#define WORLD_OFFSET_X 35

#define BCSize 9
__device__ __constant__ uint blockedColors[BCSize] = {
	0x00ac6eU, //load_pixel_scene4_alt
	0x70d79eU, //load_gunpowderpool_01
	0x70d79fU, //???
	0x70d7a1U, //load_gunpowderpool_04
	0x7868ffU, //load_gunpowderpool_02
	0xc35700U, //load_oiltank
	0xff0080U, //load_pixel_scene2
	0xff00ffU, //???
	0xff0affU, //load_pixel_scene
};

__device__ bool contains(const uint arr[BCSize], uint val)
{
	for (int i = 0; i < BCSize; i++)
		if (arr[i] == val) return true;
	return false;
};
__device__ NollaPrng GetRNG(uint world_seed, int map_w)
{
	NollaPrng rng = NollaPrng();
	rng.SetRandomFromWorldSeed(world_seed);
	rng.Next();

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

__device__ uint getPos(const uint w, const uint s, const uint x, const uint y)
{
	return s * (w * y + x);
}
__device__ void doCoalMineHax(byte* map, uint map_w, uint map_h)
{
	for (int y = 0; y < map_h; y++)
	{
		for (int x = 0; x < map_w; x++)
		{
			uint overlayPos = getPos(256, 3, x, y);
			uint i = getPos(map_w, 3, x, y);
			uint pix = createRGB(coalmine_overlay[overlayPos], coalmine_overlay[overlayPos + 1], coalmine_overlay[overlayPos + 2]);
			if (pix == 0x4000)
			{
				// pudy248 note: is not actually air, this is the main rock portion of the overlay
				map[i] = 0xFF;
				map[i + 1] = 0xFF;
				map[i + 2] = 0xFF;
			}
			if (pix == 0x0040)
			{ // blue. Looks like air?
				map[i] = 0x00;
				map[i + 1] = 0x00;
				map[i + 2] = 0x00;
			}
			if (pix == 0xFEFEFE)
			{ // white. Stairs. rock_static_intro
			   // In the debug it's not shown, but used in path finding.
				map[i] = 0x0a;
				map[i + 1] = 0x33;
				map[i + 2] = 0x44;
			}
		}
	}
}

__device__ uint getPixelColor(const byte* map, uint pos)
{
	byte r = map[pos];
	byte g = map[pos + 1];
	byte b = map[pos + 2];
	return createRGB(r, g, b);
}
__device__ uint getPixelColor(const byte* map, const uint w, const uint x, const uint y)
{
	uint pos = getPos(w, 3, x, y);
	return getPixelColor(map, pos);
}
__device__ void setPixelColor(byte* map, uint pos, uint color)
{
	byte r = ((color >> 16) & 0xff);
	byte g = ((color >> 8) & 0xff);
	byte b = ((color) & 0xff);
	map[pos] = r;
	map[pos + 1] = g;
	map[pos + 2] = b;
}
__device__ void setPixelColor(byte* map, uint w, uint x, uint y, uint color)
{
	uint pos = getPos(w, 3, x, y);
	setPixelColor(map, pos, color);
}

__device__ void fill(byte* map, int w, int x1, int x2, int y1, int y2, uint color)
{
	for (int x = x1; x <= x2; x++)
	{
		for (int y = y1; y <= y2; y++)
		{
			setPixelColor(map, w, x, y, color);
		}
	}
}

__device__ void blockOutRooms(byte* map, uint map_w, uint map_h, uint targetColor)
{
	uint posMax = map_w * map_h;

	for (uint pos = 0; pos < posMax; pos++)
	{
		uint color = getPixelColor(map, pos * 3);
		// ~70% of pixels are black or white, so skip them
		if (color == COLOR_BLACK)
		{
			continue;
		}
		if (color == COLOR_WHITE)
		{
			continue;
		}
		if (!contains(blockedColors, color))
		{
			continue;
		}
		int x = pos % map_w;
		int y = pos / map_w;

		int startX = x + 1;
		int endX = x + 1;
		int startY = y + 1;
		int endY = y + 1;
		bool foundEnd = false;
		while (!foundEnd && endX < map_w)
		{
			uint c = getPixelColor(map, map_w, endX, startY);
			if (c == COLOR_BLACK)
			{
				endX += 1;
				continue;
			};
			endX -= 1;
			foundEnd = true;
		}
		if (endX >= map_w)
		{
			endX = map_w - 1;
		}
		foundEnd = false;
		while (!foundEnd && endY < map_h)
		{
			uint c = getPixelColor(map, map_w, startX, endY);
			if (c == COLOR_BLACK)
			{
				endY += 1;
				continue;
			};
			endY -= 1;
			foundEnd = true;
		}
		if (endY >= map_h)
		{
			endY = map_h - 1;
		}
		fill(map, map_w, startX, endX, startY, endY, targetColor);
	}
}

__device__ bool isMainPath(uint map_w, int worldX)
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	return fill_x_to > 0 && fill_x_from > 0 && map_w > fill_x_from && fill_x_to < map_w + fill_x_from;
}
__device__ int fillMainPath(byte* map, uint map_w, int worldX)
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	fill(map, map_w, fill_x_from, fill_x_to, 0, 6, COLOR_BLACK);
	return fill_x_from;
}