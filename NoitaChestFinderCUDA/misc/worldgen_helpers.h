#pragma once

#include "../structs/primitives.h"

#include "noita_random.h"
#include "stb_hbwang.h"

__device__ uint8_t* coalmine_overlay;

constexpr uint32_t COLOR_PURPLE = 0x7f007fU;
constexpr uint32_t COLOR_BLACK = 0x000000U;
constexpr uint32_t COLOR_WHITE = 0xffffffU;
constexpr uint32_t COLOR_YELLOW = 0xffff00U;
constexpr uint32_t COLOR_COFFEE = 0xc0ffeeU;
constexpr uint32_t COLOR_FROZEN_VAULT_MINT = 0xcff7c8;
constexpr uint32_t COLOR_HELL_GREEN = 0x8aff80;

#define BIOME_PATH_FIND_WORLD_POS_MIN_X 159
#define BIOME_PATH_FIND_WORLD_POS_MAX_X 223
#define WORLD_OFFSET_X 35

#define BCSize 9
const uint32_t blockedColors[BCSize] = {
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

__universal__ bool contains(const uint32_t arr[BCSize], uint32_t val)
{
	for (int i = 0; i < BCSize; i++)
		if (arr[i] == val) return true;
	return false;
};
__device__ WorldgenPRNG GetRNG(uint32_t world_seed, int map_w)
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

__universal__ uint32_t getPos(const uint32_t w, const uint32_t s, const uint32_t x, const uint32_t y)
{
	return s * (w * y + x);
}

__device__ void doCoalMineHax(uint8_t* map, uint32_t map_w, uint32_t map_h)
{
	for (int y = 0; y < map_h; y++)
	{
		for (int x = 0; x < map_w; x++)
		{
			uint32_t overlayPos = getPos(256, 3, x, y);
			uint32_t i = getPos(map_w, 3, x, y);
			uint32_t pix = createRGB(coalmine_overlay[overlayPos], coalmine_overlay[overlayPos + 1], coalmine_overlay[overlayPos + 2]);
			if (pix == 0x4000)
			{
				map[i] = 0xFF;
				map[i + 1] = 0xFF;
				map[i + 2] = 0xFF;
			}
			if (pix == 0x0040)
			{
				map[i] = 0x00;
				map[i + 1] = 0x00;
				map[i + 2] = 0x00;
			}
			if (pix == 0xFEFEFE)
			{
				map[i] = 0x0a;
				map[i + 1] = 0x33;
				map[i + 2] = 0x44;
			}
		}
	}
}

__universal__ uint32_t getPixelColor(const uint8_t* map, uint32_t pos)
{
	return createRGB(map[pos], map[pos + 1], map[pos + 2]);
}
__universal__ uint32_t getPixelColor(const uint8_t* map, const uint32_t w, const uint32_t x, const uint32_t y)
{
	uint32_t pos = getPos(w, 3, x, y);
	return getPixelColor(map, pos);
}
__universal__ void setPixelColor(uint8_t* map, uint32_t pos, uint32_t color)
{
	uint8_t r = ((color >> 16) & 0xff);
	uint8_t g = ((color >> 8) & 0xff);
	uint8_t b = ((color) & 0xff);
	map[pos] = r;
	map[pos + 1] = g;
	map[pos + 2] = b;
}
__universal__ void setPixelColor(uint8_t* map, uint32_t w, uint32_t x, uint32_t y, uint32_t color)
{
	uint32_t pos = getPos(w, 3, x, y);
	setPixelColor(map, pos, color);
}

__universal__ void fill(uint8_t* map, int w, int x1, int x2, int y1, int y2, uint32_t color)
{
	for (int x = x1; x <= x2; x++)
	{
		for (int y = y1; y <= y2; y++)
		{
			setPixelColor(map, w, x, y, color);
		}
	}
}

void blockOutRooms(uint8_t* map, int map_w, int map_h, uint32_t targetColor)
{
	int posMax = map_w * map_h;

	for (int pos = 0; pos < posMax; pos++)
	{
		uint32_t color = getPixelColor(map, pos * 3);
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
			uint32_t c = getPixelColor(map, map_w, endX, startY);
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
			uint32_t c = getPixelColor(map, map_w, startX, endY);
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

__device__ bool isMainPath(uint32_t map_w, int worldX)
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	return fill_x_to > 0 && fill_x_from > 0 && map_w > fill_x_from && fill_x_to < map_w + fill_x_from;
}
__device__ int fillMainPath(uint8_t* map, uint32_t map_w, int worldX)
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	fill(map, map_w, fill_x_from, fill_x_to, 0, 6, COLOR_BLACK);
	return fill_x_from;
}