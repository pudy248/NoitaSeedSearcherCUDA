#pragma once
#include "../platforms/platform_compute_helpers.h"

#include "worldgen_helpers.h"

_compute bool traversable(uint8_t* map, int x, int y, int rmw)
{
	long c = getPixelColor(map, rmw, x, y);

	return c == COLOR_BLACK || c == COLOR_COFFEE || c == COLOR_HELL_GREEN || c == COLOR_FROZEN_VAULT_MINT;
}

_compute
void tryNext(int x, int y, uint8_t* map, Vec2i* stackCache, int& stackSize, uint8_t* visited, int rmw, int rmh)
{
	if (x >= 0 && y >= 0 && x < rmw && y < rmh) {
		if (visited[y * rmw + x] == 0 && traversable(map, x, y, rmw))
		{
			visited[y * rmw + x] = 1;
			stackCache[stackSize++] = { x, y };
		}
	}
}

_compute bool findPath(uint8_t* map, uint8_t* stackMemArea, uint8_t* visited, const uint32_t map_w, const uint32_t map_h, int x, int y)
{
	int rmw = map_w; //register map width
	int rmh = map_h; //register map height

	bool pathFound = false;

	int stackSize = 1;
	Vec2i* stackMem = (Vec2i*)stackMemArea;
	cMemset(visited, 0, map_w * map_h);

	stackMem[0] = { x , y };

	while (stackSize > 0 && pathFound != 1)
	{
		Vec2i n = stackMem[--stackSize];
		//if((n.x + n.y) % 2 == 0) 
			setPixelColor(map, rmw, n.x, n.y, COLOR_PURPLE);
		if (n.x != -1) {
			if (n.y == rmh - 1)
			{
				pathFound = 1;
			}
			tryNext(n.x, n.y - 1, map, stackMem, stackSize, visited, rmw, rmh);
			tryNext(n.x - 1, n.y, map, stackMem, stackSize, visited, rmw, rmh);
			tryNext(n.x + 1, n.y, map, stackMem, stackSize, visited, rmw, rmh);
			tryNext(n.x, n.y + 1, map, stackMem, stackSize, visited, rmw, rmh);
		}
	}
	return pathFound;
}

_compute bool HasPathToBottom(uint8_t* map, uint8_t* stackMemArea, uint8_t* visited, uint32_t map_w, uint32_t map_h, uint32_t path_start_x, bool fixed_x)
{
	if (fixed_x)
	{
		bool hasPath = findPath(map, stackMemArea, visited, map_w, map_h, path_start_x, 0);
		if (hasPath)
		{
			return true;
		}
		return false;
	}

	int x = path_start_x;

	while (x < map_w)
	{
		long c = getPixelColor(map, map_w, x, 0);
		if (c != COLOR_BLACK && c != COLOR_COFFEE)
		{
			x++;
			continue;
		}

		bool hasPath = findPath(map, stackMemArea, visited, map_w, map_h, x, 0);
		if (hasPath)
		{
			return true;
		}
		x++;
	}
	return false;
}

_compute bool isValid(uint8_t* map, uint8_t* stackMemArea, uint8_t* visited, uint32_t map_w, uint32_t map_h, int worldX, int worldY, bool isCoalMine) {
	uint32_t path_start_x = 0;
	bool mainPath = isMainPath(map_w, worldX);
	if (isCoalMine)
	{
		path_start_x = 0x8e;
	}
	else if (mainPath)
	{
		path_start_x = fillMainPath(map, map_w, worldX);
	}

	bool hasPath = HasPathToBottom(map, stackMemArea, visited, map_w, map_h, path_start_x, mainPath);

	return hasPath;
}