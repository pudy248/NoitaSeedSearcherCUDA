#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "datatypes.h"
#include "worldgen_helpers.h"

__device__ IntPair Pop(IntPair* stackMem, int* stackSize) {
	return stackMem[--(*stackSize)];
}

__device__ void Push(IntPair* stackMem, IntPair* stackCache, byte* cacheSize, int* stackSize) {
	memcpy(stackMem + *stackSize, stackCache, sizeof(IntPair) * *cacheSize);
	*stackSize += *cacheSize;
	*cacheSize = 0;
}

__device__ bool traversable(byte* map, int x, int y, int rmw)
{
	long c = getPixelColor(map, rmw, x, y);

	return c == COLOR_BLACK || c == COLOR_COFFEE;
}

__device__ void tryNext(IntPair position, IntPair offset, byte* map, byte* visited, IntPair* stackCache, byte* cacheSize, int rmw, int rmh)
{
	IntPair next = position + offset;
	if (next.x >= 0 && next.y >= 0 && next.x < rmw && next.y < rmh) {
		if (visited[next.y * rmw + next.x] == 0 && traversable(map, next.x, next.y, rmw))
		{
			visited[next.y * rmw + next.x] = 1;
			stackCache[(*cacheSize)++] = { next.x, next.y };
		}
	}
}

__device__ bool atTarget(int targetY, IntPair n)
{
	return targetY == n.y;
}

__device__ bool findPath(byte* map, byte* stackMemArea, byte* visited, const uint map_w, const uint map_h, int x, int y)
{
	int rmw = map_w; //register map width
	int rmh = map_h; //register map height
	bool pathFound = false;

	int stackSize = 1;
	byte cacheSize = 0;
	IntPair* stackMem = (IntPair*)stackMemArea;
	IntPair stackCacke[4];
	memset(visited, 0, map_w * map_h);

	stackMem[0] = { x , y };

	while (stackSize > 0 && !pathFound)
	{
		cacheSize = 0;
		IntPair n = Pop(stackMem, &stackSize);
		//if((n.x + n.y) % 2 == 0) setPixelColor(map, register_mapW, n.x, n.y, COLOR_PURPLE);
		if (n.x != -1) {
			if (atTarget(map_h - 1, n))
			{
				pathFound = 1;
			}
			tryNext(n, { 0, 1 }, map, visited, stackCacke, &cacheSize, rmw, rmh);
			tryNext(n, { -1, 0 }, map, visited, stackCacke, &cacheSize, rmw, rmh);
			tryNext(n, { 1, 0 }, map, visited, stackCacke, &cacheSize, rmw, rmh);
			tryNext(n, { 0, -1 }, map, visited, stackCacke, &cacheSize, rmw, rmh);
		}
		Push(stackMem, stackCacke, &cacheSize, &stackSize);
	}

	return pathFound;
}

__device__ bool HasPathToBottom(byte* map, byte* stackMemArea, byte* visited, uint map_w, uint map_h, uint path_start_x, bool fixed_x)
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

__device__ bool isValid(byte* map, byte* stackMemArea, byte* visited, uint map_w, uint map_h, int worldX, int worldY, bool isCoalMine) {
	uint path_start_x = 0;
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