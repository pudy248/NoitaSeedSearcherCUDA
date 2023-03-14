#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "datatypes.h"
#include "worldgen_helpers.h"

/*
__device__ void Push(IntPair* stackMem, IntPair* stackCache, int* cacheSize, int* stackSize) {
	//memcpy(stackMem + *stackSize, stackCache, sizeof(IntPair) * *cacheSize);
	//*stackSize += *cacheSize;
	//*cacheSize = 0;
	while (stackSize > 0) {
		stackMem[(*stackSize)++] = stackCache[--(*cacheSize)];
	}
}

__device__ bool traversable(byte* map, int x, int y, int rmw)
{
	long c = getPixelColor(map, rmw, x, y);

	return c == COLOR_BLACK;// || c == COLOR_COFFEE;
}

__device__ void tryNext(IntPair position, byte *map, byte* visited, IntPair* stackCache, int& cacheSize, int rmw, int rmh)
{
	if (position.x >= 0 && position.y >= 0 && position.x < rmw && position.y < rmh) {
		if (visited[position.y * rmw + position.x] == 0 && traversable(map, position.x, position.y, rmw))
		{
			visited[position.y * rmw + position.x] = 1;
			stackCache[cacheSize++] = position;
		}
	}
}

__device__ bool findPath(byte* map, byte* stackMemArea, byte* visited, const uint map_w, const uint map_h, int x, int y)
{
	int rmw = map_w; //register map width
	int rmh = map_h; //register map height
	bool pathFound = false;

	int stackSize = 1;
	IntPair* stackMem = (IntPair*)stackMemArea;
	memset(visited, 0, map_w * map_h);
	memset(stackMem, 0, sizeof(IntPair) * rmw * rmh);

	stackMem[0] = { x , y };

	while (stackSize > 0 && !pathFound)
	{
		if(blockDim.x * blockIdx.x + threadIdx.x == 55) printf("%i\n", stackSize);
		IntPair n = stackMem[--stackSize];
		//if((n.x + n.y) % 2 == 0) 
			setPixelColor(map, rmw, n.x, n.y, COLOR_PURPLE);
		if (n.x != -1) {
			if (n.y == rmh - 1)
			{
				pathFound = 1;
			}
			tryNext(n + IntPair(0, -1), map, visited, stackMem, stackSize, rmw, rmh);
			tryNext(n + IntPair(-1, 0), map, visited, stackMem, stackSize, rmw, rmh);
			tryNext(n + IntPair(1, 0), map, visited, stackMem, stackSize, rmw, rmh);
			tryNext(n + IntPair(0, 1), map, visited, stackMem, stackSize, rmw, rmh);
		}
	}

	return pathFound;
}*/

__device__ bool traversable(byte* map, int x, int y, int rmw)
{
	long c = getPixelColor(map, rmw, x, y);

	return c == COLOR_BLACK || c == COLOR_COFFEE;
}

__device__
void tryNext(int x, int y, byte* map, IntPair* stackCache, int& stackSize, byte* visited, int rmw, int rmh)
{
	if (x >= 0 && y >= 0 && x < rmw && y < rmh) {
		if (visited[y * rmw + x] == 0 && traversable(map, x, y, rmw))
		{
			visited[y * rmw + x] = 1;
			stackCache[stackSize++] = { x, y };
		}
	}
}

__device__ bool findPath(byte* map, byte* stackMemArea, byte* visited, const uint map_w, const uint map_h, int x, int y)
{
	int rmw = map_w; //register map width
	int rmh = map_h; //register map height

	bool pathFound = false;

	int stackSize = 1;
	IntPair* stackMem = (IntPair*)stackMemArea;//(IntPair*)malloc(sizeof(IntPair) * (map_w + map_h));
	memset(visited, 0, map_w * map_h);
	//memset(stackMem, 0, sizeof(IntPair) * rmw * rmh);

	stackMem[0] = { x , y };

	while (stackSize > 0 && pathFound != 1)
	{
		//if (stackSize >= map_w + map_h) {
		//	printf("Blew up stack!");
		//	return false;
		//}
		IntPair n = stackMem[--stackSize];
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
	//free(stackMem);
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