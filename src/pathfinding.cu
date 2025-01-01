
#include "../platforms/platform_implementation.h"

#include "../include/worldgen_structs.h"
#include "../include/noita_random.h"
#include "../include/compute.h"
#include "../include/misc_funcs.h"

template <bool skip_coalmine, bool skip_fill>
_compute uint32_t get_pixel(const GeneratedBiome& s, const MainPathFill& f, int x, int y, int ssl) {
	if constexpr (!skip_coalmine) {
		if (s.scope.bSec.b == B_COALMINE && y >= 0) {
			if (coalmine_overlay[(y * 256 + x) * 3 + 2] == 0x42)
				return COLOR_BLACK;
			else if (coalmine_overlay[(y * 256 + x) * 3 + 1] > 0X10)
				return COLOR_WHITE;
		}
	}
	else {
		if (f.active) {
			if (y < 6 && x >= f.x1 && x <= f.x2)
				return COLOR_BLACK;
		}
	}

	y += 4;

	int tx = x / ssl;
	int ty = y / ssl;

	int tile = s.indices[s.scope.bSec.wang_w * ty + tx];
	bool v = tile & 0x4000;
	if (tile & 0x8000) {
		if (v)
			ty--;
		else
			tx--;
	}
	tile &= 0x3fff;

	int off_x = x - tx * ssl;
	int off_y = y - ty * ssl;

	if constexpr (!skip_fill) {
		while (1) {
			if (!(x - off_x >= 0 && y - off_y - 4 >= 0 && (v ? s.scope.ts.vTiles : s.scope.ts.hTiles)[tile].should_block))
				break;
			if (s.scope.bSec.b == B_COALMINE && coalmine_overlay[((y - off_y - 4) * 256 + x - off_x) * 3 + 1] > 0x10)
				break;
			if (v && !(off_x < s.scope.ts.short_side_len - 1 && off_y < 2 * s.scope.ts.short_side_len - 1))
				break;
			if (!v && !(off_x < 2 * s.scope.ts.short_side_len - 1 && off_y < s.scope.ts.short_side_len - 1))
				break;
			uint32_t c = get_pixel<true, true>(s, f, x - off_x, y - off_y - 4, ssl);
			if (c != COLOR_BLACK && c != COLOR_WHITE)
				return COLOR_WHITE;
			break;
		}
	}

	if (v)
		return s.scope.ts.v_tile_at(tile % s.scope.ts.widthV, tile / s.scope.ts.widthV, off_x, off_y);
	else
		return s.scope.ts.h_tile_at(tile % s.scope.ts.widthH, tile / s.scope.ts.widthH, off_x, off_y);
}

_compute static void tryNext(const GeneratedBiome& s, const MainPathFill& f, int x, int y, MemSpan stackCache, int& stackSize, MemSpan visited, int rmw, int rmh, int ssl)
{
	if (x >= 0 && y >= 0 && x < rmw && y < rmh)
	{
		if (visited.ptr[y * rmw + x]) return;
		uint32_t c = get_pixel(s, f, x, y, ssl);
		if (c == COLOR_BLACK || c == COLOR_COFFEE || c == COLOR_HELL_GREEN || c == COLOR_FROZEN_VAULT_MINT) {
			((Vec2i*)stackCache.ptr)[stackSize++] = { x, y };
			visited.ptr[y * rmw + x] = 2;
		}
		else
			visited.ptr[y * rmw + x] = 1;
	}
}

_compute bool findPath(const GeneratedBiome& s, const MainPathFill& f, MemSpan stackMemArea, MemSpan visited, int x, int y)
{
	int rmw = s.scope.bSec.map_w; //register map width
	int rmh = s.scope.bSec.map_h; //register map height
	int ssl = s.scope.ts.short_side_len;

	bool pathFound = false;

	int stackSize = 1;
	Vec2i* stackMem = (Vec2i*)stackMemArea.ptr;

	stackMem[0] = { x , y };

	while (stackSize > 0 && pathFound != 1)
	{
		Vec2i n = stackMem[--stackSize];
		if (n.y == rmh - 1) {
			pathFound = 1;
			break;
		}
		if (n.x != -1)
		{
			tryNext(s, f, n.x, n.y - 1, stackMemArea, stackSize, visited, rmw, rmh, ssl);
			tryNext(s, f, n.x - 1, n.y, stackMemArea, stackSize, visited, rmw, rmh, ssl);
			tryNext(s, f, n.x + 1, n.y, stackMemArea, stackSize, visited, rmw, rmh, ssl);
			tryNext(s, f, n.x, n.y + 1, stackMemArea, stackSize, visited, rmw, rmh, ssl);
			if (!stackMemArea.is_safe(stackSize, sizeof(Vec2i)))
				printf("findPath(): stack mem too small\n");
		}
	}
	return pathFound;
}

_compute bool HasPathToBottom(const GeneratedBiome& s, const MainPathFill& f, MemSpan stackMemArea, MemSpan visited, uint32_t path_start_x, bool fixed_x)
{
	if (!visited.is_safe(max(0, s.scope.bSec.map_w * s.scope.bSec.map_h - 1)))
		printf("findPath(): visited mem too small\n");
	cMemset(visited.ptr, 0, s.scope.bSec.map_w * s.scope.bSec.map_h);

	if (fixed_x)
		return findPath(s, f, stackMemArea, visited, path_start_x, 0);

	int x = path_start_x;

	while (x < s.scope.bSec.map_w)
	{
		uint32_t c = get_pixel(s, f, x, 0, s.scope.ts.short_side_len);
		if (c != COLOR_BLACK && c != COLOR_COFFEE)
		{
			x++;
			continue;
		}

		bool hasPath = findPath(s, f, stackMemArea, visited, x, 0);
		if (hasPath)
			return true;
		x++;
		while (x < s.scope.bSec.map_w) {
			uint32_t c = get_pixel(s, f, x, 0, s.scope.ts.short_side_len);
			if (c != COLOR_BLACK && c != COLOR_COFFEE)
				break;
			x++;
		}
	}
	return false;
}

_compute bool isValid(const GeneratedBiome& s, MemSpan stackMemArea, MemSpan visited)
{
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (s.scope.bSec.worldX - WORLD_OFFSET_X) * 512.0) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	bool active = fill_x_to > 0 && fill_x_from > 0 && s.scope.bSec.map_w > fill_x_from && fill_x_to < s.scope.bSec.map_w + fill_x_from;
	MainPathFill f = { active, fill_x_from, fill_x_to };

	uint32_t path_start_x = 0;
	if (s.scope.bSec.b == B_COALMINE)
		path_start_x = 0x8e;
	else if (active)
		path_start_x = fill_x_from;

	return HasPathToBottom(s, f, stackMemArea, visited, path_start_x, active);
}