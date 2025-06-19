
#include "../platforms/platform_implementation.h"

#include "../include/compute.h"
#include "../include/misc_funcs.h"
#include "../include/noita_random.h"
#include "../include/worldgen_structs.h"

constexpr int BIOME_PATH_FIND_WORLD_POS_MIN_X = 159;
constexpr int BIOME_PATH_FIND_WORLD_POS_MAX_X = 223;
constexpr int WORLD_OFFSET_X = 35;

template <int coalmine_mode, bool skip_fill>
_compute uint32_t get_pixel(const GeneratedBiome& s, const MainPathFill& f, int x, int y, int ssl, const uint8_t* rcoal_overlay) {
	if constexpr (coalmine_mode) {
		if (s.scope.bSec.b == B_COALMINE && y >= 0) {
			// Somehow this is really slow if it's not "forced" to be in a register. Apparently not caching well, not worth the time to investigate
			if (rcoal_overlay[y * 256 + x] == 1)
				return COLOR_BLACK;
			else if (rcoal_overlay[y * 256 + x] > 1)
				return COLOR_WHITE;
		}
	}
	if constexpr (coalmine_mode == 2)
		return COLOR_BLACK;

	if (f.active) {
		if (y < 6 && x >= f.x1 && x <= f.x2)
			return COLOR_BLACK;
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
		while (s.scope.bSec.b == B_COALMINE || s.scope.bSec.b == B_EXCAVATIONSITE) {
			if (!(x - off_x >= 0 && y - off_y - 4 >= 0 &&
					(v ? s.scope.ts.vTiles : s.scope.ts.hTiles)[tile].should_block))
				break;
			//if (s.scope.bSec.b == B_COALMINE && rcoal_overlay[(y - off_y - 4) * 256 + x - off_x] > 1)
			//	break;
			if (v && !(off_x < s.scope.ts.short_side_len - 1 && off_y < 2 * s.scope.ts.short_side_len - 1))
				break;
			if (!v && !(off_x < 2 * s.scope.ts.short_side_len - 1 && off_y < s.scope.ts.short_side_len - 1))
				break;
			uint32_t c = get_pixel<0, true>(s, f, x - off_x, y - off_y - 4, ssl, coalmine_overlay);
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

_compute static void tryNext(const GeneratedBiome& s, const MainPathFill& f, int x, int y, MemSpan stackCache,
	int& stackSize, MemSpan visited, int rmw, int rmh, int ssl) {
	if (x >= 0 && y >= 0 && x < rmw && y < rmh) {
		if (visited.ptr[y * rmw + x])
			return;
		uint32_t c = get_pixel(s, f, x, y, ssl, coalmine_overlay);
		if (c == COLOR_BLACK || c == COLOR_COFFEE || c == COLOR_HELL_GREEN) {
			((Vec2i*)stackCache.ptr)[stackSize++] = {x, y};
			visited.ptr[y * rmw + x] = 2;
		} else
			visited.ptr[y * rmw + x] = 1;
	}
}

_compute static bool findPath(
	const GeneratedBiome& s, const MainPathFill& f, MemSpan stackMemArea, MemSpan visited, int x, int y) {
	int rmw = s.scope.bSec.map_w; //register map width
	int rmh = s.scope.bSec.map_h; //register map height
	int ssl = s.scope.ts.short_side_len;
	const uint8_t* rcoal_overlay = coalmine_overlay;

	bool pathFound = false;

	int stackSize = 3;
	Vec2i* stackMem = (Vec2i*)stackMemArea.ptr;

	stackMem[0] = {x - 1, y};
	stackMem[1] = {x + 1, y};
	stackMem[2] = {x, y + 1};

	while (stackSize > 0 && pathFound != 1) {
		Vec2i n = stackMem[--stackSize];
		uint32_t c = get_pixel(s, f, n.x, n.y, ssl, rcoal_overlay);
		if (c == COLOR_BLACK || c == COLOR_COFFEE || c == COLOR_HELL_GREEN)
			visited.ptr[n.y * rmw + n.x] = 2;
		else
			continue;
		if (n.y == rmh - 1) {
			pathFound = 1;
			break;
		}
		if (n.x != -1) {
			if (n.y > 0 && !visited.ptr[(n.y - 1) * rmw + n.x]) {
				stackMem[stackSize++] = {n.x, n.y - 1};
				visited.ptr[(n.y - 1) * rmw + n.x] = 1;
			}
			if (n.x > 0 && !visited.ptr[n.y * rmw + (n.x - 1)]) {
				stackMem[stackSize++] = {n.x - 1, n.y};
				visited.ptr[n.y * rmw + (n.x - 1)] = 1;
			}
			if (n.x < rmw - 1 && !visited.ptr[n.y * rmw + (n.x + 1)]) {
				stackMem[stackSize++] = {n.x + 1, n.y};
				visited.ptr[n.y * rmw + (n.x + 1)] = 1;
			}
			if (n.y < rmh - 1 && !visited.ptr[(n.y + 1) * rmw + n.x]) {
				stackMem[stackSize++] = {n.x, n.y + 1};
				visited.ptr[(n.y + 1) * rmw + n.x] = 1;
			}
			stackMemArea.is_safe("findPath", "stack (misc)", stackSize, sizeof(Vec2i));
		}
	}
	return pathFound;
}

_compute static bool HasPathToBottom(const GeneratedBiome& s, const MainPathFill& f, MemSpan stackMemArea,
	MemSpan visited, uint32_t path_start_x, bool fixed_x) {
	visited.is_safe("findPath", "visited", max(0, s.scope.bSec.map_w * s.scope.bSec.map_h - 1));
	cMemset(visited.ptr, 0, s.scope.bSec.map_w * s.scope.bSec.map_h);

	if (fixed_x)
		return findPath(s, f, stackMemArea, visited, path_start_x, 0);

	for (uint32_t x = path_start_x; x < s.scope.bSec.map_w; x++) {
		uint32_t c = get_pixel(s, f, x, 0, s.scope.ts.short_side_len, coalmine_overlay);
		if (c != COLOR_BLACK && c != COLOR_COFFEE)
			continue;
		if (visited.ptr[x])
			continue;

		cMemset(visited.ptr, 0, s.scope.bSec.map_w * s.scope.bSec.map_h);
		bool hasPath = findPath(s, f, stackMemArea, visited, x, 0);
		if (hasPath)
			return true;
	}
	return false;
}

_compute bool isValid(const GeneratedBiome& s, MemSpan stackMemArea, MemSpan visited) {
	int fill_x_from = (BIOME_PATH_FIND_WORLD_POS_MIN_X - (s.scope.bSec.worldX - WORLD_OFFSET_X) * 512) / 10;
	int fill_x_to = fill_x_from + (BIOME_PATH_FIND_WORLD_POS_MAX_X - BIOME_PATH_FIND_WORLD_POS_MIN_X) / 10;
	bool active = fill_x_to > 0 && fill_x_from > 0 && s.scope.bSec.map_w > fill_x_from &&
				  fill_x_to < s.scope.bSec.map_w + fill_x_from;
	MainPathFill f = {active, fill_x_from, fill_x_to};

	uint32_t path_start_x = 0;
	if (s.scope.bSec.b == B_COALMINE)
		path_start_x = 0x8e;
	else if (active)
		path_start_x = fill_x_from;

	return HasPathToBottom(s, f, stackMemArea, visited, path_start_x, active);
}