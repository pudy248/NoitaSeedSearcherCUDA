#include "../platforms/platform_implementation.h"

#include "../include/noita_random.h"
#include "../include/worldgen_structs.h"

#include <cmath>
#include <cstdio>

static constexpr uint32_t blockedColors[] = {
	0xff00ffU,
	0xff0080U,
	0xff0affU,
	0xc35700U,
	0x00ac64U,
	0x00ac6eU,
	0x7868ffU,
	0x70d79eU,
	0x70d79fU,
	0x70d7a0U,
	0x70d7a1U,
};
static constexpr int BCSize = sizeof(blockedColors) / sizeof(blockedColors[0]);
_universal constexpr static bool contains(const uint32_t arr[BCSize], uint32_t val) {
	for (int i = 0; i < BCSize; i++)
		if (arr[i] == val) return true;
	return false;
};

struct WangProcess
{
	int biome;
	int corner_type_color_template[4][4];
	uint32_t* colors;
	int w, h, sx, sy;
};

_universal uint32_t WangTileset::h_tile_at(int tx, int ty, int xoff, int yoff) const {
	int xpos = tx * (2 * short_side_len + 3) + 1;
	int ypos = ty * (short_side_len + 3) + 3;
	uint8_t* pix = &tileData[(ypos + yoff) * tdStride + (xpos + xoff) * 3];
	return (pix[0] << 16) | (pix[1] << 8) | pix[2];
}
_universal uint32_t WangTileset::v_tile_at(int tx, int ty, int xoff, int yoff) const {
	int xpos = tx * (short_side_len + 3) + 1;
	int ypos = heightH * (short_side_len + 3) + ty * (2 * short_side_len + 3) + 5;
	uint8_t* pix = &tileData[(ypos + yoff) * tdStride + (xpos + xoff) * 3];
	return (pix[0] << 16) | (pix[1] << 8) | pix[2];
}

static void stbhw__get_template_info(WangProcess& p, WangTileset& ts) {
	int size_x, size_y;
	int horz_w, horz_h, vert_w, vert_h;

	if (ts.is_corner) {
		horz_w = ts.num_color[1] * ts.num_color[2] * ts.num_color[3] * ts.num_vary[0];
		horz_h = ts.num_color[0] * ts.num_color[1] * ts.num_color[2] * ts.num_vary[1];

		vert_w = ts.num_color[0] * ts.num_color[3] * ts.num_color[2] * ts.num_vary[1];
		vert_h = ts.num_color[1] * ts.num_color[0] * ts.num_color[3] * ts.num_vary[0];

		int horz_x = horz_w * (2 * ts.short_side_len + 3);
		int horz_y = horz_h * (ts.short_side_len + 3);

		int vert_x = vert_w * (ts.short_side_len + 3);
		int vert_y = vert_h * (2 * ts.short_side_len + 3);

		size_x = horz_x > vert_x ? horz_x : vert_x;
		size_y = 2 + horz_y + 2 + vert_y;
	}
	else {
		horz_w = ts.num_color[0] * ts.num_color[1] * ts.num_color[2] * ts.num_vary[0];
		horz_h = ts.num_color[3] * ts.num_color[4] * ts.num_color[2] * ts.num_vary[1];

		vert_w = ts.num_color[0] * ts.num_color[5] * ts.num_color[1] * ts.num_vary[1];
		vert_h = ts.num_color[3] * ts.num_color[4] * ts.num_color[5] * ts.num_vary[0];

		int horz_x = horz_w * (2 * ts.short_side_len + 3);
		int horz_y = horz_h * (ts.short_side_len + 3);

		int vert_x = vert_w * (ts.short_side_len + 3);
		int vert_y = vert_h * (2 * ts.short_side_len + 3);

		size_x = horz_x > vert_x ? horz_x : vert_x;
		size_y = 2 + horz_y + 2 + vert_y;
	}

	p.sx = size_x;
	p.sy = size_y;
	ts.widthH = horz_w;
	ts.heightH = horz_h;
	ts.widthV = vert_w;
	ts.heightV = vert_h;
}

static void stbhw__parse_h_rect(WangProcess& p, WangTileset& ts, int tx, int ty,
	char a, char b, char c, char d, char e, char f, int idx)
{
	int sIdx = 0;
	int len = ts.short_side_len;
	WangTile& h = ts.hTiles[idx];
	h.should_block = contains(blockedColors, ts.h_tile_at(tx, ty, 0, 0));
	for (int i = 0; i < _WangTileMaxSpawns; i++)
		h.spawns[i] = { 0, 0, -1 };
	h.colors[0] = a;
	h.colors[1] = b;
	h.colors[2] = c;
	h.colors[3] = d;
	h.colors[4] = e;
	h.colors[5] = f;
#ifdef DEBUG_SPAWN_PIXELS
	printf("H %i: %i %i %i %i %i %i\n", idx, a, b, c, d, e, f);
#endif

	for (uint8_t j = 0; j < len; ++j)
		for (uint8_t i = 0; i < len * 2; ++i) {
			uint32_t pix = ts.h_tile_at(tx, ty, i, j);
			for (int16_t z = 0; z < HostSpawnColors[p.biome].count; z++) {
				if (pix == HostSpawnColors[p.biome].colors[z]) {
					h.spawns[sIdx++] = { i, j, z };
#ifdef DEBUG_SPAWN_PIXELS
					printf("Wang Spawn (%i, %i): Biome %i\n", i, j, z);
#endif
					goto h_end;
				}
			}
			for (int16_t z = 0; z < HostSpawnColors[0].count; z++)
			{
				if (pix == HostSpawnColors[0].colors[z]) {
					h.spawns[sIdx++] = { i, j, (int16_t)(HostSpawnColors[p.biome].count + z) };
#ifdef DEBUG_SPAWN_PIXELS
					printf("Wang Spawn (%i, %i): Global %i\n", i, j, z);
#endif
					goto h_end;
				}
			}
			h_end:
		}
	if (sIdx > _WangTileMaxSpawns)
		printf("H Tile %i: Ran out of spawns! %i of %i.\n", idx, sIdx, _WangTileMaxSpawns);
}

static void stbhw__parse_v_rect(WangProcess& p, WangTileset& ts, int tx, int ty,
	char a, char b, char c, char d, char e, char f, int idx)
{
	int sIdx = 0;
	int len = ts.short_side_len;
	WangTile& h = ts.vTiles[idx];
	h.should_block = contains(blockedColors, ts.v_tile_at(tx, ty, 0, 0));
	for (int i = 0; i < _WangTileMaxSpawns; i++)
		h.spawns[i] = { 0, 0, -1 };
	h.colors[0] = a;
	h.colors[1] = b;
	h.colors[2] = c;
	h.colors[3] = d;
	h.colors[4] = e;
	h.colors[5] = f;
#ifdef DEBUG_SPAWN_PIXELS
	printf("V %i: %i %i %i %i %i %i\n", idx, a, b, c, d, e, f);
#endif

	for (uint8_t j = 0; j < len * 2; ++j)
		for (uint8_t i = 0; i < len; ++i)
		{
			uint32_t pix = ts.v_tile_at(tx, ty, i, j);
			for (int16_t z = 0; z < HostSpawnColors[p.biome].count; z++) {
				if (pix == HostSpawnColors[p.biome].colors[z]) {
					h.spawns[sIdx++] = { i, j, z };
#ifdef DEBUG_SPAWN_PIXELS
					printf("Wang Spawn (%i, %i): Biome %i\n", i, j, z);
#endif
					goto v_end;
				}
			}
			for (int16_t z = 0; z < HostSpawnColors[0].count; z++) {
				if (pix == HostSpawnColors[0].colors[z]) {
					h.spawns[sIdx++] = { i, j, (int16_t)(HostSpawnColors[p.biome].count + z) };
#ifdef DEBUG_SPAWN_PIXELS
					printf("Wang Spawn (%i, %i): Global %i\n", i, j, z);
#endif
					goto v_end;
				}
			}
			v_end:
		}
	if (sIdx > _WangTileMaxSpawns)
		printf("V Tile %i: Ran out of spawns! %i of %i.\n", idx, sIdx, _WangTileMaxSpawns);
}

static void stbhw__process_h_row(WangProcess& p, WangTileset& ts,
	int tx, int ty,
	int a0, int a1,
	int b0, int b1,
	int c0, int c1,
	int d0, int d1,
	int e0, int e1,
	int f0, int f1,
	int variants, int& i)
{
	for (char v = 0; v < variants; ++v)
		for (char f = f0; f <= f1; ++f)
			for (char e = e0; e <= e1; ++e)
				for (char d = d0; d <= d1; ++d)
					for (char c = c0; c <= c1; ++c)
						for (char b = b0; b <= b1; ++b)
							for (char a = a0; a <= a1; ++a)
							{
								stbhw__parse_h_rect(p, ts, tx, ty, a, b, c, d, e, f, i++);
								tx++;
							}
}

static void stbhw__process_v_row(WangProcess& p, WangTileset& ts,
	int tx, int ty,
	int a0, int a1,
	int b0, int b1,
	int c0, int c1,
	int d0, int d1,
	int e0, int e1,
	int f0, int f1,
	int variants, int& i)
{
	for (char v = 0; v < variants; ++v)
		for (char f = f0; f <= f1; ++f)
			for (char e = e0; e <= e1; ++e)
				for (char d = d0; d <= d1; ++d)
					for (char c = c0; c <= c1; ++c)
						for (char b = b0; b <= b1; ++b)
							for (char a = a0; a <= a1; ++a)
							{
								stbhw__parse_v_rect(p, ts, tx, ty, a, b, c, d, e, f, i++);
								tx++;
							}
}

static int stbhw__process_template(WangProcess& p, WangTileset& ts)
{
	int i, j, k, q, ty;

	int vi = 0;
	int hi = 0;

	// printf("process_template: %i %i [%i %i %i %i]\n", ts.num_vary[0], ts.num_vary[1], ts.num_color[0], ts.num_color[1], ts.num_color[2], ts.num_color[3]);

	if (ts.is_corner)
	{
		ty = 0;
		for (k = 0; k < ts.num_color[2]; ++k)
		{
			for (j = 0; j < ts.num_color[1]; ++j)
			{
				for (i = 0; i < ts.num_color[0]; ++i)
				{
					for (q = 0; q < ts.num_vary[1]; ++q)
					{
						stbhw__process_h_row(p, ts, 0, ty,
							0, ts.num_color[1] - 1, 0, ts.num_color[2] - 1, 0, ts.num_color[3] - 1,
							i, i, j, j, k, k,
							ts.num_vary[0], hi);
						ty++;
					}
				}
			}
		}
		ty = 0;
		for (k = 0; k < ts.num_color[3]; ++k)
		{
			for (j = 0; j < ts.num_color[0]; ++j)
			{
				for (i = 0; i < ts.num_color[1]; ++i)
				{
					for (q = 0; q < ts.num_vary[0]; ++q)
					{
						stbhw__process_v_row(p, ts, 0, ty,
							0, ts.num_color[0] - 1, 0, ts.num_color[3] - 1, 0, ts.num_color[2] - 1,
							i, i, j, j, k, k,
							ts.num_vary[1], vi);
						ty++;
					}
				}
			}
		}
	}
	else
	{
		ty = 0;
		for (k = 0; k < ts.num_color[3]; ++k)
		{
			for (j = 0; j < ts.num_color[4]; ++j)
			{
				for (i = 0; i < ts.num_color[2]; ++i)
				{
					for (q = 0; q < ts.num_vary[1]; ++q)
					{
						stbhw__process_h_row(p, ts, 0, ty,
							0, ts.num_color[2] - 1, k, k,
							0, ts.num_color[1] - 1, j, j,
							0, ts.num_color[0] - 1, i, i,
							ts.num_vary[0], hi);
						ty++;
					}
				}
			}
		}
		ty = 0;
		for (k = 0; k < ts.num_color[3]; ++k)
		{
			for (j = 0; j < ts.num_color[4]; ++j)
			{
				for (i = 0; i < ts.num_color[5]; ++i)
				{
					for (q = 0; q < ts.num_vary[0]; ++q)
					{
						stbhw__process_v_row(p, ts, 0, ty,
							0, ts.num_color[0] - 1, i, i,
							0, ts.num_color[1] - 1, j, j,
							0, ts.num_color[5] - 1, k, k,
							ts.num_vary[1], vi);
						ty++;
					}
				}
			}
		}
	}
	return 0;
}

static Vec2i stbhw_get_index_stride(WangTile* list, int numlist, char a, char b, char c, char d, char e, char f)
{
	//printf("%i %i %i %i %i %i:\n", a, b, c, d, e, f);
	int first = -1;
	int second = -1;
	for (int i = 0; i < numlist; ++i)
	{
		WangTile& h = list[i];
		if ((a < 0 || a == h.colors[0]) &&
			(b < 0 || b == h.colors[1]) &&
			(c < 0 || c == h.colors[2]) &&
			(d < 0 || d == h.colors[3]) &&
			(e < 0 || e == h.colors[4]) &&
			(f < 0 || f == h.colors[5]))
		{
			//printf("%i ", i);
			if (first < 0) first = i;
			else if (second < 0) second = i;
		}
	}
	if (first < 0)
	{
		//printf("NO TILE\n");
		return { 0, 0 };
	}
	//printf("\n");
	return { first, second == -1 ? 0 : second - first };
}

static uint32_t stbhw_get_index_num(char a, char b, char c, char d, char e, char f) {
	constexpr int base = 3;
	return base * base * base * base * base * a
		+ base * base * base * base * b
		+ base * base * base * c
		+ base * base * d
		+ base * e
		+ f;
}

static void stbhw_get_all_indices(WangTileset& ts)
{
	for (char a = 0; a < ts.max_colors; a++)
		for (char b = 0; b < ts.max_colors; b++)
			for (char c = 0; c < ts.max_colors; c++)
				for (char d = 0; d < ts.max_colors; d++)
					for (char e = 0; e < ts.max_colors; e++)
						for (char f = 0; f < ts.max_colors; f++) {
							Vec2i h = stbhw_get_index_stride(ts.hTiles, ts.widthH * ts.heightH, a, b, c, d, e, f);
							Vec2i v = stbhw_get_index_stride(ts.vTiles, ts.widthV * ts.heightV, a, b, c, d, e, f);
							ts.hIndices[stbhw_get_index_num(a, b, c, d, e, f)] = ((h.x & 0xff) << 8 | (h.y & 0xff));
							ts.vIndices[stbhw_get_index_num(a, b, c, d, e, f)] = ((v.x & 0xff) << 8 | (v.y & 0xff));
						}
}

WangTileset stbhw_build_tileset_from_image(uint8_t* data, int biome, int stride, int w, int h)
{
	uint8_t header[9];
	WangTileset ts = {};
	WangProcess p = {};
	p.biome = biome;

	for (int i = 0; i < 9; ++i)
	{
		header[i] = data[w * 3 - 1 - i] ^ (i * 55);
	}

	for (int i = 0; i < 6; i++) ts.num_color[i] = 0;

	// extract header info
	if (header[7] == 0xc0)
	{
		// corner-type
		ts.is_corner = 1;
		for (int i = 0; i < 4; ++i) {
			ts.num_color[i] = header[i];
			ts.max_colors = max(ts.max_colors, header[i]);
		}
		ts.num_vary[0] = header[4];
		ts.num_vary[1] = header[5];
		ts.short_side_len = header[6];
	}
	else
	{
		ts.is_corner = 0;
		// edge-type
		for (int i = 0; i < 6; ++i) {
			ts.num_color[i] = header[i];
			ts.max_colors = max(ts.max_colors, header[i]);
		}
		ts.num_vary[0] = header[6];
		ts.num_vary[1] = header[7];
		ts.short_side_len = header[8];
	}

	if (ts.max_colors > 3) {
		printf("ERR: TILESET HAS MORE THAN 3 COLOR CHANNELS\n");
	}

	//if (ts.num_vary[0] < 0 || ts.num_vary[0] > 64 || ts.num_vary[1] < 0 || ts.num_vary[1] > 64)
	//	return tileSet;
	//if (ts.short_side_len == 0)
	//	return tileSet;
	//if (ts.num_color[0] > 32 || ts.num_color[1] > 32 || ts.num_color[2] > 32 || ts.num_color[3] > 32)
	//	return tileSet;


	ts.tileData = data;
	ts.tdStride = stride;
	p.w = w;
	p.h = h;
	stbhw__get_template_info(p, ts);
	
	int ret = stbhw__process_template(p, ts);
	stbhw_get_all_indices(ts);
	return ts;
}

#if 1
_compute static int stbhw__choose_tile(const WangTile* list, const uint16_t* indices, int max_colors, int numVary, WorldgenPRNG& prng,
	signed char& a, signed char& b, signed char& c, signed char& d, signed char& e, signed char& f)
{
	uint16_t index = indices[stbhw_get_index_num(a, b, c, d, e, f)];
	uint8_t start = index >> 8;
	uint8_t stride = index & 0xff;

	int m = prng.NextU() % numVary;
	int i = start + m * stride;
	const WangTile& h = list[i];
	a = h.colors[0];
	b = h.colors[1];
	c = h.colors[2];
	d = h.colors[3];
	e = h.colors[4];
	f = h.colors[5];
	return i;
}
#else
// randomly choose a tile that fits constraints for a given spot, and update the constraints
_compute static int stbhw__choose_tile(const WangTile* list, const uint16_t* indices, int max_colors, int numlist, WorldgenPRNG& prng,
	signed char& a, signed char& b, signed char& c, signed char& d, signed char& e, signed char& f)
{
	int m = 1 << 30;
	for (int pass = 0; pass < 2; ++pass)
	{
		int n = 0;
		// pass #1:
		//   count number of variants that match this partial set of constraints
		// pass #2:
		//   stop on randomly selected match
		for (int i = 0; i < numlist; ++i)
		{
			const WangTile& h = list[i];
			printf("C %i %i %i %i %i %i\n", h.colors[0], h.colors[1], h.colors[2], h.colors[3], h.colors[4], h.colors[5]);
			if ((a < 0 || a == h.colors[0]) &&
				(b < 0 || b == h.colors[1]) &&
				(c < 0 || c == h.colors[2]) &&
				(d < 0 || d == h.colors[3]) &&
				(e < 0 || e == h.colors[4]) &&
				(f < 0 || f == h.colors[5]))
			{
				n += 1;
				if (n > m)
				{
					printf("\n");
					// use list[i]
					// update constraints to reflect what we placed
					a = h.colors[0];
					b = h.colors[1];
					c = h.colors[2];
					d = h.colors[3];
					e = h.colors[4];
					f = h.colors[5];
					return i;
				}
			}
		}
		printf("%i %i %i %i %i %i: %i %i\n", a, b, c, d, e, f, n, m);
		if (n == 0) {
			return -1;
		}
		m = prng.NextU() % n;
	}
	return -1;
}
#endif

_compute
static int stbhw__match(int x, int y, signed char (&c_color)[64][64])
{
	return c_color[y][x] == c_color[y + 1][x + 1];
}

_compute
static int stbhw__change_color(int old_color, int num_options, WorldgenPRNG& prng)
{
	int offset = 1 + prng.NextU() % (num_options - 1);
	return (old_color + offset) % num_options;
}

// generate a map that is w * h pixels (3-bytes each)
// returns 1 on success, 0 on error
_compute int stbhw_generate_image(WangTileIndex* output, const BiomeWangScope& scope, int w, int h, WorldgenPRNG& prng)
{
	signed char c_color[64][64];
	signed char v_color[64][64];
	signed char h_color[64][64];

	int sidelen = scope.ts.short_side_len;
	int xmax = (w / sidelen) + 6;
	int ymax = (h / sidelen) + 6;
	if (xmax > 64 || ymax > 64)
	{
		printf("STBHW_GENERATE_IMAGE: RAN OUT OF COLORS!\n");
		return 0;
	}

	int yIdx = -1;
	int xIdx = 0;
	int yWidth = scope.bSec.wang_w;

	if (scope.ts.is_corner)
	{
		// 2 1 3 1 0 0
		const int* cc = scope.ts.num_color;
		// num_vary 2 1

		for (int j = 0; j < ymax; ++j)
		{
			for (int i = 0; i < xmax; ++i)
			{
				int p = (i - j + 1) & 3; // corner type
				c_color[j][i] = prng.NextU() % cc[p];
			}
		}
		for (int j = 0; j < ymax - 3; ++j)
		{
			for (int i = 0; i < xmax - 3; ++i)
			{
				// int p = (i-j+1) & 3; // corner type   // unused, not sure what the intent was so commenting it out
				if (stbhw__match(i, j, c_color) && stbhw__match(i, j + 1, c_color) && stbhw__match(i, j + 2, c_color) && stbhw__match(i + 1, j, c_color) && stbhw__match(i + 1, j + 1, c_color) && stbhw__match(i + 1, j + 2, c_color))
				{
					int p = ((i + 1) - (j + 1) + 1) & 3;
					if (cc[p] > 1) {
						c_color[j + 1][i + 1] = stbhw__change_color(c_color[j + 1][i + 1], cc[p], prng);
					}
				}
				if (stbhw__match(i, j, c_color) && stbhw__match(i + 1, j, c_color) && stbhw__match(i + 2, j, c_color) && 
					stbhw__match(i, j + 1, c_color) && stbhw__match(i + 1, j + 1, c_color) && stbhw__match(i + 2, j + 1, c_color))
				{
					int p = ((i + 2) - (j + 1) + 1) & 3;
					if (cc[p] > 1) {
						c_color[j + 1][i + 2] = stbhw__change_color(c_color[j + 1][i + 2], cc[p], prng);
					}
				}
			}
		}

		int i = 0;
		for (int j = -1; yIdx < scope.bSec.wang_h; ++j)
		{
			int phase = (j & 3);
			if (phase == 0)
				i = 0;
			else
				i = phase - 4;
			for (;; i += 4)
			{
				xIdx = i;
				if (xIdx >= scope.bSec.wang_w)
					break;
				if (xIdx + 2 >= 0 && yIdx >= 0)
				{
					int ti = stbhw__choose_tile(
						scope.ts.hTiles, scope.ts.hIndices, scope.ts.max_colors , scope.ts.num_vary[0] * scope.ts.num_vary[1], prng,
						c_color[j + 2][i + 2], c_color[j + 2][i + 3], c_color[j + 2][i + 4],
						c_color[j + 3][i + 2], c_color[j + 3][i + 3], c_color[j + 3][i + 4]);
					if (ti == -1)
						return 0;
					if (xIdx >= 0) output[yIdx * yWidth + xIdx] = ti;
					if (xIdx + 1 >= 0) output[yIdx * yWidth + xIdx + 1] = ti | 0x8000;
				}
				xIdx += 3;
				if (xIdx < scope.bSec.wang_w)
				{
					int ti = stbhw__choose_tile(
						scope.ts.vTiles, scope.ts.vIndices, scope.ts.max_colors, scope.ts.num_vary[0] * scope.ts.num_vary[1], prng,
						c_color[j + 2][i + 5], c_color[j + 3][i + 5], c_color[j + 4][i + 5],
						c_color[j + 2][i + 6], c_color[j + 3][i + 6], c_color[j + 4][i + 6]);
					if (ti == -1)
						return 0;
					output[yIdx * yWidth + xIdx] = ti | 0x4000;
					output[(yIdx + 1) * yWidth + xIdx] = ti | 0xC000;
				}
			}
			yIdx++;
		}
	}
	else
	{
		// @TODO edge-color repetition reduction
		int i, j;
		cMemset(v_color, -1, sizeof(v_color));
		cMemset(h_color, -1, sizeof(h_color));

		for (j = -1; yIdx < scope.bSec.wang_h; ++j)
		{
			// a general herringbone row consists of:
			//    horizontal left block, the bottom of a previous vertical, the top of a new vertical
			int phase = (j & 3);
			// displace horizontally according to pattern
			if (phase == 0)
			{
				i = 0;
			}
			else
			{
				i = phase - 4;
			}
			for (;; i += 4)
			{
				xIdx = i;
				if (xIdx >= scope.bSec.wang_w)
					break;
				// horizontal left-block
				if (xIdx + 2 >= 0 && yIdx >= 0)
				{
					int ti = stbhw__choose_tile(
						scope.ts.hTiles, scope.ts.hIndices, scope.ts.max_colors, scope.ts.num_vary[0] * scope.ts.num_vary[1], prng,
						h_color[j + 2][i + 2], h_color[j + 2][i + 3],
						v_color[j + 2][i + 2], v_color[j + 2][i + 4],
						h_color[j + 3][i + 2], h_color[j + 3][i + 3]);
					if (ti == -1)
						return 0;
					if (xIdx >= 0) output[yIdx * yWidth + xIdx] = ti;
					if (xIdx + 1 >= 0) output[yIdx * yWidth + xIdx + 1] = ti | 0x8000;
				}
				xIdx += 2;
				// now we're at the end of a previous vertical one
				xIdx++;
				// now we're at the start of a new vertical one
				if (xIdx < scope.bSec.wang_w)
				{
					int ti = stbhw__choose_tile(
						scope.ts.vTiles, scope.ts.vIndices, scope.ts.max_colors, scope.ts.num_vary[0] * scope.ts.num_vary[1], prng,
						h_color[j + 2][i + 5],
						v_color[j + 2][i + 5], v_color[j + 2][i + 6],
						v_color[j + 3][i + 5], v_color[j + 3][i + 6],
						h_color[j + 4][i + 5]);
					if (ti == -1)
						return 0;
					if (yIdx >= 0)output[yIdx * yWidth + xIdx] = ti | 0x4000;
					output[(yIdx + 1) * yWidth + xIdx] = ti | 0xC000;
				}
			}
			yIdx++;
		}
	}
	return 1;
}
