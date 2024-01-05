#include "../platforms/platform_implementation.h"

#include "../include/noita_random.h"
#include "../include/worldgen_structs.h"

#include <cmath>
#include <cstdio>

struct WangProcess
{
	BiomeSpawnFunctions** funcs;
	WangConfig c;
	uint8_t* data;
	uint32_t* colors;
	int stride, w, h, sx, sy;
};

static void stbhw__get_template_info(WangProcess* p, WangTileset* ts)
{
	int size_x, size_y;
	int horz_count, vert_count;

	if (p->c.is_corner)
	{
		int horz_w = p->c.num_color[1] * p->c.num_color[2] * p->c.num_color[3] * p->c.num_vary_x;
		int horz_h = p->c.num_color[0] * p->c.num_color[1] * p->c.num_color[2] * p->c.num_vary_y;

		int vert_w = p->c.num_color[0] * p->c.num_color[3] * p->c.num_color[2] * p->c.num_vary_y;
		int vert_h = p->c.num_color[1] * p->c.num_color[0] * p->c.num_color[3] * p->c.num_vary_x;

		int horz_x = horz_w * (2 * p->c.short_side_len + 3);
		int horz_y = horz_h * (p->c.short_side_len + 3);

		int vert_x = vert_w * (p->c.short_side_len + 3);
		int vert_y = vert_h * (2 * p->c.short_side_len + 3);

		horz_count = horz_w * horz_h;
		vert_count = vert_w * vert_h;

		size_x = horz_x > vert_x ? horz_x : vert_x;
		size_y = 2 + horz_y + 2 + vert_y;
	}
	else
	{
		int horz_w = p->c.num_color[0] * p->c.num_color[1] * p->c.num_color[2] * p->c.num_vary_x;
		int horz_h = p->c.num_color[3] * p->c.num_color[4] * p->c.num_color[2] * p->c.num_vary_y;

		int vert_w = p->c.num_color[0] * p->c.num_color[5] * p->c.num_color[1] * p->c.num_vary_y;
		int vert_h = p->c.num_color[3] * p->c.num_color[4] * p->c.num_color[5] * p->c.num_vary_x;

		int horz_x = horz_w * (2 * p->c.short_side_len + 3);
		int horz_y = horz_h * (p->c.short_side_len + 3);

		int vert_x = vert_w * (p->c.short_side_len + 3);
		int vert_y = vert_h * (2 * p->c.short_side_len + 3);

		horz_count = horz_w * horz_h;
		vert_count = vert_w * vert_h;

		size_x = horz_x > vert_x ? horz_x : vert_x;
		size_y = 2 + horz_y + 2 + vert_y;
	}

	p->sx = size_x;
	p->sy = size_y;
	ts->maxH = horz_count;
	ts->maxV = vert_count;
}

static void stbhw__parse_h_rect(WangProcess* p, WangTileset* tileSet, int xpos, int ypos,
	char a, char b, char c, char d, char e, char f, int idx)
{
	int sIdx = 0;
	int len = p->c.short_side_len;
	WangTile* h = &tileSet->hTiles[idx];
	for (int i = 0; i < _WangTileMaxSpawns; i++)
		h->spawns[i] = { 0, 0, -1 };
	h->s[0] = a;
	h->s[1] = b;
	h->s[2] = c;
	h->s[3] = d;
	h->s[4] = e;
	h->s[5] = f;
	//printf("H %i: %i %i %i %i %i %i\n", idx, a, b, c, d, e, f);
	for (uint8_t j = 0; j < len; ++j)
		for (uint8_t i = 0; i < len * 2; ++i)
		{
			uint8_t* buf = (p->data + (ypos + j + 1) * p->stride + (xpos + i + 1) * 3);
			uint32_t color = (buf[0] << 16) | (buf[1] << 8) | buf[2];
			for (int16_t z = 0; z < p->funcs[0]->count; z++)
			{
				if (color == p->funcs[0]->funcs[z].color)
					h->spawns[sIdx++] = { i, j, z };
			}
			for (int16_t z = 0; z < p->funcs[1]->count; z++)
			{
				if (color == p->funcs[1]->funcs[z].color)
					h->spawns[sIdx++] = { i, j, (int16_t)(p->funcs[0]->count + z) };
			}
		}
	if (sIdx >= _WangTileMaxSpawns)
	{
		printf("H Tile %i: Ran out of spawns! %i of %i.\n", idx, sIdx, _WangTileMaxSpawns);
	}
	tileSet->numH++;
}

static void stbhw__parse_v_rect(WangProcess* p, WangTileset* tileSet, int xpos, int ypos,
	char a, char b, char c, char d, char e, char f, int idx)
{
	int sIdx = 0;
	int len = p->c.short_side_len;
	WangTile* h = &tileSet->vTiles[idx];
	for (int i = 0; i < _WangTileMaxSpawns; i++)
		h->spawns[i] = { 0, 0, -1 };
	h->s[0] = a;
	h->s[1] = b;
	h->s[2] = c;
	h->s[3] = d;
	h->s[4] = e;
	h->s[5] = f;
	//printf("V %i: %i %i %i %i %i %i\n", idx, a, b, c, d, e, f);
	for (uint8_t j = 0; j < len * 2; ++j)
		for (uint8_t i = 0; i < len; ++i)
		{
			uint8_t* buf = (p->data + (ypos + j + 1) * p->stride + (xpos + i + 1) * 3);
			uint32_t color = (buf[0] << 16) | (buf[1] << 8) | buf[2];
			for (int16_t z = 0; z < p->funcs[0]->count; z++)
			{
				if (color == p->funcs[0]->funcs[z].color)
					h->spawns[sIdx++] = { i, j, z };
			}
			for (int16_t z = 0; z < p->funcs[1]->count; z++)
			{
				if (color == p->funcs[1]->funcs[z].color)
					h->spawns[sIdx++] = { i, j, (int16_t)(p->funcs[0]->count + z) };
			}
		}
	if (sIdx >= _WangTileMaxSpawns)
	{
		printf("V Tile %i: Ran out of spawns! %i of %i.\n", idx, sIdx, _WangTileMaxSpawns);
	}
	tileSet->numV++;
}

static void stbhw__process_h_row(WangProcess* p, WangTileset* tileSet,
	int xpos, int ypos,
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
								stbhw__parse_h_rect(p, tileSet, xpos, ypos, a, b, c, d, e, f, i++);
								xpos += 2 * p->c.short_side_len + 3;
							}
}

static void stbhw__process_v_row(WangProcess* p, WangTileset* tileSet,
	int xpos, int ypos,
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
								stbhw__parse_v_rect(p, tileSet, xpos, ypos, a, b, c, d, e, f, i++);
								xpos += p->c.short_side_len + 3;
							}
}

static int stbhw__process_template(WangProcess* p, WangTileset* tileSet)
{
	int i, j, k, q, ypos;
	WangConfig* c = &p->c;

	int vi = 0;
	int hi = 0;

	printf("%i %i [%i %i %i %i]\n", c->num_vary_x, c->num_vary_y, c->num_color[0], c->num_color[1], c->num_color[2], c->num_color[3]);

	if (p->c.is_corner)
	{
		ypos = 2;
		for (k = 0; k < c->num_color[2]; ++k)
		{
			for (j = 0; j < c->num_color[1]; ++j)
			{
				for (i = 0; i < c->num_color[0]; ++i)
				{
					for (q = 0; q < c->num_vary_y; ++q)
					{
						stbhw__process_h_row(p, tileSet, 0, ypos,
							0, c->num_color[1] - 1, 0, c->num_color[2] - 1, 0, c->num_color[3] - 1,
							i, i, j, j, k, k,
							c->num_vary_x, hi);
						ypos += c->short_side_len + 3;
					}
				}
			}
		}
		ypos += 2;
		for (k = 0; k < c->num_color[3]; ++k)
		{
			for (j = 0; j < c->num_color[0]; ++j)
			{
				for (i = 0; i < c->num_color[1]; ++i)
				{
					for (q = 0; q < c->num_vary_x; ++q)
					{
						stbhw__process_v_row(p, tileSet, 0, ypos,
							0, c->num_color[0] - 1, 0, c->num_color[3] - 1, 0, c->num_color[2] - 1,
							i, i, j, j, k, k,
							c->num_vary_y, vi);
						ypos += (c->short_side_len * 2) + 3;
					}
				}
			}
		}
	}
	else
	{
		ypos = 2;
		for (k = 0; k < c->num_color[3]; ++k)
		{
			for (j = 0; j < c->num_color[4]; ++j)
			{
				for (i = 0; i < c->num_color[2]; ++i)
				{
					for (q = 0; q < c->num_vary_y; ++q)
					{
						stbhw__process_h_row(p, tileSet, 0, ypos,
							0, c->num_color[2] - 1, k, k,
							0, c->num_color[1] - 1, j, j,
							0, c->num_color[0] - 1, i, i,
							c->num_vary_x, hi);
						ypos += c->short_side_len + 3;
					}
				}
			}
		}
		ypos += 2;
		for (k = 0; k < c->num_color[3]; ++k)
		{
			for (j = 0; j < c->num_color[4]; ++j)
			{
				for (i = 0; i < c->num_color[5]; ++i)
				{
					for (q = 0; q < c->num_vary_x; ++q)
					{
						stbhw__process_v_row(p, tileSet, 0, ypos,
							0, c->num_color[0] - 1, i, i,
							0, c->num_color[1] - 1, j, j,
							0, c->num_color[5] - 1, k, k,
							c->num_vary_y, vi);
						ypos += (c->short_side_len * 2) + 3;
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
		WangTile* h = &list[i];
		if ((a < 0 || a == h->s[0]) &&
			(b < 0 || b == h->s[1]) &&
			(c < 0 || c == h->s[2]) &&
			(d < 0 || d == h->s[3]) &&
			(e < 0 || e == h->s[4]) &&
			(f < 0 || f == h->s[5]))
		{
			//printf("%i ", i);
			if (first < 0) first = i;
			else if (second < 0) second = i;
		}
	}
	//printf("\n");
	if (second < 0)
	{
		//printf("NO TILE\n");
		return { 0, 0 };
	}
	return { first, second - first };
}

static void stbhw_get_all_indices(WangTileset* tileSet)
{
	for (char a = 0; a < 2; a++)
		for (char b = 0; b < 2; b++)
			for (char c = 0; c < 2; c++)
				for (char d = 0; d < 2; d++)
					for (char e = 0; e < 2; e++)
						for (char f = 0; f < 2; f++)
						{
							Vec2i h = stbhw_get_index_stride(tileSet->hTiles, tileSet->numH, a, b, c, d, e, f);
							Vec2i v = stbhw_get_index_stride(tileSet->vTiles, tileSet->numV, a, b, c, d, e, f);
							tileSet->hIndices[32 * a + 16 * b + 8 * c + 4 * d + 2 * e + f] = ((h.x & 0xff) << 8 | (h.y & 0xff));
							tileSet->vIndices[32 * a + 16 * b + 8 * c + 4 * d + 2 * e + f] = ((v.x & 0xff) << 8 | (v.y & 0xff));
						}
}

int stbhw_build_tileset_from_image(uint8_t* data, WangTileset* tileSet, BiomeSpawnFunctions** funcs, int stride, int w, int h)
{
	uint8_t header[9];
	WangProcess p = {};
	p.funcs = funcs;

	for (int i = 0; i < 9; ++i)
	{
		header[i] = data[w * 3 - 1 - i] ^ (i * 55);
	}

	for (int i = 0; i < 6; i++) p.c.num_color[i] = 0;

	// extract header info
	if (header[7] == 0xc0)
	{
		// corner-type
		p.c.is_corner = 1;
		for (int i = 0; i < 4; ++i)
			p.c.num_color[i] = header[i];
		p.c.num_vary_x = header[4];
		p.c.num_vary_y = header[5];
		p.c.short_side_len = header[6];
	}
	else
	{
		p.c.is_corner = 0;
		// edge-type
		for (int i = 0; i < 6; ++i)
			p.c.num_color[i] = header[i];
		p.c.num_vary_x = header[6];
		p.c.num_vary_y = header[7];
		p.c.short_side_len = header[8];
	}

	//if (p.c.num_vary_x < 0 || p.c.num_vary_x > 64 || p.c.num_vary_y < 0 || p.c.num_vary_y > 64)
	//	return tileSet;
	//if (p.c.short_side_len == 0)
	//	return tileSet;
	//if (p.c.num_color[0] > 32 || p.c.num_color[1] > 32 || p.c.num_color[2] > 32 || p.c.num_color[3] > 32)
	//	return tileSet;

	stbhw__get_template_info(&p, tileSet);

	tileSet->num_vary[0] = p.c.num_vary_x;
	tileSet->num_vary[1] = p.c.num_vary_y;
	for (int i = 0; i < 6; i++) tileSet->num_color[i] = p.c.num_color[i];
	tileSet->numH = tileSet->numV = 0;
	tileSet->short_side_len = p.c.short_side_len;
	tileSet->is_corner = p.c.is_corner;

	p.data = data;
	p.stride = stride;
	p.w = w;
	p.h = h;

	int ret = stbhw__process_template(&p, tileSet);
	stbhw_get_all_indices(tileSet);
	return ret;
}


#if 1
_compute static int stbhw__choose_tile(WangTile* list, uint16_t* indices, int numVary, WorldgenPRNG* prng,
	signed char* a, signed char* b, signed char* c, signed char* d, signed char* e, signed char* f)
{
	uint16_t index = indices[32 * *a + 16 * *b + 8 * *c + 4 * *d + 2 * *e + *f];
	uint8_t start = index >> 8;
	uint8_t stride = index & 0xff;

	int m = prng->NextU() % numVary;
	int i = start + m * stride;
	WangTile* h = &list[i];
	*a = h->s[0];
	*b = h->s[1];
	*c = h->s[2];
	*d = h->s[3];
	*e = h->s[4];
	*f = h->s[5];
	return i;
}
#else
// randomly choose a tile that fits constraints for a given spot, and update the constraints
_compute static int stbhw__choose_tile(WangTile* list, int numlist,
	signed char* a, signed char* b, signed char* c,
	signed char* d, signed char* e, signed char* f,
	WorldgenPRNG* prng)
{
	printf("%i %i %i %i %i %i:\n", *a, *b, *c, *d, *e, *f);
	int i, n, m = 1 << 30, pass;
	for (pass = 0; pass < 2; ++pass)
	{
		n = 0;
		// pass #1:
		//   count number of variants that match this partial set of constraints
		// pass #2:
		//   stop on randomly selected match
		for (i = 0; i < numlist; ++i)
		{
			WangTile* h = &list[i];
			if ((*a < 0 || *a == h->s[0]) &&
				(*b < 0 || *b == h->s[1]) &&
				(*c < 0 || *c == h->s[2]) &&
				(*d < 0 || *d == h->s[3]) &&
				(*e < 0 || *e == h->s[4]) &&
				(*f < 0 || *f == h->s[5]))
			{
				n += 1;
				printf("%i ", i);
				if (n > m)
				{
					printf("\n");
					// use list[i]
					// update constraints to reflect what we placed
					*a = h->s[0];
					*b = h->s[1];
					*c = h->s[2];
					*d = h->s[3];
					*e = h->s[4];
					*f = h->s[5];
					return i;
				}
			}
		}
		if (n == 0)
		{
			printf("NO TILE\n");
			return -1;
		}
		printf("-- %i\n", n);
		m = prng->NextU() % n;
	}
	return -1;
}
#endif

_compute
static int stbhw__match(int x, int y, signed char c_color[64][64])
{
	return c_color[y][x] == c_color[y + 1][x + 1];
}

_compute
static int stbhw__change_color(int old_color, int num_options, WorldgenPRNG* prng)
{
	int offset = 1 + prng->NextU() % (num_options - 1);
	return (old_color + offset) % num_options;
}

// generate a map that is w * h pixels (3-bytes each)
// returns 1 on success, 0 on error
_compute int stbhw_generate_image(WangTileIndex* output, WangTileset* tileSet, int w, int h, WorldgenPRNG* prng)
{
	signed char c_color[64][64];
	signed char v_color[64][64];
	signed char h_color[64][64];

	int sidelen = tileSet->short_side_len;
	int xmax = (w / sidelen) + 6;
	int ymax = (h / sidelen) + 6;
	if (xmax > 64 || ymax > 64)
	{
		printf("STBHW_GENERATE_IMAGE: RAN OUT OF COLORS!\n");
		return 0;
	}

	int yIdx = 0;
	int xIdx = 0;
	int yWidth = (w + sidelen - 1) / sidelen;

	if (tileSet->is_corner)
	{
		int i, j, ypos;
		int* cc = tileSet->num_color;

		for (j = 0; j < ymax; ++j)
		{
			for (i = 0; i < xmax; ++i)
			{
				int p = (i - j + 1) & 3; // corner type
				c_color[j][i] = prng->NextU() % cc[p];
			}
		}
		// now go back through and make sure we don't have adjancent 3x2 vertices that are identical,
		// to avoid really obvious repetition (which happens easily with extreme weights)
		for (j = 0; j < ymax - 3; ++j)
		{
			for (i = 0; i < xmax - 3; ++i)
			{
				// int p = (i-j+1) & 3; // corner type   // unused, not sure what the intent was so commenting it out
				if (stbhw__match(i, j, c_color) && stbhw__match(i, j + 1, c_color) && stbhw__match(i, j + 2, c_color) && stbhw__match(i + 1, j, c_color) && stbhw__match(i + 1, j + 1, c_color) && stbhw__match(i + 1, j + 2, c_color))
				{
					int p = ((i + 1) - (j + 1) + 1) & 3;
					if (cc[p] > 1)
						c_color[j + 1][i + 1] = stbhw__change_color(c_color[j + 1][i + 1], cc[p], prng);
				}
				if (stbhw__match(i, j, c_color) && stbhw__match(i + 1, j, c_color) && stbhw__match(i + 2, j, c_color) && stbhw__match(i, j + 1, c_color) && stbhw__match(i + 1, j + 1, c_color) && stbhw__match(i + 2, j + 1, c_color))
				{
					int p = ((i + 2) - (j + 1) + 1) & 3;
					if (cc[p] > 1)
						c_color[j + 1][i + 2] = stbhw__change_color(c_color[j + 1][i + 2], cc[p], prng);
				}
			}
		}

		ypos = -1 * sidelen;
		for (j = -1; ypos < h; ++j)
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
				int xpos = i * sidelen;
				xIdx = i;
				if (xpos >= w)
					break;
				// horizontal left-block
				if (xpos + sidelen * 2 >= 0 && ypos >= 0)
				{
					int ti = stbhw__choose_tile(
						tileSet->hTiles, tileSet->hIndices, tileSet->num_vary[0] * tileSet->num_vary[1], prng,
						&c_color[j + 2][i + 2], &c_color[j + 2][i + 3], &c_color[j + 2][i + 4],
						&c_color[j + 3][i + 2], &c_color[j + 3][i + 3], &c_color[j + 3][i + 4]);
					if (ti == -1)
						return 0;
					output[yIdx * yWidth + xIdx] = ti;
					output[yIdx * yWidth + xIdx + 1] = 0x8000;
				}
				xpos += sidelen * 2;
				xIdx += 2;
				// now we're at the end of a previous vertical one
				xpos += sidelen;
				xIdx++;
				// now we're at the start of a new vertical one
				if (xpos < w)
				{
					int ti = stbhw__choose_tile(
						tileSet->vTiles, tileSet->vIndices, tileSet->num_vary[0] * tileSet->num_vary[1], prng,
						&c_color[j + 2][i + 5], &c_color[j + 3][i + 5], &c_color[j + 4][i + 5],
						&c_color[j + 2][i + 6], &c_color[j + 3][i + 6], &c_color[j + 4][i + 6]);
					if (ti == -1)
						return 0;
					output[yIdx * yWidth + xIdx] = ti | 0x4000;
					output[(yIdx + 1) * yWidth + xIdx] = 0xC000;
				}
			}
			ypos += sidelen;
			yIdx++;
		}
	}
	else
	{
		// @TODO edge-color repetition reduction
		int i, j, ypos;
		cMemset(v_color, -1, sizeof(v_color));
		cMemset(h_color, -1, sizeof(h_color));

		ypos = -1 * sidelen;
		for (j = -1; ypos < h; ++j)
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
				int xpos = i * sidelen;
				xIdx = i;
				if (xpos >= w)
					break;
				// horizontal left-block
				if (xpos + sidelen * 2 >= 0 && ypos >= 0)
				{
					int ti = stbhw__choose_tile(
						tileSet->hTiles, tileSet->hIndices, tileSet->num_vary[0] * tileSet->num_vary[1], prng,
						&h_color[j + 2][i + 2], &h_color[j + 2][i + 3],
						&v_color[j + 2][i + 2], &v_color[j + 2][i + 4],
						&h_color[j + 3][i + 2], &h_color[j + 3][i + 3]);
					if (ti == -1)
						return 0;
					output[yIdx * yWidth + xIdx] = ti;
					output[yIdx * yWidth + xIdx + 1] = 0x8000;
				}
				xpos += sidelen * 2;
				xIdx += 2;
				// now we're at the end of a previous vertical one
				xpos += sidelen;
				xIdx++;
				// now we're at the start of a new vertical one
				if (xpos < w)
				{
					int ti = stbhw__choose_tile(
						tileSet->vTiles, tileSet->vIndices, tileSet->num_vary[0] * tileSet->num_vary[1], prng,
						&h_color[j + 2][i + 5],
						&v_color[j + 2][i + 5], &v_color[j + 2][i + 6],
						&v_color[j + 3][i + 5], &v_color[j + 3][i + 6],
						&h_color[j + 4][i + 5]);
					if (ti == -1)
						return 0;
					output[yIdx * yWidth + xIdx] = ti | 0x4000;
					output[(yIdx + 1) * yWidth + xIdx] = 0xC000;
				}
			}
			ypos += sidelen;
			yIdx++;
		}
	}
	return 1;
}
