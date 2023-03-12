#pragma once

#include "noita_random.h"
#include <iostream>

/* stbhw - v0.7 -  http://nothings.org/gamedev/herringbone
   Herringbone Wang Tile Generator - Sean Barrett 2014 - public domain

== LICENSE ==============================

This software is dual-licensed to the public domain and under the following
license: you are granted a perpetual, irrevocable license to copy, modify,
publish, and distribute this file as you see fit.
*/

#define INCLUDE_STB_HWANG_H

typedef struct
{
    signed char a, b, c, d, e, f;

    unsigned char pixels[1];
} stbhw_tile;

struct stbhw_tileset
{
    int is_corner;
    int num_color[6]; // number of colors for each of 6 edge types or 4 corner types
    int short_side_len;
    stbhw_tile** h_tiles;
    stbhw_tile** v_tiles;
    int num_h_tiles, max_h_tiles;
    int num_v_tiles, max_v_tiles;
};

__device__ stbhw_tileset dTileSet;

typedef struct
{
    int is_corner;      // using corner colors or edge colors?
    int short_side_len; // rectangles is 2n x n, n = short_side_len
    int num_color[6];   // see below diagram for meaning of the index to this;
                        // 6 values if edge (!is_corner), 4 values if is_corner
                        // legal numbers: 1..8 if edge, 1..4 if is_corner
    int num_vary_x;     // additional number of variations along x axis in the template
    int num_vary_y;     // additional number of variations along y axis in the template
    int corner_type_color_template[4][4];
    // if corner_type_color_template[s][t] is non-zero, then any
    // corner of type s generated as color t will get a little
    // corner sample markup in the template image data

} stbhw_config;

__device__ int stbhw_build_tileset_from_image(unsigned char* pixels, int stride_in_bytes, int w, int h);

__device__ void stbhw_free_tileset();

__device__ int stbhw_generate_image(unsigned char* pixels, int stride_in_bytes, int w, int h, uint(*getRandom)(NollaPrng*), NollaPrng* prng);

__device__ void stbhw_get_template_size(stbhw_config* c, int* w, int* h);

__device__ int stbhw_make_template(stbhw_config* c, unsigned char* data, int w, int h, int stride_in_bytes);

#include <string.h> // memcpy
#include <stdlib.h> // malloc

// map size
#ifndef STB_HBWANG_MAX_X
#define STB_HBWANG_MAX_X 100
#endif

#ifndef STB_HBWANG_MAX_Y
#define STB_HBWANG_MAX_Y 100
#endif

typedef struct stbhw__process
{
    stbhw_config* c;
    unsigned char* data;
    int stride, w, h;
} stbhw__process;

__device__
static void stbhw__parse_h_rect(stbhw__process* p, int xpos, int ypos,
    int a, int b, int c, int d, int e, int f)
{
    int len = p->c->short_side_len;
    stbhw_tile* h = (stbhw_tile*)malloc(sizeof(*h) - 1 + 3 * (len * 2) * len);
    int i, j;
    ++xpos;
    ++ypos;
    h->a = a, h->b = b, h->c = c, h->d = d, h->e = e, h->f = f;
    for (j = 0; j < len; ++j)
        for (i = 0; i < len * 2; ++i)
            memcpy(h->pixels + j * (3 * len * 2) + i * 3, p->data + (ypos + j) * p->stride + (xpos + i) * 3, 3);
    dTileSet.h_tiles[dTileSet.num_h_tiles++] = h;
}

__device__
static void stbhw__parse_v_rect(stbhw__process* p, int xpos, int ypos,
    int a, int b, int c, int d, int e, int f)
{
    int len = p->c->short_side_len;
    stbhw_tile* h = (stbhw_tile*)malloc(sizeof(*h) - 1 + 3 * (len * 2) * len);
    int i, j;
    ++xpos;
    ++ypos;
    h->a = a, h->b = b, h->c = c, h->d = d, h->e = e, h->f = f;
    for (j = 0; j < len * 2; ++j)
        for (i = 0; i < len; ++i)
            memcpy(h->pixels + j * (3 * len) + i * 3, p->data + (ypos + j) * p->stride + (xpos + i) * 3, 3);
    dTileSet.v_tiles[dTileSet.num_v_tiles++] = h;
}

__device__
static void stbhw__process_h_row(stbhw__process* p,
    int xpos, int ypos,
    int a0, int a1,
    int b0, int b1,
    int c0, int c1,
    int d0, int d1,
    int e0, int e1,
    int f0, int f1,
    int variants)
{
    int a, b, c, d, e, f, v;

    for (v = 0; v < variants; ++v)
        for (f = f0; f <= f1; ++f)
            for (e = e0; e <= e1; ++e)
                for (d = d0; d <= d1; ++d)
                    for (c = c0; c <= c1; ++c)
                        for (b = b0; b <= b1; ++b)
                            for (a = a0; a <= a1; ++a)
                            {
                                stbhw__parse_h_rect(p, xpos, ypos, a, b, c, d, e, f);
                                xpos += 2 * p->c->short_side_len + 3;
                            }
}

__device__
static void stbhw__process_v_row(stbhw__process* p,
    int xpos, int ypos,
    int a0, int a1,
    int b0, int b1,
    int c0, int c1,
    int d0, int d1,
    int e0, int e1,
    int f0, int f1,
    int variants)
{
    int a, b, c, d, e, f, v;

    for (v = 0; v < variants; ++v)
        for (f = f0; f <= f1; ++f)
            for (e = e0; e <= e1; ++e)
                for (d = d0; d <= d1; ++d)
                    for (c = c0; c <= c1; ++c)
                        for (b = b0; b <= b1; ++b)
                            for (a = a0; a <= a1; ++a)
                            {
                                stbhw__parse_v_rect(p, xpos, ypos, a, b, c, d, e, f);
                                xpos += p->c->short_side_len + 3;
                            }
}

__device__
static void stbhw__get_template_info(stbhw_config* c, int* w, int* h, int* h_count, int* v_count)
{
    int size_x, size_y;
    int horz_count, vert_count;

    if (c->is_corner)
    {
        int horz_w = c->num_color[1] * c->num_color[2] * c->num_color[3] * c->num_vary_x;
        int horz_h = c->num_color[0] * c->num_color[1] * c->num_color[2] * c->num_vary_y;

        int vert_w = c->num_color[0] * c->num_color[3] * c->num_color[2] * c->num_vary_y;
        int vert_h = c->num_color[1] * c->num_color[0] * c->num_color[3] * c->num_vary_x;

        int horz_x = horz_w * (2 * c->short_side_len + 3);
        int horz_y = horz_h * (c->short_side_len + 3);

        int vert_x = vert_w * (c->short_side_len + 3);
        int vert_y = vert_h * (2 * c->short_side_len + 3);

        horz_count = horz_w * horz_h;
        vert_count = vert_w * vert_h;

        size_x = horz_x > vert_x ? horz_x : vert_x;
        size_y = 2 + horz_y + 2 + vert_y;
    }
    else
    {
        int horz_w = c->num_color[0] * c->num_color[1] * c->num_color[2] * c->num_vary_x;
        int horz_h = c->num_color[3] * c->num_color[4] * c->num_color[2] * c->num_vary_y;

        int vert_w = c->num_color[0] * c->num_color[5] * c->num_color[1] * c->num_vary_y;
        int vert_h = c->num_color[3] * c->num_color[4] * c->num_color[5] * c->num_vary_x;

        int horz_x = horz_w * (2 * c->short_side_len + 3);
        int horz_y = horz_h * (c->short_side_len + 3);

        int vert_x = vert_w * (c->short_side_len + 3);
        int vert_y = vert_h * (2 * c->short_side_len + 3);

        horz_count = horz_w * horz_h;
        vert_count = vert_w * vert_h;

        size_x = horz_x > vert_x ? horz_x : vert_x;
        size_y = 2 + horz_y + 2 + vert_y;
    }
    if (w)
        *w = size_x;
    if (h)
        *h = size_y;
    if (h_count)
        *h_count = horz_count;
    if (v_count)
        *v_count = vert_count;
}

__device__ void stbhw_get_template_size(stbhw_config* c, int* w, int* h)
{
    stbhw__get_template_info(c, w, h, NULL, NULL);
}

__device__
static int stbhw__process_template(stbhw__process* p)
{
    int i, j, k, q, ypos;
    int size_x, size_y;
    stbhw_config* c = p->c;

    stbhw__get_template_info(c, &size_x, &size_y, NULL, NULL);

    if (c->is_corner)
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
                        stbhw__process_h_row(p, 0, ypos,
                            0, c->num_color[1] - 1, 0, c->num_color[2] - 1, 0, c->num_color[3] - 1,
                            i, i, j, j, k, k,
                            c->num_vary_x);
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
                        stbhw__process_v_row(p, 0, ypos,
                            0, c->num_color[0] - 1, 0, c->num_color[3] - 1, 0, c->num_color[2] - 1,
                            i, i, j, j, k, k,
                            c->num_vary_y);
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
                        stbhw__process_h_row(p, 0, ypos,
                            0, c->num_color[2] - 1, k, k,
                            0, c->num_color[1] - 1, j, j,
                            0, c->num_color[0] - 1, i, i,
                            c->num_vary_x);
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
                        stbhw__process_v_row(p, 0, ypos,
                            0, c->num_color[0] - 1, i, i,
                            0, c->num_color[1] - 1, j, j,
                            0, c->num_color[5] - 1, k, k,
                            c->num_vary_y);
                        ypos += (c->short_side_len * 2) + 3;
                    }
                }
            }
        }
    }
    return 1;
}

__device__
static void stbhw__draw_pixel(unsigned char* output, int stride, int x, int y, unsigned char c[3])
{
    memcpy(output + y * stride + x * 3, c, 3);
}

__device__
static void stbhw__draw_h_tile(unsigned char* output, int stride, int xmax, int ymax, int x, int y, stbhw_tile* h, int sz)
{
    int i, j;
    for (j = 0; j < sz; ++j)
        if (y + j >= 0 && y + j < ymax)
            for (i = 0; i < sz * 2; ++i)
                if (x + i >= 0 && x + i < xmax)
                    stbhw__draw_pixel(output, stride, x + i, y + j, &h->pixels[(j * sz * 2 + i) * 3]);
}

__device__
static void stbhw__draw_v_tile(unsigned char* output, int stride, int xmax, int ymax, int x, int y, stbhw_tile* h, int sz)
{
    int i, j;
    for (j = 0; j < sz * 2; ++j)
        if (y + j >= 0 && y + j < ymax)
            for (i = 0; i < sz; ++i)
                if (x + i >= 0 && x + i < xmax)
                    stbhw__draw_pixel(output, stride, x + i, y + j, &h->pixels[(j * sz + i) * 3]);
}

// randomly choose a tile that fits constraints for a given spot, and update the constraints
__device__
static stbhw_tile* stbhw__choose_tile(stbhw_tile** list, int numlist,
    signed char* a, signed char* b, signed char* c,
    signed char* d, signed char* e, signed char* f,
    uint(*getRandom)(NollaPrng*), NollaPrng* prng)
{
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
            stbhw_tile* h = list[i];
            if ((*a < 0 || *a == h->a) &&
                (*b < 0 || *b == h->b) &&
                (*c < 0 || *c == h->c) &&
                (*d < 0 || *d == h->d) &&
                (*e < 0 || *e == h->e) &&
                (*f < 0 || *f == h->f))
            {
                n += 1;
                if (n > m)
                {
                    // use list[i]
                    // update constraints to reflect what we placed
                    *a = h->a;
                    *b = h->b;
                    *c = h->c;
                    *d = h->d;
                    *e = h->e;
                    *f = h->f;
                    return h;
                }
            }
        }
        if (n == 0)
        {
            printf("NO TILE\n");
            return NULL;
        }
        m = getRandom(prng) % n;
    }
    return NULL;
}

__device__
static int stbhw__match(int x, int y, signed char c_color[STB_HBWANG_MAX_Y + 6][STB_HBWANG_MAX_X + 6])
{
    return c_color[y][x] == c_color[y + 1][x + 1];
}

__device__
static int stbhw__change_color(int old_color, int num_options, uint(*getRandom)(NollaPrng*), NollaPrng* prng)
{
    int offset = 1 + getRandom(prng) % (num_options - 1);
    return (old_color + offset) % num_options;
}

// generate a map that is w * h pixels (3-bytes each)
// returns 1 on success, 0 on error
__device__
int stbhw_generate_image(unsigned char* output, int stride, int w, int h, uint(*getRandom)(NollaPrng*), NollaPrng* prng)
{
    signed char c_color[STB_HBWANG_MAX_Y + 6][STB_HBWANG_MAX_X + 6];
    signed char v_color[STB_HBWANG_MAX_Y + 6][STB_HBWANG_MAX_X + 5];
    signed char h_color[STB_HBWANG_MAX_Y + 5][STB_HBWANG_MAX_X + 6];

    int sidelen = dTileSet.short_side_len;
    int xmax = (w / sidelen) + 6;
    int ymax = (h / sidelen) + 6;
    if (xmax > STB_HBWANG_MAX_X + 6 || ymax > STB_HBWANG_MAX_Y + 6)
    {
        return 0;
    }

    if (dTileSet.is_corner)
    {
        int i, j, ypos;
        int* cc = dTileSet.num_color;

        for (j = 0; j < ymax; ++j)
        {
            for (i = 0; i < xmax; ++i)
            {
                int p = (i - j + 1) & 3; // corner type
                c_color[j][i] = getRandom(prng) % cc[p];
            }
        }
#ifndef STB_HBWANG_NO_REPITITION_REDUCTION
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
                        c_color[j + 1][i + 1] = stbhw__change_color(c_color[j + 1][i + 1], cc[p], getRandom, prng);
                }
                if (stbhw__match(i, j, c_color) && stbhw__match(i + 1, j, c_color) && stbhw__match(i + 2, j, c_color) && stbhw__match(i, j + 1, c_color) && stbhw__match(i + 1, j + 1, c_color) && stbhw__match(i + 2, j + 1, c_color))
                {
                    int p = ((i + 2) - (j + 1) + 1) & 3;
                    if (cc[p] > 1)
                        c_color[j + 1][i + 2] = stbhw__change_color(c_color[j + 1][i + 2], cc[p], getRandom, prng);
                }
            }
        }
#endif

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
                if (xpos >= w)
                    break;
                // horizontal left-block
                if (xpos + sidelen * 2 >= 0 && ypos >= 0)
                {
                    stbhw_tile* t = stbhw__choose_tile(
                        dTileSet.h_tiles, dTileSet.num_h_tiles,
                        &c_color[j + 2][i + 2], &c_color[j + 2][i + 3], &c_color[j + 2][i + 4],
                        &c_color[j + 3][i + 2], &c_color[j + 3][i + 3], &c_color[j + 3][i + 4],
                        getRandom, prng);
                    if (t == NULL)
                        return 0;
                    stbhw__draw_h_tile(output, stride, w, h, xpos, ypos, t, sidelen);
                }
                xpos += sidelen * 2;
                // now we're at the end of a previous vertical one
                xpos += sidelen;
                // now we're at the start of a new vertical one
                if (xpos < w)
                {
                    stbhw_tile* t = stbhw__choose_tile(
                        dTileSet.v_tiles, dTileSet.num_v_tiles,
                        &c_color[j + 2][i + 5], &c_color[j + 3][i + 5], &c_color[j + 4][i + 5],
                        &c_color[j + 2][i + 6], &c_color[j + 3][i + 6], &c_color[j + 4][i + 6],
                        getRandom, prng);
                    if (t == NULL)
                        return 0;
                    stbhw__draw_v_tile(output, stride, w, h, xpos, ypos, t, sidelen);
                }
            }
            ypos += sidelen;
        }
    }
    else
    {
        // @TODO edge-color repetition reduction
        int i, j, ypos;
        memset(v_color, -1, sizeof(v_color));
        memset(h_color, -1, sizeof(h_color));

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
                if (xpos >= w)
                    break;
                // horizontal left-block
                if (xpos + sidelen * 2 >= 0 && ypos >= 0)
                {
                    stbhw_tile* t = stbhw__choose_tile(
                        dTileSet.h_tiles, dTileSet.num_h_tiles,
                        &h_color[j + 2][i + 2], &h_color[j + 2][i + 3],
                        &v_color[j + 2][i + 2], &v_color[j + 2][i + 4],
                        &h_color[j + 3][i + 2], &h_color[j + 3][i + 3],
                        getRandom, prng);
                    if (t == NULL)
                        return 0;
                    stbhw__draw_h_tile(output, stride, w, h, xpos, ypos, t, sidelen);
                }
                xpos += sidelen * 2;
                // now we're at the end of a previous vertical one
                xpos += sidelen;
                // now we're at the start of a new vertical one
                if (xpos < w)
                {
                    stbhw_tile* t = stbhw__choose_tile(
                        dTileSet.v_tiles, dTileSet.num_v_tiles,
                        &h_color[j + 2][i + 5],
                        &v_color[j + 2][i + 5], &v_color[j + 2][i + 6],
                        &v_color[j + 3][i + 5], &v_color[j + 3][i + 6],
                        &h_color[j + 4][i + 5],
                        getRandom, prng);
                    if (t == NULL)
                        return 0;
                    stbhw__draw_v_tile(output, stride, w, h, xpos, ypos, t, sidelen);
                }
            }
            ypos += sidelen;
        }
    }
    return 1;
}

__device__
int stbhw_build_tileset_from_image(unsigned char* data, int stride, int w, int h)
{
    int i, h_count, v_count;
    unsigned char header[9];
    stbhw_config c = { 0 };
    stbhw__process p = { 0 };

    // extract binary header

    // remove encoding that makes it more visually obvious it encodes actual data
    for (i = 0; i < 9; ++i) {
        header[i] = data[w * 3 - 1 - i] ^ (i * 55);
    }

    // extract header info
    if (header[7] == 0xc0)
    {
        // corner-type
        c.is_corner = 1;
        for (i = 0; i < 4; ++i)
            c.num_color[i] = header[i];
        c.num_vary_x = header[4];
        c.num_vary_y = header[5];
        c.short_side_len = header[6];
    }
    else
    {
        c.is_corner = 0;
        // edge-type
        for (i = 0; i < 6; ++i)
            c.num_color[i] = header[i];
        c.num_vary_x = header[6];
        c.num_vary_y = header[7];
        c.short_side_len = header[8];
    }


    if (c.num_vary_x < 0 || c.num_vary_x > 64 || c.num_vary_y < 0 || c.num_vary_y > 64)
        return 0;
    if (c.short_side_len == 0)
        return 0;
    if (c.num_color[0] > 32 || c.num_color[1] > 32 || c.num_color[2] > 32 || c.num_color[3] > 32)
        return 0;

    stbhw__get_template_info(&c, NULL, NULL, &h_count, &v_count);

    dTileSet.is_corner = c.is_corner;
    dTileSet.short_side_len = c.short_side_len;
    memcpy(dTileSet.num_color, c.num_color, sizeof(dTileSet.num_color));

    dTileSet.max_h_tiles = h_count;
    dTileSet.max_v_tiles = v_count;

    dTileSet.num_h_tiles = dTileSet.num_v_tiles = 0;

    dTileSet.h_tiles = (stbhw_tile**)malloc(sizeof(*dTileSet.h_tiles) * h_count);
    dTileSet.v_tiles = (stbhw_tile**)malloc(sizeof(*dTileSet.v_tiles) * v_count);

    p.data = data;
    p.stride = stride;
    p.w = w;
    p.h = h;
    p.c = &c;

    // load all the tiles out of the image
    return stbhw__process_template(&p);
}

__device__
void stbhw_free_tileset()
{
    int i;
    for (i = 0; i < dTileSet.num_h_tiles; ++i)
        free(dTileSet.h_tiles[i]);
    for (i = 0; i < dTileSet.num_v_tiles; ++i)
        free(dTileSet.v_tiles[i]);
    free(dTileSet.h_tiles);
    free(dTileSet.v_tiles);
    dTileSet.h_tiles = NULL;
    dTileSet.v_tiles = NULL;
    dTileSet.num_h_tiles = dTileSet.max_h_tiles = 0;
    dTileSet.num_v_tiles = dTileSet.max_v_tiles = 0;
}