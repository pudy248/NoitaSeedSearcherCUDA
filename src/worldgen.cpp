#include "../platforms/platform_implementation.h"

#include "../data/ui_names.h"
#include "../include/compute.h"
#include "../include/misc_funcs.h"
#include "../include/noita_random.h"
#include "../include/worldgen_structs.h"

_universal static WorldgenPRNG GetRNG(uint32_t world_seed, int map_w) {
	WorldgenPRNG rng = WorldgenPRNG(world_seed);

	int iters = map_w + world_seed + 11 * (map_w / -11) - 12 * (world_seed / 12);

	if (iters > 0) {
		do {
			rng.Next();
			iters -= 1;
		} while (iters != 0);
	}
	return rng;
}
_universal static uint32_t getPos(const uint32_t w, const uint32_t s, const uint32_t x, const uint32_t y) {
	return s * (w * y + x);
}
_universal static uint32_t getPixelColor(const uint8_t* map, uint32_t pos) {
	return createRGB(map[pos], map[pos + 1], map[pos + 2]);
}
_universal static uint32_t getPixelColor(const uint8_t* map, const uint32_t w, const uint32_t x, const uint32_t y) {
	uint32_t pos = getPos(w, 3, x, y);
	return getPixelColor(map, pos);
}
_universal static void setPixelColor(uint8_t* map, uint32_t pos, uint32_t color) {
	uint8_t r = ((color >> 16) & 0xff);
	uint8_t g = ((color >> 8) & 0xff);
	uint8_t b = ((color) & 0xff);
	map[pos] = r;
	map[pos + 1] = g;
	map[pos + 2] = b;
}
_universal static void setPixelColor(uint8_t* map, uint32_t w, uint32_t x, uint32_t y, uint32_t color) {
	uint32_t pos = getPos(w, 3, x, y);
	setPixelColor(map, pos, color);
}
_universal static void fill(uint8_t* map, int w, int x1, int x2, int y1, int y2, uint32_t color) {
	for (int x = x1; x <= x2; x++) {
		for (int y = y1; y <= y2; y++) {
			setPixelColor(map, w, x, y, color);
		}
	}
}
_compute GeneratedBiome GenerateMap(
	uint32_t worldSeed, const BiomeWangScope& scope, MemSpan output, MemSpan res, MemSpan visited, MemSpan miscMem) {
#ifdef SEEDS_AS_TRIES
	int MAX_TRIES = worldSeed;
	worldSeed = SEEDS_AS_TRIES;
#else
	constexpr int MAX_TRIES = 100;
#endif

	WorldgenPRNG rng = GetRNG(worldSeed, scope.bSec.map_w);
	//if (scope.bSec.isNightmare) rng.Next();

	memset(res.ptr, 0x80, scope.bSec.wang_w * 2);
	WangTileIndex* idxs = (WangTileIndex*)res.ptr + scope.bSec.wang_w;
	GeneratedBiome b = {scope, idxs, 0};

	int tries = 0;
	while (tries < MAX_TRIES) {
		tries++;
		WorldgenPRNG rng2 = WorldgenPRNG(rng.NextU());
		stbhw_generate_image(idxs, scope, scope.bSec.map_w, scope.bSec.map_h + 4, rng2);
#if 0
		printf("SEED %i, TRY %i\n", worldSeed, tries);
		for (int y = -1; y < scope.bSec.wang_h; y++) {
			for (int x = 0; x < scope.bSec.wang_w; x++) {
				WangTileIndex tile = idxs[y * scope.bSec.wang_w + x];
				printf(tile & 0x8000 ? "%s-- " : "%s%02i ", tile & 0x4000 ? "V" : "H", tile & 0x3fff);
			}
			printf("\n");
		}
		printf("\n");
#endif
#ifdef SEEDS_AS_TRIES
		isValid(b, miscMem, visited);
#else
		if ((DEBUG_FLAGS & DEBUG::NO_PATHFINDING) || isValid(b, miscMem, visited))
			break;
#endif
	}
	//printf("found path %i in %i tries\n", worldSeed, tries);
	//if (tries > 60)
	//	printf("Seed %i: %i tries\n", worldSeed, tries);

#ifdef IMAGE_OUTPUT
	output.is_safe("GenerateMap", "image output", 3 * scope.bSec.map_w * scope.bSec.map_h + 11);
	memcpy(output.ptr + 4, &scope.bSec.map_w, 4);
	memcpy(output.ptr + 8, &scope.bSec.map_h, 4);
	uint8_t* img = output.ptr + 12;
	for (int y = 0; y < scope.bSec.map_h; y++) {
		for (int x = 0; x < scope.bSec.map_w; x++) {
			uint32_t c = get_pixel<0, false>(b, {}, x, y, scope.ts.short_side_len, coalmine_overlay);
			if (visited.ptr[y * scope.bSec.map_w + x] == 2 && !false)
				c = 0xff00ffU;
			int i = (y * scope.bSec.map_w + x) * 3;
			img[i + 0] = (c >> 16) & 0xff;
			img[i + 1] = (c >> 8) & 0xff;
			img[i + 2] = (c >> 0) & 0xff;
		}
	}
#endif

	return b;
}

void UploadBiomeData() {
	for (int i = 1; i < B_BIOME_COUNT; i++) {
		for (int j = 0; j < HostPixelSceneLists[i].count; j++) {
			for (int k = 0; k < HostPixelSceneLists[i].lists[j].count; k++) {
				PixelSceneData& d = HostPixelSceneLists[i].lists[j].scenes[k];
				d.spawnCount = 0;
				if (!d.scene)
					continue;

				const uint8_t* png = (const uint8_t*)get_wak_file(HTables::ps_paths[d.scene]).data();
				Vec2i dims = GetBufferImageDimensions(png);
				uint8_t* buf = (uint8_t*)malloc(3 * dims.x * dims.y);
				ReadBufferImage(png, buf, false);

				if (DEBUG_FLAGS & DEBUG::LOG_SPAWN_PIXELS)
					fprintf(stderr, "\n%s\n", HTables::ps_paths[d.scene]);
				for (int16_t y = 0; y < dims.y; y++) {
					for (int16_t x = 0; x < dims.x; x++) {
						uint32_t pix = (buf[3 * (y * dims.x + x)] << 16) + (buf[3 * (y * dims.x + x) + 1] << 8) +
									   buf[3 * (y * dims.x + x) + 2];
						for (int16_t z = 0; z < HostSpawnColors[0].count; z++) {
							if (pix == HostSpawnColors[0].colors[z]) {
								d.spawns[d.spawnCount++] = {z, x, y};
								if (DEBUG_FLAGS & DEBUG::LOG_SPAWN_PIXELS)
									fprintf(stderr, "PS Spawn (%i, %i): Global %i\n", x, y, z);
							}
						}
						for (int16_t z = 0; z < HostSpawnColors[i].count; z++) {
							if (pix == HostSpawnColors[i].colors[z]) {
								d.spawns[d.spawnCount++] = {(int16_t)(HostSpawnColors[0].count + z), x, y};
								if (DEBUG_FLAGS & DEBUG::LOG_SPAWN_PIXELS) {
									fprintf(stderr, "PS Spawn (%i, %i): Biome %i\n", x, y, z);
									if (HostSpawnColors[i].colors[z] != 0x00ff00)
										fprintf(stderr,
											"WARNING: BIOME-SPECIFIC PIXEL SCENE SPAWNS UNSUPPORTED:\n%s @ %i, %i: Biome %i\n",
											HTables::ps_paths[d.scene], x, y, z);
								}
							}
						}
					}
				}
				free(buf);
			}
		}
	}

	HSetBiomeData2(HostPixelSceneLists);
}