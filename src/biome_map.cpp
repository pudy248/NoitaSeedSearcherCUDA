#include "../include/misc_funcs.h"
#include "../include/pngutils.h"
#include "../include/wak.h"
#include "../platforms/platform_implementation.h"
#include "../data/ids.h"

#include <queue>

constexpr uint32_t biomeColors[] = {0, 0xd57917, 0xd56516, 0x124445, 0x1775d5, 0x0046ff, 0x808000, 0xa08400, 0x008000,
	0x786c42, 0xe861f0, 0xa861ff, 0x375c00, 0x006c42, 0x3c0f0a, 0xd3e6f0, 0x726186, 0xe1cd32, 0x967f11, 0x4e5267,
	0x89a04b, 0x0080a8, 0x572828};

BiomeMapChunks load_biome_map(const char* path, int ngplus) {
	const uint8_t* png = (const uint8_t*)get_wak_file(path).data();
	Vec2i dims = GetBufferImageDimensions(png);
	uint8_t* buf = (uint8_t*)malloc(3 * dims.x * dims.y);
	ReadBufferImage(png, buf, false);
	Biome* map = (Biome*)malloc(dims.x * dims.y);
	BiomeMapChunks out = {dims.x, dims.y, map};
	for (int16_t y = 0; y < dims.y; y++) {
		for (int16_t x = 0; x < dims.x; x++) {
			uint32_t pix = (buf[3 * (y * dims.x + x)] << 16) + (buf[3 * (y * dims.x + x) + 1] << 8) +
						   buf[3 * (y * dims.x + x) + 2];
			map[y * dims.x + x] = B_NONE;
			for (int i = 0; i < sizeof(biomeColors) / 4; i++) {
				if (pix == biomeColors[i])
					map[y * dims.x + x] = (Biome)i;
			}
		}
	}
	free(buf);

	Biome* map2 = (Biome*)malloc(dims.x * dims.y);
	memcpy(map2, map, dims.x * dims.y);
	for (int16_t y = 0; y < dims.y; y++) {
		for (int16_t x = 0; x < dims.x; x++) {
			if (map2[y * dims.x + x] == B_NONE)
				continue;
			Biome cur = map2[y * dims.x + x];
			int minX = x, maxX = x, minY = y, maxY = y;
			std::queue<Vec2i> q;
			q.push({x, y});
			while (!q.empty()) {
				Vec2i pos = q.front();
				q.pop();
				if (x < 0 || y < 0 || x >= dims.x || y >= dims.y || map2[pos.y * dims.x + pos.x] != cur)
					continue;
				map2[pos.y * dims.x + pos.x] = B_NONE;
				minX = std::min(minX, pos.x);
				minY = std::min(minY, pos.y);
				maxX = std::max(maxX, pos.x);
				maxY = std::max(maxY, pos.y);
				q.push({pos.x - 1, pos.y});
				q.push({pos.x + 1, pos.y});
				q.push({pos.x, pos.y - 1});
				q.push({pos.x, pos.y + 1});
			}
			if (DEBUG_FLAGS & DEBUG::LOG_BIOME_SECTORS)
				printf("Biome sector: b:%s x:%i y:%i w:%i h:%i\n", IDs::biomes[0][cur], minX, minY, maxX - minX + 1,
					maxY - minY + 1);
			out.chunks.emplace_back(cur, minX, minY, maxX - minX + 1, maxY - minY + 1);
		}
	}
	free(map2);
	return out;
}