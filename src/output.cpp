#pragma once
#include "../platforms/platform_implementation.h"

#include "../data/ui_names.h"
#include "../include/misc_funcs.h"
#include "../include/pngutils.h"
#include "../include/search_structs.h"

#include <filesystem>
#include <iostream>

_compute void WriteOutputBlock(MemSpan output, const SpawnableBlock& b) {
	int offset = 0;
	writeInt(output, offset, b.seed);
	writeInt(output, offset, b.count);

	for (int i = 0; i < b.count; i++) {
		Spawnable* sPtr = b.spawnables[i];
		Spawnable s = readMisalignedSpawnable(sPtr);
		writeInt(output, offset, s.x);
		writeInt(output, offset, s.y);
		writeByte(output, offset, s.sType);
		writeInt(output, offset, s.count);
		output.is_safe("WriteOutputBlock", "output", offset + s.count - 1);
		memcpy(output.ptr + offset, &sPtr->contents, s.count);
		offset += s.count;
	}
}

#define sprintfc(buf, ...) bufOffset += sprintf(buf + bufOffset, __VA_ARGS__)

void PrintOutputBlock(
	uint8_t* output, int time[2], FILE* outputFile, OutputConfig outputCfg, void (*appendOutput)(char*, char*))
//write output
{
	char* seedNum = (char*)malloc(12);
	const int outputSize = DEBUG_OUTPUT_SIZE_OVERRIDE ? 2 * DEBUG_OUTPUT_SIZE_OVERRIDE : 16384;
	char* seedInfo = (char*)malloc(outputSize);
	int memOffset = 0;
	int bufOffset = 0;
	int seed = readInt(output, memOffset);
	sprintf(seedNum, "%i", seed);
	if (outputCfg.outputMode == 3) {
		int w = readInt(output, memOffset);
		int h = readInt(output, memOffset);
		char buffer[30];
#ifndef __CUDA_ARCH__
		std::filesystem::create_directory("outputs");
#endif
		sprintf(buffer, "outputs/%i.png", seed);
		WriteImage(buffer, output + memOffset, w, h);
	} else if (outputCfg.outputMode == 1 || outputCfg.outputMode == 2) {
		constexpr int NEWLINE_CHAR_LIMIT = 100;
		int lineCtr = 1;

		int sCount = readInt(output, memOffset);
		//sprintfc(seedInfo, "in %i seconds [UNIX %i]\n", time[0], (int)(time[1] + time[0]),
		//	pick_world_seed(time[1] + time[0]));
		sprintfc(seedInfo, "%i: ", seed);
		if (sCount > 0) {
			for (int i = 0; i < sCount; i++) {
				Spawnable* sPtr = (Spawnable*)(output + memOffset);
				Spawnable s = *sPtr;
				Vec2i chunkCoords = GetLocalPos(s.x, s.y);

				sprintfc(seedInfo, "%s at (%i", SpawnableTypeNames[s.sType - TYPE_CHEST], s.x);
				if (abs(chunkCoords.x - 35) > 35) {
					int pwPos = abs((int)rintf((chunkCoords.x - 35) / 70.0f));
					sprintfc(seedInfo, s.x > 0 ? "[E%i]" : "[W%i]", pwPos);
				}
				sprintfc(seedInfo, ", %i", s.y);
				if (abs(chunkCoords.y - 24) > 24) {
					int pwPos = abs((int)rintf((chunkCoords.y - 24) / 48.0f));
					sprintfc(seedInfo, s.y > 0 ? "[H%i]" : "[S%i]", pwPos);
				}
				sprintfc(seedInfo, ") - [");

				if (outputCfg.outputMode == 2) {
					for (int n = 0; n < s.count; n += 16) {
						sprintfc(seedInfo, "\n    %i{", n);
						for (int m = 0; m < 16 && m + n < s.count; m++) {
							uint8_t item = *(&sPtr->contents + n + m);
							sprintfc(seedInfo, " 0x%x", item);
						}
						sprintfc(seedInfo, " }");
					}
					sprintfc(seedInfo, "\n]\n");
					memOffset += s.count + 13;
					continue;
				}
				for (int n = 0; n < s.count; n++) {
					Item item = *(&sPtr->contents + n);
					if (item == DATA_MATERIAL) {
						int offset2 = n + 1;
						short m = readShort((uint8_t*)(&sPtr->contents), offset2);
						sprintfc(seedInfo, "Potion (%s)", MaterialNames[m]);
						n += 2;
					} else if (item == DATA_SPELL) {
						int offset2 = n + 1;
						short m = readShort((uint8_t*)(&sPtr->contents), offset2);
						sprintfc(seedInfo, "%s", SpellNames[m]);
						n += 2;
					} else if (item == DATA_PIXEL_SCENE) {
						int offset2 = n + 1;
						short ps = readShort((uint8_t*)(&sPtr->contents), offset2);
						short m = readShort((uint8_t*)(&sPtr->contents), offset2);
						sprintfc(seedInfo, "%s", PixelSceneNames[ps]);
						if (m != MATERIAL_NONE) {
							sprintfc(seedInfo, "[%s]", MaterialNames[m]);
						}
						n += 4;
					} else if (item == DATA_WAND) {
						n++;
						WandData dat = readMisalignedWand((WandData*)(&sPtr->contents + n));
						sprintfc(seedInfo,
							"[%i capacity, %i S/C, %.2fsec CD, %.2fsec RT, %u Mana, %u Regen, %.3fx Speed, %ideg Spread, %s]",
							(int)dat.capacity, dat.multicast, dat.delay / 60.f, dat.reload / 60.f, dat.mana, dat.regen,
							dat.speed, dat.spread, dat.shuffle ? "Shuffle" : "Non-shuffle");
						if (dat.alwaysCast.s)
							sprintfc(seedInfo, " AC: ");

						n += 19 + (3 * !dat.alwaysCast.s);
						continue;
					} else if (GOLD_NUGGETS > item || item > TRUE_ORB) {
						sprintfc(seedInfo, "0x%x", item);
					} else {
						sprintfc(seedInfo, "%s", ItemNames[item]);
					}

					if (n < s.count - 1) {
						sprintfc(seedInfo, ", ");
						if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT) {
							lineCtr++;
							//_putstr_offset("\n", seedInfo, bufOffset);
						}
					}
				}
				sprintfc(seedInfo, "]\n");
				memOffset += s.count + 13;
			}
		} else
			sprintfc(seedInfo, "(no objects)\n");
		seedInfo[bufOffset++] = '\0';
		if (outputCfg.printOutputToFile) {
			fprintf(outputFile, "%s", seedInfo);
			fflush(outputFile);
		}
		if (outputCfg.printOutputToConsole)
			printf("%s", seedInfo);
	} else {
		//sprintfc(
		//	seedInfo, "in %i seconds [UNIX %i]: \n", time[0], (int)(time[1] + time[0]), pick_world_seed(time[1] + time[0]));
		sprintfc(seedInfo, "%s", seedNum);
		if (outputCfg.printOutputToFile) {
			fprintf(outputFile, "%s\n", seedInfo);
			fflush(outputFile);
		}
		if (outputCfg.printOutputToConsole)
			printf("%s\n", seedInfo);
	}
	if (bufOffset > outputSize)
		fprintf(stderr, "Buffer overflow in output with size %i\n", bufOffset);
	if (appendOutput != NULL)
		appendOutput(seedNum, seedInfo);
	else {
		free(seedNum);
		free(seedInfo);
	}
}