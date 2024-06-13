#pragma once
#include "../platforms/platform_implementation.h"

#include "../include/search_structs.h"
#include "../include/misc_funcs.h"

#include "../data/uiNames.h"

#include <iostream>

_compute void WriteOutputBlock(uint8_t* output, int seed, Spawnable** spawnables, int sCount)
{
	int offset = 0;
	writeInt(output, offset, seed);
	writeInt(output, offset, sCount);

	for (int i = 0; i < sCount; i++)
	{
		Spawnable* sPtr = spawnables[i];
		Spawnable s = readMisalignedSpawnable(sPtr);
		writeInt(output, offset, s.x);
		writeInt(output, offset, s.y);
		writeByte(output, offset, s.sType);
		writeInt(output, offset, s.count);
		memcpy(output + offset, &sPtr->contents, s.count);
		offset += s.count;
	}
}

#define sprintfc(buf, ...) bufOffset += sprintf(buf + bufOffset, __VA_ARGS__)

void PrintOutputBlock(uint8_t* output, int time[2], FILE* outputFile, OutputConfig outputCfg, void(*appendOutput)(char*, char*))
//write output
{
#ifdef DISABLE_OUTPUT
	return;
#endif
	char* seedNum = (char*)malloc(12);
	char* seedInfo = (char*)malloc(8192);
	int memOffset = 0;
	int bufOffset = 0;
	int seed = readInt(output, memOffset);
	sprintf(seedNum, "%i", seed);
#ifdef IMAGE_OUTPUT
	int w = readInt(output, memOffset);
	int h = readInt(output, memOffset);
	char buffer[30];
	sprintf(buffer, "outputs/%i.png", seed);
	WriteImage(buffer, output + memOffset, w, h);
#else
#ifdef SPAWNABLE_OUTPUT
	constexpr int NEWLINE_CHAR_LIMIT = 100;
	int lineCtr = 1;

	int sCount = readInt(output, memOffset);
#ifdef REALTIME_SEEDS
	sprintfc(seedInfo, "in %i seconds [UNIX %i]\n", time[0], (int)(time[1] + time[0]), pick_world_seed(time[1] + time[0]));
#endif
	sprintfc(seedInfo, "%i: ", seed);
	if (sCount > 0) {
		for (int i = 0; i < sCount; i++) {
			Spawnable* sPtr = (Spawnable*)(output + memOffset);
			Spawnable s = *sPtr;
			Vec2i chunkCoords = GetLocalPos(s.x, s.y);

			sprintfc(seedInfo, "%s(%i", SpawnableTypeNames[s.sType - TYPE_CHEST], s.x);
			if (abs(chunkCoords.x - 35) > 35) {
				int pwPos = abs((int)rintf((chunkCoords.x - 35) / 70.0f));
				sprintfc(seedInfo, s.x > 0 ? "[E%i]" : "[W%i]", pwPos);
			}
			sprintfc(seedInfo, ", %i", s.y);
			if (abs(chunkCoords.y - 24) > 24) {
				int pwPos = abs((int)rintf((chunkCoords.y - 24) / 48.0f));
				sprintfc(seedInfo, s.y > 0 ? "[H%i]" : "[S%i]", pwPos);
			}
			sprintfc(seedInfo, ") - %ib[", s.count);

			for (int n = 0; n < s.count; n++) {
				Item item = *(&sPtr->contents + n);
				if (item == DATA_MATERIAL) {
					int offset2 = n + 1;
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					sprintfc(seedInfo, "Potion (%s)", MaterialNames[m]);
					n += 2;
				}
				else if (item == DATA_SPELL) {
					int offset2 = n + 1;
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					sprintfc(seedInfo, "%s", SpellNames[m]);
					n += 2;
				}
				else if (item == DATA_PIXEL_SCENE) {
					int offset2 = n + 1;
					short ps = readShort((uint8_t*)(&sPtr->contents), offset2);
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					sprintfc(seedInfo, "%s", PixelSceneNames[ps]);
					if (m != MATERIAL_NONE) {
						sprintfc(seedInfo, "[%s]", MaterialNames[m]);
					}
					n += 4;
				}
				else if (item == DATA_WAND) {
					n++;
					WandData dat = *(WandData*)(&sPtr->contents + n);
					sprintfc(seedInfo, "[%i capacity, %i S/C, %.2fsec CD, %.2fsec RT, %i Mana, %i Regen, %.3fx Speed, %ideg Spread, %s]",
						(int)dat.capacity, dat.multicast, dat.delay / 60.f, dat.reload / 60.f, dat.mana, dat.regen, dat.speed, dat.spread, dat.shuffle ? "Shuffle" : "Non-shuffle");
					if (dat.alwaysCast.s) sprintfc(seedInfo, " AC: ");

					n += 33;
					continue;
				}
				else if (GOLD_NUGGETS > item || item > TRUE_ORB) {
					sprintfc(seedInfo, "0x%x", item);
				}
				else {
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
	}
	else seedInfo[bufOffset++] = '\n';
	seedInfo[bufOffset++] = '\0';
	if (outputCfg.printOutputToFile) fprintf(outputFile, "%s", seedInfo);
	if (outputCfg.printOutputToConsole) printf("%s", seedInfo);
#else
#ifdef REALTIME_SEEDS
	sprintfc(seedInfo, "in %i seconds [UNIX %i]: \n", time[0], (int)(time[1] + time[0]), pick_world_seed(time[1] + time[0]));
#endif
	sprintfc(seedInfo, "%s", seedNum);
	fprintf(outputFile, "%s\n", seedInfo);
	if (outputCfg.printOutputToConsole) printf("%s\n", seedInfo);
#endif
#endif
	if (bufOffset > 8192) printf("ERR! Buffer overflow in output with size %i\n", bufOffset);
	if (appendOutput != NULL) appendOutput(seedNum, seedInfo);
	else { free(seedNum); free(seedInfo); }
}