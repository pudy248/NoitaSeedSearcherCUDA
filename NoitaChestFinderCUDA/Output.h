#pragma once
#include "platforms/platform_compute_helpers.h"

#include "structs/primitives.h"
#include "structs/enums.h"
#include "structs/spawnableStructs.h"

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

void PrintOutputBlock(uint8_t* output, FILE* outputFile, OutputConfig outputCfg, void(*appendOutput)(char*, char*))
//write output
{
	char* seedNum = (char*)malloc(12);
	char* seedInfo = (char*)malloc(4096);
	int memOffset = 0;
	int bufOffset = 0;
	int seed = readInt(output, memOffset);
	_itoa_offset(seed, 10, seedNum, bufOffset);
	seedNum[bufOffset++] = '\0';
	bufOffset = 0;
#ifdef IMAGE_OUTPUT
	int w = readInt(output, memOffset);
	int h = readInt(output, memOffset);
	char buffer[30];
	_putstr_offset("outputs/", buffer, bufOffset);
	_itoa_offset(seed, 10, buffer, bufOffset);
	_putstr_offset(".png", buffer, bufOffset);
	buffer[bufOffset++] = '\0';
	WriteImage(buffer, output + memOffset, w, h);
#else
#ifdef SPAWNABLE_OUTPUT
	constexpr int NEWLINE_CHAR_LIMIT = 100;
	int lineCtr = 1;

	int sCount = readInt(output, memOffset);
#ifdef REALTIME_SEEDS
	_itoa_offset(time, 10, seedInfo, bufOffset);
	_putstr_offset(" secs, seed ", seedInfo, bufOffset);
#endif
	_itoa_offset(seed, 10, seedInfo, bufOffset);
	if (sCount > 0)
	{
		_putstr_offset(": ", seedInfo, bufOffset);
		Spawnable* sPtr;
		for (int i = 0; i < sCount; i++)
		{
			sPtr = (Spawnable*)(output + memOffset);
			Spawnable s = readMisalignedSpawnable(sPtr);
			Vec2i chunkCoords = GetLocalPos(s.x, s.y);

			_putstr_offset(" ", seedInfo, bufOffset);
			_putstr_offset(SpawnableTypeNames[s.sType - TYPE_CHEST], seedInfo, bufOffset);
			_putstr_offset("(", seedInfo, bufOffset);
			_itoa_offset(s.x, 10, seedInfo, bufOffset);
			if (abs(chunkCoords.x - 35) > 35)
			{
				_putstr_offset("[", seedInfo, bufOffset);
				_putstr_offset(s.x > 0 ? "E" : "W", seedInfo, bufOffset);
				int pwPos = abs((int)rintf((chunkCoords.x - 35) / 70.0f));
				_itoa_offset(pwPos, 10, seedInfo, bufOffset);
				_putstr_offset("]", seedInfo, bufOffset);
			}
			_putstr_offset(", ", seedInfo, bufOffset);
			_itoa_offset(s.y, 10, seedInfo, bufOffset);
			if (abs(chunkCoords.y - 24) > 24)
			{
				_putstr_offset("[", seedInfo, bufOffset);
				_putstr_offset(s.y > 0 ? "H" : "S", seedInfo, bufOffset);
				int pwPos = abs((int)rintf((chunkCoords.y - 24) / 48.0f));
				_itoa_offset(pwPos, 10, seedInfo, bufOffset);
				_putstr_offset("]", seedInfo, bufOffset);
			}
			_putstr_offset(") - [", seedInfo, bufOffset);

			for (int n = 0; n < s.count; n++)
			{
				Item item = *(&sPtr->contents + n);
				if (item == DATA_MATERIAL)
				{
					int offset2 = n + 1;
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					_putstr_offset("Potion (", seedInfo, bufOffset);
					_putstr_offset(MaterialNames[m], seedInfo, bufOffset);
					_putstr_offset(")", seedInfo, bufOffset);
					n += 2;
				}
				else if (item == DATA_SPELL)
				{
					int offset2 = n + 1;
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					_putstr_offset(SpellNames[m], seedInfo, bufOffset);
					n += 2;
				}
				else if (item == DATA_PIXEL_SCENE)
				{
					int offset2 = n + 1;
					short ps = readShort((uint8_t*)(&sPtr->contents), offset2);
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					_putstr_offset(PixelSceneNames[ps], seedInfo, bufOffset);
					if (m != MATERIAL_NONE)
					{
						_putstr_offset("[", seedInfo, bufOffset);
						_putstr_offset(MaterialNames[m], seedInfo, bufOffset);
						_putstr_offset("]", seedInfo, bufOffset);
					}
					n += 4;
				}
				else if (item == DATA_WAND)
				{
					n++;
					WandData dat = readMisalignedWand((WandData*)(&sPtr->contents + n));
					_putstr_offset("[", seedInfo, bufOffset);

					_itoa_offset(dat.capacity, 10, seedInfo, bufOffset);
					_putstr_offset(" Capacity, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset(dat.multicast, 10, seedInfo, bufOffset);
					_putstr_offset(" Spells/Cast, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset_decimal((int)rintf(dat.delay * 100 / 60.0f), 10, 2, seedInfo, bufOffset);
					_putstr_offset("sec Cast Delay, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset_decimal((int)rintf(dat.reload * 100 / 60.0f), 10, 2, seedInfo, bufOffset);
					_putstr_offset("sec Reload Time, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset(dat.mana, 10, seedInfo, bufOffset);
					_putstr_offset(" Max Mana, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset(dat.regen, 10, seedInfo, bufOffset);
					_putstr_offset(" Mana Regen, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset_decimal((int)(dat.speed * 100), 10, 2, seedInfo, bufOffset);
					_putstr_offset("x Speed, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_itoa_offset(dat.spread, 10, seedInfo, bufOffset);
					_putstr_offset(" Spread, ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}

					_putstr_offset(dat.shuffle ? "Shuffle] AC: " : "Non-shuffle] AC: ", seedInfo, bufOffset);
					n += 33;
					continue;
				}
				else if (GOLD_NUGGETS > item || item > TRUE_ORB)
				{
					_putstr_offset("0x", seedInfo, bufOffset);
					_itoa_offset_zeroes(item, 16, 2, seedInfo, bufOffset);
				}
				else
				{
					int idx = item - GOLD_NUGGETS;
					_putstr_offset(ItemNames[idx], seedInfo, bufOffset);
				}

				if (n < s.count - 1)
				{
					_putstr_offset(", ", seedInfo, bufOffset);
					if (bufOffset > lineCtr * NEWLINE_CHAR_LIMIT)
					{
						lineCtr++;
						_putstr_offset("\n", seedInfo, bufOffset);
					}
				}
			}
			_putstr_offset("]\n\n", seedInfo, bufOffset);
			memOffset += s.count + 13;
		}
	}
	seedInfo[bufOffset++] = '\0';
	fprintf(outputFile, "%s", seedInfo);
	if(outputCfg.printOutputToConsole) printf("%s", seedInfo);
#else
#ifdef REALTIME_SEEDS
	printf("in %i seconds [UNIX %i]: seed %i\n", times[i], (int)(startTime + times[i]), GenerateSeed(startTime + times[i]));
#else
	strcpy(seedInfo, seedNum);
	fprintf(outputFile, "%s\n", seedInfo);
	if (outputCfg.printOutputToConsole) printf("%s\n", seedInfo);
#endif
#endif
#endif
	if (appendOutput != NULL) appendOutput(seedNum, seedInfo);
	else { free(seedNum); free(seedInfo); }
}