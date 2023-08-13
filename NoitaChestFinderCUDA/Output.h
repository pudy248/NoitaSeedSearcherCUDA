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
		cMemcpy(output + offset, &sPtr->contents, s.count);
		offset += s.count;
	}
}

void PrintOutputBlock(uint8_t* output, FILE* outputFile, OutputConfig outputCfg)
//write output
{
#ifdef IMAGE_OUTPUT
	int memOffset = 0;
	int seed = readInt(output, memOffset);
	int w = readInt(output, memOffset);
	int h = readInt(output, memOffset);
	char buffer[30];
	int bufOffset = 0;
	_putstr_offset("outputs/", buffer, bufOffset);
	_itoa_offset(seed, 10, buffer, bufOffset);
	_putstr_offset(".png", buffer, bufOffset);
	buffer[bufOffset++] = '\0';
	WriteImage(buffer, output + memOffset, w, h);
#else
#ifdef SPAWNABLE_OUTPUT
	char buffer[4096];
	int bufOffset = 0;
	int memOffset = 0;
	int seed = readInt(output, memOffset);
	int sCount = readInt(output, memOffset);
	if (seed == 0) return;

#ifdef REALTIME_SEEDS
	_itoa_offset(time, 10, buffer, bufOffset);
	_putstr_offset(" secs, seed ", buffer, bufOffset);
#endif
	_itoa_offset(seed, 10, buffer, bufOffset);
	if (sCount > 0)
	{
		_putstr_offset(": ", buffer, bufOffset);
		Spawnable* sPtr;
		for (int i = 0; i < sCount; i++)
		{
			sPtr = (Spawnable*)(output + memOffset);
			Spawnable s = readMisalignedSpawnable(sPtr);
			Vec2i chunkCoords = GetLocalPos(s.x, s.y);

			_putstr_offset(" ", buffer, bufOffset);
			_putstr_offset(SpawnableTypeNames[s.sType - TYPE_CHEST], buffer, bufOffset);
			_putstr_offset("(", buffer, bufOffset);
			_itoa_offset(s.x, 10, buffer, bufOffset);
			if (abs(chunkCoords.x - 35) > 35)
			{
				_putstr_offset("[", buffer, bufOffset);
				_putstr_offset(s.x > 0 ? "E" : "W", buffer, bufOffset);
				int pwPos = abs((int)rintf((chunkCoords.x - 35) / 70.0f));
				_itoa_offset(pwPos, 10, buffer, bufOffset);
				_putstr_offset("]", buffer, bufOffset);
			}
			_putstr_offset(", ", buffer, bufOffset);
			_itoa_offset(s.y, 10, buffer, bufOffset);
			if (abs(chunkCoords.y - 24) > 24)
			{
				_putstr_offset("[", buffer, bufOffset);
				_putstr_offset(s.y > 0 ? "H" : "S", buffer, bufOffset);
				int pwPos = abs((int)rintf((chunkCoords.y - 24) / 48.0f));
				_itoa_offset(pwPos, 10, buffer, bufOffset);
				_putstr_offset("]", buffer, bufOffset);
			}
			_putstr_offset("){", buffer, bufOffset);

			for (int n = 0; n < s.count; n++)
			{
				Item item = *(&sPtr->contents + n);
				if (item == DATA_MATERIAL)
				{
					int offset2 = n + 1;
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					_putstr_offset("POTION_", buffer, bufOffset);
					_putstr_offset(MaterialNames[m], buffer, bufOffset);
					n += 2;
				}
				else if (item == DATA_SPELL)
				{
					int offset2 = n + 1;
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					_putstr_offset("SPELL_", buffer, bufOffset);
					_putstr_offset(SpellNames[m], buffer, bufOffset);
					n += 2;
				}
				else if (item == DATA_PIXEL_SCENE)
				{
					int offset2 = n + 1;
					short ps = readShort((uint8_t*)(&sPtr->contents), offset2);
					short m = readShort((uint8_t*)(&sPtr->contents), offset2);
					_putstr_offset(PixelSceneNames[ps], buffer, bufOffset);
					if (m != MATERIAL_NONE)
					{
						_putstr_offset("[", buffer, bufOffset);
						_putstr_offset(MaterialNames[m], buffer, bufOffset);
						_putstr_offset("]", buffer, bufOffset);
					}
					n += 4;
				}
				else if (item == DATA_WAND)
				{
					n++;
					WandData dat = readMisalignedWand((WandData*)(&sPtr->contents + n));
					_putstr_offset("[", buffer, bufOffset);

					_itoa_offset_decimal((int)(dat.capacity * 100), 10, 2, buffer, bufOffset);
					_putstr_offset(" CAPACITY, ", buffer, bufOffset);

					_itoa_offset(dat.multicast, 10, buffer, bufOffset);
					_putstr_offset(" MULTI, ", buffer, bufOffset);

					_itoa_offset(dat.delay, 10, buffer, bufOffset);
					_putstr_offset(" DELAY, ", buffer, bufOffset);

					_itoa_offset(dat.reload, 10, buffer, bufOffset);
					_putstr_offset(" RELOAD, ", buffer, bufOffset);

					_itoa_offset(dat.mana, 10, buffer, bufOffset);
					_putstr_offset(" MANA, ", buffer, bufOffset);

					_itoa_offset(dat.regen, 10, buffer, bufOffset);
					_putstr_offset(" REGEN, ", buffer, bufOffset);

					//speed... float?

					_itoa_offset(dat.spread, 10, buffer, bufOffset);
					_putstr_offset(" SPREAD, ", buffer, bufOffset);

					_putstr_offset(dat.shuffle ? "SHUFFLE] AC_" : "NON-SHUFFLE] AC_", buffer, bufOffset);
					n += 33;
					continue;
				}
				else if (GOLD_NUGGETS > item || item > TRUE_ORB)
				{
					_putstr_offset("0x", buffer, bufOffset);
					_itoa_offset_zeroes(item, 16, 2, buffer, bufOffset);
				}
				else
				{
					int idx = item - GOLD_NUGGETS;
					_putstr_offset(ItemNames[idx], buffer, bufOffset);
				}

				if (n < s.count - 1)
					_putstr_offset(" ", buffer, bufOffset);
			}
			_putstr_offset("}", buffer, bufOffset);
			memOffset += s.count + 13;
		}
	}
	buffer[bufOffset++] = '\n';
	buffer[bufOffset++] = '\0';
	fprintf(outputFile, "%s", buffer);
	if(outputCfg.printOutputToConsole) printf("%s", buffer);
#else
#ifdef REALTIME_SEEDS
	printf("in %i seconds [UNIX %i]: seed %i\n", times[i], (int)(startTime + times[i]), GenerateSeed(startTime + times[i]));
#else
	char buffer[12];
	int bufOffset = 0;
	int seed = *(int*)output;
	_itoa_offset(seed, 10, buffer, bufOffset);
	buffer[bufOffset++] = '\n';
	buffer[bufOffset++] = '\0';
	fprintf(outputFile, "%s", buffer);
	if (outputCfg.printOutputToConsole) printf("%s", buffer);
#endif
#endif
#endif
}