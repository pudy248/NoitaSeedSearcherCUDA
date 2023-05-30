#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "structs/primitives.h"
#include "structs/enums.h"
#include "structs/spawnableStructs.h"

#include <iostream>

enum ThreadState : uint8_t
{
	Running, //Computation ongoing
	SeedFound, //Waiting for host output read
	QueueEmpty, //Waiting for host seed dispatch
	HostLock, //Host accessing shared data
	DeviceLock, //Device accessing shared data
	ThreadStop, //Thread execution ended
};

struct UnifiedOutputFlags
{
	uint32_t seed;
	ThreadState state;
};

__device__ void WriteOutputBlock(uint8_t* output, int seed, Spawnable** spawnables, int sCount)
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

/*
__device__ void PrintSpawnableBlock(int seed, Spawnable** spawnables, int sCount, byte* bufferMem)
{
	for (int i = 0; i < sCount; i++)
	{
		Spawnable* sPtr = spawnables[i];
		Spawnable s = readMisalignedSpawnable(sPtr);

		//printf("%i %i %i %i\n", seed, s.x, s.y, s.sType);
		//continue;

		char* buffer = (char*)bufferMem;
		int offset = 0;

		_itoa_offset(seed, 10, buffer, offset);
		_putstr_offset(" @ (", buffer, offset);
		_itoa_offset(s.x, 10, buffer, offset);
		if (abs(s.x) > 35 * 512)
		{
			_putstr_offset(" [", buffer, offset);
			_putstr_offset(s.x > 0 ? "E" : "W", buffer, offset);
			int pwPos = abs((int)rintf(s.x / (70.0f * 512)));
			_itoa_offset(pwPos, 10, buffer, offset);
			_putstr_offset("]", buffer, offset);
		}
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(s.y, 10, buffer, offset);
		_putstr_offset("): ", buffer, offset);
		_putstr_offset(SpawnableTypeNames[s.sType - TYPE_CHEST], buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(s.count, 10, buffer, offset);
		_putstr_offset(" bytes: (", buffer, offset);

		for (int n = 0; n < s.count; n++)
		{
			Item item = *(&sPtr->contents + n);
			if (item == DATA_MATERIAL)
			{
				int offset2 = n + 1;
				short m = readShort((byte*)(&sPtr->contents), offset2);
				_putstr_offset("POTION_", buffer, offset);
				_putstr_offset(MaterialNames[m], buffer, offset);
				n += 2;
			}
			else if (item == DATA_SPELL)
			{
				int offset2 = n + 1;
				short m = readShort((byte*)(&sPtr->contents), offset2);
				_putstr_offset("SPELL_", buffer, offset);
				_putstr_offset(SpellNames[m], buffer, offset);
				n += 2;
			}
			else if (item == DATA_WAND)
			{
				n++;
				WandData dat = readMisalignedWand((WandData*)(&sPtr->contents + n));
				_putstr_offset("[", buffer, offset);

				_itoa_offset_decimal((int)(dat.capacity * 100), 10, 2, buffer, offset);
				_putstr_offset(" CAPACITY, ", buffer, offset);

				_itoa_offset(dat.multicast, 10, buffer, offset);
				_putstr_offset(" MULTI, ", buffer, offset);

				_itoa_offset(dat.delay, 10, buffer, offset);
				_putstr_offset(" DELAY, ", buffer, offset);

				_itoa_offset(dat.reload, 10, buffer, offset);
				_putstr_offset(" RELOAD, ", buffer, offset);

				_itoa_offset(dat.mana, 10, buffer, offset);
				_putstr_offset(" MANA, ", buffer, offset);

				_itoa_offset(dat.regen, 10, buffer, offset);
				_putstr_offset(" REGEN, ", buffer, offset);

				//speed... float?

				_itoa_offset(dat.spread, 10, buffer, offset);
				_putstr_offset(" SPREAD, ", buffer, offset);

				_putstr_offset(dat.shuffle ? "SHUFFLE] AC_" : "NON-SHUFFLE] AC_", buffer, offset);
				n += 33;
				continue;
			}
			else if (GOLD_NUGGETS > item || item > TRUE_ORB)
			{
				_putstr_offset("0x", buffer, offset);
				_itoa_offset_zeroes(item, 16, 2, buffer, offset);
			}
			else
			{
				int idx = item - GOLD_NUGGETS;
				_putstr_offset(ItemNames[idx], buffer, offset);
			}

			if (n < s.count - 1)
				_putstr_offset(" ", buffer, offset);
		}
		_putstr_offset(")\n", buffer, offset);
		buffer[offset] = '\0';
		printf("%s", buffer);
	}
}*/