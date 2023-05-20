#pragma once

#include "misc/datatypes.h"
#include "data/items.h"
#include "data/spells.h"

#include "WorldgenSearch.h"
#include "misc/wandgen.h"

__device__ void PrintSpawnableBlock(int seed, Spawnable** spawnables, int sCount)
{
	for (int i = 0; i < sCount; i++)
	{
		Spawnable* sPtr = spawnables[i];
		Spawnable s = readMisalignedSpawnable(sPtr);

		constexpr int buffer_size = 3000;
		char buffer[buffer_size];
		int offset = 0;

		_itoa_offset(seed, buffer, 10, offset);
		_putstr_offset(" @ (", buffer, offset);
		_itoa_offset(s.x, buffer, 10, offset);
		if (abs(s.x) > 35 * 512)
		{
			_putstr_offset(" [", buffer, offset);
			_putstr_offset(s.x > 0 ? "E" : "W", buffer, offset);
			int pwPos = abs((int)roundf(s.x / (70.0f * 512)));
			_itoa_offset(pwPos, buffer, 10, offset);
			_putstr_offset("]", buffer, offset);
		}
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(s.y, buffer, 10, offset);
		_putstr_offset("): ", buffer, offset);
		_putstr_offset(SpawnableTypeNames[s.sType - TYPE_CHEST], buffer, offset);
		_putstr_offset(", ", buffer, offset);
		_itoa_offset(s.count, buffer, 10, offset);
		_putstr_offset(" bytes: (", buffer, offset);

		for (int n = 0; n < s.count; n++)
		{
			if (offset > buffer_size - 50) printf("Dangerously high offset reached! Offset: %i, buffer size %i\n", offset, buffer_size);
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

				_itoa_offset_decimal((int)(dat.capacity * 100), buffer, 10, 2, offset);
				_putstr_offset(" CAPACITY, ", buffer, offset);

				_itoa_offset(dat.multicast, buffer, 10, offset);
				_putstr_offset(" MULTI, ", buffer, offset);

				_itoa_offset(dat.delay, buffer, 10, offset);
				_putstr_offset(" DELAY, ", buffer, offset);

				_itoa_offset(dat.reload, buffer, 10, offset);
				_putstr_offset(" RELOAD, ", buffer, offset);

				_itoa_offset(dat.mana, buffer, 10, offset);
				_putstr_offset(" MANA, ", buffer, offset);

				_itoa_offset(dat.regen, buffer, 10, offset);
				_putstr_offset(" REGEN, ", buffer, offset);

				//speed... float?

				_itoa_offset(dat.spread, buffer, 10, offset);
				_putstr_offset(" SPREAD, ", buffer, offset);

				_putstr_offset(dat.shuffle ? "SHUFFLE] AC_" : "NON-SHUFFLE] AC_", buffer, offset);
				n += 33;
				continue;
			}
			else if (GOLD_NUGGETS > item || item > MIMIC_SIGN)
			{
				_putstr_offset("0x", buffer, offset);
				_itoa_offset_zeroes(item, buffer, 16, 2, offset);
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
}