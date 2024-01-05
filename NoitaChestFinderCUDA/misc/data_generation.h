#pragma once

#include "Windows.h"
#include <cstdlib>
#include <cstdio>

#include "../data/spells.h"
#include "../data/uiNames.h"
#include "utilities.h"

void FindInterestingColors(const char* path, int initialPathLen)
{
	char path2[100];
	int i = 0;
	while (path[i] != '/')
	{
		path2[i] = toupper(path[i]);
		i++;
	}
	path2[i++] = '_';
	while (path[i] != '.')
	{
		path2[i] = toupper(path[i]);
		i++;
	}
	path2[i++] = '\0';

#if 1
	printf("PS_%s,\n", path2);
#else
	const uint32_t interestingColors[] = { 0x78ffff, 0x55ff8c, 0x50a000, 0x00ff00, 0xff0000, 0x800000 };
	const char* typeEnum[] = { "PSST_SpawnHeart", "PSST_SpawnChest", "PSST_SpawnFlask", "PSST_SpawnItem", "PSST_SmallEnemy", "PSST_LargeEnemy" };

	png_byte color_type = GetColorType(path);
	if (color_type != PNG_COLOR_TYPE_RGB) ConvertRGBAToRGB(path);
	Vec2i dims = GetImageDimensions(path);
	uint8_t* data = (uint8_t*)malloc(3 * dims.x * dims.y);
	ReadImage(path, data);

	printf("%s = {\n", path2);

	for (int x = 0; x < dims.x; x++)
	{
		for (int y = 0; y < dims.y; y++)
		{
			int idx = 3 * (y * dims.x + x);
			uint32_t pix = createRGB(data[idx], data[idx + 1], data[idx + 2]);
			for (int i = 0; i < 6; i++)
			{
				if (pix == interestingColors[i])
				{
					printf("	\"PixelSceneSpawn(%s, %i, %i),\",\n", typeEnum[i], x, y);
				}
			}
		}
	}
	printf("},\n");
#endif
}

void GetAllInterestingPixelsInFolder(const char* path)
{
	WIN32_FIND_DATA fd;

	char buffer[50];
	int offset = 0;
	_putstr_offset(path, buffer, offset);
	_putstr_offset("*.*", buffer, offset);
	buffer[offset] = '\0';

	HANDLE hFind = ::FindFirstFile(buffer, &fd);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				offset = 0;
				_putstr_offset(path, buffer, offset);
				_putstr_offset(fd.cFileName, buffer, offset);
				buffer[offset] = '\0';
				FindInterestingColors(buffer, strlen(path));
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
}

void GenerateSpellData()
{
	printf("_data const static bool spellSpawnableInChests[] = {\n");
	for (int j = 0; j < SpellCount; j++)
	{
		bool passed = false;
		for (int t = 0; t < 11; t++)
		{
			if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL || allSpells[j].s == SPELL_SEA_SWAMP)
			{
				passed = true;
				break;
			}
		}
		printf(passed ? "true" : "false");
		printf(",\n");
	}
	printf("};\n");

	printf("_data const static bool spellSpawnableInBoxes[] = {\n");
	for (int j = 0; j < SpellCount; j++)
	{
		bool passed = false;
		if (allSpells[j].type == MODIFIER || allSpells[j].type == UTILITY)
		{
			for (int t = 0; t < 11; t++)
			{
				if (allSpells[j].spawn_probabilities[t] > 0 || allSpells[j].s == SPELL_SUMMON_PORTAL || allSpells[j].s == SPELL_SEA_SWAMP)
				{
					passed = true;
					break;
				}
			}
		}
		printf(passed ? "true" : "false");
		printf(",\n");
	}
	printf("};\n");

	int counters2[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
	double sums[11] = { 0,0,0,0,0,0,0,0,0,0,0 };
	for (int t = 0; t < 11; t++)
	{
		printf("_data const static SpellProb spellProbs_%i[] = {\n", t);
		for (int j = 0; j < SpellCount; j++)
		{
			if (allSpells[j].spawn_probabilities[t] > 0)
			{
				counters2[t]++;
				sums[t] += allSpells[j].spawn_probabilities[t];
				printf("{%f,SPELL_%s},\n", sums[t], SpellNames[j + 1]);
			}
		}
		printf("};\n");
	}

	printf("_data const static int spellTierCounts[] = {\n");
	for (int t = 0; t < 11; t++)
	{
		printf("%i,\n", counters2[t]);
	}
	printf("};\n");

	printf("_data const static float spellTierSums[] = {\n");
	for (int t = 0; t < 11; t++)
	{
		printf("%f,\n", sums[t]);
	}
	printf("};\n\n");


	for (int tier = 0; tier < 11; tier++)
	{
		int counters[8] = { 0,0,0,0,0,0,0,0 };
		for (int t = 0; t < 8; t++)
		{
			for (int j = 0; j < SpellCount; j++)
			{
				if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
				{
					counters[t]++;
				}
			}
		}
		for (int t = 0; t < 8; t++)
		{
			if (counters[t] > 0)
			{
				double sum = 0;
				printf("_data const static SpellProb spellProbs_%i_T%i[] = {\n", tier, t);
				for (int j = 0; j < SpellCount; j++)
				{
					if ((int)allSpells[j].type == t && allSpells[j].spawn_probabilities[tier] > 0)
					{
						sum += allSpells[j].spawn_probabilities[tier];
						printf("{%f,SPELL_%s},\n", sum, SpellNames[j + 1]);
					}
				}
				printf("};\n");
			}
		}
		printf("_data const static SpellProb* spellProbs_%i_Types[] = {\n", tier);
		for (int t = 0; t < 8; t++)
		{
			if (counters[t] > 0)
				printf("spellProbs_%i_T%i,\n", tier, t);
			else
				printf("NULL,\n");
		}
		printf("};\n");

		printf("_data const static int spellProbs_%i_Counts[] = {\n", tier);
		for (int t = 0; t < 8; t++)
		{
			printf("%i,\n", counters[t]);
		}
		printf("};\n\n");
	}
}
