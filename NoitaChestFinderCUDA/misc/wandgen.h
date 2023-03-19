#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#include "noita_random.h"
#include "../data/spells.h"

enum WandStat {
	RELOAD,
	CAST_DELAY,
	SPREAD,
	SPEED_MULT,
	CAPACITY,
	MULTICAST,
	SHUFFLE
};

struct Wand
{
	int level;
	bool isBetter;

	float cost;
	float capacity;
	int multicast;
	int mana;
	int regen;
	int delay;
	int reload;
	float speed;
	int spread;
	bool shuffle;

	float prob_unshuffle;
	float prob_draw_many;
	bool force_unshuffle;
	bool is_rare;

	Spell alwaysCast;
	int spellIdx;
	Spell spells[69];
};


struct StatProb
{
	float prob;
	float min;
	float max;
	float mean;
	float sharpness;
};

struct StatProbBlock {
	WandStat stat;
	int count;
	StatProb probs[10];
};

__device__ StatProbBlock statProbabilities[] = {
	{
		CAPACITY, 7,
		{
			{ 1, 3, 10, 6, 2 },
			{ 0.1f, 2, 7, 4, 4 },
			{ 0.05f, 1, 5, 3, 4 },
			{ 0.15f, 5, 11, 8, 2 },
			{ 0.12f, 2, 20, 8, 4 },
			{ 0.15f, 3, 12, 6, 6 },
			{ 1, 1, 20, 6, 0 }
		}
	},
	{
		RELOAD, 4,
		{
			{ 1, 5, 60, 30, 2 },
			{ 0.5f, 1, 100, 40, 2 },
			{ 0.02f, 1, 100, 40, 0 },
			{ 0.35f, 1, 240, 40, 0 }
		}
	},
	{
		CAST_DELAY, 4,
		{
			{ 1, 1, 30, 5, 2 },
			{ 0.1f, 1, 50, 15, 3 },
			{ 0.1f, -15, 15, 0, 3 },
			{ 0.45f, 0, 35, 12, 0 }
		}
	},
	{
		SPREAD, 2,
		{
			{ 1, -5, 10, 0, 3 },
			{ 0.1f, -35, 35, 0, 0 }
		}
	},
	{
		SPEED_MULT, 5,
		{
			{ 1, 0.8f, 1.2f, 1, 6 },
			{ 0.05f, 1, 2, 1.1f, 3 },
			{ 0.05f, 0.5f, 1, 0.9f, 3 },
			{ 1, 0.8f, 1.2f, 1, 0 },
			{ 0.001f, 1, 10, 5, 2 }
		}
	},
	{
		MULTICAST, 4,
		{
			{ 1, 1, 3, 1, 3 },
			{ 0.2f, 2, 4, 2, 8 },
			{ 0.05f, 1, 5, 2, 2 },
			{ 1, 1, 5, 2, 0 }
		}
	},
	{
		SHUFFLE,
		0
	}
};

__device__ StatProbBlock statProbabilitiesBetter[] = {
	{
		CAPACITY, 1,
		{
			{ 1, 5, 13, 8, 2 }
		}
	},
	{
		RELOAD, 1,
		{
			{ 1, 5, 40, 20, 2 }
		}
	},
	{
		CAST_DELAY, 1,
		{
			{ 1, 1, 35, 5, 2 }
		}
	},
	{
		SPREAD, 1,
		{
			{ 1, -1, 2, 0, 3 }
		}
	},
	{
		SPEED_MULT, 1,
		{
			{ 1, 0.8f, 1.2f, 1, 6 }
		}
	},
	{
		MULTICAST, 1,
		{
			{ 1, 1, 3, 1, 3 }
		}
	},
	{
		SHUFFLE, 0
	}
};

__device__ Spell GetRandomActionWithType(uint seed, double x, double y, int level, ACTION_TYPE type, int offset)
{
	/*NoitaRandom random = NoitaRandom((uint)(seed + offset));
	random.SetRandomSeed(x, y);
	double sum = 0;
	level = min(level, 10);
	SpellData spellsOfType[] = spellsByType[type];
	// all_spells length is 393
	for (int i = 0; i < spellsOfType.Length; i++)
	{
		sum += spellsOfType[i].spawn_probabilities[level];
	}

	double multiplier = random.Next();
	double accumulated = sum * multiplier;

	for (int i = 0; i < spellsOfType.Length; i++)
	{
		SpellData spell2 = spellsOfType[i];

		double probability = 0;
		probability = spell2.spawn_probabilities[level];
		if (probability > 0.0 && probability >= accumulated)
		{
			return (Spell)(i + 1);
		}
		accumulated -= probability;
	}
	int rand = (int)(random.Next() * 393);
	for (int j = 0; j < 393; j++)
	{
		SpellData spell = all_spells[(j + rand) % 393];
		if (spell.type == type && spell.spawn_probabilities[level] > 0.0)
		{
			return (Spell)(((j + rand) % 393) + 1);
		}
		j++;
	}*/
	return SPELL_NONE;
}

__device__ StatProb getGunProbs(WandStat s, StatProbBlock dict[7], NoitaRandom* random) {
	StatProbBlock probs = dict[0];
	for (int i = 0; i < 7; i++) if(s == dict[i].stat) probs = dict[i];
	if (probs.count == 0) return {};
	float sum = 0;
	for (int i = 0; i < probs.count; i++) sum += probs.probs[i].prob;
	float rnd = (float)random->Next() * sum;
	for (int i = 0; i < probs.count; i++)
	{
		if (rnd < probs.probs[i].prob) return probs.probs[i];
		rnd -= probs.probs[i].prob;
	}
	return {};
}

__device__ void shuffleTable(WandStat table[4], int length, NoitaRandom* random)
{
	for (int i = length - 1; i >= 1; i--)
	{
		int j = random->Random(0, i);
		WandStat temp = table[i];
		table[i] = table[j];
		table[j] = temp;
	}
}

__device__ void applyRandomVariable(Wand* gun, WandStat s, StatProbBlock dict[7], NoitaRandom* random) {
	float cost = gun->cost;
	StatProb prob = getGunProbs(s, dict, random);
	float min, max;
	int rnd;
	float temp_cost;
	
	float actionCosts[] = {
			0,
			5 + (gun->capacity * 2),
			15 + (gun->capacity * 3.5f),
			35 + (gun->capacity * 5),
			45 + (gun->capacity * gun->capacity)
	};

	switch (s)
	{
	case RELOAD:
		min = fminf(fmaxf(60 - (cost * 5), 1), 240);
		max = 1024;
		gun->reload = (int)fminf(fmaxf(random->RandomDistribution(prob.min, prob.max, prob.mean, prob.sharpness), min), max);
		gun->cost -= (60 - gun->reload) / 5;
		return;
	case CAST_DELAY:
		min = fminf(fmaxf(16 - cost, -50), 50);
		max = 50;
		gun->delay = (int)fminf(fmaxf(random->RandomDistribution(prob.min, prob.max, prob.mean, prob.sharpness), min), max);
		gun->cost -= 16 - gun->delay;
		return;
	case SPREAD:
		min = fminf(fmaxf(cost / -1.5f, -35), 35);
		max = 35;
		gun->spread = (int)fminf(fmaxf(random->RandomDistribution(prob.min, prob.max, prob.mean, prob.sharpness), min), max);
		gun->cost -= 16 - gun->spread;
		return;
	case SPEED_MULT:
		gun->speed = random->RandomDistributionf(prob.min, prob.max, prob.mean, prob.sharpness);
		return;
	case CAPACITY:
		min = 1;
		max = fminf(fmaxf((cost / 5) + 6, 1), 20);
		if (gun->force_unshuffle)
		{
			max = (cost - 15) / 5;
			if (max > 6)
				max = 6 + (cost - 45) / 10;
		}

		max = fminf(fmaxf(max, 1), 20);

		gun->capacity = fminf(fmaxf(random->RandomDistribution(prob.min, prob.max, prob.mean, prob.sharpness), min), max);
		gun->cost -= (gun->capacity - 6) * 5;
		return;
	case SHUFFLE:
		rnd = random->Random(0, 1);
		if (gun->force_unshuffle)
			rnd = 1;
		if (rnd == 1 && cost >= (15 + gun->capacity * 5) && gun->capacity <= 9)
		{
			gun->shuffle = false;
			gun->cost -= 15 + gun->capacity * 5;
		}
		return;
	case MULTICAST:
		min = 1;
		max = 1;
		for (int i = 0; i < 5; i++)
		{
			if (actionCosts[i] <= cost) max = actionCosts[i];
		}
		max = fminf(fmaxf(max, 1), gun->capacity);

		gun->multicast = (int)floor(fminf(fmaxf(random->RandomDistribution(prob.min, prob.max, prob.mean, prob.sharpness), min), max));
		temp_cost = actionCosts[(int)(fminf(fmaxf(gun->multicast, 1), 5) - 1)];
		gun->cost -= temp_cost;
		return;
	default:
		return;
	}
}

__device__ void AddRandomCards(Wand* gun, uint seed, double x, double y, int _level, NoitaRandom* random)
{
	bool is_rare = gun->is_rare;
	int goodCards = 5;
	if (random->Random(0, 100) < 7) goodCards = random->Random(20, 50);
	if (is_rare) goodCards *= 2;

	int orig_level = _level;
	int level = _level - 1;
	int capacity = (int)gun->capacity;
	int multicast = gun->multicast;
	int cardCount = random->Random(1, 3);
	Spell bulletCard = GetRandomActionWithType(seed, x, y, level, PROJECTILE, 0);
	Spell card = SPELL_NONE;
	int randomBullets = 0;
	int good_card_count = 0;

	if (random->Random(0, 100) < 50 && cardCount < 3) cardCount++;
	if (random->Random(0, 100) < 10 || is_rare) cardCount += random->Random(1, 2);

	goodCards = random->Random(5, 45);
	cardCount = random->Random((int)(0.51f * capacity), capacity);
	cardCount = (int)fminf(fmaxf(cardCount, 1), capacity - 1);

	if (random->Random(0, 100) < (orig_level * 10) - 5) randomBullets = 1;

	if (random->Random(0, 100) < 4 || is_rare)
	{
		int p = random->Random(0, 100);
		if (p < 77)
			card = GetRandomActionWithType(seed, x, y, level + 1, MODIFIER, 666);
		else if (p < 85)
		{
			card = GetRandomActionWithType(seed, x, y, level + 1, MODIFIER, 666);
			good_card_count++;
		}
		else if (p < 93)
			card = GetRandomActionWithType(seed, x, y, level + 1, STATIC_PROJECTILE, 666);
		else
			card = GetRandomActionWithType(seed, x, y, level + 1, PROJECTILE, 666);
		gun->alwaysCast = card;
	}

	if (random->Random(0, 100) < 50)
	{
		int extraLevel = level;
		while (random->Random(1, 10) == 10)
		{
			extraLevel++;
			bulletCard = GetRandomActionWithType(seed, x, y, extraLevel, PROJECTILE, 0);
		}

		if (cardCount < 3)
		{
			if (cardCount < 1 && random->Random(0, 100) < 20)
			{
				card = GetRandomActionWithType(seed, x, y, level, MODIFIER, 2);
				gun->spells[gun->spellIdx++] = card;
				cardCount--;
			}

			for (int i = 0; i < cardCount; i++)
				gun->spells[gun->spellIdx++] = bulletCard;
		}
		else
		{
			if (random->Random(0, 100) < 40)
			{
				card = GetRandomActionWithType(seed, x, y, level, DRAW_MANY, 1);
				gun->spells[gun->spellIdx++] = card;
				cardCount--;
			}
			if (cardCount > 3 && random->Random(0, 100) < 40)
			{
				card = GetRandomActionWithType(seed, x, y, level, DRAW_MANY, 1);
				gun->spells[gun->spellIdx++] = card;
				cardCount--;
			}
			if (random->Random(0, 100) < 80)
			{
				card = GetRandomActionWithType(seed, x, y, level, MODIFIER, 2);
				gun->spells[gun->spellIdx++] = card;
				cardCount--;
			}

			for (int i = 0; i < cardCount; i++)
				gun->spells[gun->spellIdx++] = bulletCard;
		}
	}
	else
	{
		for (int i = 0; i < cardCount; i++)
		{
			if (random->Random(0, 100) < goodCards && cardCount > 2)
			{
				if (good_card_count == 0 && multicast == 1)
				{
					card = GetRandomActionWithType(seed, x, y, level, DRAW_MANY, i + 1);
					good_card_count++;
				}
				else
				{
					if (random->Random(0, 100) < 83)
						card = GetRandomActionWithType(seed, x, y, level, MODIFIER, i + 1);
					else
						card = GetRandomActionWithType(seed, x, y, level, DRAW_MANY, i + 1);
				}

				gun->spells[gun->spellIdx++] = card;
			}
			else
			{
				gun->spells[gun->spellIdx++] = bulletCard;
				if (randomBullets == 1)
				{
					bulletCard = GetRandomActionWithType(seed, x, y, level, PROJECTILE, i + 1);
				}
			}
		}
	}
}

__device__ void AddRandomCardsBetter(Wand* gun, uint seed, double x, double y, int _level, NoitaRandom* random)
{
	bool is_rare = gun->is_rare;
	int goodCards = 5;
	if (random->Random(0, 100) < 7) goodCards = random->Random(20, 50);
	if (is_rare) goodCards *= 2;

	int orig_level = _level;
	int level = _level - 1;
	int capacity = (int)gun->capacity;
	int cardCount = random->Random(1, 3);
	Spell bulletCard = GetRandomActionWithType(seed, x, y, level, PROJECTILE, 0);
	Spell card = SPELL_NONE;
	int good_card_count = 0;

	if (random->Random(0, 100) < 50 && cardCount < 3) cardCount++;
	if (random->Random(0, 100) < 10 || is_rare) cardCount += random->Random(1, 2);

	goodCards = random->Random(5, 45);
	cardCount = random->Random((int)(0.51f * capacity), capacity);
	cardCount = (int)fminf(fmaxf(cardCount, 1), capacity - 1);

	if (random->Random(0, 100) < (orig_level * 10) - 5) {}

	if (random->Random(0, 100) < 4 || is_rare)
	{
		int p = random->Random(0, 100);
		if (p < 77)
			card = GetRandomActionWithType(seed, x, y, level + 1, MODIFIER, 666);
		else if (p < 85)
		{
			card = GetRandomActionWithType(seed, x, y, level + 1, MODIFIER, 666);
			good_card_count++;
		}
		else if (p < 93)
			card = GetRandomActionWithType(seed, x, y, level + 1, STATIC_PROJECTILE, 666);
		else
			card = GetRandomActionWithType(seed, x, y, level + 1, PROJECTILE, 666);
		gun->alwaysCast = card;
	}

	if (cardCount < 3)
	{
		if (cardCount < 1 && random->Random(0, 100) < 20)
		{
			card = GetRandomActionWithType(seed, x, y, level, MODIFIER, 2);
			gun->spells[gun->spellIdx++] = card;
			cardCount--;
		}

		for (int i = 0; i < cardCount; i++)
			gun->spells[gun->spellIdx++] = bulletCard;
	}
	else
	{
		if (random->Random(0, 100) < 40)
		{
			card = GetRandomActionWithType(seed, x, y, level, DRAW_MANY, 1);
			gun->spells[gun->spellIdx++] = card;
			cardCount--;
		}
		if (cardCount > 3 && random->Random(0, 100) < 40)
		{
			card = GetRandomActionWithType(seed, x, y, level, DRAW_MANY, 1);
			gun->spells[gun->spellIdx++] = card;
			cardCount--;
		}
		if (random->Random(0, 100) < 80)
		{
			card = GetRandomActionWithType(seed, x, y, level, MODIFIER, 2);
			gun->spells[gun->spellIdx++] = card;
			cardCount--;
		}

		for (int i = 0; i < cardCount; i++)
			gun->spells[gun->spellIdx++] = bulletCard;
	}
}

__device__ Wand GetWandStats(int _cost, int level, bool force_unshuffle, NoitaRandom* random)
{
	Wand gun = { level };
	int cost = _cost;

	if (level == 1 && random->Random(0, 100) < 50)
		cost += 5;

	cost += random->Random(-3, 3);
	gun.cost = cost;
	gun.capacity = 0;
	gun.multicast = 0;
	gun.reload = 0;
	gun.shuffle = true;
	gun.delay = 0;
	gun.spread = 0;
	gun.speed = 0;
	gun.prob_unshuffle = 0.1f;
	gun.prob_draw_many = 0.15f;
	gun.regen = 50 * level + random->Random(-5, 5 * level);
	gun.mana = 50 + (150 * level) + random->Random(-5, 5) * 10;
	gun.force_unshuffle = false;
	gun.is_rare = false;

	int p = random->Random(0, 100);
	if (p < 20)
	{
		gun.regen = (50 * level + random->Random(-5, 5 * level)) / 5;
		gun.mana = (50 + (150 * level) + random->Random(5, 5) * 10) * 3;
	}

	p = random->Random(0, 100);
	if (p < 15)
	{
		gun.regen = (50 * level + random->Random(-5, 5 * level)) * 5;
		gun.mana = (50 + (150 * level) + random->Random(-5, 5) * 10) / 3;
	}

	if (gun.mana < 50) gun.mana = 50;
	if (gun.regen < 10) gun.regen = 10;

	p = random->Random(0, 100);
	if (p < 15 + level * 6)
		gun.force_unshuffle = true;

	p = random->Random(0, 100);
	if (p < 5)
	{
		gun.is_rare = true;
		gun.cost += 65;
	}

	WandStat variables_01[4] = { RELOAD, CAST_DELAY, SPREAD, SPEED_MULT };
	WandStat variables_03[4] = { SHUFFLE, MULTICAST };

	shuffleTable(variables_01, 4, random);
	if (!gun.force_unshuffle) shuffleTable(variables_03, 2, random);

	for (int i = 0; i < 4; i++)
		applyRandomVariable(&gun, variables_01[i], statProbabilities, random);

	applyRandomVariable(&gun, CAPACITY, statProbabilities, random);
	for (int i = 0; i < 2; i++)
		applyRandomVariable(&gun, variables_03[i], statProbabilities, random);
	
	if (gun.cost > 5 && random->Random(0, 1000) < 995)
	{
		if (gun.shuffle)
			gun.capacity += (gun.cost / 5.0f);
		else
			gun.capacity += (gun.cost / 10.0f);
		gun.cost = 0;
	}
	gun.capacity = (float)floor(gun.capacity - 0.1f);

	if (force_unshuffle) gun.shuffle = false;
	if (random->Random(0, 10000) <= 9999)
	{
		gun.capacity = fminf(fmaxf(gun.capacity, 2), 26);
	}

	gun.capacity = fmaxf(gun.capacity, 2);

	if (gun.reload >= 60)
	{
		int rnd = 0;
		while (rnd < 70)
		{
			gun.multicast++;
			rnd = random->Random(0, 100);
		}

		if (random->Random(0, 100) < 50)
		{
			int new_multicast = (int)gun.capacity;
			for (int i = 1; i <= 6; i++)
			{
				int temp = random->Random(gun.multicast, (int)gun.capacity);
				if (temp < new_multicast)
					new_multicast = temp;
			}
			gun.multicast = new_multicast;
		}
	}

	gun.multicast = (int)fminf(fmaxf(gun.multicast, 1), (int)gun.capacity);

	return gun;
}

__device__ Wand GetWandStatsBetter(int _cost, int level, NoitaRandom* random)
{
	Wand gun = { level, true };
	int cost = _cost;

	if (level == 1 && random->Random(0, 100) < 50)
		cost += 5;

	cost += random->Random(-3, 3);
	gun.cost = cost;
	gun.capacity = 0;
	gun.multicast = 0;
	gun.reload = 0;
	gun.shuffle = true;
	gun.delay = 0;
	gun.spread = 0;
	gun.speed = 0;
	gun.prob_unshuffle = 0.1f;
	gun.prob_draw_many = 0.15f;
	gun.regen = 50 * level + random->Random(-5, 5 * level);
	gun.mana = 50 + (150 * level) + random->Random(-5, 5) * 10;
	gun.force_unshuffle = false;
	gun.is_rare = false;

	int p = random->Random(0, 100);
	if (p < 20)
	{
		gun.regen = (50 * level + random->Random(-5, 5 * level)) / 5;
		gun.mana = (50 + (150 * level) + random->Random(5, 5) * 10) * 3;

		if (gun.mana < 50) gun.mana = 50;
		if (gun.regen < 10) gun.regen = 10;
	}

	p = random->Random(0, 100);
	if (p < 15 + level * 6)
		gun.force_unshuffle = true;

	p = random->Random(0, 100);
	if (p < 5)
	{
		gun.is_rare = true;
		gun.cost += 65;
	}

	WandStat variables_01[4] = { RELOAD, CAST_DELAY, SPREAD, SPEED_MULT };
	WandStat variables_03[4] = { SHUFFLE, MULTICAST };

	shuffleTable(variables_01, 4, random);
	if (!gun.force_unshuffle) shuffleTable(variables_03, 2, random);

	for (int i = 0; i < 4; i++)
		applyRandomVariable(&gun, variables_01[i], statProbabilitiesBetter, random);

	applyRandomVariable(&gun, CAPACITY, statProbabilitiesBetter, random);
	for (int i = 0; i < 2; i++)
		applyRandomVariable(&gun, variables_03[i], statProbabilitiesBetter, random);

	if (gun.cost > 5 && random->Random(0, 1000) < 995)
	{
		if (gun.shuffle)
			gun.capacity += (gun.cost / 5.0f);
		else
			gun.capacity += (gun.cost / 10.0f);
		gun.cost = 0;
	}
	gun.capacity = floor(gun.capacity - 0.1f);

	if (random->Random(0, 10000) <= 9999)
	{
		gun.capacity = fminf(fmaxf(gun.capacity, 2), 26);
	}

	gun.capacity = fmaxf(gun.capacity, 2);

	if (gun.reload >= 60)
	{
		int rnd = 0;
		while (rnd < 70)
		{
			gun.multicast++;
			rnd = random->Random(0, 100);
		}

		if (random->Random(0, 100) < 50)
		{
			int new_multicast = (int)gun.capacity;
			for (int i = 1; i < 6; i++)
			{
				int temp = random->Random(gun.multicast, (int)gun.capacity);
				if (temp < new_multicast)
					new_multicast = temp;
			}
			gun.multicast = new_multicast;
		}
	}

	gun.multicast = fminf(fmaxf(gun.multicast, 1), (int)gun.capacity);

	return gun;
}

__device__ Wand GetWand(uint seed, double x, double y, int cost, int level, bool force_unshuffle)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	Wand wand = GetWandStats(cost, level, force_unshuffle, &random);
	AddRandomCards(&wand, seed, x, y, level, &random);

	return wand;
}

__device__ Wand GetWandBetter(uint seed, double x, double y, int cost, int level)
{
	NoitaRandom random = NoitaRandom(seed);
	random.SetRandomSeed(x, y);
	Wand wand = GetWandStatsBetter(cost, level, &random);
	AddRandomCardsBetter(&wand, seed, x, y, level, &random);

	return wand;
}

__device__ Wand GetWandWithLevel(uint seed, double x, double y, int level, bool nonshuffle, bool better)
{
	if (nonshuffle)
		switch (level)
		{
		case 1:
			return GetWand(seed, x, y, 25, 1, true);
		case 2:
			return GetWand(seed, x, y, 40, 2, true);
		case 3:
			return GetWand(seed, x, y, 60, 3, true);
		case 4:
			return GetWand(seed, x, y, 80, 4, true);
		case 5:
			return GetWand(seed, x, y, 100, 5, true);
		case 6:
			return GetWand(seed, x, y, 120, 6, true);
		default:
			return GetWand(seed, x, y, 180, 11, true);
		}
	else if (better)
		switch (level)
		{
		case 1:
			return GetWandBetter(seed, x, y, 30, 1);
		case 2:
			return GetWandBetter(seed, x, y, 40, 2);
		case 3:
			return GetWandBetter(seed, x, y, 60, 3);
		case 4:
			return GetWandBetter(seed, x, y, 80, 4);
		case 5:
			return GetWandBetter(seed, x, y, 100, 5);
		case 6:
			return GetWandBetter(seed, x, y, 120, 6);
		}
	else
		switch (level)
		{
		case 1:
			return GetWand(seed, x, y, 30, 1, false);
		case 2:
			return GetWand(seed, x, y, 40, 2, false);
		case 3:
			return GetWand(seed, x, y, 60, 3, false);
		case 4:
			return GetWand(seed, x, y, 80, 4, false);
		case 5:
			return GetWand(seed, x, y, 100, 5, false);
		case 6:
			return GetWand(seed, x, y, 120, 6, false);
		}
	return GetWand(seed, x, y, 10, 1, false);
}