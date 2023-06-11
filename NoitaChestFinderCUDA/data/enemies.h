#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../structs/enums.h"
#include "../structs/spawnableStructs.h"

__device__ const EnemyList coalmine_small_enemies = {
	6,
	1.25f,
{
	{0.1f, 0, 0, ENEMY_NONE},
	{0.5f, 1, 2, ENEMY_ZOMBIE_WEAK},
	{0.1f, 1, 1, ENEMY_SLIMESHOOTER_WEAK},
	{0.2f, 1, 3, ENEMY_HAMIS},
	{0.25f, 1, 2, ENEMY_MINER_WEAK},
	{0.1f, 1, 1, ENEMY_SHOTGUNNER_WEAK},
}
};

__device__ const EnemyList coalmine_large_enemies = {
	10,
	1.7f,
{
	{0.7f, 0, 0, ENEMY_NONE},
	{0.2f, 1, 1, ENEMY_FIREMAGE_WEAK},
	{0.01f, 1, 1, ENEMY_WORM},
	{0.2f, 5, 10, ENEMY_HAMIS},
	{0.1f, 2, 2, ENEMY_MINER_WEAK},
	//{0.3f, 1, 2, ENEMY_MINER_SANTA},
	{0.2f, 1, 1, ENEMY_SHOTGUNNER_WEAK},
	{0.1f, 1, 1, ENEMY_ACIDSHOOTER_WEAK},
	{0.08f, 1, 1, ENEMY_GIANTSHOOTER_WEAK},
	{0.09f, 1, 1, ENEMY_FIRESKULL},
	//{0.3f, 1, 2, ENEMY_MINER_SANTA},
	{0.02f, 1, 1, ENEMY_SHAMAN},
}
};