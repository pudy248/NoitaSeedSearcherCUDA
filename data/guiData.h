#pragma once
#include "../include/enums.h"

constexpr int _startingProjectileCount = 5;
const Spell _startingProjectiles[_startingProjectileCount] = {
	SPELL_LIGHT_BULLET, SPELL_SPITTER, SPELL_RUBBER_BALL, SPELL_BOUNCY_ORB, SPELL_NONE};
constexpr int _startingBombCount = 6;
const Spell _startingBombs[_startingBombCount] = {
	SPELL_BOMB, SPELL_DYNAMITE, SPELL_MINE, SPELL_ROCKET, SPELL_GRENADE, SPELL_NONE};
