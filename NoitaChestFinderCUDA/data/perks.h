#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../structs/primitives.h"
#include "../structs/enums.h"
#include "../structs/staticPrecheckStructs.h"

constexpr int perkCount = 106;
__device__ const PerkData perkAttrs[] =
{
	{true}, //CRITICAL_HIT
	{true, true}, //BREATH_UNDERWATER
	{true}, //EXTRA_MONEY
	{true}, //EXTRA_MONEY_TRICK_KILL
	{}, //GOLD_IS_FOREVER
	{}, //TRICK_BLOOD_MONEY
	{true, true, 6, 1}, //EXPLODING_GOLD
	{true, false, 0, 1}, //HOVER_BOOST
	{true, false, 0, 1}, //FASTER_LEVITATION
	{true, false, 0, 2}, //MOVEMENT_FASTER
	{true, false, 0, 1}, //STRONG_KICK
	{}, //TELEKINESIS
	{true, true, 8, 2}, //REPELLING_CAPE
	{}, //EXPLODING_CORPSES
	{}, //SAVING_GRACE
	{}, //INVISIBILITY
	{true, false, 0, 1}, //GLOBAL_GORE
	{}, //REMOVE_FOG_OF_WAR
	{true, true, 0, 2}, //LEVITATION_TRAIL
	{}, //VAMPIRISM
	{true, false, 0, 3}, //EXTRA_HP
	{true, true, 9, 2}, //HEARTS_MORE_EXTRA_HP
	{true, true, 2, 2}, //GLASS_CANNON
	{true, true, 0, 2}, //LOW_HP_DAMAGE_BOOST
	{true, true}, //RESPAWN
	{true, true}, //WORM_ATTRACTOR
	{}, //RADAR_ENEMY
	{}, //FOOD_CLOCK
	{}, //IRON STOMACH
	{}, //WAND_RADAR
	{}, //ITEM_RADAR
	{false, false, 0, 0, true}, //MOON_RADAR
	{false, false, 0, 0, true}, //MAP
	{}, //PROTECTION_FIRE
	{}, //PROTECTION_RADIOACTIVITY
	{}, //PROTECTION_EXPLOSION
	{}, //PROTECTION_MELEE
	{}, //PROTECTION_ELECTRICITY
	{}, //TELEPORTITIS
	{}, //TELEPORTITIS_DODGE
	{true, true}, //STAINLESS_ARMOUR
	{}, //EDIT_WANDS_EVERYWHERE
	{}, //NO_WAND_EDITING
	{true, true}, //WAND_EXPERIMENTER
	{}, //ADVENTURER
	{}, //ABILITY_ACTIONS_MATERIALIZED
	{}, //PROJECTILE_HOMING
	{}, //PROJECTILE_HOMING_SHOOTER
	{}, //UNLIMITED_SPELLS
	{}, //FREEZE_FIELD
	{}, //FIRE_GAS
	{}, //DISSOLVE_POWDERS
	{true, true}, //BLEED_SLIME
	{}, //BLEED_OIL
	{}, //BLEED_GAS
	{true, false, 5, 2, false, 10}, //SHIELD
	{true, true}, //REVENGE_EXPLOSION
	{true, true}, //REVENGE_TENTACLE
	{}, //REVENGE_RATS
	{true, true}, //REVENGE_BULLET
	{true, true, 3, 2}, //ATTACK_FOOT
	{true, true, 0, 0, true}, //LEGGY_FEET
	{true, true, 5, 2}, //PLAGUE_RATS
	{}, //VOMIT_RATS
	{}, //CORDYCEPS
	{}, //MOLD
	{}, //WORM_SMALLER_HOLES
	{true, true}, //PROJECTILE_REPULSION
	{true, true, 3, 2}, //RISKY_CRITICAL
	{true, true, 3, 2}, //FUNGAL_DISEASE
	{true, true}, //PROJECTILE_SLOW_FIELD
	{true, true}, //PROJECTILE_REPULSION_SECTOR
	{}, //PROJECTILE_EATER_SECTOR
	{true, false, 0, 2, false, 10}, //ORBIT
	{true}, //ANGRY_GHOST
	{true, false, 5, 2}, //HUNGRY_GHOST
	{true, true}, //DEATH_GHOST
	{true, false, 10, 2}, //HOMUNCULUS
	{}, //LUKKI_MINION
	{}, //ELECTRICITY
	{true, false, 6, 1}, //ATTRACT_ITEMS
	{true, true}, //EXTRA_KNOCKBACK
	{true, true}, //LOWER_SPREAD
	{}, //LOW_RECOIL
	{}, //BOUNCE
	{}, //FAST_PROJECTILES
	{true}, //ALWAYS_CAST
	{true}, //EXTRA_MANA
	{}, //NO_MORE_SHUFFLE
	{}, //NO_MORE_KNOCKBACK
	{true}, //DUPLICATE_PROJECTILE
	{true}, //FASTER_WANDS
	{true}, //EXTRA_SLOTS
	{}, //CONTACT_DAMAGE
	{true, false, 5, 3}, //EXTRA_PERK
	{true, true, 6, 3}, //PERKS_LOTTERY
	{true}, //GAMBLE
	{true, false, 5, 2}, //EXTRA_SHOP_ITEM
	{true}, //GENOME_MORE_HATRED
	{true}, //GENOME_MORE_LOVE
	{}, //PEACE_WITH_GODS
	{}, //MANA_FROM_KILLS
	{}, //ANGRY_LEVITATION
	{true, true}, //LASER_AIM
	{true, true, 5, 2}, //PERSONAL_LASER
	{true} //MEGA_BEAM_STONE
};

__device__ const signed char perkStackableDistances[] = {
	4,
	4,
	4,
	4,
	-1,
	-1,
	4,
	4,
	4,
	4,
	4,
	-1,
	4,
	-1,
	-1,
	-1,
	4,
	-1,
	4,
	-1,
	4,
	4,
	4,
	4,
	4,
	4,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	4,
	-1,
	-1,
	4,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	-1,
	4,
	-1,
	-1,
	10,
	4,
	4,
	-1,
	4,
	4,
	-1,
	4,
	-1,
	-1,
	-1,
	-1,
	4,
	4,
	4,
	4,
	4,
	-1,
	10,
	4,
	4,
	4,
	4,
	-1,
	-1,
	4,
	4,
	4,
	-1,
	-1,
	-1,
	4,
	4,
	-1,
	-1,
	4,
	4,
	4,
	-1,
	4,
	4,
	4,
	4,
	4,
	4,
	-1,
	-1,
	-1,
	4,
	4,
	4
};