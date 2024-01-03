#pragma once
#include "../platforms/platform_implementation.h"

_data constexpr int biomeModifierCount = 22;
_data constexpr float biomeModifierProbSum = 9.71025f;
_data const float biomeModifierProbs[biomeModifierCount] = {
	0.7f,
	1,
	0.5f,
	0.5f,
	0.2f,
	0.6f,
	0.01f,
	0.00025f,
	1,
	0.5f,
	0.75f,
	0.75f,
	0.5f,
	0.3f,
	0.5f,
	0.75f,
	0.5f,
	0.1f,
	0.2f,
	0.2f,
	0.1f,
	0.05f
};