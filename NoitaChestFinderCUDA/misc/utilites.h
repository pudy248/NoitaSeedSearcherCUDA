#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "datatypes.h"
#include "noita_random.h"

__device__ float random_next(float min, float max, NoitaRandom* random, IntPair* rnd) {
	float result = random->ProceduralRandomf(rnd->x, rnd->y, min, max);
	rnd->y += 1;
	return result;
}

__device__ int pick_random_from_table_backwards(const float* probs, int length, NoitaRandom* random, IntPair* rnd) {
	for (int i = length - 1; i > 0; i--) {
		if (random_next(0, 1, random, rnd) <= probs[i]) return i;
	}
	return 0;
}

__device__ int pick_random_from_table_weighted(const float* probs, float sum, int length, NoitaRandom* random, IntPair* rnd) {
	float val = random_next(0, sum, random, rnd);
	for (int i = 0; i < length; i++) {
		if (val < probs[i]) return i;
		val -= probs[i];
	}
	return 0;
}