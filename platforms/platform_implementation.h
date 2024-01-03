#pragma once
#include "../defines.h"

#ifdef BACKEND_CPU
#include "cpu_defines.h"
#endif
#ifdef BACKEND_CUDA
#include "cuda_defines.h"
#endif

#include "platform_compute_helpers.h"
#include "platform.h"
