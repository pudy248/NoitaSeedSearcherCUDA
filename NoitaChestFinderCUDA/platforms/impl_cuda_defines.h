#pragma once
#include "platform_defines.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//platform_compute_helpers.h impl.
#undef _compute
#define _compute __device__
#undef _universal
#define _universal __host__ __device__
#undef _data
#define _data __device__
#undef _noinline
#define _noinline __noinline__