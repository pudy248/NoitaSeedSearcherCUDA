#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//platform_compute_helpers.h impl.
#define _compute __device__
#define _universal __host__ __device__
#define _data __device__
#define _noinline __noinline__