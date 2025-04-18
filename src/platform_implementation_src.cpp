#include "../platforms/platform_implementation.h"

#ifdef BACKEND_CPU
#include "../platforms/impl_cpu.h"
#include "../platforms/impl_cpu_compute_helpers.h"
#endif
#ifdef BACKEND_CUDA
#include "../platforms/impl_cuda.h"
#include "../platforms/impl_cuda_compute_helpers.h"
#endif
