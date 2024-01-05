#include "defines.h"

#ifdef BACKEND_CPU
#include "platforms/impl_cpu_defines.h"
#include "platforms/impl_cpu_compute_helpers.h"
#include "platforms/impl_cpu.h"
#endif
#ifdef BACKEND_CUDA
#include "platforms/impl_cuda_defines.h"
#include "platforms/impl_cuda_compute_helpers.h"
#include "platforms/impl_cuda.h"
#endif

#include "gui/guiMain.h"

int main()
{
	SfmlMain();
	return;
}
