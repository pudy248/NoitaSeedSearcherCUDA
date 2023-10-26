//This is a separate file because there were some cases where not having it as such caused annoying circular includes.
#pragma once
#include "platform_defines.h"

//Misc. math functions that may or may not need to be separately defined depending on device architecture.
_compute void dAtomicAdd(int* ptr, int val);

int min(int a, int b);
int max(int a, int b);

//Default memcpy and memset are slow on some architectures so there is an option to replace it with a platform-specific substitute.
//Note that this is only used for large-ish memcpys where the difference is significant.
_compute void cMemcpy(void* dest, void* source, size_t size);
_compute void cMemset(void* dest, int val, size_t size);

//For workers with threads that benefit from more parallel execution.
_compute void threadSync();