#pragma once
#include <cstddef>

//Misc. math functions that need to be separately defined depending on device architecture.
_compute void dAtomicAdd(int* ptr, int val);

//Default memcpy and memset are slow on some architectures so there is an option to replace it with a platform-specific substitute.
//Note that this is only used for large memcpys where the difference is significant.
_universal void cMemcpy(void* dest, const void* source, size_t size);
_universal void cMemcpyU(void* dest, const void* source, size_t size);
_universal void cMemset(void* dest, int val, size_t size);

//For workers with threads that benefit from more parallel execution.
_compute void threadSync();