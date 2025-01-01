#pragma once
#include "platform_compute_helpers.h"

#include <cstring>

_compute void dAtomicAdd(int* ptr, int val)
{
	*ptr += val;
}
_universal void cMemcpy(void* dest, const void* source, size_t size)
{
	memcpy(dest, source, size);
};
_universal void cMemcpyU(void* dest, const void* source, size_t size)
{
	memcpy(dest, source, size);
};
_universal void cMemset(void* dest, int val, size_t size)
{
	memset(dest, val, size);
};
_compute void threadSync()
{

}