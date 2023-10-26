#pragma once
#include "platform_compute_helpers.h"

#include <cstring>

_compute void dAtomicAdd(int* ptr, int val)
{
	*ptr += val;
}
_compute void cMemcpy(void* dest, void* source, size_t size)
{
	memcpy(dest, source, size);
};
_compute void cMemset(void* dest, int val, size_t size)
{
	memset(dest, val, size);
};
_compute void threadSync()
{

}