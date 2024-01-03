/////////////////////////////////////////////////////////////////////////////////////////////////
// This file contains all platform-specific code for compatibility with various GPUs and CPUs. //
// Build targets must implement definitions for every function and define in this file.        //
/////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "platform_api.h"
#include "../include/search_structs.h"
#include "../include/worldgen_structs.h"

//This struct should contain handles to threads, streams, and etc. which are persistent between batch jobs.
struct Worker;

////////////////////
// Host Functions //
////////////////////

//This is called before memory is allocated, and should be used for configuring the target platform, 
// determining the amount of resources available.
void InitializePlatform();
//This is called after all memory is freed and is the last function called before the application closes.
//It should ensure the platform resources are properly freed and that the application is completely safe to terminate.
void DestroyPlatform();

//These functions can be used to initialize memory for use by the platform.
//Note that it is not required that all memory be handled here, as memory can be freely
// allocated and freed in batch handling functions as needed.
void AllocateComputeMemory();
void FreeComputeMemory();

//Generates compute-accessible data from host pointers.
void* UploadToDevice(void* hostMem, size_t size);
BiomeSpawnFunctions* GetSpawnFunc(Biome b);

//Creates or destroys a persistent worker that can recieve jobs.
Worker CreateWorker();
void DestroyWorker(Worker& worker);

////////////////////////////////////
// Host/Compute Interop Functions //
////////////////////////////////////

//This should dispatch a single workload to the worker, which consists of a number of spans equal to the appetite set
// in initialization.
void DispatchJob(Worker& worker, SpanParams* spans);

//Checks if a worker is done with its job, at which point SubmitJob is called.
bool QueryWorker(Worker& worker);

//This is called when a batch is finished computing, and should collect the return values from the job and pass them back to the API.
SpanRet* SubmitJob(Worker& worker);

//This is called when a job is cancelled early, and should safely end any ongoing work. DestroyWorker() will be called on the worker immediately after evaluation finishes.
void AbortJob(Worker& worker);