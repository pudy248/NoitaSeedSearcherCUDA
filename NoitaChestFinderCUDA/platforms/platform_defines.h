//This is a separate file because there were some cases where not having it as such caused annoying circular includes.
#pragma once

#ifndef _compute
//Function annotation for functions which run on a high-performance computing platform.
#define _compute
#endif
#ifndef _universal
//Function annotation for functions which are platform-agnostic for use on compute platforms as well as in host code.
#define _universal
#endif
#ifndef _data
//Annotation for blocks of static data that should be accessible by the compute platform.
#define _data
#endif

#ifndef _noinline
//Used for performance reasons on some platforms (because the CUDA compiler is dumb sometimes). 
//This can be ignored on platforms where inlining is more performant.
//Performance should be tested with and without force-noinlining on a per-platform basis.
#define _noinline
#endif/