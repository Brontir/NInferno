#pragma once

// cudaStream_t for headers that only name a stream in a declaration. Under NINFERNO_HOST_ONLY (the
// host part of the selector, built with no CUDA runtime, #620 step 11) the handle type is spelled
// as the runtime spells it; otherwise this is cuda_runtime.h.

#ifdef NINFERNO_HOST_ONLY
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#else
#    include <cuda_runtime.h>
#endif
