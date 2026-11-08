#pragma once

// NINFERNO_ARCH_FAMILY_SPECIFIC: the value nvcc gives __CUDA_ARCH_FAMILY_SPECIFIC__ (1000 for
// sm_100a/sm_100f, 1200 for sm_120a/sm_120f, 1210 for sm_121a), or 0 on a plain target and in
// host code. nvcc 12.8 does not define __CUDA_ARCH_FAMILY_SPECIFIC__ at all: for an
// architecture-specific target it defines only __CUDA_ARCH_FEAT_SM<nnn>_ALL, so the family macro
// alone would silently select the portable forms for sm_120a there (and leave the NVFP4 MMA
// undeclared). Every gate on the Blackwell instruction families reads this macro instead.
// The 12.9+ spelling is used whenever it exists, so newer toolkits behave exactly as before.
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#    define NINFERNO_ARCH_FAMILY_SPECIFIC __CUDA_ARCH_FAMILY_SPECIFIC__
#elif defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#    define NINFERNO_ARCH_FAMILY_SPECIFIC 1200
#elif defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM100_ALL)
#    define NINFERNO_ARCH_FAMILY_SPECIFIC 1000
#else
// With nvcc 12.8, sm_101a (Thor, renamed sm_110 in 13.0) defines only __CUDA_ARCH_FEAT_SM101_ALL, so it
// takes the portable path: safe, and not a supported target.
#    define NINFERNO_ARCH_FAMILY_SPECIFIC 0
#endif
