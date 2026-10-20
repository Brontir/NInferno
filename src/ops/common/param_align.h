#pragma once

// The alignment of a struct passed to a kernel by value. MSVC refuses parameters aligned past 16
// bytes in the host stubs nvcc writes (C2719), so its host pass gives the struct no alignment of its
// own: with __cplusplus left at MSVC's 199711L (no /Zc:__cplusplus, cmake/Windows.cmake) cuda.h's
// CUtensorMap carries none either, the size stays, and the device pass, which places the
// parameter, keeps the full alignment the TMA descriptors need. Every other build is unchanged.
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#define NINFERNO_PARAM_ALIGN(bytes)
#else
#define NINFERNO_PARAM_ALIGN(bytes) alignas(bytes)
#endif