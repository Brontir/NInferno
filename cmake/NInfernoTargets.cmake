# Internal headers are private to each compile owner. External headers come from
# the dependency targets (or a component-local include for bundled C sources).
function(ninferno_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
endfunction()

# -lineinfo where NINFERNO_CUDA_LINEINFO asks for it (see CMakeLists.txt).
function(ninferno_cuda_lineinfo target)
  if(NINFERNO_CUDA_LINEINFO)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()
endfunction()

# The libraries' device code. Relocatable device code changes what ptxas makes of the kernels, and
# which way pays depends on the architecture; no library links a device symbol from another, so
# either links. 24K prefill of the 27B q4_g64 without RDC: RTX 5090 4232 -> 4458 tok/s (+5 %), RTX
# 4090 3415 -> 3272 and A10 936 -> 904 (the generic int8 GEMM gets 128-137 registers instead of
# 181-205 there); decode unchanged. So the full 120a build compiles whole-program per translation
# unit, and the generic builds keep RDC.
function(ninferno_cuda_archive target)
  if(NINFERNO_WITH_NVFP4)
    ninferno_cuda_non_rdc_archive(${target})
  else()
    set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    ninferno_cuda_lineinfo(${target})
  endif()
endfunction()

function(ninferno_cuda_non_rdc_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  ninferno_cuda_lineinfo(${target})
endfunction()

# target_sources with the carve-out of the native NVFP4 kernels: when NINFERNO_WITH_NVFP4 is OFF
# (every architecture but 120a) the sm_120a NVFP4 families are dropped from the build instead of
# failing to compile against Blackwell-only instructions: the native linear and fused projection
# nvfp4/ directories, the warp-specialized NVFP4 KV prefill kernel (tiled_launch.cu), and the
# generic W4A4 and NVFP4-tiled routes. The generic register-MMA and grouped-MoE NVFP4 codecs stay:
# they decode E2M1 on any tensor-core GPU, and ops/common/native_nvfp4.h sends NVFP4 weights to
# them wherever the native Ops are absent. So do the NVFP4 KV append and the grouped KV attention,
# which decode the cache to FP16 in software. The filter is one place, not five.
function(ninferno_target_sources target)
  # Skip the visibility keyword the callers pass; it is not a source.
  set(args)
  foreach(a IN LISTS ARGN)
    if(a MATCHES "^(PRIVATE|PUBLIC|INTERFACE)$")
      continue()
    endif()
    list(APPEND args "${a}")
  endforeach()
  set(filtered)
  foreach(src IN LISTS args)
    if(NOT NINFERNO_WITH_NVFP4 AND src MATCHES
       "/(linear|linear_add|linear_swiglu|gdn_input_proj|attn_input_proj)/nvfp4/|/causal_cache/nvfp4/tiled_launch\\.cu$|generic_w4a4\\.cu$|generic_nvfp4_tiled\\.cu$")
      continue()
    endif()
    list(APPEND filtered "${src}")
  endforeach()
  if(NOT filtered)
    return()
  endif()
  if(NOT TARGET ${target})
    # The Ada build drops the non-RDC NVFP4 archive entirely; a call whose list filters to
    # empty must not try to add sources to a target that does not exist.
    return()
  endif()
  target_sources(${target} PRIVATE ${filtered})
endfunction()
