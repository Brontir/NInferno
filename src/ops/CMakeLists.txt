# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninferno_ops STATIC)
ninferno_internal_includes(ninferno_ops)
ninferno_cuda_archive(ninferno_ops)
# ninferno_ops_host first: its kernel_facts.cpp calls the inline device_class(), so the reference to
# current_device_class() (defined in ninferno_core) first appears while that archive is scanned, and
# ninferno_core must come after it on the link line. CMake keeps the order of this list, so it is
# part of the contract; tests/test_ops_link_only.cpp links nothing but ninferno_ops and fails to
# link if the two are swapped.
if(NINFERNO_WITH_NVFP4)
  target_link_libraries(ninferno_ops PUBLIC ninferno_ops_host ninferno_core PRIVATE ninferno_nvfp4_non_rdc)
else()
  target_link_libraries(ninferno_ops PUBLIC ninferno_ops_host ninferno_core)
endif()
# The FP8 and BF16 TMA routes encode their tensor maps through the driver API on every build.
target_link_libraries(ninferno_ops PUBLIC CUDA::cuda_driver)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
# The Ada build drops every NVFP4 TU, so the archive has no sources and is not created.
if(NINFERNO_WITH_NVFP4)
  add_library(ninferno_nvfp4_non_rdc STATIC)
  ninferno_internal_includes(ninferno_nvfp4_non_rdc)
  ninferno_cuda_non_rdc_archive(ninferno_nvfp4_non_rdc)
  target_link_libraries(ninferno_nvfp4_non_rdc PRIVATE ninferno_core ninferno::cudart CUDA::cuda_driver)
endif()

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/generic/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/mamba2/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gemma4_moe/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/sources.cmake")
