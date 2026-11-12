find_package(Threads REQUIRED)

# Repository-pinned header dependencies, vendored: nothing here is downloaded at configure time (the
# optional downloads are CUTLASS for sm_70 trees and ninferno-ui's chat bundle, NINFERNO_UI_CHAT).
add_library(ninferno::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninferno::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# A decisions-only tree (NINFERNO_DECISIONS_ONLY, CMakeLists.txt) has no CUDA language and no
# toolkit: of the toolkit it reads one header, cuda_occupancy.h (src/core/kernel_occupancy.cpp, a
# header-only occupancy calculator with no CUDA include of its own), found by path. The header is
# NVIDIA's, under the CUDA toolkit's license; it is not vendored, so that the repository carries no
# copy under those terms: any toolkit install, or the runtime's development package alone
# (cuda-cudart-dev-<version>, no compiler), has it. Its toolkit version can move an occupancy
# answer (a decision input), so the configure prints it (CUDART_VERSION of the cuda_runtime_api.h
# beside it); the decisions workflow reads CUDA 12.8's.
if(NINFERNO_DECISIONS_ONLY)
  # Each root as include/ and as targets/<platform>/include/ (where the packages put the files;
  # include/ is a link to it when the toolkit's links are installed).
  set(ninferno_cuda_roots "$ENV{CUDA_PATH}" "$ENV{CUDA_HOME}" "$ENV{CUDAToolkit_ROOT}")
  file(GLOB ninferno_versioned_cuda_roots LIST_DIRECTORIES true "/usr/local/cuda-*")
  list(SORT ninferno_versioned_cuda_roots COMPARE NATURAL ORDER DESCENDING)
  set(ninferno_cuda_hints "")
  set(ninferno_cuda_paths "")
  foreach(ninferno_cuda_root IN LISTS ninferno_cuda_roots)
    if(ninferno_cuda_root)
      file(GLOB ninferno_cuda_targets LIST_DIRECTORIES true "${ninferno_cuda_root}/targets/*/include")
      list(APPEND ninferno_cuda_hints "${ninferno_cuda_root}/include" ${ninferno_cuda_targets})
    endif()
  endforeach()
  foreach(ninferno_cuda_root IN ITEMS /usr/local/cuda ${ninferno_versioned_cuda_roots} /opt/cuda)
    file(GLOB ninferno_cuda_targets LIST_DIRECTORIES true "${ninferno_cuda_root}/targets/*/include")
    list(APPEND ninferno_cuda_paths "${ninferno_cuda_root}/include" ${ninferno_cuda_targets})
  endforeach()
  find_path(NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR cuda_occupancy.h
    HINTS ${ninferno_cuda_hints}
    PATHS ${ninferno_cuda_paths}
    DOC "The directory that holds the CUDA toolkit's cuda_occupancy.h (a decisions-only tree)")
  if(NOT NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR OR
     NOT EXISTS "${NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR}/cuda_occupancy.h")
    message(FATAL_ERROR
      "NINFERNO_DECISIONS_ONLY needs the CUDA toolkit's cuda_occupancy.h (a header, no compiler): "
      "not found (NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR='${NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR}'). "
      "Install a CUDA toolkit or its runtime development package alone (Ubuntu: cuda-cudart-dev-12-8 "
      "from NVIDIA's repository), or pass -DNINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR=<dir holding "
      "cuda_occupancy.h>.")
  endif()
  set(NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIRS "${NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR}")
  set(ninferno_cudart_version "unknown: no cuda_runtime_api.h beside it")
  if(EXISTS "${NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR}/cuda_runtime_api.h")
    file(STRINGS "${NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR}/cuda_runtime_api.h" ninferno_cudart_line
         REGEX "^#define[ \t]+CUDART_VERSION[ \t]+[0-9]+" LIMIT_COUNT 1)
    if(ninferno_cudart_line MATCHES "CUDART_VERSION[ \t]+([0-9]+)")
      math(EXPR ninferno_cudart_major "${CMAKE_MATCH_1} / 1000")
      math(EXPR ninferno_cudart_minor "(${CMAKE_MATCH_1} % 1000) / 10")
      set(ninferno_cudart_version "CUDA ${ninferno_cudart_major}.${ninferno_cudart_minor}")
    endif()
  endif()
  message(STATUS "Decisions only: cuda_occupancy.h from ${NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIR} "
                 "(${ninferno_cudart_version})")
  return()
endif()

find_package(CUDAToolkit REQUIRED)
# The toolkit's headers, for the one that host-only code reads (cuda_occupancy.h, src/core).
set(NINFERNO_CUDA_OCCUPANCY_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
if(NOT WIN32)
  find_package(PkgConfig REQUIRED)
endif()

# The CUDA runtime every target links: static in the product, so a release binary needs only the
# NVIDIA driver on the user's machine. In the shared test tree (NINFERNO_SHARED_OP_TESTS) it
# resolves on the executable or library being linked: shared for ninferno_ops_shared and the tests
# on it (property NINFERNO_SHARED_CUDART), which must use one runtime; static for everything else.
add_library(ninferno::cudart INTERFACE IMPORTED GLOBAL)
if(NINFERNO_SHARED_OP_TESTS)
  set_property(TARGET ninferno::cudart PROPERTY INTERFACE_LINK_LIBRARIES
    "$<IF:$<BOOL:$<TARGET_PROPERTY:NINFERNO_SHARED_CUDART>>,CUDA::cudart,CUDA::cudart_static>")
  # Read by the closure walks of cmake/NInfernoTargets.cmake instead of the generator expression.
  set_target_properties(ninferno::cudart PROPERTIES
    NINFERNO_REDIRECT_KEY NINFERNO_SHARED_CUDART
    NINFERNO_REDIRECT_STATIC CUDA::cudart_static
    NINFERNO_REDIRECT_SHARED CUDA::cudart)
else()
  target_link_libraries(ninferno::cudart INTERFACE CUDA::cudart_static)
endif()
if(WIN32)
  # vcpkg's FFmpeg (vcpkg.json), under the name the Linux tree's pkg-config gives it.
  find_package(FFMPEG REQUIRED)
  add_library(PkgConfig::FFMPEG INTERFACE IMPORTED GLOBAL)
  target_include_directories(PkgConfig::FFMPEG INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(PkgConfig::FFMPEG INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(PkgConfig::FFMPEG INTERFACE ${FFMPEG_LIBRARIES})
else()
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
endif()

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# Grammar compiler and token-mask matcher for constrained decoding (structured output).
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

# libcurl for the converter's fetches (src/convert, built in every tree) and media acquisition,
# which uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR, introduced in libcurl 7.85 (not
# merely the version of the maintainer environment). Looked up whatever NINFERNO_BUILD_PRODUCT_SUPPORT
# says: a benchmarks-only tree (NINFERNO_BUILD_APPS=OFF) failed to generate on ninferno_convert's link.
if(WIN32)
  find_package(CURL 7.85 REQUIRED)
  add_library(PkgConfig::LIBCURL INTERFACE IMPORTED GLOBAL)
  target_link_libraries(PkgConfig::LIBCURL INTERFACE CURL::libcurl)
else()
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()

if(NINFERNO_BUILD_PRODUCT_SUPPORT)
  add_library(ninferno::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninferno::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  if(WIN32)
    target_link_libraries(ninferno::httplib INTERFACE ws2_32 crypt32)
  endif()
  add_subdirectory(third_party/spdlog)
endif()
