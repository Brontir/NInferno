cmake_minimum_required(VERSION 3.28)

# One architecture per build tree. 120a (Blackwell GB20x) is the full engine: NVFP4 weights and
# the TMA W4A4 GEMMs. 89 (Ada) builds the same engine without the NVFP4 family: the format is
# rejected at admission ("NVFP4 weights need SM120"), and every other path (BF16, FP8, GGUF K/IQ,
# int8) runs on the sm_80-class MMA instructions both architectures share. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic host-toolchain
# default or accepts an unsupported target.
# Any other architecture number from 70 on (a later GPU generation: "100", "120", "130", ...)
# builds that generic engine too: device code gates on the instruction families the target
# carries (ops/common/mma.cuh, nvfp4_codec.cuh), never on the number, and the runtime on
# DeviceFeatures (core/device_class.cpp), so a new generation needs no source change to run.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
# 120f (the sm_12x family target, every consumer Blackwell including later members) and 121a
# build the full engine too: its kernels gate on the sm_12x family (__CUDA_ARCH_FAMILY_SPECIFIC__
# 1200 or 1210).
set(NINFERNO_FULL_CUDA_ARCHS "120a" "121a" "120f")
set(NINFERNO_CUDA_ARCHS ${NINFERNO_FULL_CUDA_ARCHS} "89" "86" "80" "75" "70")
if(NOT CMAKE_CUDA_ARCHITECTURES IN_LIST NINFERNO_CUDA_ARCHS)
  if(CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$" AND CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 70)
    message(WARNING
      "CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}: building the generic engine (no "
      "NVFP4 kernels), not measured on such a GPU. Measured: 120a (full), 89, 86, 80, 75, 70.")
  else()
    message(FATAL_ERROR
      "NInferno supports CMAKE_CUDA_ARCHITECTURES 120a (Blackwell), 89 (Ada), 86 and 80 (Ampere), "
      "75 (Turing), 70 (Volta) or a plain architecture number >= 70 (generic engine); got "
      "'${CMAKE_CUDA_ARCHITECTURES}'")
  endif()
endif()
if(CMAKE_CUDA_ARCHITECTURES IN_LIST NINFERNO_FULL_CUDA_ARCHS)
  set(NINFERNO_WITH_NVFP4 ON)
else()
  set(NINFERNO_WITH_NVFP4 OFF)
endif()
add_compile_definitions($<$<NOT:$<BOOL:${NINFERNO_WITH_NVFP4}>>:NINFERNO_WITHOUT_NVFP4>)
# The architecture the tree compiles for ("120a" -> 120) and whether its target is architecture-
# or family-specific: device_class() keeps no feature this build carries no code for
# (build_features, core/device_class.h), so a GPU newer than the build runs the build's paths.
string(REGEX MATCH "^[0-9]+" NINFERNO_BUILD_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
add_compile_definitions(NINFERNO_BUILD_ARCH=${NINFERNO_BUILD_ARCH})
if(CMAKE_CUDA_ARCHITECTURES MATCHES "[af]$")
  add_compile_definitions(NINFERNO_BUILD_ARCH_SPECIFIC)
endif()
# A tree runs on the one architecture it names: embed that architecture's SASS alone. The PTX
# beside it (for a JIT onto a later GPU) doubled every fat binary, and on sm_75, whose composed MMA
# shapes quadruple the GEMM code, it pushed the executables past x86-64's 2 GiB reach.
# NINFERNO_CUDA_EMBED_PTX keeps the PTX for a GPU this nvcc does not know yet: "120" with it runs the
# generic engine on any later generation, the driver compiling each kernel at its first launch.
# Only a plain number's PTX carries forward; an "a" or "f" target's is bound to its own family.
option(NINFERNO_CUDA_EMBED_PTX "Embed the architecture's PTX beside its SASS (JIT onto a later GPU)" OFF)
if(NINFERNO_CUDA_EMBED_PTX)
  if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "NINFERNO_CUDA_EMBED_PTX needs a plain architecture number (its PTX runs on later GPUs); "
      "'${CMAKE_CUDA_ARCHITECTURES}' is bound to its own family")
  endif()
else()
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}-real")
endif()

project(ninferno VERSION 0.1.0 LANGUAGES C CXX CUDA)
# The version the executables report (`ninferno-serve --version`); a release build passes its tag.
set(NINFERNO_VERSION_STRING "${PROJECT_VERSION}" CACHE STRING "Version the executables report")
# A build run from its own tree finds routes/, the drafters' rankings and the fetched chat page in
# the source and build trees, whose paths it embeds. A release turns that off: those files ship
# beside the binaries, and the source paths CUDA errors name (__FILE__) are written relative to
# the tree, so no binary carries the path of the machine that built it.
option(NINFERNO_SOURCE_TREE_PATHS "Embed the source and build tree paths, for runs from the tree" ON)
if(NOT NINFERNO_SOURCE_TREE_PATHS)
  if(MSVC)
    file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/" ninferno_source_prefix)
    set(ninferno_file_map "/d1trimfile:${ninferno_source_prefix}")
  else()
    set(ninferno_file_map "-ffile-prefix-map=${PROJECT_SOURCE_DIR}/=")
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${ninferno_file_map}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${ninferno_file_map}>")
endif()

# Volta (sm_70) prefill multiplies on CUTLASS's sm_70 tensor-core GEMM (generic_volta_cutlass.cu):
# header-only, fetched for the sm_70 tree alone. FETCHCONTENT_SOURCE_DIR_CUTLASS points at a local
# checkout instead of the download.
option(NINFERNO_VOLTA_CUTLASS "Volta prefill on CUTLASS's sm_70 GEMM (sm_70 trees)" ON)
set(NINFERNO_CUTLASS_INCLUDE "")
if(NINFERNO_BUILD_ARCH STREQUAL "70" AND NINFERNO_VOLTA_CUTLASS)
  include(FetchContent)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR headers-only-no-cmake)
  FetchContent_MakeAvailable(cutlass)
  set(NINFERNO_CUTLASS_INCLUDE "${cutlass_SOURCE_DIR}/include")
endif()

# --- Global settings ---------------------------------------------------------
option(NINFERNO_BUILD_APPS "Build the NInferno CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInferno tests" OFF)
option(NINFERNO_BUILD_BENCHMARKS "Build NInferno benchmarks" OFF)
# Op tests against one shared ninferno_ops instead of each statically linking the whole engine
# (620 MB per test binary on 120a, linked one at a time through the ninferno_link pool): a test
# tree for fast gates. Every object is then position-independent and the CUDA runtime shared, so
# the library and the tests use one runtime. Not for the product tree.
option(NINFERNO_SHARED_OP_TESTS "Link the op tests against one shared ninferno_ops" OFF)
if(NINFERNO_SHARED_OP_TESTS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()
# -lineinfo (source lines for Nsight Compute) also embeds every kernel's PTX text in the fat
# binary (.nv_debug_ptx_txt): on sm_75 2.9 of the 4.7 GB device-link cubin, which put every
# executable past x86-64's 2 GiB reach. On for the 120a tree, where the engine is profiled; off
# on the bring-up architectures, whose images it would at least double, and in a release
# (NINFERNO_SOURCE_TREE_PATHS off): its line tables name every source by its full path.
if(NINFERNO_WITH_NVFP4 AND NINFERNO_SOURCE_TREE_PATHS)
  set(NINFERNO_CUDA_LINEINFO_DEFAULT ON)
else()
  set(NINFERNO_CUDA_LINEINFO_DEFAULT OFF)
endif()
option(NINFERNO_CUDA_LINEINFO "Compile CUDA with -lineinfo" ${NINFERNO_CUDA_LINEINFO_DEFAULT})
if(NINFERNO_CUDA_LINEINFO AND NOT NINFERNO_SOURCE_TREE_PATHS)
  message(WARNING "NINFERNO_CUDA_LINEINFO embeds the full source paths that "
                  "NINFERNO_SOURCE_TREE_PATHS=OFF keeps out of the binaries")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninferno_link=1)
  set(CMAKE_JOB_POOL_LINK ninferno_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFERNO_BUILD_PRODUCT_SUPPORT OFF)
if(NINFERNO_BUILD_APPS OR BUILD_TESTING)
  set(NINFERNO_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Windows.cmake)
include(cmake/Dependencies.cmake)
include(cmake/NInfernoTargets.cmake)
include(cmake/EmbedText.cmake)

# The operating system calls (src/core/sys.h): every NInferno target links them.
add_library(ninferno_sys STATIC src/core/sys.cpp $<$<BOOL:${WIN32}>:src/core/sys_win.cpp>)
target_include_directories(ninferno_sys PUBLIC ${PROJECT_SOURCE_DIR}/src)
if(WIN32)
  target_link_libraries(ninferno_sys PUBLIC bcrypt ws2_32 shell32 winmm)
else()
  target_link_libraries(ninferno_sys PUBLIC ${CMAKE_DL_LIBS})
endif()
link_libraries(ninferno_sys)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFERNO_BUILD_APPS)
  add_subdirectory(apps)
  # `cmake --install`: the executables, the route tables and token rankings they look up beside
  # themselves (<prefix>/share/ninferno/routes and .../rankings), ninferno-ui's chat window
  # (src/ui) and the licenses. ninferno-dispatch is not installed: the release scripts copy it and
  # one engine per GPU generation from their build trees and give them the same run path
  # (tools/release/package.sh).
  include(GNUInstallDirs)
  # An installed executable also looks in <prefix>/lib for the shared libraries a release archive
  # carries beside it (tools/release/package.sh), before the system's.
  foreach(app ninferno ninferno-serve ninferno-convert ninferno-perplexity ninferno-ui ninferno-dispatch)
    if(TARGET ${app})
      set_target_properties(${app} PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
      if(WIN32)
        ninferno_windows_executable(${app})
      endif()
    endif()
  endforeach()
  install(TARGETS ninferno ninferno-serve ninferno-convert ninferno-perplexity ninferno-ui OPTIONAL
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY routes/ DESTINATION ${CMAKE_INSTALL_DATADIR}/ninferno/routes)
  install(DIRECTORY tools/spec/rankings/ DESTINATION ${CMAKE_INSTALL_DATADIR}/ninferno/rankings
          FILES_MATCHING PATTERN "*.i64")
  # The default ranking a proposal head is restricted by (ninferno-convert --ranking).
  install(FILES tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64
          DESTINATION ${CMAKE_INSTALL_DATADIR}/ninferno/rankings)
  # The licenses and notices of the vendored code the executables carry (NOTICE), one directory
  # each under share/licenses; CUTLASS's where an sm_70 tree compiled it in.
  install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/ninferno)
  foreach(component cpp-httplib ggml-quants llama-jinja nlohmann sm70_flash_d256 spdlog utf8proc
                    xgrammar)
    file(GLOB component_licenses third_party/${component}/LICENSE* third_party/${component}/NOTICE*
         third_party/${component}/AUTHORS third_party/${component}/UNICODE-LICENSE)
    install(FILES ${component_licenses} DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/${component})
  endforeach()
  install(FILES third_party/xgrammar/3rdparty/dlpack/LICENSE
          DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/xgrammar/dlpack)
  if(NINFERNO_CUTLASS_INCLUDE)
    install(FILES "${cutlass_SOURCE_DIR}/LICENSE.txt" DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/cutlass)
  endif()
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFERNO_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
