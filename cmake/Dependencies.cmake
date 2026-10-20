find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(NOT WIN32)
  find_package(PkgConfig REQUIRED)
endif()

# The CUDA runtime every target links: static in the product, so a release binary needs only the
# NVIDIA driver on the user's machine; shared in the shared-op test tree, where the library and the
# tests must use one runtime.
add_library(ninferno::cudart INTERFACE IMPORTED GLOBAL)
if(NINFERNO_SHARED_OP_TESTS)
  target_link_libraries(ninferno::cudart INTERFACE CUDA::cudart)
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

# Repository-pinned header dependencies, vendored: nothing here is downloaded at configure time (the
# optional downloads are CUTLASS for sm_70 trees and ninferno-ui's chat bundle, NINFERNO_UI_CHAT).
add_library(ninferno::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninferno::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

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
