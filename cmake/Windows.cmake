# Windows: MSVC (Visual Studio 2022 Build Tools) as the host compiler of C++ and of nvcc, Ninja,
# and the libraries Linux takes from the distribution from vcpkg (vcpkg.json):
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a ^
#     -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#
# from a "x64 Native Tools" prompt (tools/release/windows/package-native.ps1 sets it up).
if(NOT WIN32)
  return()
endif()
if(NOT MSVC)
  message(FATAL_ERROR "NInferno on Windows builds with MSVC (Visual Studio 2022 Build Tools)")
endif()

# windows.h's min/max macros collide with std::min/max and numeric_limits; LEAN keeps winsock 1
# out of it (the files that talk to sockets include winsock2.h). CCCL's headers want the
# conforming preprocessor and say so with an #error under the traditional one; fmt (spdlog) wants
# /utf-8.
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES
  CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
add_compile_options(
  "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor;/utf-8;/bigobj;/EHsc>"
  "$<$<COMPILE_LANGUAGE:CXX>:/permissive->"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler=/Zc:preprocessor,/utf-8,/bigobj,/EHsc>")
# One C runtime in every object: the DLL one, as vcpkg's x64-windows libraries use.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# The programs a user runs read and write paths in UTF-8 (the narrow Win32 calls and the C runtime take
# UTF-8 with this manifest; without it a non-ASCII user name breaks the hub cache's paths).
set(NINFERNO_WINDOWS_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/windows-utf8.manifest")
function(ninferno_windows_executable target)
  target_sources(${target} PRIVATE "${NINFERNO_WINDOWS_MANIFEST}")
endfunction()
