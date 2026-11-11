# NINFERNO_SHARED_OP_TESTS: the member archives (NINFERNO_SHARED_MEMBERS, cmake/NInfernoTargets.cmake)
# whole in one shared library, which every test with device code links instead of its own static
# copy. Whole, so that no test ever needs a member archive beside it. An RDC archive already carries
# its device link (CUDA_RESOLVE_DEVICE_SYMBOLS), so the library links no device code of its own.
# Its members are compiled position-independent (an x86-64 shared library takes no other code);
# that is the only change the option makes to the objects the apps link.
add_library(ninferno_ops_shared SHARED "${PROJECT_SOURCE_DIR}/tests/cmake/ops_shared_anchor.cpp")
set(ninferno_whole "")
foreach(m IN LISTS NINFERNO_SHARED_MEMBERS)
  if(TARGET ninferno_${m})
    set_property(TARGET ninferno_${m} PROPERTY POSITION_INDEPENDENT_CODE ON)
    list(APPEND ninferno_whole $<TARGET_FILE:ninferno_${m}>)
    # The dependency on the archive and its third-party closure.
    target_link_libraries(ninferno_ops_shared PRIVATE ninferno_${m})
  endif()
endforeach()
target_link_options(ninferno_ops_shared PRIVATE
  "LINKER:--whole-archive" ${ninferno_whole} "LINKER:--no-whole-archive")
target_link_libraries(ninferno_ops_shared PUBLIC CUDA::cudart CUDA::cuda_driver Threads::Threads)
# NINFERNO_SHARED_CUDART: ninferno::cudart in the members' closure is the shared runtime here. The
# library does not take NINFERNO_SHARED_OPS, so the redirects in that closure stay the archives.
set_target_properties(ninferno_ops_shared PROPERTIES
  LINKER_LANGUAGE CXX
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  CUDA_RUNTIME_LIBRARY Shared
  NINFERNO_SHARED_CUDART ON)
