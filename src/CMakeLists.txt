# Components own targets and dependencies; source manifests remain explicit.
if(NINFERNO_DECISIONS_ONLY)
  # The decision core alone (CMakeLists.txt): core's host part, the selector's host objects and the
  # configuration choice; each of these directories stops after them.
  add_subdirectory(core)
  add_subdirectory(ops)
  add_subdirectory(product)
else()
  add_subdirectory(core)
  add_subdirectory(artifact)
  add_subdirectory(spec)
  add_subdirectory(convert)
  add_subdirectory(text)
  add_subdirectory(chat)
  add_subdirectory(structured)
  add_subdirectory(media)
  add_subdirectory(ops)
  add_subdirectory(models)
  add_subdirectory(runtime)

  if(NINFERNO_BUILD_PRODUCT_SUPPORT)
    add_subdirectory(product)
    add_subdirectory(serve)
    add_subdirectory(connect)
    add_subdirectory(ui)
  endif()
endif()

# The decision core with no CUDA runtime (#620 step 13): the selector's host objects
# (ninferno_ops_host), core's host part (ninferno_core_host) and the configuration choice
# (ninferno_product_choice), with the stubs of the device seams compiled into each consumer
# (core/*_nodevice.cpp: the device class and ordinal of a build with no device; generic_nodevice.cpp:
# no tuned route, no executed kernel's description) and NINFERNO_HOST_ONLY. A target that links
# it and no CUDA library builds and runs on a machine with no GPU and no toolkit (checked at
# configure); a decisions-only tree (NINFERNO_DECISIONS_ONLY) builds it alone.
if(TARGET ninferno_product_choice)
  add_library(ninferno_decisions INTERFACE)
  target_sources(ninferno_decisions INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core/device_class_nodevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core/per_device_nodevice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/generic/generic_nodevice.cpp)
  target_compile_definitions(ninferno_decisions INTERFACE NINFERNO_HOST_ONLY)
  target_link_libraries(ninferno_decisions INTERFACE
    ninferno_ops_host ninferno_core_host ninferno_product_choice)
  cmake_language(DEFER CALL ninferno_assert_host_only ninferno_decisions)
endif()
