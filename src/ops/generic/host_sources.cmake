# The selector, with no CUDA runtime (#620 step 11): everything that decides a route and reads no
# device but through the seams (core/device_class.h, core/per_device.h). Compiled once, as objects
# with NINFERNO_HOST_ONLY, and used twice:
# - ninferno_ops_host, a static library for a host-only target (the selector golden's host twin, the
#   route-name test), which links no CUDA runtime and no ninferno_core: the stubs of the device seams
#   (core/*_nodevice.cpp, generic_nodevice.cpp) take their place in that target's own sources;
# - ninferno_ops (sources.cmake), which holds the same objects, so a product binary resolves the
#   seams against the real ninferno_core and the launch code of ninferno_ops in one archive, with no
#   link order between archives to depend on.
# Everything that shapes these objects is here (sources, kernel-facts snapshots, compile
# definitions and options), so a decisions-only tree (NINFERNO_DECISIONS_ONLY, which includes this
# file alone of src/ops) compiles them exactly as a full tree does.

# The kernel-facts snapshots (#612): data/kernel_facts/sm_<arch>.txt, one per build architecture,
# all compiled into every build (ops/generic/kernel_facts.h reads them; the tool that writes them
# is tools/kernel_facts/kernel_facts.py). A new file is picked up at the next configure.
file(GLOB NINFERNO_KERNEL_FACTS_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/data/kernel_facts/sm_*.txt")
list(SORT NINFERNO_KERNEL_FACTS_FILES)
set(NINFERNO_KERNEL_FACTS_ARRAYS "")
set(NINFERNO_KERNEL_FACTS_ENTRIES "")
foreach(file IN LISTS NINFERNO_KERNEL_FACTS_FILES)
  get_filename_component(stem "${file}" NAME_WE)
  string(REGEX REPLACE "^sm_" "" label "${stem}")
  ninferno_embed_text("${file}" NINFERNO_KF "${stem}_text")
  string(APPEND NINFERNO_KERNEL_FACTS_ARRAYS
    "inline constexpr char kKernelFacts_${stem}[] = ${${stem}_text};
")
  string(APPEND NINFERNO_KERNEL_FACTS_ENTRIES
    "    {\"${label}\", std::string_view{kKernelFacts_${stem}, sizeof(kKernelFacts_${stem}) - 1}},
")
endforeach()
configure_file("${CMAKE_CURRENT_LIST_DIR}/embedded_kernel_facts.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/ops/generic/embedded_kernel_facts.h" @ONLY)

add_library(ninferno_ops_host_objects OBJECT
  "${CMAKE_CURRENT_LIST_DIR}/kernel_facts.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernel_facts_device.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_routes.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_select.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_rmma_describe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_gates.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_cost.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_workspace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_volta_cutlass_sizes.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_query.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_authorities.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../prefill_routes.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../route_pins.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../route_table.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_ggml_host.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_mmq_host.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../moe/moe_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/selector_facts_host.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_facts_sink.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/generic_resolve.cpp")
ninferno_internal_includes(ninferno_ops_host_objects)
target_include_directories(ninferno_ops_host_objects PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_compile_definitions(ninferno_ops_host_objects PRIVATE NINFERNO_HOST_ONLY)
target_link_libraries(ninferno_ops_host_objects PRIVATE ninferno_core_host ninferno::json)
add_library(ninferno_ops_host STATIC $<TARGET_OBJECTS:ninferno_ops_host_objects>)
target_link_libraries(ninferno_ops_host PUBLIC ninferno_core_host)

# The per-GPU-class route table (routes/<class key>.json, include/ninferno/ops/route_table.h): a
# build run from its own tree reads the source tree's routes/ when none sits beside the executable.
if(NINFERNO_SOURCE_TREE_PATHS)
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/../route_table.cpp"
    TARGET_DIRECTORY ninferno_ops_host_objects
    PROPERTIES COMPILE_DEFINITIONS "NINFERNO_DEFAULT_ROUTE_DIR=\"${PROJECT_SOURCE_DIR}/routes\"")
endif()

# The files that compute the kernel roofline's price on the host (kernel_roofline.h: the selector's
# generic_cost.cpp and the price tool, which ninferno_ops compiles in the same directory): no
# multiply-add is fused there, so the price is the same bits on every host and build set, which the
# golden test (test_kernel_roofline.cpp) compares exactly. (The baseline x86-64 target has no FMA;
# this keeps it so under -march or on aarch64.)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/generic_cost.cpp"
                      "${CMAKE_CURRENT_LIST_DIR}/price_tool.cpp"
    TARGET_DIRECTORY ninferno_ops_host_objects
    APPEND PROPERTY COMPILE_OPTIONS "-ffp-contract=off")
endif()

# The Volta CUTLASS route's sizes (host code) answer for the route an sm_70 tree with CUTLASS
# compiles (generic_volta_cutlass.cu, sources.cmake): NINFERNO_VOLTA_CUTLASS where CMakeLists.txt
# fetched CUTLASS, in a decisions-only tree too.
if(NINFERNO_CUTLASS_INCLUDE)
  set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/generic_volta_cutlass_sizes.cpp"
    TARGET_DIRECTORY ninferno_ops_host_objects
    PROPERTIES COMPILE_DEFINITIONS NINFERNO_VOLTA_CUTLASS)
endif()
