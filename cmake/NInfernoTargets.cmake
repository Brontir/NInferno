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

# Fails the configure when `target` reaches the CUDA runtime through its link closure: the host-only
# targets (ninferno_core_host and what links only it) must build and run on a machine with no
# toolkit (#620 step 11). Call it deferred, once every target exists.
function(ninferno_assert_host_only target)
  set(todo ${target})
  set(seen "")
  while(todo)
    list(POP_FRONT todo item)
    if(item IN_LIST seen)
      continue()
    endif()
    list(APPEND seen ${item})
    if(item MATCHES "cudart|^CUDA::|^ninferno_core$")
      message(FATAL_ERROR "${target} is host-only but its link closure reaches ${item}")
    endif()
    if(NOT TARGET ${item})
      continue()
    endif()
    # A redirect of the shared test tree (ninferno::core_host, ...): a host-only target never takes
    # the shared test library, so the redirect is its static archive.
    get_target_property(redirect ${item} NINFERNO_REDIRECT_STATIC)
    if(redirect)
      list(APPEND todo ${redirect})
      continue()
    endif()
    get_target_property(kind ${item} TYPE)
    set(props INTERFACE_LINK_LIBRARIES)
    if(NOT kind STREQUAL "INTERFACE_LIBRARY")
      list(APPEND props LINK_LIBRARIES)
    endif()
    foreach(prop IN LISTS props)
      get_target_property(deps ${item} ${prop})
      if(NOT deps)
        continue()
      endif()
      foreach(dep IN LISTS deps)
        string(REGEX REPLACE "^\\$<(LINK_ONLY|BUILD_INTERFACE):(.*)>$" "\\2" dep "${dep}")
        if(dep MATCHES "^\\$<INSTALL_INTERFACE:")
          continue() # install-time names, not this build's closure
        endif()
        if(dep MATCHES "\\$<")
          message(FATAL_ERROR "${target}: the host-only check cannot read the generator expression "
                              "${dep} of ${item}; teach ninferno_assert_host_only about it")
        endif()
        list(APPEND todo ${dep})
      endforeach()
    endforeach()
  endwhile()
endfunction()

# --- The shared test tree (NINFERNO_SHARED_OP_TESTS) ---------------------------------------------
# The archives that go whole into ninferno_ops_shared (cmake/SharedTestLibrary.cmake). Each holds
# per-process or per-device state (the substituted device class, PerDevice slots, the facts cache,
# the CUDA runtime's registrations) that must exist once in a process, so an archive is in the
# library whole or not at all, and an executable that takes the library takes none of them
# (ninferno_assert_link_mode). ninferno_sys, which every target links by link_libraries(), is not
# a member: it holds no state (operating-system calls and one constant table), so the copy an
# executable links beside the library's is harmless, and under its own name it leaves every app's
# link line as it is (a redirect of it reorders them).
set(NINFERNO_SHARED_MEMBERS core_host core ops nvfp4_non_rdc ggml_mmq)

# ninferno::<m> is the name every non-member library and every test links a member by. With the
# option off it is an alias of the archive ninferno_<m>, so every link line is the one the archive's
# own name gives. With it on it resolves on the executable being linked (the head): the archive for
# an executable without the NINFERNO_SHARED_OPS property (every app and bench: the product link,
# unchanged), the shared test library plus the archive's compile requirements for one with it.
# Members link each other by their own names. Called once the members exist (after src/).
function(ninferno_define_redirects)
  foreach(m IN LISTS NINFERNO_SHARED_MEMBERS)
    if(NOT TARGET ninferno_${m})
      continue() # the NVFP4 archive outside the NVFP4 trees
    endif()
    if(NOT NINFERNO_SHARED_OP_TESTS)
      add_library(ninferno::${m} ALIAS ninferno_${m})
      continue()
    endif()
    add_library(ninferno::${m} INTERFACE IMPORTED GLOBAL)
    set(shared "$<BOOL:$<TARGET_PROPERTY:NINFERNO_SHARED_OPS>>")
    set_property(TARGET ninferno::${m} PROPERTY INTERFACE_LINK_LIBRARIES
      "$<${shared}:ninferno_ops_shared>"
      "$<IF:${shared},$<COMPILE_ONLY:ninferno_${m}>,ninferno_${m}>")
    # Read by the closure walks below instead of the generator expression.
    set_target_properties(ninferno::${m} PROPERTIES
      NINFERNO_REDIRECT_KEY NINFERNO_SHARED_OPS
      NINFERNO_REDIRECT_STATIC ninferno_${m}
      NINFERNO_REDIRECT_SHARED ninferno_ops_shared)
  endforeach()
endfunction()

# The link closure of `root` as its link line holds it: every target and name it reaches, each
# redirect resolved by the root's own properties. Compile-only and install-time items are not
# followed. A generator expression it cannot read fails the configure when it could name a member,
# the shared library or a CUDA runtime.
function(ninferno_link_closure out root)
  get_target_property(todo ${root} LINK_LIBRARIES)
  if(NOT todo)
    set(todo "")
  endif()
  set(seen "")
  while(todo)
    list(POP_FRONT todo item)
    # A target linked from another directory is listed between ::@ markers.
    string(REGEX REPLACE "^::@\\([^)]*\\)" "" item "${item}")
    if(item STREQUAL "" OR item STREQUAL "::@")
      continue()
    endif()
    while(item MATCHES "^\\$<(LINK_ONLY|BUILD_INTERFACE):(.*)>$")
      set(item "${CMAKE_MATCH_2}")
    endwhile()
    if(item MATCHES "^\\$<(COMPILE_ONLY|INSTALL_INTERFACE):")
      continue()
    endif()
    if(item MATCHES "\\$<")
      if(item MATCHES "ninferno|cudart")
        message(FATAL_ERROR "${root}: the link-mode check cannot read the generator expression "
                            "${item}; teach ninferno_link_closure about it")
      endif()
      continue()
    endif()
    if(item IN_LIST seen)
      continue()
    endif()
    list(APPEND seen ${item})
    if(NOT TARGET ${item})
      continue()
    endif()
    get_target_property(key ${item} NINFERNO_REDIRECT_KEY)
    if(key)
      get_target_property(root_shared ${root} ${key})
      if(root_shared)
        get_target_property(to ${item} NINFERNO_REDIRECT_SHARED)
      else()
        get_target_property(to ${item} NINFERNO_REDIRECT_STATIC)
      endif()
      list(APPEND todo ${to})
      continue()
    endif()
    get_target_property(kind ${item} TYPE)
    get_target_property(imported ${item} IMPORTED)
    set(props INTERFACE_LINK_LIBRARIES)
    if(kind STREQUAL "STATIC_LIBRARY" AND NOT imported)
      list(APPEND props LINK_LIBRARIES) # a static library's private links reach its consumer
    endif()
    foreach(prop IN LISTS props)
      get_target_property(deps ${item} ${prop})
      if(deps)
        list(APPEND todo ${deps})
      endif()
    endforeach()
  endwhile()
  set(${out} ${seen} PARENT_SCOPE)
endfunction()

# Fails the configure when `target`'s link closure is not of `mode`:
# - shared: a test on the shared library: it reaches ninferno_ops_shared, no member archive (a second
#   copy of the state above) and no static CUDA runtime (a second runtime);
# - static: the product link: neither the shared library nor the shared CUDA runtime;
# - members: the shared library itself: the members, ninferno_sys, third-party libraries and the
#   shared CUDA runtime only.
function(ninferno_assert_link_mode target mode)
  ninferno_link_closure(closure ${target})
  set(members "")
  foreach(m IN LISTS NINFERNO_SHARED_MEMBERS)
    list(APPEND members ninferno_${m})
  endforeach()
  set(bad "")
  if(mode STREQUAL "shared")
    if(NOT "ninferno_ops_shared" IN_LIST closure)
      list(APPEND bad "no ninferno_ops_shared")
    endif()
    foreach(item IN LISTS closure)
      if(item IN_LIST members OR item STREQUAL "CUDA::cudart_static")
        list(APPEND bad ${item})
      endif()
    endforeach()
  elseif(mode STREQUAL "static")
    foreach(item IN LISTS closure)
      if(item STREQUAL "ninferno_ops_shared" OR item STREQUAL "CUDA::cudart")
        list(APPEND bad ${item})
      endif()
    endforeach()
  elseif(mode STREQUAL "members")
    foreach(item IN LISTS closure)
      if(item STREQUAL "ninferno_ops_shared" OR item STREQUAL "CUDA::cudart_static" OR
         (item MATCHES "^ninferno_" AND NOT item IN_LIST members AND NOT item STREQUAL "ninferno_sys"))
        list(APPEND bad ${item})
      endif()
    endforeach()
  else()
    message(FATAL_ERROR "ninferno_assert_link_mode: unknown mode ${mode}")
  endif()
  if(bad)
    list(JOIN bad ", " bad)
    message(FATAL_ERROR "${target} must link ${mode} but its link closure holds: ${bad}. A test "
                        "links a member archive by its redirect (ninferno::<m>), never by name.")
  endif()
endfunction()

# The shared test tree's link modes, decided once every target exists (called at the end of the
# top-level CMakeLists.txt): a test executable whose static closure holds device code (ninferno_core,
# ninferno_ops or their device archives) takes the shared library and the shared CUDA runtime,
# unless it was declared STATIC_LINK (ninferno_add_test); a test with no device code keeps its
# static link (it would only gain a dependency on the CUDA runtime). Apps and benches stay static.
# Each decision is asserted, and the list is written to <build>/ninferno_link_modes.txt for
# tools/audit/link_mode.sh.
function(ninferno_decide_link_modes)
  set(device ninferno_core ninferno_ops ninferno_nvfp4_non_rdc ninferno_ggml_mmq)
  set(lines "")
  get_property(tests DIRECTORY ${PROJECT_SOURCE_DIR}/tests PROPERTY BUILDSYSTEM_TARGETS)
  foreach(t IN LISTS tests)
    get_target_property(kind ${t} TYPE)
    if(NOT kind STREQUAL "EXECUTABLE")
      continue()
    endif()
    set(mode static)
    get_target_property(static_link ${t} NINFERNO_STATIC_LINK)
    if(NOT static_link)
      ninferno_link_closure(closure ${t})
      foreach(item IN LISTS device)
        if(item IN_LIST closure)
          set(mode shared)
        endif()
      endforeach()
    endif()
    if(mode STREQUAL "shared")
      set_target_properties(${t} PROPERTIES
        NINFERNO_SHARED_OPS ON
        NINFERNO_SHARED_CUDART ON
        CUDA_RUNTIME_LIBRARY Shared)
    endif()
    ninferno_assert_link_mode(${t} ${mode})
    string(APPEND lines "${mode} ${t} $<TARGET_FILE:${t}>\n")
  endforeach()
  set(product_dirs "")
  if(NINFERNO_BUILD_APPS)
    list(APPEND product_dirs apps)
  endif()
  if(NINFERNO_BUILD_BENCHMARKS)
    list(APPEND product_dirs bench)
  endif()
  foreach(dir IN LISTS product_dirs)
    get_property(targets DIRECTORY ${PROJECT_SOURCE_DIR}/${dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(t IN LISTS targets)
      get_target_property(kind ${t} TYPE)
      if(kind STREQUAL "EXECUTABLE")
        ninferno_assert_link_mode(${t} static)
        string(APPEND lines "product ${t} $<TARGET_FILE:${t}>\n")
      endif()
    endforeach()
  endforeach()
  ninferno_assert_link_mode(ninferno_ops_shared members)
  string(APPEND lines "library ninferno_ops_shared $<TARGET_FILE:ninferno_ops_shared>\n")
  file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/ninferno_link_modes.txt CONTENT "${lines}")
endfunction()
