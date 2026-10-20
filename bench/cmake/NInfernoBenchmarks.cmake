function(ninferno_add_op_bench name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
  # A build without the NVFP4 family (89, 86, 75) drops its sources from the engine
  # (ninferno_target_sources); a bench of that family has nothing to measure there.
  if(NOT NINFERNO_WITH_NVFP4)
    foreach(src IN LISTS arg_SOURCES)
      if(src MATCHES "/nvfp4/|nvfp4[^/]*\\.(cu|cpp)$|generic_w4a4\\.cu$")
        return()
      endif()
    endforeach()
  endif()
  add_executable(${name} ${arg_SOURCES})
  ninferno_internal_includes(${name})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops)
  target_link_libraries(${name} PRIVATE ninferno_ops)
  ninferno_cuda_lineinfo(${name})
endfunction()
