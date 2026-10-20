add_library(ninferno_linear_add_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_add_test_common.cpp")
ninferno_test_includes(ninferno_linear_add_test_support)
ninferno_op_oracle_options(ninferno_linear_add_test_support)
ninferno_link_ops(ninferno_linear_add_test_support PUBLIC)

ninferno_add_op_test(ninferno_linear_add_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninferno_linear_add_test_support)

ninferno_add_op_test(ninferno_linear_add_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninferno_linear_add_test_support)

ninferno_add_op_test(ninferno_linear_add_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninferno_linear_add_test_support)

ninferno_add_op_test(ninferno_linear_add_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninferno_linear_add_test_support)

# NVFP4 kernels ship only in the 120a build.
if(NINFERNO_WITH_NVFP4)
  ninferno_add_op_test(ninferno_linear_add_nvfp4_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
    LIBRARIES ninferno_ops)
endif()

ninferno_add_op_test(ninferno_linear_add_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES ninferno_ops)
