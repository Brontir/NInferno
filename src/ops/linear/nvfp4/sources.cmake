ninferno_target_sources(ninferno_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n8192_k2816.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k2816.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2816_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2816_k8192.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2816_k2112.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5376.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n18432_k5376.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k5376.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n43008_k5376.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5376_k8192.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5376_k16384.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5376_k21504.cu"
)

ninferno_target_sources(ninferno_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma.cu"
)
