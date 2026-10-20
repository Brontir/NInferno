add_library(ninferno_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninferno_internal_includes(ninferno_media_acquire)
target_link_libraries(ninferno_media_acquire PRIVATE PkgConfig::LIBCURL)

add_library(ninferno_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninferno_internal_includes(ninferno_product_prompt_input)
target_link_libraries(ninferno_product_prompt_input PRIVATE ninferno_media_acquire ninferno::json)

add_library(ninferno_product_logging STATIC
  logging/logging.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninferno_internal_includes(ninferno_product_logging)
target_link_libraries(ninferno_product_logging PUBLIC spdlog::spdlog)

# Whether a newer release is out (one small manifest over https, at most daily).
add_library(ninferno_product_update STATIC
  update/update_check.cpp
)
ninferno_internal_includes(ninferno_product_update)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
target_link_libraries(ninferno_product_update PRIVATE PkgConfig::LIBCURL OpenSSL::Crypto ninferno::json)

# The speculation a Qwen artifact came with (its resolved profile), as the CLI and the server run it.
add_library(ninferno_product_speculation STATIC
  speculation/speculation.cpp
)
ninferno_internal_includes(ninferno_product_speculation)
target_link_libraries(ninferno_product_speculation PUBLIC ninferno_spec PRIVATE ninferno_core)
