# Vendored XGrammar core (C++ only). See README.ninferno.md for provenance and local policy.
file(GLOB NINFERNO_XGRAMMAR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/support/*.cc)

add_library(ninferno_third_party_xgrammar STATIC ${NINFERNO_XGRAMMAR_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/ninferno/log_hook.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ninferno/tool_syntax_grammars.cc)
add_library(ninferno::xgrammar ALIAS ninferno_third_party_xgrammar)

target_compile_features(ninferno_third_party_xgrammar PUBLIC cxx_std_17)
set_target_properties(ninferno_third_party_xgrammar PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ninferno_third_party_xgrammar PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ninferno)
target_include_directories(ninferno_third_party_xgrammar SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/picojson
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include)
target_include_directories(ninferno_third_party_xgrammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
# XGRAMMAR_LOG_CUSTOMIZE is upstream's logging hook. The NInferno-owned ninferno/log_hook.cc
# implements LogFatalImpl and LogMessageImpl so a compile-time degradation warning becomes a
# request error rather than a stderr line. The upstream source changes (object key order and a
# caller-made converter, in cpp/json_schema_converter.*) are described in README.ninferno.md.
target_compile_definitions(ninferno_third_party_xgrammar PUBLIC
  XGRAMMAR_LOG_CUSTOMIZE=1
  XGRAMMAR_ENABLE_CPPTRACE=0
  XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ninferno_third_party_xgrammar PRIVATE -O3 -w)
endif()
target_link_libraries(ninferno_third_party_xgrammar PUBLIC Threads::Threads)
