# ninferno-connect's library (#339, docs/connect.md): the worker's reversed connection to a router.
# Built where pkg-config finds libnghttp2 (the HTTP/2 session) and CMake finds OpenSSL 3 (TLS); not
# on Windows. Without them the tree builds as before, with no ninferno-connect.
if(WIN32)
  return()
endif()
pkg_check_modules(NGHTTP2 IMPORTED_TARGET GLOBAL libnghttp2>=1.60)
find_package(OpenSSL 3)
if(NOT NGHTTP2_FOUND OR NOT OpenSSL_FOUND)
  message(STATUS "ninferno-connect: not built (it needs libnghttp2 >= 1.60 and OpenSSL 3)")
  return()
endif()
# Seen by the tests too (a stand-in router's TLS).
set_target_properties(OpenSSL::SSL OpenSSL::Crypto PROPERTIES IMPORTED_GLOBAL TRUE)

add_library(ninferno_connect STATIC
  link.cpp
  tunnel.cpp
  connector.cpp
)
ninferno_internal_includes(ninferno_connect)
target_link_libraries(ninferno_connect
  PUBLIC Threads::Threads ninferno::json ninferno::httplib
  PRIVATE PkgConfig::NGHTTP2 OpenSSL::SSL OpenSSL::Crypto)
