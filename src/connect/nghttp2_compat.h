#pragma once

// The libnghttp2 calls whose form changed in 1.60, for 1.59 (Ubuntu 24.04) and later alike.
// 1.60 added the "2" calls, which return nghttp2_ssize, and deprecated the older ones, which
// return ssize_t; 1.59 has only the older ones. Each pair takes the same arguments and gives the
// same results; only the type of the signed length differs (nghttp2_ssize is ptrdiff_t, as wide as
// ssize_t on Linux). The data provider's read callback differs the same way and nothing else.
#include <nghttp2/nghttp2.h>
#include <sys/types.h>

namespace ninferno::connect::h2 {

#if NGHTTP2_VERSION_NUM >= 0x013c00
using ssize                           = nghttp2_ssize;
using DataProvider                    = nghttp2_data_provider2;
inline constexpr auto mem_recv        = &nghttp2_session_mem_recv2;
inline constexpr auto mem_send        = &nghttp2_session_mem_send2;
inline constexpr auto submit_request  = &nghttp2_submit_request2;
inline constexpr auto submit_response = &nghttp2_submit_response2;
#else
using ssize                           = ssize_t;
using DataProvider                    = nghttp2_data_provider;
inline constexpr auto mem_recv        = &nghttp2_session_mem_recv;
inline constexpr auto mem_send        = &nghttp2_session_mem_send;
inline constexpr auto submit_request  = &nghttp2_submit_request;
inline constexpr auto submit_response = &nghttp2_submit_response;
#endif

} // namespace ninferno::connect::h2
