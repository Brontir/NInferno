#!/usr/bin/env python3
"""ffmpeg_check.py LIB_DIR: check that the FFmpeg in LIB_DIR is the one a release may carry.

It loads LIB_DIR's libavutil, libswscale, libavcodec and libavformat and asks each for its
licence and configure line, then lists what the build holds. It fails (exit 1) unless every
library reports an LGPL licence, the configure line has neither --enable-gpl nor --enable-nonfree,
and the build has no encoder, no muxer and no protocol; and when LIB_DIR also holds a library that
only a GPL or a full distribution FFmpeg brings (x264, x265, xvid, libpostproc, libavfilter,
libavdevice, libswresample, VA-API, VDPAU, the OpenCL loader). The report, on standard output, is
the BUILD.txt of the release's share/licenses/ffmpeg.
"""

import ctypes
import os
import re
import sys

LIBRARIES = ("avutil", "swscale", "avcodec", "avformat")  # the load order: each needs the ones before
FORBIDDEN = re.compile(
    r"^lib(x264|x265|xvidcore|postproc|avfilter|avdevice|swresample|va|va-drm|va-x11|vdpau|OpenCL)\.so"
)


class Named(ctypes.Structure):
    """The first field of AVCodec and AVInputFormat/AVOutputFormat: const char *name."""

    _fields_ = [("name", ctypes.c_char_p)]


def soname_path(lib_dir, name):
    pattern = re.compile(rf"^lib{name}\.so\.\d+$")
    found = sorted(f for f in os.listdir(lib_dir) if pattern.match(f))
    if len(found) != 1:
        raise SystemExit(f"ffmpeg_check.py: want one lib{name}.so.<major> in {lib_dir}, found {found or 'none'}")
    return os.path.join(lib_dir, found[0])


def iterate(function):
    """The names an av_*_iterate(void **opaque) function yields."""
    function.restype = ctypes.POINTER(Named)
    function.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    opaque = ctypes.c_void_p()
    names = []
    while True:
        item = function(ctypes.byref(opaque))
        if not item:
            return names
        names.append(item.contents.name.decode())


def text(function):
    function.restype = ctypes.c_char_p
    function.argtypes = []
    return function().decode()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: ffmpeg_check.py LIB_DIR")
    lib_dir = os.path.realpath(sys.argv[1])
    libs = {}
    for name in LIBRARIES:
        libs[name] = ctypes.CDLL(soname_path(lib_dir, name), mode=ctypes.RTLD_GLOBAL)
    problems = []
    lines = [f"FFmpeg {text(libs['avutil'].av_version_info)}"]

    for name in LIBRARIES:
        licence = text(getattr(libs[name], f"{name}_license"))
        lines.append(f"lib{name}: {licence}")
        if not licence.startswith("LGPL"):
            problems.append(f"lib{name} reports the licence '{licence}'")
    configurations = {text(getattr(libs[name], f"{name}_configuration")) for name in LIBRARIES}
    if len(configurations) != 1:
        problems.append("the libraries come from different builds (their configure lines differ)")
    for configuration in sorted(configurations):
        lines.append(f"configure: {configuration}")
        for flag in ("--enable-gpl", "--enable-nonfree"):
            if re.search(rf"(^|\s){flag}(\s|$)", configuration):
                problems.append(f"the build was configured with {flag}")

    iterate_codec = libs["avcodec"].av_codec_iterate
    iterate_codec.restype = ctypes.POINTER(Named)
    iterate_codec.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    is_encoder = libs["avcodec"].av_codec_is_encoder
    is_encoder.restype = ctypes.c_int
    is_encoder.argtypes = [ctypes.POINTER(Named)]
    opaque = ctypes.c_void_p()
    decoders, encoders = [], []
    while codec := iterate_codec(ctypes.byref(opaque)):
        (encoders if is_encoder(codec) else decoders).append(codec.contents.name.decode())
    demuxers = iterate(libs["avformat"].av_demuxer_iterate)
    muxers = iterate(libs["avformat"].av_muxer_iterate)
    enum_protocols = libs["avformat"].avio_enum_protocols
    enum_protocols.restype = ctypes.c_char_p
    enum_protocols.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    protocols = set()
    for output in (0, 1):
        opaque = ctypes.c_void_p()
        while (protocol := enum_protocols(ctypes.byref(opaque), output)) is not None:
            protocols.add(protocol.decode())

    lines.append(f"decoders ({len(decoders)}): {' '.join(sorted(decoders))}")
    lines.append(f"demuxers ({len(demuxers)}): {' '.join(sorted(demuxers))}")
    lines.append(f"encoders: {' '.join(sorted(encoders)) or 'none'}")
    lines.append(f"muxers: {' '.join(sorted(muxers)) or 'none'}")
    lines.append(f"protocols: {' '.join(sorted(protocols)) or 'none'}")
    if encoders:
        problems.append(f"the build has encoders: {' '.join(sorted(encoders))}")
    if muxers:
        problems.append(f"the build has muxers: {' '.join(sorted(muxers))}")
    if protocols:
        problems.append(f"the build has protocols: {' '.join(sorted(protocols))}")
    if not decoders or not demuxers:
        problems.append("the build has no decoder or no demuxer")
    forbidden = sorted(f for f in os.listdir(lib_dir) if FORBIDDEN.match(f))
    if forbidden:
        problems.append(f"{lib_dir} holds libraries a release must not carry: {' '.join(forbidden)}")

    print("\n".join(lines))
    if problems:
        for problem in problems:
            print(f"ffmpeg_check.py: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
