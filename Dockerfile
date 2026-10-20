# syntax=docker/dockerfile:1

# CUDA 13 no longer compiles for Volta: a V100 image takes --build-arg CUDA_VERSION=12.8.1.
ARG CUDA_VERSION=13.4.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        git \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# The GPU architecture to build for: 120a, 89, 86, 80, 75 or 70 (see README).
ARG CUDA_ARCH=120a
# The version the binaries report (default: the project version).
ARG NINFERNO_VERSION=
WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DNINFERNO_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFERNO_BUILD_BENCHMARKS=OFF \
        -DNINFERNO_SOURCE_TREE_PATHS=OFF \
        -DNINFERNO_UI_CHAT=OFF \
        ${NINFERNO_VERSION:+-DNINFERNO_VERSION_STRING=${NINFERNO_VERSION}} \
    && cmake --build /build --parallel --target ninferno ninferno-serve \
    && bash tools/release/pdfium.sh linux-x64 /pdfium/lib /pdfium/licenses \
    && mkdir -p /licenses \
    && { [ ! -f /build/_deps/cutlass-src/LICENSE.txt ] \
         || install -D -m 0644 /build/_deps/cutlass-src/LICENSE.txt /licenses/cutlass/LICENSE.txt; }

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libssl3t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninferno /usr/local/bin/ninferno
COPY --from=build /build/apps/ninferno-serve /usr/local/bin/ninferno-serve
# The per-GPU route tables and the drafters' token rankings, where the binaries look beside
# themselves: without them the kernels run untuned routes.
COPY --from=build /src/routes /usr/local/share/ninferno/routes
COPY --from=build /src/tools/spec/rankings/*.i64 /usr/local/share/ninferno/rankings/
COPY --from=build /src/tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64 /usr/local/share/ninferno/rankings/
# pdfium, which ninferno-serve loads for PDF documents, and the licenses.
COPY --from=build /pdfium/lib/ /usr/local/lib/
COPY --from=build /pdfium/licenses /usr/local/share/licenses/pdfium
COPY LICENSE NOTICE /usr/local/share/licenses/ninferno/
# CUTLASS's, where a 70 build compiled it in.
COPY --from=build /licenses/ /usr/local/share/licenses/
COPY third_party /tmp/third_party
RUN cd /tmp/third_party \
    && find . -maxdepth 4 \( -name 'LICENSE*' -o -name 'NOTICE*' -o -name AUTHORS -o -name 'UNICODE-LICENSE' \) \
        -exec install -D -m 0644 {} /usr/local/share/licenses/{} \; \
    && rm -rf /tmp/third_party \
    && ldconfig

# The server needs no root: it reads mounted models and listens on an unprivileged port. The
# Ubuntu base image's own unprivileged user (uid 1000) runs it and owns the working directory.
WORKDIR /workspace
RUN chown ubuntu:ubuntu /workspace
USER ubuntu
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninferno-serve", "--help"]
