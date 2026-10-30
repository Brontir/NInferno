#pragma once

// What is already in a local hub cache, read once for every plan of it: the GGUF quantizations a
// repository has here (a download, or the artifact converted from one where the download was
// removed) and each one's own tensor table, so that ninferno-serve's start, its --plan --no-load
// and ninferno-ui's /api/plan price a cached quantization by the same file the start loads. The
// hub cache root is the caller's (convert::default_hub_cache for ninferno-serve; ninferno-ui's
// --hub-cache, which it passes to the servers it starts as HF_HUB_CACHE).

#include "convert/gguf.h"
#include "product/plan/plan_quants.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::product {

// A GGUF header from the first bytes of a file, read longer until its metadata fits.
[[nodiscard]] convert::GgufFile read_local_head(const std::filesystem::path& path);
// Every part of a split GGUF beside `path` ("x-00001-of-00003.gguf"), or the file alone.
[[nodiscard]] std::uint64_t local_gguf_bytes(const std::filesystem::path& path);
// The GGUF quantizations of hub repository `repo` on `hub` ("huggingface" | "modelscope") at
// `revision` ("": main, as a start without a revision) in the hub cache at `hub_root`, each with
// its file: a downloaded GGUF (the first part of a split one) in the revision's snapshot; where
// the snapshot was removed, the conversion a start serves in its place (convert::kept_artifact).
[[nodiscard]] std::map<std::string, std::filesystem::path>
cached_quants(const std::filesystem::path& hub_root, const std::string& hub,
              const std::string& repo, const std::string& revision);
// What cached_quants found as a key (labels, files, sizes, times): a plan of the repository
// priced by them is planned again when a download or conversion changes it.
[[nodiscard]] std::string cached_stamp(const std::map<std::string, std::filesystem::path>& cached);
// A quantization here: its own tensor table (nullopt: unreadable, a split part). A GGUF's from its
// header; a converted artifact's from its tensors with its own drafter and MTP layers. Read once
// per file as it is (path, size, time).
[[nodiscard]] std::optional<GgufTensorTable> local_tensor_table(const std::filesystem::path& path);
// The listing's quantizations that are here priced by their own files (over a hub header's).
void tables_of_cached(std::vector<QuantListing>& listing,
                      const std::map<std::string, std::filesystem::path>& cached);

} // namespace ninferno::product
