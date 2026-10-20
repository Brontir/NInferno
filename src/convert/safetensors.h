#pragma once

// A Hugging Face checkpoint directory: config.json, the safetensors index (or one
// model.safetensors), and every shard memory-mapped read-only. Tensor bytes are addressed in
// place; nothing is copied until a consumer converts it.
//
// A GGUF file (or a directory holding one model's GGUF files) opens as a checkpoint too: its
// metadata becomes config.json, and its tensors appear under their Hugging Face names as views
// (tensor_view.h) that undo llama.cpp's conversion and decode ggml blocks on read.

#include "convert/gguf_request.h"
#include "convert/gguf.h"
#include "convert/pyjson.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace ninferno::convert {

enum class SourceDType { BF16, F16, F32, F64, I32, I64, I8, U8, F8_E4M3 };

[[nodiscard]] std::size_t dtype_bytes(SourceDType dtype) noexcept;

class TensorView;

struct TensorInfo {
    std::string name;
    std::string dtype_name; // as the header spells it
    std::vector<std::uint64_t> shape;
    std::uint64_t bytes   = 0;
    const std::byte* data = nullptr; // inside the shard's mapping
    int file              = -1;
    bool known_dtype      = false;
    SourceDType dtype     = SourceDType::U8;
    // A GGUF tensor: the ggml type of its blocks (data and bytes as stored).
    std::uint32_t ggml_type = 0;
    // A computed tensor (no data): values come from the view, as `dtype` words.
    std::shared_ptr<const TensorView> view;
    [[nodiscard]] std::uint64_t elements() const noexcept;
};

class Checkpoint {
public:
    // `path` is the --model argument: a directory (the published form) or one .safetensors file.
    explicit Checkpoint(const std::filesystem::path& path,
                        GgufRequest request = GgufRequest::Model);
    ~Checkpoint();
    Checkpoint(const Checkpoint&)            = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    [[nodiscard]] const OJson& config() const noexcept { return config_; }

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

    // str(pathlib.Path(argument)), which source labels and provenance quote.
    [[nodiscard]] const std::string& path_string() const noexcept { return path_string_; }

    [[nodiscard]] bool has(const std::string& name) const;
    // Throws when the tensor is absent.
    [[nodiscard]] const TensorInfo& describe(const std::string& name) const;
    // The tensor's bytes, after checking its dtype is readable and its size matches its shape.
    [[nodiscard]] const TensorInfo& readable(const std::string& name) const;

    // Drop mapped pages from this process (bounded RSS) and, with `drop_cache`, from the page
    // cache as well (only pages the range covers whole). Data stays valid to re-read.
    void release(const std::byte* begin, std::size_t bytes, bool drop_cache) const noexcept;
    void prefetch(const std::byte* begin, std::size_t bytes) const noexcept;
    void release_all() const noexcept;
    // Copy a whole tensor with pread, without mapping it into the process: for the thousands of
    // scattered scalars a plan reads, which fault-around would otherwise inflate. The tensor is
    // recorded as read at plan time (pinned): a source-consuming conversion never punches it,
    // because a resumed run reads it again to rebuild the same plan.
    void read(const TensorInfo& tensor, void* out) const;
    // Bytes [offset, offset + bytes) of a tensor, read and pinned the same way.
    void read_range(const TensorInfo& tensor, std::uint64_t offset, std::uint64_t bytes,
                    void* out) const;
    [[nodiscard]] bool pinned(const TensorInfo& tensor) const;
    // Unmap every page from this process; the page cache keeps them.
    void unmap_all() const noexcept;

    // Shards, for source-consuming conversion.
    [[nodiscard]] std::size_t shard_count() const noexcept { return shards_.size(); }

    [[nodiscard]] const std::filesystem::path& shard_path(std::size_t shard) const {
        return shards_.at(shard).path;
    }

    [[nodiscard]] std::uint64_t shard_bytes(std::size_t shard) const {
        return shards_.at(shard).bytes;
    }

    // Bytes of the shard's safetensors header (length word included).
    [[nodiscard]] std::uint64_t shard_header_bytes(std::size_t shard) const {
        return shards_.at(shard).header_bytes;
    }

    [[nodiscard]] std::uint64_t shard_offset(const TensorInfo& tensor) const;

    // The stored tensors, each once (for a GGUF checkpoint: under their GGUF names).
    [[nodiscard]] const std::map<std::string, TensorInfo, std::less<>>& tensors() const noexcept {
        return tensors_;
    }

    // Deallocate shard bytes (FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE). Destroys the source.
    void punch(std::size_t shard, std::uint64_t offset, std::uint64_t bytes) const;

    // The parsed GGUF file (the first split) when this checkpoint is one, else null.
    [[nodiscard]] const GgufFile* gguf() const noexcept {
        return gguf_.empty() ? nullptr : &gguf_.front();
    }

    // What opening a GGUF checkpoint mapped: one line per kind of rename or inverse transform.
    [[nodiscard]] const std::vector<std::string>& mapping_notes() const noexcept {
        return mapping_notes_;
    }
    // The GGUF mapping's version where it has one (GgufMapping::mapper).
    [[nodiscard]] const std::string& gguf_mapper() const noexcept { return mapper_; }

    // The Hugging Face-named computed tensors of a GGUF checkpoint.
    [[nodiscard]] const std::map<std::string, TensorInfo, std::less<>>& views() const noexcept {
        return views_;
    }

    // Frontend resources a GGUF checkpoint's metadata rebuilds (gguf_resources.h), by role, and
    // what was rebuilt or could not be; empty for a safetensors checkpoint.
    [[nodiscard]] const std::map<std::string, std::string>& rebuilt_resources() const noexcept {
        return rebuilt_resources_;
    }

    [[nodiscard]] const std::vector<std::string>& rebuilt_notes() const noexcept {
        return rebuilt_notes_;
    }

private:
    struct Shard {
        std::filesystem::path path;
        int fd                     = -1;
        mutable int write_fd       = -1; // opened on the first punch
        std::uint64_t bytes        = 0;
        std::uint64_t header_bytes = 0;
        std::byte* base            = nullptr;
    };

    int shard_for(const std::byte* pointer) const noexcept;
    void load_shard(int index);
    void open_gguf(const std::vector<std::filesystem::path>& files, GgufRequest request);
    void map_file(Shard& shard);
    const TensorInfo* find(std::string_view name) const;

    std::filesystem::path root_;
    std::string path_string_;
    OJson config_;
    std::vector<Shard> shards_;
    std::map<std::string, int, std::less<>> weight_map_;
    std::map<std::string, TensorInfo, std::less<>> tensors_;
    std::map<std::string, TensorInfo, std::less<>> views_;
    std::vector<GgufFile> gguf_;
    std::vector<std::string> mapping_notes_;
    std::string mapper_;
    std::map<std::string, std::string> rebuilt_resources_;
    std::vector<std::string> rebuilt_notes_;
    mutable std::mutex pinned_mutex_;
    mutable std::set<const TensorInfo*> pinned_;
};

// The GGUF files of the model at `path` (a .gguf file, possibly the first of a split, or a
// directory holding one model's GGUF files and no safetensors), or empty when it is not one.
[[nodiscard]] std::vector<std::filesystem::path> gguf_model_files(const std::filesystem::path& path);

} // namespace ninferno::convert
