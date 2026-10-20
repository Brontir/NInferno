#include "convert/safetensors.h"

#include "convert/files.h"
#include "core/ascii.h"

#include "convert/ggml_quants.h"
#include "convert/gguf_model.h"
#include "convert/gguf_resources.h"
#include "convert/tensor_view.h"

#include "core/checked.h"
#include "core/sys.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ninferno::convert {
namespace {

// A JSON document of the checkpoint; a malformed one is an error that names its file.
template <class Json>
Json parse_named(std::string_view text, const std::filesystem::path& file) {
    try {
        return Json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(file.string() + ": " + error.what());
    }
}

bool parse_dtype(const std::string& name, SourceDType& out) {
    static const std::pair<const char*, SourceDType> kNames[] = {
        {"BF16", SourceDType::BF16}, {"F16", SourceDType::F16}, {"F32", SourceDType::F32},
        {"F64", SourceDType::F64},   {"I32", SourceDType::I32}, {"I64", SourceDType::I64},
        {"I8", SourceDType::I8},     {"U8", SourceDType::U8},   {"F8_E4M3", SourceDType::F8_E4M3},
    };
    for (const auto& [spelling, dtype] : kNames) {
        if (name == spelling) {
            out = dtype;
            return true;
        }
    }
    return false;
}

std::uint64_t page_size() {
    static const auto size = static_cast<std::uint64_t>(sys::page_size());
    return size;
}

// llama.cpp's gguf-split naming: <stem>-00001-of-00003.gguf.
const std::regex& split_name() {
    static const std::regex pattern(R"((.*)-(\d{5})-of-(\d{5})\.gguf)", std::regex::icase);
    return pattern;
}

// Every part of the split `file` belongs to, in order (the file alone when it is not a split).
std::vector<std::filesystem::path> split_parts(const std::filesystem::path& file) {
    std::smatch match;
    const auto name = file.filename().string();
    if (!std::regex_match(name, match, split_name())) { return {file}; }
    const auto count = std::stoul(match[3].str());
    std::vector<std::filesystem::path> out;
    for (unsigned long i = 1; i <= count; ++i) {
        char number[16];
        std::snprintf(number, sizeof(number), "%05lu", i);
        auto part = file.parent_path() /
                    (match[1].str() + "-" + number + "-of-" + match[3].str() + ".gguf");
        if (!std::filesystem::is_regular_file(part)) {
            throw std::runtime_error("GGUF split part missing: " + part.string());
        }
        out.push_back(std::move(part));
    }
    return out;
}

} // namespace

std::vector<std::filesystem::path> gguf_model_files(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    if (fs::is_regular_file(path)) {
        return ascii_lower(path.extension().string()) == ".gguf" ? split_parts(path)
                                                                 : std::vector<fs::path>{};
    }
    if (!fs::is_directory(path) || fs::exists(path / "model.safetensors.index.json") ||
        fs::exists(path / "model.safetensors")) {
        return {};
    }
    std::map<std::string, fs::path> models; // split stem or file name -> first file
    for (const auto& entry : fs::directory_iterator(path)) {
        if (!entry.is_regular_file()) { continue; }
        const auto name      = entry.path().filename().string();
        const auto extension = ascii_lower(entry.path().extension().string());
        if (extension == ".safetensors") { return {}; }
        if (extension != ".gguf" || gguf_auxiliary_file(name)) { continue; }
        std::smatch match;
        const auto key = std::regex_match(name, match, split_name()) ? match[1].str() : name;
        auto& first    = models[key];
        if (first.empty() || entry.path() < first) { first = entry.path(); }
    }
    if (models.empty()) { return {}; }
    if (models.size() > 1) {
        std::string names;
        for (const auto& [key, file] : models) {
            names += (names.empty() ? "" : ", ") + file.filename().string();
        }
        throw std::runtime_error(path.string() + " holds several GGUF models (" + names +
                                 "); pass one file");
    }
    return split_parts(models.begin()->second);
}

std::size_t dtype_bytes(SourceDType dtype) noexcept {
    switch (dtype) {
    case SourceDType::F64:
    case SourceDType::I64:
        return 8;
    case SourceDType::F32:
    case SourceDType::I32:
        return 4;
    case SourceDType::BF16:
    case SourceDType::F16:
        return 2;
    default:
        return 1;
    }
}

std::uint64_t TensorInfo::elements() const noexcept {
    std::uint64_t out = 1;
    for (const auto dim : shape) { out *= dim; }
    return out;
}

Checkpoint::Checkpoint(const std::filesystem::path& path, GgufRequest request) {
    namespace fs         = std::filesystem;
    path_string_         = py_path_str(path.string());
    const bool directory = fs::is_directory(path);
    root_                = directory ? path : path.parent_path();
    if (root_.empty()) { root_ = "."; }
    if (const auto files = gguf_model_files(path); !files.empty()) {
        open_gguf(files, request);
        return;
    }
    const auto config_path = root_ / "config.json";
    config_                = fs::is_regular_file(config_path)
                                 ? parse_named<OJson>(read_file(config_path), config_path)
                                 : OJson::object();

    std::vector<std::pair<std::string, fs::path>> entries;
    const auto add_whole_file = [&](const fs::path& file) {
        shards_.push_back({file});
        load_shard(0);
        for (const auto& [name, info] : tensors_) { weight_map_[name] = 0; }
    };
    if (fs::is_regular_file(path) && path.extension() == ".safetensors") {
        add_whole_file(path);
        return;
    }
    if (fs::is_regular_file(path) && path.extension() != ".json") {
        throw std::runtime_error(path.string() +
                                 ": neither a safetensors file, a safetensors index "
                                 "(.json) nor a GGUF model");
    }
    const auto index = fs::is_regular_file(path) ? path : root_ / "model.safetensors.index.json";
    if (fs::is_regular_file(index)) {
        // nlohmann::json, not ordered_json: an index maps tens of thousands of names, and the
        // ordered map inserts in linear time. Order is irrelevant here.
        const auto data = parse_named<nlohmann::json>(read_file(index), index);
        std::map<std::string, int> files;
        for (const auto& [name, file] : data.at("weight_map").items()) {
            const auto shard    = index.parent_path() / file.get<std::string>();
            auto [it, inserted] = files.emplace(shard.string(), static_cast<int>(shards_.size()));
            if (inserted) { shards_.push_back({shard}); }
            weight_map_[name] = it->second;
        }
        for (int i = 0; i < static_cast<int>(shards_.size()); ++i) { load_shard(i); }
        // Python reads a tensor only through the index: a name the shards hold but the index
        // omits is not visible, and one the index names but no shard holds fails on use. The
        // index's file for a name is not trusted over the shard headers: transformers loads every
        // listed shard whole, so a published index that files a tensor under the wrong shard of
        // the same checkpoint (ibm-granite/granite-4.2-3b maps 101 of 363 that way) still loads,
        // and the tensor is read from the shard that holds it.
        for (auto it = tensors_.begin(); it != tensors_.end();) {
            const auto mapped = weight_map_.find(it->first);
            if (mapped == weight_map_.end()) {
                it = tensors_.erase(it);
            } else {
                mapped->second = it->second.file;
                ++it;
            }
        }
        return;
    }
    const auto file = root_ / "model.safetensors";
    if (fs::is_regular_file(file)) {
        add_whole_file(file);
        return;
    }
    if (!directory) { throw std::runtime_error("no such checkpoint: " + path.string()); }
}

Checkpoint::~Checkpoint() {
    for (auto& shard : shards_) {
        if (shard.base) { sys::unmap(shard.base, shard.bytes); }
        if (shard.fd >= 0) { sys::close(shard.fd); }
        if (shard.write_fd >= 0) { sys::close(shard.write_fd); }
    }
}

void Checkpoint::map_file(Shard& shard) {
    shard.fd = sys::open(shard.path, O_RDONLY | O_CLOEXEC);
    if (shard.fd < 0) { throw std::runtime_error("cannot open " + shard.path.string()); }
    const std::int64_t size = sys::file_size(shard.fd);
    if (size < 0) { throw std::runtime_error("cannot stat " + shard.path.string()); }
    shard.bytes = static_cast<std::uint64_t>(size);
    if (shard.bytes < 8) {
        throw std::runtime_error(shard.path.string() + ": truncated checkpoint header");
    }
    void* base = sys::map(shard.fd, shard.bytes, sys::Map::ReadShared);
    if (base == nullptr) { throw std::runtime_error("cannot map " + shard.path.string()); }
    shard.base = static_cast<std::byte*>(base);
}

void Checkpoint::open_gguf(const std::vector<std::filesystem::path>& files, GgufRequest request) {
    for (std::size_t i = 0; i < files.size(); ++i) {
        shards_.push_back({files[i]});
        auto& shard = shards_.back();
        map_file(shard);
        gguf_.push_back(parse_gguf(shard.base, shard.bytes, shard.path.string()));
        const auto& file   = gguf_.back();
        shard.header_bytes = file.data_offset;
        for (const auto& tensor : file.tensors) {
            TensorInfo info;
            info.name       = tensor.name;
            info.shape      = tensor.shape();
            info.bytes      = tensor.bytes;
            info.data       = shard.base + file.data_offset + tensor.offset;
            info.file       = static_cast<int>(i);
            info.ggml_type  = tensor.type;
            info.dtype_name = ascii_upper(ggml::type_name(tensor.type));
            if (tensor.type == ggml::F32 || tensor.type == ggml::F16 ||
                tensor.type == ggml::BF16) {
                info.known_dtype = parse_dtype(info.dtype_name, info.dtype);
            }
            if (!tensors_.emplace(tensor.name, std::move(info)).second) {
                throw std::runtime_error(shard.path.string() + ": tensor " + tensor.name +
                                         " appears in two GGUF splits");
            }
            weight_map_[tensor.name] = static_cast<int>(i);
        }
    }
    const StoredLookup lookup = [this](const std::string& name) -> const TensorInfo* {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    };
    const StoredRead read = [this](const TensorInfo& tensor, std::uint64_t offset,
                                   std::uint64_t bytes, void* out) {
        read_range(tensor, offset, bytes, out);
    };
    auto mapping = map_gguf_model(gguf_.front(), lookup, read, path_string_, request);
    config_      = std::move(mapping.config);
    for (auto& view : mapping.views) {
        auto name = view.name;
        if (!views_.emplace(std::move(name), std::move(view)).second) {
            throw std::logic_error("GGUF mapping produced a name twice");
        }
    }
    mapping_notes_ = std::move(mapping.notes);
    mapper_        = std::move(mapping.mapper);
    auto rebuilt       = gguf_frontend_resources(gguf_.front());
    rebuilt_resources_ = std::move(rebuilt.files);
    rebuilt_notes_     = std::move(rebuilt.notes);
}

void Checkpoint::load_shard(int index) {
    auto& shard = shards_[static_cast<std::size_t>(index)];
    map_file(shard);
    std::uint64_t length = 0;
    std::memcpy(&length, shard.base, 8);
    shard.header_bytes = 8 + length;
    if (length > shard.bytes - 8) {
        throw std::runtime_error(shard.path.string() + ": safetensors header exceeds file");
    }
    const auto header = parse_named<nlohmann::json>(
        std::string_view(reinterpret_cast<const char*>(shard.base) + 8, length), shard.path);
    const auto data_bytes = shard.bytes - length - 8;
    for (const auto& [name, item] : header.items()) {
        if (name == "__metadata__") { continue; }
        TensorInfo info;
        info.name           = name;
        std::uint64_t begin = 0;
        std::uint64_t end   = 0;
        try {
            info.dtype_name = item.at("dtype").get<std::string>();
            info.shape      = item.at("shape").get<std::vector<std::uint64_t>>();
            begin           = item.at("data_offsets").at(0).get<std::uint64_t>();
            end             = item.at("data_offsets").at(1).get<std::uint64_t>();
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error(shard.path.string() + "/" + name + ": " + error.what());
        }
        std::uint64_t elements = 1;
        for (const auto dim : info.shape) {
            if (mul_overflow(elements, dim, &elements)) {
                throw std::runtime_error(shard.path.string() + "/" + name +
                                         ": shape overflows 64 bits");
            }
        }
        if (!(begin <= end && end <= data_bytes)) {
            throw std::runtime_error(shard.path.string() + "/" + name + ": invalid source range");
        }
        info.bytes       = end - begin;
        info.data        = shard.base + 8 + length + begin;
        info.file        = index;
        info.known_dtype = parse_dtype(info.dtype_name, info.dtype);
        tensors_.insert_or_assign(name, std::move(info));
    }
    sys::advise_file(shard.fd, 0, 0, sys::Advice::Normal);
}

bool Checkpoint::has(const std::string& name) const {
    return weight_map_.contains(name) || views_.contains(name);
}

const TensorInfo* Checkpoint::find(std::string_view name) const {
    // A GGUF is read under the transformers names its mapping gives; a mapped name may coincide
    // with a stored GGUF name (a DFlash2 drafter's fc.weight), and then the mapping's view is it.
    if (!gguf_.empty()) {
        if (const auto it = views_.find(name); it != views_.end()) { return &it->second; }
    }
    if (const auto it = tensors_.find(name); it != tensors_.end()) { return &it->second; }
    if (const auto it = views_.find(name); it != views_.end()) { return &it->second; }
    return nullptr;
}

const TensorInfo& Checkpoint::describe(const std::string& name) const {
    const auto* info = find(name);
    if (!info) {
        throw std::runtime_error(path_string_ + ": missing source tensor '" + name + "'");
    }
    return *info;
}

const TensorInfo& Checkpoint::readable(const std::string& name) const {
    const auto& info = describe(name);
    if (info.view) { return info; }
    if (!info.known_dtype) {
        throw std::runtime_error(name + ": unsupported source dtype " + info.dtype_name);
    }
    std::uint64_t expected = 0;
    if (mul_overflow(info.elements(), dtype_bytes(info.dtype), &expected) ||
        info.bytes != expected) {
        throw std::runtime_error(name + ": source byte count disagrees with shape/dtype");
    }
    return info;
}

int Checkpoint::shard_for(const std::byte* pointer) const noexcept {
    for (int i = 0; i < static_cast<int>(shards_.size()); ++i) {
        const auto& shard = shards_[static_cast<std::size_t>(i)];
        if (pointer >= shard.base && pointer < shard.base + shard.bytes) { return i; }
    }
    return -1;
}

void Checkpoint::release(const std::byte* begin, std::size_t bytes,
                         bool drop_cache) const noexcept {
    const int index = shard_for(begin);
    if (index < 0 || !bytes) { return; }
    const auto& shard = shards_[static_cast<std::size_t>(index)];
    const auto offset = static_cast<std::uint64_t>(begin - shard.base);
    const auto page   = page_size();
    // Unmapping is safe to round outward (a neighbour still reading a boundary page faults it
    // back from the cache); dropping cached pages rounds inward, so no neighbour loses data.
    const auto outer_first = offset / page * page;
    const auto outer_last =
        std::min<std::uint64_t>((offset + bytes + page - 1) / page * page, shard.bytes);
    sys::advise(shard.base + outer_first, outer_last - outer_first, sys::Advice::DontNeed);
    const auto first = (offset + page - 1) / page * page;
    const auto last  = (offset + bytes) / page * page;
    if (drop_cache && last > first) {
        sys::advise_file(shard.fd, static_cast<std::int64_t>(first),
                         static_cast<std::int64_t>(last - first), sys::Advice::DontNeed);
    }
}

void Checkpoint::prefetch(const std::byte* begin, std::size_t bytes) const noexcept {
    const int index = shard_for(begin);
    if (index < 0 || !bytes) { return; }
    const auto& shard = shards_[static_cast<std::size_t>(index)];
    const auto offset = static_cast<std::uint64_t>(begin - shard.base);
    const auto page   = page_size();
    const auto first  = offset / page * page;
    sys::advise(shard.base + first, offset + bytes - first, sys::Advice::WillNeed);
}

void Checkpoint::release_all() const noexcept {
    unmap_all();
    for (const auto& shard : shards_) {
        if (shard.fd >= 0) { sys::advise_file(shard.fd, 0, 0, sys::Advice::DontNeed); }
    }
}

void Checkpoint::read(const TensorInfo& tensor, void* out) const {
    read_range(tensor, 0, tensor.bytes, out);
}

void Checkpoint::read_range(const TensorInfo& tensor, std::uint64_t first, std::uint64_t count,
                            void* out) const {
    if (first + count > tensor.bytes) { throw std::logic_error("read beyond " + tensor.name); }
    if (tensor.view) {
        // A computed tensor: its value words, from stored tensors pinned like any plan read.
        const auto word = dtype_bytes(tensor.dtype);
        if (first % word || count % word) {
            throw std::logic_error("unaligned read of " + tensor.name);
        }
        tensor.view->values(first / word, (first + count) / word, out, nullptr);
        std::vector<const TensorInfo*> stored;
        tensor.view->tensors(stored);
        std::lock_guard lock(pinned_mutex_);
        pinned_.insert(stored.begin(), stored.end());
        return;
    }
    const auto& shard = shards_.at(static_cast<std::size_t>(tensor.file));
    auto offset       = static_cast<std::int64_t>(tensor.data - shard.base + first);
    auto* cursor      = static_cast<char*>(out);
    auto bytes        = static_cast<std::size_t>(count);
    while (bytes) {
        const auto count = sys::pread(shard.fd, cursor, bytes, offset);
        if (count <= 0) { throw std::runtime_error("short read from " + shard.path.string()); }
        cursor += count;
        offset += count;
        bytes -= static_cast<std::size_t>(count);
    }
    std::lock_guard lock(pinned_mutex_);
    pinned_.insert(&tensor);
}

bool Checkpoint::pinned(const TensorInfo& tensor) const {
    std::lock_guard lock(pinned_mutex_);
    return pinned_.contains(&tensor);
}

std::uint64_t Checkpoint::shard_offset(const TensorInfo& tensor) const {
    if (tensor.view) { throw std::logic_error(tensor.name + " is computed, not stored"); }
    return static_cast<std::uint64_t>(tensor.data -
                                      shards_.at(static_cast<std::size_t>(tensor.file)).base);
}

void Checkpoint::punch(std::size_t index, std::uint64_t offset, std::uint64_t bytes) const {
    const auto& shard = shards_.at(index);
    if (shard.write_fd < 0) {
        shard.write_fd = sys::open(shard.path, O_WRONLY | O_CLOEXEC);
        if (shard.write_fd < 0) {
            throw std::runtime_error("cannot open " + shard.path.string() +
                                     " to consume it: " + sys::error_text(errno));
        }
    }
    if (sys::punch_hole(shard.write_fd, static_cast<std::int64_t>(offset),
                        static_cast<std::int64_t>(bytes)) != 0) {
        throw std::runtime_error("cannot punch " + shard.path.string() + ": " +
                                 sys::error_text(errno));
    }
}

void Checkpoint::unmap_all() const noexcept {
    for (const auto& shard : shards_) {
        if (shard.base) { sys::advise(shard.base, shard.bytes, sys::Advice::DontNeed); }
    }
}

} // namespace ninferno::convert
