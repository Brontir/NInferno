#include "convert/gguf.h"
#include "core/ascii.h"
#include "core/checked.h"

#include "convert/ggml_quants.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninferno::convert {
namespace {

constexpr std::uint32_t kMagic = 0x46554747; // "GGUF" little-endian

// Bounds-checked little-endian reader over the file bytes.
class Reader {
public:
    Reader(const std::byte* data, std::size_t size, const std::string& label)
        : data_(data), size_(size), label_(label) {}

    template <typename T> T read() {
        need(sizeof(T));
        T value;
        std::memcpy(&value, data_ + at_, sizeof(T));
        at_ += sizeof(T);
        return value;
    }

    std::string string() {
        const auto length = read<std::uint64_t>();
        need(length);
        std::string out(reinterpret_cast<const char*>(data_ + at_), length);
        at_ += length;
        return out;
    }

    [[nodiscard]] std::size_t position() const noexcept { return at_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return size_ - at_; }

    // Refuses `count` items of at least `bytes` each that the bytes held cannot contain: a
    // damaged count, or a head that ends inside them.
    void need_items(std::uint64_t count, std::uint64_t bytes) const {
        if (count > remaining() / bytes) {
            throw GgufTruncated(label_ + ": truncated GGUF (" + std::to_string(count) +
                                " items at byte " + std::to_string(at_) + ")");
        }
    }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error(label_ + ": " + what + " (at byte " + std::to_string(at_) + ")");
    }

private:
    void need(std::uint64_t bytes) const {
        if (bytes > size_ - at_) {
            throw GgufTruncated(label_ + ": truncated GGUF (at byte " + std::to_string(at_) + ")");
        }
    }

    const std::byte* data_;
    std::size_t size_;
    const std::string& label_;
    std::size_t at_ = 0;
};

nlohmann::json scalar(Reader& in, GgufValueType type) {
    switch (type) {
    case GgufValueType::U8:
        return in.read<std::uint8_t>();
    case GgufValueType::I8:
        return in.read<std::int8_t>();
    case GgufValueType::U16:
        return in.read<std::uint16_t>();
    case GgufValueType::I16:
        return in.read<std::int16_t>();
    case GgufValueType::U32:
        return in.read<std::uint32_t>();
    case GgufValueType::I32:
        return in.read<std::int32_t>();
    case GgufValueType::U64:
        return in.read<std::uint64_t>();
    case GgufValueType::I64:
        return in.read<std::int64_t>();
    case GgufValueType::F32:
        return f32_shortest(in.read<float>());
    case GgufValueType::F64:
        return in.read<double>();
    case GgufValueType::BOOL: {
        const auto value = in.read<std::uint8_t>();
        if (value > 1) { in.fail("invalid BOOL value"); }
        return value == 1;
    }
    case GgufValueType::STRING:
        return in.string();
    case GgufValueType::ARRAY:
        break;
    }
    in.fail("invalid metadata value type " + std::to_string(static_cast<std::uint32_t>(type)));
}

bool valid_type(std::uint32_t type) { return type <= 12; }

// The fewest bytes a value of `type` takes (a string: its length word).
std::uint64_t min_bytes(GgufValueType type) {
    switch (type) {
    case GgufValueType::U8:
    case GgufValueType::I8:
    case GgufValueType::BOOL:
        return 1;
    case GgufValueType::U16:
    case GgufValueType::I16:
        return 2;
    case GgufValueType::U32:
    case GgufValueType::I32:
    case GgufValueType::F32:
        return 4;
    default:
        return 8;
    }
}

} // namespace

std::vector<std::uint64_t> GgufTensor::shape() const { return {ne.rbegin(), ne.rend()}; }

std::uint64_t GgufTensor::elements() const noexcept {
    std::uint64_t out = 1;
    for (const auto n : ne) { out *= n; }
    return out;
}

double f32_shortest(float value) {
    char text[64];
    const auto result = std::to_chars(text, text + sizeof(text), value);
    *result.ptr       = '\0';
    return std::strtod(text, nullptr);
}

bool gguf_auxiliary_file(const std::string& file_name) {
    const std::string name = ascii_lower(file_name.substr(file_name.find_last_of('/') + 1));
    return name.find("mmproj") != std::string::npos || name.find("imatrix") != std::string::npos ||
           name.find("mtp-only") != std::string::npos ||
           name.find("mtp_only") != std::string::npos || name.starts_with("mtp-") ||
           name.starts_with("mtp_");
}

namespace {

// The file's first `size` bytes of `file_size`: the header, the metadata and the tensor infos read
// from them, the tensors checked against the whole file.
GgufFile parse(const std::byte* data, std::size_t size, std::uint64_t file_size,
               const std::string& label) {
    Reader in(data, size, label);
    GgufFile out;
    if (in.read<std::uint32_t>() != kMagic) { in.fail("not a GGUF file"); }
    out.version = in.read<std::uint32_t>();
    if (out.version == 0x03000000 || out.version == 0x02000000) {
        in.fail("big-endian GGUF files are not supported");
    }
    if (out.version != 2 && out.version != 3) {
        in.fail("unsupported GGUF version " + std::to_string(out.version));
    }
    const auto tensor_count = in.read<std::uint64_t>();
    const auto kv_count     = in.read<std::uint64_t>();
    if (tensor_count > file_size / 24 || kv_count > file_size / 12) {
        in.fail("implausible GGUF counts");
    }
    for (std::uint64_t i = 0; i < kv_count; ++i) {
        auto key        = in.string();
        const auto type = in.read<std::uint32_t>();
        if (!valid_type(type)) { in.fail(key + ": invalid metadata type " + std::to_string(type)); }
        if (out.metadata.contains(key)) { in.fail("duplicate metadata key " + key); }
        if (static_cast<GgufValueType>(type) == GgufValueType::ARRAY) {
            const auto element = in.read<std::uint32_t>();
            const auto count   = in.read<std::uint64_t>();
            if (!valid_type(element) || static_cast<GgufValueType>(element) == GgufValueType::ARRAY) {
                in.fail(key + ": unsupported array element type " + std::to_string(element));
            }
            if (count > file_size) { in.fail(key + ": implausible array length"); }
            in.need_items(count, min_bytes(static_cast<GgufValueType>(element)));
            auto values = nlohmann::json::array();
            values.get_ref<nlohmann::json::array_t&>().reserve(count);
            for (std::uint64_t j = 0; j < count; ++j) {
                values.push_back(scalar(in, static_cast<GgufValueType>(element)));
            }
            out.types[key]    = {static_cast<GgufValueType>(element), true};
            out.metadata[key] = std::move(values);
        } else {
            out.types[key]    = {static_cast<GgufValueType>(type), false};
            out.metadata[key] = scalar(in, static_cast<GgufValueType>(type));
        }
    }
    if (const auto* alignment = out.find("general.alignment")) {
        if (!alignment->is_number_unsigned() || alignment->get<std::uint64_t>() == 0 ||
            (alignment->get<std::uint64_t>() & (alignment->get<std::uint64_t>() - 1))) {
            in.fail("general.alignment must be a positive power of two");
        }
        out.alignment = alignment->get<std::uint64_t>();
    }
    // A tensor info holds at least a name length, a dimension count, one dimension, a type and an
    // offset.
    in.need_items(tensor_count, 32);
    out.tensors.reserve(tensor_count);
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensor tensor;
        tensor.name        = in.string();
        const auto n_dims  = in.read<std::uint32_t>();
        if (n_dims == 0 || n_dims > 4) {
            in.fail(tensor.name + ": " + std::to_string(n_dims) + " dimensions");
        }
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            const auto n = in.read<std::uint64_t>();
            if (n == 0 || n > (1ULL << 40)) { in.fail(tensor.name + ": invalid dimension"); }
            tensor.ne.push_back(n);
        }
        tensor.type   = in.read<std::uint32_t>();
        tensor.offset = in.read<std::uint64_t>();
        const auto traits = ggml::traits(tensor.type);
        if (!traits) {
            in.fail(tensor.name + ": unknown ggml type " + std::to_string(tensor.type));
        }
        if (tensor.ne[0] % traits->block) {
            in.fail(tensor.name + ": row of " + std::to_string(tensor.ne[0]) +
                    " values does not fill whole " + ggml::type_name(tensor.type) + " blocks");
        }
        tensor.bytes      = tensor.ne[0] / traits->block * traits->bytes;
        std::uint64_t all = tensor.ne[0];
        for (std::size_t d = 1; d < tensor.ne.size(); ++d) {
            if (mul_overflow(tensor.bytes, tensor.ne[d], &tensor.bytes) ||
                mul_overflow(all, tensor.ne[d], &all)) {
                in.fail(tensor.name + ": shape overflows 64 bits");
            }
        }
        out.tensors.push_back(std::move(tensor));
    }
    const auto end_of_infos = static_cast<std::uint64_t>(in.position());
    out.data_offset = (end_of_infos + out.alignment - 1) / out.alignment * out.alignment;
    // A shard of metadata alone (gguf-split --no-tensor-first-split) ends right after the infos,
    // without padding to a data section it does not have.
    if (out.tensors.empty()) { return out; }
    if (out.data_offset > file_size) { in.fail("data section starts beyond the end of the file"); }
    const auto data_bytes = file_size - out.data_offset;
    std::map<std::string, int> seen;
    for (const auto& tensor : out.tensors) {
        if (!seen.emplace(tensor.name, 0).second) { in.fail("duplicate tensor " + tensor.name); }
        if (tensor.offset % out.alignment) { in.fail(tensor.name + ": misaligned tensor data"); }
        if (tensor.offset > data_bytes || tensor.bytes > data_bytes - tensor.offset) {
            in.fail(tensor.name + ": tensor data beyond the end of the file");
        }
    }
    return out;
}

} // namespace

GgufFile parse_gguf(const std::byte* data, std::size_t size, const std::string& label) {
    return parse(data, size, size, label);
}

GgufFile parse_gguf_head(const std::byte* data, std::size_t size, std::uint64_t file_size,
                         const std::string& label) {
    if (size > file_size) { throw std::invalid_argument(label + ": a head longer than its file"); }
    return parse(data, size, file_size, label);
}

} // namespace ninferno::convert
