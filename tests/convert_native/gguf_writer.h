#pragma once

// A minimal GGUF v3 writer for tests: metadata of every value type, tensors as raw ggml blocks.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ninferno::convert::testing {

class GgufWriter {
public:
    explicit GgufWriter(std::uint64_t alignment = 32) : alignment_(alignment) {}

    void kv_u32(const std::string& key, std::uint32_t value) {
        header(key, 4);
        put(meta_, value);
        ++kv_count_;
    }

    void kv_i32(const std::string& key, std::int32_t value) {
        header(key, 5);
        put(meta_, value);
        ++kv_count_;
    }

    void kv_u64(const std::string& key, std::uint64_t value) {
        header(key, 10);
        put(meta_, value);
        ++kv_count_;
    }

    void kv_f32(const std::string& key, float value) {
        header(key, 6);
        put(meta_, value);
        ++kv_count_;
    }

    void kv_bool(const std::string& key, bool value) {
        header(key, 7);
        meta_.push_back(value ? 1 : 0);
        ++kv_count_;
    }

    void kv_str(const std::string& key, const std::string& value) {
        header(key, 8);
        str(meta_, value);
        ++kv_count_;
    }

    void kv_arr_str(const std::string& key, const std::vector<std::string>& values) {
        header(key, 9);
        put(meta_, std::uint32_t{8});
        put(meta_, static_cast<std::uint64_t>(values.size()));
        for (const auto& v : values) { str(meta_, v); }
        ++kv_count_;
    }

    void kv_arr_i32(const std::string& key, const std::vector<std::int32_t>& values) {
        header(key, 9);
        put(meta_, std::uint32_t{5});
        put(meta_, static_cast<std::uint64_t>(values.size()));
        for (const auto v : values) { put(meta_, v); }
        ++kv_count_;
    }

    void kv_arr_f32(const std::string& key, const std::vector<float>& values) {
        header(key, 9);
        put(meta_, std::uint32_t{6});
        put(meta_, static_cast<std::uint64_t>(values.size()));
        for (const auto v : values) { put(meta_, v); }
        ++kv_count_;
    }

    void kv_arr_bool(const std::string& key, const std::vector<bool>& values) {
        header(key, 9);
        put(meta_, std::uint32_t{7});
        put(meta_, static_cast<std::uint64_t>(values.size()));
        for (const bool v : values) { meta_.push_back(v ? 1 : 0); }
        ++kv_count_;
    }

    // ne in ggml order (ne[0] contiguous); bytes are the raw blocks.
    void tensor(const std::string& name, const std::vector<std::uint64_t>& ne, std::uint32_t type,
                const std::vector<std::uint8_t>& bytes) {
        while (data_.size() % alignment_) { data_.push_back(0); }
        str(infos_, name);
        put(infos_, static_cast<std::uint32_t>(ne.size()));
        for (const auto n : ne) { put(infos_, n); }
        put(infos_, type);
        put(infos_, static_cast<std::uint64_t>(data_.size()));
        data_.insert(data_.end(), bytes.begin(), bytes.end());
        ++tensor_count_;
    }

    [[nodiscard]] std::vector<std::uint8_t> finish(std::uint32_t version = 3) const {
        std::vector<std::uint8_t> out;
        put(out, std::uint32_t{0x46554747});
        put(out, version);
        put(out, tensor_count_);
        put(out, kv_count_);
        out.insert(out.end(), meta_.begin(), meta_.end());
        out.insert(out.end(), infos_.begin(), infos_.end());
        while (out.size() % alignment_) { out.push_back(0); }
        out.insert(out.end(), data_.begin(), data_.end());
        return out;
    }

private:
    template <typename T> static void put(std::vector<std::uint8_t>& out, T value) {
        const auto at = out.size();
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &value, sizeof(T));
    }

    static void str(std::vector<std::uint8_t>& out, const std::string& value) {
        put(out, static_cast<std::uint64_t>(value.size()));
        out.insert(out.end(), value.begin(), value.end());
    }

    void header(const std::string& key, std::uint32_t type) {
        str(meta_, key);
        put(meta_, type);
    }

    std::uint64_t alignment_;
    std::uint64_t kv_count_     = 0;
    std::uint64_t tensor_count_ = 0;
    std::vector<std::uint8_t> meta_;
    std::vector<std::uint8_t> infos_;
    std::vector<std::uint8_t> data_;
};

} // namespace ninferno::convert::testing
