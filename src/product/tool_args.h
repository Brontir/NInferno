#pragma once

// What the maintainer tools ninferno-dense and ninferno-gemma4 share on their command lines: the
// artifact, the command and --key value options, token-id files and their comma-joined echo.

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninferno::product {

struct ToolArguments {
    std::string artifact, command;
    std::map<std::string, std::string> options;

    bool flag(const std::string& key) const { return options.contains(key); }

    std::string text(const std::string& key, std::string fallback = {}) const {
        const auto it = options.find(key);
        return it == options.end() ? fallback : it->second;
    }

    std::int64_t number(const std::string& key, std::int64_t fallback) const {
        const auto it = options.find(key);
        return it == options.end() ? fallback : std::stoll(it->second);
    }
};

// A file of little-endian int32 token ids; an empty one is an error.
inline std::vector<std::int32_t> read_tokens(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path); }
    file.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(file.tellg());
    file.seekg(0);
    std::vector<std::int32_t> tokens(bytes / 4);
    file.read(reinterpret_cast<char*>(tokens.data()),
              static_cast<std::streamsize>(tokens.size() * 4));
    if (tokens.empty()) { throw std::runtime_error(path + " holds no tokens"); }
    return tokens;
}

// "1,2,3".
inline std::string join_ids(const std::vector<std::int32_t>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        out += (i ? "," : "") + std::to_string(values[i]);
    }
    return out;
}

} // namespace ninferno::product
