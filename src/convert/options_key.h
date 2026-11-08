#pragma once

// The identity of the options a kept conversion was made with (#553): what acquire_model and the
// drafter conversion key their cache entries by.

#include <filesystem>
#include <string>
#include <vector>

namespace ninferno::convert {

// One option of a conversion: its flag, a value that identifies it (a file by its content) and
// what to show the user for it.
struct OptionPart {
    std::string name;
    std::string value;
    std::string shown;
};

// A file or a directory tree by its content: a file of at most 64 MiB by the SHA-256 of its bytes,
// a larger one (a GGUF or checkpoint shard) by its size, modification time and the SHA-256 of its
// first 16 MiB (the header and the tensor index) and its last MiB, which hashing it whole would not
// repay for a key; a directory by every file under it, in name order. "unreadable:<path>" when it
// cannot be read.
[[nodiscard]] std::string content_identity(const std::filesystem::path& path);

// "" when there is no part; else 12 hex digits of the parts' identity, and `described` (when
// given) one "name shown" line per part.
[[nodiscard]] std::string options_key(const std::vector<OptionPart>& parts,
                                      std::string* described = nullptr);

} // namespace ninferno::convert
