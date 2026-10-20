// Tokenization parity with Hugging Face transformers on a real checkpoint's tokenizer files.
//
//   python tests/models/qwen3_5/hf_tokenizer_parity.py <checkpoint or tokenizer dir> <out dir>
//
// writes <out dir>/{tokenizer.json, tokenizer_config.json, generation_config.json,
// expected.jsonl} with the ids and the decoded text transformers produces for every text of its
// corpus; then
//
//   NINFERNO_TOKENIZER_PARITY_DIR=<out dir> ninferno_qwen3_5_tokenizer_parity_test
//
// encodes and decodes the same texts with NInferno's Qwen tokenizer and compares. Without the
// variable the test is skipped (77): the files are the checkpoint's, not the repository's.

#include "models/qwen3_5/frontend/tokenizer.h"
#include "support/files.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fi = ninferno::models::qwen3_5::frontend;

} // namespace

int main() {
    const char* dir_text = std::getenv("NINFERNO_TOKENIZER_PARITY_DIR");
    if (dir_text == nullptr || *dir_text == '\0') {
        std::cout << "tokenizer parity: NINFERNO_TOKENIZER_PARITY_DIR not set, skipped\n";
        return 77;
    }
    const std::filesystem::path dir(dir_text);
    const std::string tokenizer_json  = read_file(dir / "tokenizer.json");
    const std::string config_json     = read_file(dir / "tokenizer_config.json");
    const std::string generation_json = read_file(dir / "generation_config.json");
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = config_json,
                                   .generation_config_json = generation_json});

    std::ifstream expected(dir / "expected.jsonl");
    if (!expected) { throw std::runtime_error("cannot read expected.jsonl in " + dir.string()); }
    std::size_t texts = 0, tokens = 0, encode_mismatches = 0, decode_mismatches = 0;
    std::string line;
    while (std::getline(expected, line)) {
        if (line.empty()) { continue; }
        const auto item     = nlohmann::json::parse(line);
        const auto text     = item.at("text").get<std::string>();
        const auto ids      = item.at("ids").get<std::vector<int>>();
        const auto decoded  = item.at("decoded").get<std::string>();
        const auto got      = tokenizer.encode(text);
        ++texts;
        tokens += ids.size();
        if (got != ids) {
            std::size_t at = 0;
            while (at < got.size() && at < ids.size() && got[at] == ids[at]) { ++at; }
            if (++encode_mismatches <= 5) {
                std::cerr << "encode mismatch in text " << texts << " at token " << at << " of "
                          << ids.size() << " (ninferno " << (at < got.size() ? got[at] : -1)
                          << ", transformers " << (at < ids.size() ? ids[at] : -1) << ")\n";
            }
        }
        if (tokenizer.decode(ids) != decoded && ++decode_mismatches <= 5) {
            std::cerr << "decode mismatch in text " << texts << "\n";
        }
    }
    std::cout << nlohmann::json{{"dir", dir.string()},
                                {"texts", texts},
                                {"tokens", tokens},
                                {"encode_mismatches", encode_mismatches},
                                {"decode_mismatches", decode_mismatches}}
                     .dump()
              << '\n';
    return texts > 0 && encode_mismatches == 0 && decode_mismatches == 0 ? 0 : 1;
}
