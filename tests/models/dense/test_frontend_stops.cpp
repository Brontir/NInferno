// The dense frontend's end-of-generation ids: generation_config.json's eos_token_id (one id or a
// list) joined with the tokenizer's eos token, each once. CPU only.

#include "models/dense/frontend.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

using Ids = std::vector<std::int32_t>;
using ninferno::models::dense::end_of_generation_ids;

} // namespace

int main() {
    const auto json = [](const char* text) { return nlohmann::json::parse(text); };
    // Hermes 3 (Llama 3.2 3B): the base model's generation_config, ChatML <|im_end|> as tokenizer eos.
    check(end_of_generation_ids(json(R"({"eos_token_id": [128001, 128008, 128009]})"), 128039) ==
              Ids{128001, 128008, 128009, 128039},
          "the tokenizer's eos joins the generation_config list");
    // Llama 3.2 Instruct: the tokenizer's eos (<|eot_id|>) is already listed.
    check(end_of_generation_ids(json(R"({"eos_token_id": [128001, 128008, 128009]})"), 128009) ==
              Ids{128001, 128008, 128009},
          "an eos already listed is not repeated");
    check(end_of_generation_ids(json(R"({"eos_token_id": 2})"), 2) == Ids{2}, "a single id");
    check(end_of_generation_ids(json(R"({"eos_token_id": 2})"), std::nullopt) == Ids{2},
          "no tokenizer eos");
    check(end_of_generation_ids(json(R"({})"), 7) == Ids{7}, "no generation_config eos");
    try {
        (void)end_of_generation_ids(json(R"({})"), std::nullopt);
        check(false, "no end-of-generation id accepted");
    } catch (const std::invalid_argument&) {
    }
    if (failures != 0) { return 1; }
    std::cout << "dense frontend stops: all checks passed\n";
    return 0;
}
