// A dense-decoder artifact end to end on the device. NINFERNO_TEST_DENSE_ARTIFACTS names one or
// more artifacts (':'-separated), NINFERNO_TEST_DENSE_TOKENS a token file (little-endian int32 ids
// of real text in that model's vocabulary; one per artifact, ':'-separated); without them the test
// skips. Per artifact:
//   - chunk invariance: scoring the same 700 tokens with chunks 128 and 2048 gives bit-identical
//     log-probabilities (the prefill route profile);
//     and a 704-slot context under the 2048 chunk scores them identically too;
//   - the teacher-forced NLL is finite and plausible for real text (below 5 nats);
//   - greedy decode is deterministic: two sessions give the same 32 tokens, and the argmax the
//     scorer predicts for the prompt's last position is the first decoded token.

#include "core/device.h"
#include "models/dense/session.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::models::dense;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ':')) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
}

std::vector<std::int32_t> read_tokens(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::int32_t> out;
    std::int32_t v = 0;
    while (file.read(reinterpret_cast<char*>(&v), 4)) { out.push_back(v); }
    return out;
}

void run(const std::string& artifact, const std::vector<std::int32_t>& all) {
    const std::string label = artifact.substr(artifact.find_last_of('/') + 1);
    check(all.size() >= 700, label + ": the token file holds fewer than 700 ids");
    if (all.size() < 700) { return; }
    const std::vector<std::int32_t> tokens(all.begin(), all.begin() + 700);
    DeviceContext device(0);
    auto model = Model::load(artifact, device, ActivationProfile::Artifact);
    std::vector<float> scores[2];
    std::vector<std::int32_t> predictions;
    const std::int32_t chunks[2] = {128, 2048};
    for (int i = 0; i < 2; ++i) {
        SessionOptions options;
        options.max_context   = 4096;
        options.prefill_chunk = chunks[i];
        Session session(*model, device, options);
        scores[i] = session.score(tokens, i == 1 ? &predictions : nullptr);
    }
    check(scores[0].size() == 699 && scores[1].size() == 699, label + ": 699 scores");
    check(std::memcmp(scores[0].data(), scores[1].data(), scores[0].size() * 4) == 0,
          label + ": log-probabilities depend on the prefill chunk");
    // A context shorter than the prefill chunk (2048 against 704 slots): the session runs the chunk
    // as the context and scores the same log-probabilities.
    {
        SessionOptions options;
        options.max_context   = 704;
        options.prefill_chunk = 2048;
        Session session(*model, device, options);
        const auto short_context = session.score(tokens, nullptr);
        check(short_context.size() == 699 &&
                  std::memcmp(short_context.data(), scores[1].data(), 699 * 4) == 0,
              label + ": a context below the chunk changes the log-probabilities");
    }
    double nll = 0;
    for (const float v : scores[1]) { nll -= v; }
    nll /= static_cast<double>(scores[1].size());
    std::cout << label << ": NLL " << nll << " over 699 tokens\n";
    check(std::isfinite(nll) && nll > 0 && nll < 5.0, label + ": implausible NLL");

    const std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + 300);
    std::vector<std::int32_t> decoded[2];
    for (int i = 0; i < 2; ++i) {
        SessionOptions options;
        options.max_context   = 4096;
        options.prefill_chunk = 2048;
        Session session(*model, device, options);
        session.prefill(prompt);
        decoded[i] = session.decode(32);
    }
    check(decoded[0] == decoded[1], label + ": greedy decode is not deterministic");
    // An id past the embedding table (a tokenizer's added token the model cannot read) is refused.
    {
        SessionOptions options;
        options.max_context   = 64;
        options.prefill_chunk = 64;
        Session session(*model, device, options);
        const std::vector<std::int32_t> outside = {tokens[0], model->config().vocab_size};
        bool refused                            = false;
        try {
            session.prefill(outside);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, label + ": prefill accepted an id past the vocabulary");
    }
    // The scorer's argmax after token 299 predicts the token the decode's first step consumes.
    check(!decoded[0].empty() && decoded[0][0] == predictions[299],
          label + ": the first decoded token differs from the scorer's argmax");
}

} // namespace

int main() {
    const char* artifacts = std::getenv("NINFERNO_TEST_DENSE_ARTIFACTS");
    const char* tokens    = std::getenv("NINFERNO_TEST_DENSE_TOKENS");
    if (artifacts == nullptr || tokens == nullptr) {
        std::cout << "SKIP: set NINFERNO_TEST_DENSE_ARTIFACTS and NINFERNO_TEST_DENSE_TOKENS\n";
        return 77;
    }
    const auto paths = split(artifacts);
    const auto files = split(tokens);
    if (paths.size() != files.size()) {
        std::cerr << "one token file per artifact\n";
        return 1;
    }
    try {
        for (std::size_t i = 0; i < paths.size(); ++i) { run(paths[i], read_tokens(files[i])); }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "dense session real: all checks passed\n";
    return 0;
}
