// The engine side of the H1 diff oracle (hf_oracle.py): teacher-forced logits of an artifact over
// given token sequences, written for comparison against Hugging Face transformers.
//
//   ninferno_hf_oracle_runner FAMILY ARTIFACT TOKENS OUT [mode=score|decode] [profile=artifact|a16]
//                           [chunk=N] [context=N] [invariant=1] [last=N] [activations=exact]
//
// FAMILY is qwen3_5 (the public Engine, CausalScoring) or gemma4 (Model and Session). TOKENS holds
// little-endian int32 words: the sequence count, then each sequence as its length and its ids. OUT
// receives FP32 logits, for every sequence the positions 0 .. length-2 (position i predicts id
// i+1), each over the public token domain. mode=decode (Gemma 4) produces them one decode step at
// a time from the pending token instead of from the scoring prefill; profile=a16 loads Gemma 4
// with ActivationProfile::A16Full (no activation encoded to four bits); activations=exact runs
// Qwen with --activations exact (no int8 activations). mode=draft (Gemma 4, with
// drafter=PATH, an MTP assistant artifact) prefills each sequence in a fresh session and runs one
// speculative round at depth 1; with NINFERNO_GEMMA4_DUMP set, its eager warm-up step writes the
// drafter's input and logits (mtp_round.cu). OUT is left empty in that mode.

#include "core/device.h"
#include "models/gemma4/model.h"
#include "models/gemma4/session.h"
#include "ninferno/engine.h"
#include "ninferno/ops/prefill_routes.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::vector<std::int32_t>> read_sequences(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<std::int32_t> words;
    std::int32_t word = 0;
    while (in.read(reinterpret_cast<char*>(&word), sizeof(word))) { words.push_back(word); }
    std::vector<std::vector<std::int32_t>> out;
    std::size_t cursor = 0;
    const auto next    = [&] {
        if (cursor >= words.size()) { throw std::runtime_error("truncated token file"); }
        return words[cursor++];
    };
    const std::int32_t count = next();
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t length = next();
        if (length < 2) { throw std::runtime_error("every sequence needs two tokens"); }
        std::vector<std::int32_t> ids;
        for (std::int32_t j = 0; j < length; ++j) { ids.push_back(next()); }
        out.push_back(std::move(ids));
    }
    return out;
}

void append(std::ofstream& out, const std::vector<float>& values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

int run_qwen(const std::filesystem::path& artifact,
             const std::vector<std::vector<std::int32_t>>& sequences,
             const std::filesystem::path& out, std::uint32_t context, bool exact) {
    ninferno::EngineOptions options;
    options.artifact_path = artifact;
    if (exact) { options.activations = ninferno::ActivationMode::Exact; }
    options.purpose       = ninferno::EnginePurpose::CausalScoring;
    options.max_context   = context;
    ninferno::Engine engine(options);
    // The scoring program appends every column's logits to this file (NINFERNO_SCORE_LOGITS).
    std::ofstream(out, std::ios::binary | std::ios::trunc).close();
    setenv("NINFERNO_SCORE_LOGITS", out.c_str(), 1);
    for (const auto& ids : sequences) {
        const auto scores = engine.score_tokens(ids, 1);
        if (scores.size() + 1 != ids.size()) { throw std::runtime_error("short score"); }
    }
    unsetenv("NINFERNO_SCORE_LOGITS");
    return 0;
}

int run_gemma4(const std::filesystem::path& artifact,
               const std::vector<std::vector<std::int32_t>>& sequences,
               const std::filesystem::path& out, const std::string& mode,
               const std::string& profile, std::int32_t chunk, std::int32_t context,
               const std::string& drafter_path, bool invariant, std::int32_t last) {
    using namespace ninferno::models::gemma4;
    // invariant=1: the route profile of --activations exact (prefill_routes.h) around everything.
    const ninferno::ops::ScopedInvariantRoutes invariant_routes(invariant);
    std::optional<ninferno::ops::ScopedPrefillRoutes> prefill_routes;
    if (invariant) { prefill_routes.emplace(true); }
    ninferno::DeviceContext device(0);
    const auto model = Model::load(artifact, device,
                                   profile == "a16" ? ActivationProfile::A16Full
                                                    : ActivationProfile::Artifact);
    SessionOptions options;
    options.max_context   = context;
    options.prefill_chunk = chunk;
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    if (mode == "draft") {
        if (drafter_path.empty()) { throw std::invalid_argument("mode=draft needs drafter=PATH"); }
        const auto drafter    = Drafter::load(drafter_path, device, model->config());
        options.drafter       = drafter.get();
        options.draft_tokens  = 1;
        for (const auto& ids : sequences) {
            Session session(*model, device, options); // fresh graphs: the round runs eagerly first
            session.prefill(ids);
            (void)session.speculate(1);
        }
        return 0;
    }
    Session session(*model, device, options);
    for (const auto& ids : sequences) {
        session.reset();
        if (mode == "decode") {
            session.prefill({ids.data(), 1});
            append(file, session.last_logits());
            for (std::size_t i = 1; i + 1 < ids.size(); ++i) {
                session.set_pending_token(ids[i]);
                (void)session.decode(1);
                append(file, session.last_logits());
            }
        } else {
            std::vector<float> logits;
            (void)session.score(ids, nullptr, &logits);
            if (last > 0) {
                // Only the last `last` positions (a chunking diagnosis needs the tail, not GBs).
                const std::size_t row  = logits.size() / (ids.size() - 1);
                const std::size_t keep = std::min<std::size_t>(last, ids.size() - 1) * row;
                logits.erase(logits.begin(),
                             logits.begin() + static_cast<std::ptrdiff_t>(logits.size() - keep));
            }
            append(file, logits);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: ninferno_hf_oracle_runner FAMILY ARTIFACT TOKENS OUT [key=value]...\n";
        return 2;
    }
    std::map<std::string, std::string> options;
    for (int i = 5; i < argc; ++i) {
        const std::string item = argv[i];
        const auto eq          = item.find('=');
        if (eq == std::string::npos) {
            std::cerr << "option without a value: " << item << '\n';
            return 2;
        }
        options[item.substr(0, eq)] = item.substr(eq + 1);
    }
    const auto option = [&](const std::string& key, const std::string& fallback) {
        const auto it = options.find(key);
        return it == options.end() ? fallback : it->second;
    };
    try {
        const std::string family = argv[1];
        const auto sequences     = read_sequences(argv[3]);
        std::size_t longest      = 0;
        for (const auto& ids : sequences) { longest = std::max(longest, ids.size()); }
        const auto context = static_cast<std::int32_t>(
            std::stoi(option("context", std::to_string(std::max<std::size_t>(1024, longest)))));
        if (family == "qwen3_5") {
            return run_qwen(argv[2], sequences, argv[4], static_cast<std::uint32_t>(context),
                            option("activations", "fast") == "exact");
        }
        if (family == "gemma4") {
            return run_gemma4(argv[2], sequences, argv[4], option("mode", "score"),
                              option("profile", "artifact"), std::stoi(option("chunk", "512")),
                              context, option("drafter", ""), option("invariant", "0") == "1",
                              std::stoi(option("last", "0")));
        }
        std::cerr << "unknown family " << family << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "ninferno_hf_oracle_runner: " << error.what() << '\n';
        return 1;
    }
}
