// Which conversion a Hub cache entry is (#553): a conversion made with options that change what
// is written (--vocab q8, a chat template, ...) must never be taken by a later run without them,
// and a default run must never be replaced by it. CPU, no hub.
//
// content_options_key names the options of a run ("" for the defaults), conversion_directory the
// place the kept conversion of that key lives: the commit's directory itself for the defaults
// (existing caches stay valid), options-<key> inside it otherwise.

#include "convert/acquire.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace cv = ninferno::convert;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        ++g_failures;
    }
}

void write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << text;
}

using Change = std::function<void(cv::AcquireOptions&)>;

// The Hub flow in miniature: a run converts into conversion_directory(commit, key) unless that
// directory already holds the artifact, and says which it did.
struct Cache {
    fs::path commit;

    bool convert(const cv::AcquireOptions& options, const std::string& label) {
        const auto key  = cv::content_options_key(options);
        const auto file = cv::conversion_directory(commit, key) / "model.dense_bf16.ninferno";
        if (fs::exists(file)) { return false; } // reused
        write_text(file, label);
        return true;
    }

    std::string served(const cv::AcquireOptions& options) const {
        const auto file = cv::conversion_directory(commit, cv::content_options_key(options)) /
                          "model.dense_bf16.ninferno";
        std::ifstream in(file, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    }
};

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "ninferno_acquire_options_key";
    fs::remove_all(root);
    fs::create_directories(root);

    const cv::AcquireOptions plain;
    check(cv::content_options_key(plain).empty(), "the defaults have no key");
    check(cv::conversion_directory(root / "c0ffee", "") == root / "c0ffee",
          "the default conversion stays where existing caches keep it");

    // Options that are not content do not move the entry.
    const std::vector<std::pair<std::string, Change>> neutral = {
        {"recipe", [](auto& o) { o.convert.recipe = "dense_bf16"; }},
        {"threads", [](auto& o) { o.convert.threads = 3; }},
        {"rows per chunk", [](auto& o) { o.convert.rows_per_chunk = 64; }},
        {"quant", [](auto& o) { o.fetch.quant = "Q4_K_M"; }},
        {"vision", [](auto& o) { o.vision = true; }},
        {"runtime chat template", [&](auto& o) { o.chat_template = root / "custom.jinja"; }},
        {"reuse", [](auto& o) { o.reuse = true; }},
        {"with drafters", [](auto& o) { o.drafters = cv::Drafters::Require; }},
        {"drafters attach", [](auto& o) { o.drafters = cv::Drafters::Attach; }},
        {"declared drafter quant",
         [](auto& o) {
             o.convert.drafter_quant          = "q8";
             o.convert.drafter_quant_declared = true;
         }},
    };
    for (const auto& [what, change] : neutral) {
        cv::AcquireOptions options;
        change(options);
        check(cv::content_options_key(options).empty(), what + " is not a content option");
    }

    // Every content option has a key of its own, different from the defaults and from each other.
    write_text(root / "custom.jinja", "{{ messages }} CUSTOM");
    write_text(root / "other.jinja", "{{ messages }} OTHER");
    write_text(root / "ranking.json", "[1,2,3]");
    write_text(root / "source" / "a.bin", "aaaa");
    const std::vector<std::pair<std::string, Change>> content = {
        {"--vocab q8", [](auto& o) { o.convert.vocab = "q8"; }},
        {"--drafter-quant q8", [](auto& o) { o.convert.drafter_quant = "q8"; }},
        {"--name", [](auto& o) { o.convert.name = "mine"; }},
        {"--resource template",
         [&](auto& o) {
             o.convert.resources.emplace_back("chat_template.jinja", root / "custom.jinja");
         }},
        {"--resource other template",
         [&](auto& o) {
             o.convert.resources.emplace_back("chat_template.jinja", root / "other.jinja");
         }},
        {"--source", [&](auto& o) { o.convert.sources.emplace_back("vision", root / "source"); }},
        {"--components text,mtp",
         [](auto& o) {
             o.components_given   = true;
             o.convert.components = "text,mtp";
         }},
        {"--components text",
         [](auto& o) {
             o.components_given   = true;
             o.convert.components = "text";
         }},
        {"--proposal", [](auto& o) { o.convert.proposal = true; }},
        {"--no-proposal", [](auto& o) { o.convert.proposal_default = false; }},
        {"--proposal-rows", [](auto& o) { o.convert.proposal_rows = 4096; }},
        {"--ranking", [&](auto& o) { o.convert.ranking = root / "ranking.json"; }},
        {"--max-file-bytes", [](auto& o) { o.convert.max_file_bytes = 1ULL << 30; }},
        {"--maintainer-quantize", [](auto& o) { o.convert.maintainer_quantize = true; }},
        {"--round-tiny-scales", [](auto& o) { o.convert.round_tiny_scales = true; }},
        {"--layout frag", [](auto& o) { o.convert.expert_layout = cv::ExpertLayout::Frag; }},
        {"--layout published",
         [](auto& o) { o.convert.expert_layout = cv::ExpertLayout::Published; }},
        {"--no-drafters", [](auto& o) { o.drafters = cv::Drafters::Off; }},
        {"--drafter-artifact", [&](auto& o) { o.drafter_artifacts["dflash"] = root / "source"; }},
        {"--model-id", [](auto& o) { o.model_id = "org/other"; }},
        {"--no-base-tokenizer", [](auto& o) { o.base_tokenizer_files = false; }},
        {"--include", [](auto& o) { o.fetch.include = {"*.gguf"}; }},
    };
    std::set<std::string> keys;
    for (const auto& [what, change] : content) {
        cv::AcquireOptions options;
        change(options);
        std::string described;
        const auto key = cv::content_options_key(options, &described);
        check(!key.empty(), what + " has a key");
        check(key.size() == 12, what + " key is 12 hex digits");
        check(!described.empty(), what + " is described");
        check(keys.insert(key).second, what + " has a key of its own");
        // The same options again: the same key, so a rerun finds its own conversion.
        cv::AcquireOptions again;
        change(again);
        check(cv::content_options_key(again) == key, what + " is stable");
    }
    {
        cv::AcquireOptions options;
        options.convert.vocab = "q8";
        std::string described;
        (void)cv::content_options_key(options, &described);
        check(described == "--vocab q8\n", "the key's description names the option");
    }
    {
        // A template is its bytes, not its path.
        write_text(root / "copy" / "custom.jinja", "{{ messages }} CUSTOM");
        cv::AcquireOptions a, b;
        a.convert.resources.emplace_back("chat_template.jinja", root / "custom.jinja");
        b.convert.resources.emplace_back("chat_template.jinja", root / "copy" / "custom.jinja");
        check(cv::content_options_key(a) == cv::content_options_key(b),
              "a template is keyed by its bytes");
        write_text(root / "copy" / "custom.jinja", "edited");
        check(cv::content_options_key(a) != cv::content_options_key(b),
              "an edited template is another key");
    }


    // Canonical forms: the same selection spelled another way is the same key.
    {
        cv::AcquireOptions a, b;
        a.fetch.include = {"*.gguf", "*.json"};
        b.fetch.include = {"*.json", "*.gguf", "*.json"};
        check(!cv::content_options_key(a).empty(), "--include is an option");
        check(cv::content_options_key(a) == cv::content_options_key(b),
              "--include in another order, repeated, is the same key");
        cv::AcquireOptions native;
        native.convert.vocab = "native";
        check(cv::content_options_key(native).empty(), "--vocab native is the default");
        cv::AcquireOptions id;
        id.model_id = "org/m";
        check(!cv::content_options_key(id).empty(), "--model-id of another name is an option");
        cv::AcquireOptions same;
        same.model_id = "org/m";
        cv::OptionsKeyContext context;
        context.repo = "org/m";
        check(cv::content_options_key(same, nullptr, context).empty(),
              "--model-id equal to the repository is the default");
        // A resource that is the snapshot's own file is no option.
        write_text(root / "snapshot" / "chat_template.jinja", "{{ messages }} CUSTOM");
        cv::AcquireOptions own;
        own.convert.resources.emplace_back("chat_template.jinja", root / "custom.jinja");
        context.snapshot = root / "snapshot";
        check(cv::content_options_key(own, nullptr, context).empty(),
              "a --resource identical to the snapshot's file is the default");
        cv::AcquireOptions other;
        other.convert.resources.emplace_back("chat_template.jinja", root / "other.jinja");
        check(!cv::content_options_key(other, nullptr, context).empty(),
              "a --resource that differs from the snapshot's file is an option");
    }
    // A source or drafter artifact is keyed by its content, not its time or place.
    {
        write_text(root / "src1" / "a.bin", "aaaa");
        write_text(root / "src2" / "a.bin", "aaaa");
        cv::AcquireOptions a, b;
        a.convert.sources.emplace_back("vision", root / "src1");
        b.convert.sources.emplace_back("vision", root / "src2");
        check(cv::content_options_key(a) == cv::content_options_key(b),
              "a source is keyed by its content, not its path");
        write_text(root / "src2" / "a.bin", "aaab");
        check(cv::content_options_key(a) != cv::content_options_key(b),
              "a source with another byte is another key");
    }
    // The caller's own chat template (serve --chat-template) is applied at run time: no option.
    {
        cv::AcquireOptions options;
        options.chat_template = root / "custom.jinja";
        check(cv::content_options_key(options).empty(), "a runtime chat template is no option");
    }
    // The drafter cache: flags give a directory of their own, the copied resources only an
    // identity the conversion records.
    {
        cv::ConvertOptions plain_drafter;
        check(cv::drafter_options_key(plain_drafter).empty(), "a default drafter keeps its path");
        cv::ConvertOptions tiny;
        tiny.round_tiny_scales = true;
        cv::ConvertOptions maintainer;
        maintainer.maintainer_quantize = true;
        cv::ConvertOptions big;
        big.max_file_bytes = 1ULL << 30;
        const auto k1      = cv::drafter_options_key(tiny);
        const auto k2      = cv::drafter_options_key(maintainer);
        const auto k3      = cv::drafter_options_key(big);
        check(!k1.empty() && !k2.empty() && !k3.empty() && k1 != k2 && k2 != k3 && k1 != k3,
              "each drafter flag has a key of its own");
        cv::ConvertOptions a, b;
        a.resources.emplace_back("chat_template.jinja", root / "custom.jinja");
        b.resources.emplace_back("chat_template.jinja", root / "other.jinja");
        check(cv::drafter_options_key(a).empty(), "copied resources are not in the path");
        check(cv::drafter_content_identity(a) != cv::drafter_content_identity(b) &&
                  cv::drafter_content_identity(a) != cv::drafter_content_identity(plain_drafter),
              "a drafter converted with another template has another identity");
        check(cv::drafter_content_identity(tiny) != cv::drafter_content_identity(plain_drafter),
              "and with another flag");
    }
    // The download removed: a start finds the default conversion of the commit, and an
    // optioned one beside it (in options-<hash>/) is neither found nor in the way.
    {
        const auto cache = root / "hub";
        const auto repo  = cache / "models--org--m";
        const std::string commit(40, 'c');
        write_text(repo / "refs" / "main", commit);
        const auto dir = repo / "ninferno" / commit;
        write_text(dir / "m-Q8_0.dense_bf16.ninferno", "default");
        write_text(dir / "m-Q8_0.dense_bf16.ninferno.conversion.json", "{}");
        write_text(dir / "options-0123456789ab" / "m-Q8_0.dense_bf16.ninferno", "optioned");
        write_text(dir / "options-0123456789ab" / "m-Q8_0.dense_bf16.ninferno.conversion.json",
                   "{}");
        const auto kept = cv::kept_artifact(cache, "org/m", "", "", "");
        check(kept && *kept == dir / "m-Q8_0.dense_bf16.ninferno",
              "the removed-download lookup finds the default conversion");
        // A start with a runtime template only (key "") reuses that one: no download for it.
        cv::AcquireOptions serve;
        serve.chat_template = root / "custom.jinja";
        check(cv::content_options_key(serve).empty() && kept,
              "so a start with a runtime template needs no download");
    }

    // Default, then an option: the second run converts into its own entry, the default stays.
    {
        Cache cache{root / "cache1" / "c0ffee"};
        cv::AcquireOptions custom;
        custom.convert.resources.emplace_back("chat_template.jinja", root / "custom.jinja");
        check(cache.convert(plain, "default"), "default: converts");
        check(cache.convert(custom, "custom template"), "then --chat-template: converts again");
        check(cache.served(plain) == "default",
              "the default entry is still the default conversion");
        check(cache.served(custom) == "custom template",
              "the template entry is the custom conversion");
        check(!cache.convert(plain, "never"), "a later default run reuses the default conversion");
        check(!cache.convert(custom, "never"), "a rerun with the same template reuses its own");
    }
    // An option, then the default: the default run does not take the optioned conversion.
    {
        Cache cache{root / "cache2" / "c0ffee"};
        cv::AcquireOptions vocab;
        vocab.convert.vocab = "q8";
        check(cache.convert(vocab, "q8 vocabulary"), "--vocab q8: converts");
        check(cache.convert(plain, "default"), "then the default: converts again, not reused");
        check(cache.served(plain) == "default", "the default run serves the default conversion");
        check(cache.served(vocab) == "q8 vocabulary", "the --vocab q8 conversion is kept apart");
        check(!cache.convert(vocab, "never"), "--vocab q8 again reuses its own");
        check(fs::exists(root / "cache2" / "c0ffee" / "model.dense_bf16.ninferno"),
              "the default conversion is at the path existing caches use");
    }

    fs::remove_all(root);
    if (g_failures == 0) { std::cout << "acquire options key: ok\n"; }
    return g_failures == 0 ? 0 : 1;
}
