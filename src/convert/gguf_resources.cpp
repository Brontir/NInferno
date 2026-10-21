#include "convert/gguf_resources.h"

#include "chat/conventions.h"
#include "convert/pyjson.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <unordered_map>

namespace ninferno::convert {
namespace {

using Json = nlohmann::json;

// llama.cpp's token types (gguf-py TokenType).
constexpr int kNormal      = 1;
constexpr int kUnknown     = 2;
constexpr int kControl     = 3;
constexpr int kUserDefined = 4;
constexpr int kUnused      = 5;
constexpr int kByte        = 6;

// The Hugging Face Split regexes llama.cpp's pre-tokenizer types were taken from.
constexpr const char* kLlama3Split =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr const char* kTekkenSplit =
    R"re([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr const char* kGpt4oSplit =
    R"re([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

// Byte-level BPE pre-tokenizers by llama.cpp's tokenizer.ggml.pre, as the Hugging Face
// tokenizer.json each was taken from splits: one Split regex, or (split == nullptr) the GPT-2
// regex of ByteLevel, after each digit alone where that tokenizer has a Digits step.
struct Gpt2Pre {
    const char* name;
    const char* split;
    bool nfc;
    bool ignore_merges; // a whole pre-token in the vocabulary is one token (llama.cpp LLAMA3)
    bool add_bos;       // llama.cpp's add_bos when the GGUF has no tokenizer.ggml.add_bos_token
    bool digits;        // split == nullptr: each digit alone before the ByteLevel regex
};

constexpr Gpt2Pre kGpt2Pres[] = {
    // Qwen/Qwen3.5-*
    {"qwen35",
     R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re",
     true, false, false, false},
    // Qwen/Qwen2*, Qwen/Qwen3-*
    {"qwen2",
     R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re",
     true, false, false, false},
    // meta-llama/Llama-3*
    {"llama-bpe", kLlama3Split, false, true, true, false},
    // ibm-granite/granite-4.*, allenai/OLMo-2-*, allenai/Olmo-3-* (the Llama 3 regex, merges
    // always applied)
    {"dbrx", kLlama3Split, false, false, false, false},
    // HuggingFaceTB/SmolLM3-3B (the Llama 3 tokenizer without its BOS template)
    {"smaug-bpe", kLlama3Split, false, true, false, false},
    // mistralai/Ministral-3-*, Mistral Small 3.x (Tekken)
    {"tekken", kTekkenSplit, false, true, true, false},
    // mistral-community/pixtral-12b (its tokenizer.json: the Llama 3 regex). llama.cpp gives the
    // same hash, "pixtral", to NVIDIA Nemotron 3's Tekken vocabulary, whose tokenizer.json splits
    // with tekken's regex: a nemotron_h file takes "tekken" (gguf_frontend_resources).
    {"pixtral", kLlama3Split, false, true, true, false},
    // microsoft/Phi-4-mini-* (o200k)
    {"gpt-4o", kGpt4oSplit, false, false, false, false},
    // meta-models/Muse-Glimmer-* (the Llama 4 vocabulary: the o200k regex, a whole pre-token in
    // the vocabulary taken as one token, <|begin_of_text|> prepended)
    {"llama4", kGpt4oSplit, false, true, true, false},
    // HuggingFaceTB/SmolLM2-* (StarCoder vocabulary: Digits, then the GPT-2 regex)
    {"smollm", nullptr, false, false, false, true},
    // ibm-granite/granite-3.* (the GPT-2 regex alone; llama.cpp splits digits first, which
    // takes the lookahead from a space before a digit)
    {"refact", nullptr, false, false, false, false},
};

const Gpt2Pre* gpt2_pre(const std::string& name) {
    for (const auto& pre : kGpt2Pres) {
        if (name == pre.name) { return &pre; }
    }
    return nullptr;
}

std::string json_string(const std::string& text) {
    return Json(text).dump(-1, ' ', false, Json::error_handler_t::replace);
}

std::string compact(const OJson& value) {
    return value.dump(-1, ' ', false, Json::error_handler_t::replace);
}

struct Vocabulary {
    std::vector<std::string> tokens;
    std::vector<int> types;
    std::vector<std::string> merges;
};

OJson added_entry(std::size_t id, const std::string& content, bool special) {
    return OJson{{"id", id},          {"content", content}, {"single_word", false},
                 {"lstrip", false},   {"rstrip", false},    {"normalized", false},
                 {"special", special}};
}

// tokenizer.json text: the fixed parts as ordered JSON, vocab and merges spliced in (262k-entry
// objects are too slow to build as ordered_json).
std::string tokenizer_text(const OJson& head, const OJson& model_head, const std::string& vocab,
                           const std::string& merges) {
    std::string out = compact(head);
    out.pop_back(); // '}'
    std::string model = compact(model_head);
    model.pop_back();
    model += ",\"vocab\":" + vocab + ",\"merges\":" + merges + "}";
    out += ",\"model\":" + model + "}";
    return out;
}

// A byte-level BPE tokenizer.json; `bos` (the token add_bos_token prepends) adds a
// TemplateProcessing post-processor.
// GPT-2's bytes_to_unicode: a byte-level BPE token's text, byte by byte.
std::string byte_level(const std::string& text) {
    std::string out;
    for (const unsigned char byte : text) {
        const bool printable = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                               (byte >= 174);
        std::uint32_t code = byte;
        if (!printable) {
            code = 256;
            for (unsigned b = 0; b < byte; ++b) {
                if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174)) { ++code; }
            }
        }
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }
    return out;
}

std::string gpt2_tokenizer(const Vocabulary& v, OJson& added, const Gpt2Pre& pre,
                           std::optional<std::uint64_t> bos) {
    // An added token inside the base vocabulary's id range is also a vocabulary entry in Hugging
    // Face: the tokenizers library gives an added token its vocabulary id, and one missing from
    // the vocabulary the next id after it (SmolLM2's <|im_start|> is id 1; IBM Granite 3.x's id
    // 8441 is the added "�" and the byte-level piece "ï¿½" merges build). llama.cpp keeps
    // only the added form; the piece comes back as its byte-level text. A piece the merges build
    // or join was never an added token: llama.cpp typed Granite's and OLMo 2's "�" piece
    // CONTROL, but the Hugging Face tokenizer has it as a plain vocabulary entry.
    std::size_t base_end = 0;
    std::set<std::string> pieces;
    std::set<std::string> merged;
    for (const auto& merge : v.merges) {
        const auto at = merge.find(' ');
        if (at == std::string::npos) { continue; }
        merged.insert(merge.substr(0, at));
        merged.insert(merge.substr(at + 1));
        merged.insert(merge.substr(0, at) + merge.substr(at + 1));
    }
    for (std::size_t id = 0; id < v.tokens.size(); ++id) {
        if (v.types[id] == kNormal || v.types[id] == kByte) {
            pieces.insert(v.tokens[id]);
            base_end = id + 1;
        }
    }
    std::string vocab = "{";
    bool first        = true;
    for (std::size_t id = 0; id < v.tokens.size(); ++id) {
        const auto type = v.types[id];
        if (type == kControl || type == kUserDefined) {
            const auto piece = byte_level(v.tokens[id]);
            const bool plain = merged.contains(piece);
            if (!plain) { added.push_back(added_entry(id, v.tokens[id], type == kControl)); }
            if ((plain || id < base_end) && pieces.insert(piece).second) {
                vocab += (first ? "" : ",") + json_string(piece) + ":" + std::to_string(id);
                first = false;
            }
        } else if (type == kNormal || type == kByte) {
            vocab += (first ? "" : ",") + json_string(v.tokens[id]) + ":" + std::to_string(id);
            first = false;
        }
    }
    vocab += "}";
    std::string merges = "[";
    for (std::size_t i = 0; i < v.merges.size(); ++i) {
        merges += (i ? "," : "") + json_string(v.merges[i]);
    }
    merges += "]";
    const OJson byte_level = {{"type", "ByteLevel"},
                              {"add_prefix_space", false},
                              {"trim_offsets", false},
                              {"use_regex", false}};
    OJson head;
    head["version"]        = "1.0";
    head["truncation"]     = nullptr;
    head["padding"]        = nullptr;
    head["added_tokens"]   = added;
    head["normalizer"]     = pre.nfc ? OJson{{"type", "NFC"}} : OJson(nullptr);
    OJson steps = OJson::array();
    if (pre.split) {
        steps.push_back(OJson{{"type", "Split"},
                              {"pattern", OJson{{"Regex", pre.split}}},
                              {"behavior", "Isolated"},
                              {"invert", false}});
        steps.push_back(byte_level);
    } else {
        if (pre.digits) {
            steps.push_back(OJson{{"type", "Digits"}, {"individual_digits", true}});
        }
        steps.push_back(OJson{{"type", "ByteLevel"},
                              {"add_prefix_space", false},
                              {"trim_offsets", true},
                              {"use_regex", true}});
    }
    head["pre_tokenizer"] = OJson{{"type", "Sequence"}, {"pretokenizers", std::move(steps)}};
    head["post_processor"] = byte_level;
    if (bos) {
        const auto& text = v.tokens[*bos];
        const OJson a    = OJson{{"Sequence", OJson{{"id", "A"}, {"type_id", 0}}}};
        const OJson b    = OJson{{"Sequence", OJson{{"id", "B"}, {"type_id", 1}}}};
        const OJson s    = OJson{{"SpecialToken", OJson{{"id", text}, {"type_id", 0}}}};
        OJson specials   = OJson::object();
        specials[text]   = OJson{{"id", text},
                                 {"ids", OJson::array({*bos})},
                                 {"tokens", OJson::array({text})}};
        head["post_processor"] = OJson{
            {"type", "Sequence"},
            {"processors",
             OJson::array({byte_level,
                           OJson{{"type", "TemplateProcessing"},
                                 {"single", OJson::array({s, a})},
                                 {"pair", OJson::array({s, a, s, b})},
                                 {"special_tokens", std::move(specials)}}})}};
    }
    head["decoder"]        = byte_level;
    OJson model;
    model["type"]                      = "BPE";
    model["dropout"]                   = nullptr;
    model["unk_token"]                 = nullptr;
    model["continuing_subword_prefix"] = "";
    model["end_of_word_suffix"]        = "";
    model["fuse_unk"]                  = false;
    model["byte_fallback"]             = false;
    model["ignore_merges"]             = pre.ignore_merges;
    return tokenizer_text(head, model, vocab, merges);
}

std::string gemma4_tokenizer(const Vocabulary& v, OJson& added, const std::string& unk,
                             std::uint64_t& unresolved) {
    std::unordered_map<std::string, std::size_t> ids;
    std::string vocab = "{";
    bool first        = true;
    for (std::size_t id = 0; id < v.tokens.size(); ++id) {
        const auto type = v.types[id];
        if (type == kUnused) { continue; }
        if (type == kControl || type == kUserDefined) {
            added.push_back(added_entry(id, v.tokens[id], type == kControl));
        }
        ids.emplace(v.tokens[id], id);
        vocab += (first ? "" : ",") + json_string(v.tokens[id]) + ":" + std::to_string(id);
        first = false;
    }
    vocab += "}";
    // gguf-py joins each merge pair with a space, writing a space inside a part as U+0120.
    const auto part = [&](std::string text) {
        if (ids.contains(text)) { return text; }
        std::string decoded;
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (i + 1 < text.size() && static_cast<unsigned char>(text[i]) == 0xC4 &&
                static_cast<unsigned char>(text[i + 1]) == 0xA0) {
                decoded += ' ';
                ++i;
            } else {
                decoded += text[i];
            }
        }
        return decoded;
    };
    std::string merges = "[";
    for (std::size_t i = 0; i < v.merges.size(); ++i) {
        const auto& merge = v.merges[i];
        std::string left, right;
        bool found = false;
        for (auto at = merge.find(' '); at != std::string::npos; at = merge.find(' ', at + 1)) {
            left  = part(merge.substr(0, at));
            right = part(merge.substr(at + 1));
            if (ids.contains(left) && ids.contains(right)) {
                found = true;
                break;
            }
        }
        if (!found) {
            ++unresolved;
            continue;
        }
        merges += (i ? "," : "") + std::string("[") + json_string(left) + "," + json_string(right) + "]";
    }
    if (merges.size() > 1 && merges[1] == ',') { merges.erase(1, 1); }
    merges += "]";
    OJson head;
    head["version"]        = "1.0";
    head["truncation"]     = nullptr;
    head["padding"]        = nullptr;
    head["added_tokens"]   = added;
    head["normalizer"]     = OJson{{"type", "Replace"},
                                   {"pattern", OJson{{"String", " "}}},
                                   {"content", "\xe2\x96\x81"}};
    head["pre_tokenizer"]  = OJson{{"type", "Split"},
                                   {"pattern", OJson{{"String", " "}}},
                                   {"behavior", "MergedWithPrevious"},
                                   {"invert", false}};
    head["post_processor"] = OJson{
        {"type", "TemplateProcessing"},
        {"single", OJson::array({OJson{{"Sequence", OJson{{"id", "A"}, {"type_id", 0}}}}})},
        {"pair", OJson::array({OJson{{"Sequence", OJson{{"id", "A"}, {"type_id", 0}}}},
                               OJson{{"Sequence", OJson{{"id", "B"}, {"type_id", 1}}}}})},
        {"special_tokens", OJson::object()}};
    head["decoder"] = OJson{
        {"type", "Sequence"},
        {"decoders", OJson::array({OJson{{"type", "Replace"},
                                         {"pattern", OJson{{"String", "\xe2\x96\x81"}}},
                                         {"content", " "}},
                                   OJson{{"type", "ByteFallback"}}, OJson{{"type", "Fuse"}}})}};
    OJson model;
    model["type"]                      = "BPE";
    model["dropout"]                   = nullptr;
    model["unk_token"]                 = unk.empty() ? OJson(nullptr) : OJson(unk);
    model["continuing_subword_prefix"] = nullptr;
    model["end_of_word_suffix"]        = nullptr;
    model["fuse_unk"]                  = true;
    model["byte_fallback"]             = true;
    model["ignore_merges"]             = false;
    return tokenizer_text(head, model, vocab, merges);
}

// A SentencePiece BPE vocabulary (tokenizer.ggml.model "llama": Llama 2, Mistral 7B, TinyLlama
// and their fine-tunes) as transformers' LlamaConverter writes it: every piece in the vocabulary,
// the merges SentencePieceExtractor derives from the piece scores (each piece's splits into two
// vocabulary pieces, ordered by the ids of the halves; then every merge by descending score of
// the whole piece, stably), byte fallback, and U+2581 prepended when llama.cpp adds a space
// prefix (tokenizer.ggml.add_space_prefix, true unless stated).
std::string spm_tokenizer(const Vocabulary& v, const std::vector<float>& scores, OJson& added,
                          const std::string& unk, std::optional<std::uint64_t> bos,
                          bool prefix_space) {
    std::unordered_map<std::string, std::size_t> ids;
    std::string vocab = "{";
    for (std::size_t id = 0; id < v.tokens.size(); ++id) {
        const auto type = v.types[id];
        if (type == kControl || type == kUnknown || type == kUserDefined) {
            added.push_back(added_entry(id, v.tokens[id], type != kUserDefined));
        }
        ids[v.tokens[id]] = id; // a repeated piece resolves to its last id, as a Python dict
        vocab += (id ? "," : "") + json_string(v.tokens[id]) + ":" + std::to_string(id);
    }
    vocab += "}";
    struct Merge {
        std::size_t left, right;
        float score;
    };
    std::vector<Merge> merges;
    for (std::size_t id = 0; id < v.tokens.size(); ++id) {
        const auto& piece = v.tokens[id];
        std::vector<Merge> local;
        for (std::size_t at = 1; at < piece.size(); ++at) {
            if ((static_cast<unsigned char>(piece[at]) & 0xC0) == 0x80) { continue; } // mid code point
            const auto left  = ids.find(piece.substr(0, at));
            const auto right = ids.find(piece.substr(at));
            if (left != ids.end() && right != ids.end()) {
                local.push_back({left->second, right->second, scores[id]});
            }
        }
        std::sort(local.begin(), local.end(), [](const Merge& a, const Merge& b) {
            return a.left != b.left ? a.left < b.left : a.right < b.right;
        });
        merges.insert(merges.end(), local.begin(), local.end());
    }
    std::stable_sort(merges.begin(), merges.end(),
                     [](const Merge& a, const Merge& b) { return a.score > b.score; });
    std::string merge_text = "[";
    for (std::size_t i = 0; i < merges.size(); ++i) {
        merge_text += std::string(i ? "," : "") + "[" + json_string(v.tokens[merges[i].left]) +
                      "," + json_string(v.tokens[merges[i].right]) + "]";
    }
    merge_text += "]";
    const std::string space = "\xe2\x96\x81";
    const OJson replace     = {{"type", "Replace"}, {"pattern", OJson{{"String", " "}}},
                               {"content", space}};
    OJson head;
    head["version"]      = "1.0";
    head["truncation"]   = nullptr;
    head["padding"]      = nullptr;
    head["added_tokens"] = added;
    head["normalizer"] =
        prefix_space ? OJson{{"type", "Sequence"},
                             {"normalizers",
                              OJson::array({OJson{{"type", "Prepend"}, {"prepend", space}}, replace})}}
                     : replace;
    head["pre_tokenizer"] = nullptr;
    const OJson a = OJson{{"Sequence", OJson{{"id", "A"}, {"type_id", 0}}}};
    const OJson b = OJson{{"Sequence", OJson{{"id", "B"}, {"type_id", 1}}}};
    if (bos) {
        const auto& text = v.tokens[*bos];
        OJson specials   = OJson::object();
        specials[text]   = OJson{{"id", text},
                                 {"ids", OJson::array({*bos})},
                                 {"tokens", OJson::array({text})}};
        head["post_processor"] = OJson{
            {"type", "TemplateProcessing"},
            {"single", OJson::array({OJson{{"SpecialToken", OJson{{"id", text}, {"type_id", 0}}}}, a})},
            {"pair", OJson::array({OJson{{"SpecialToken", OJson{{"id", text}, {"type_id", 0}}}}, a,
                                   OJson{{"SpecialToken", OJson{{"id", text}, {"type_id", 1}}}}, b})},
            {"special_tokens", std::move(specials)}};
    } else {
        head["post_processor"] = OJson{{"type", "TemplateProcessing"},
                                       {"single", OJson::array({a})},
                                       {"pair", OJson::array({a, b})},
                                       {"special_tokens", OJson::object()}};
    }
    OJson decoders = OJson::array({OJson{{"type", "Replace"},
                                         {"pattern", OJson{{"String", space}}},
                                         {"content", " "}},
                                   OJson{{"type", "ByteFallback"}}, OJson{{"type", "Fuse"}}});
    if (prefix_space) {
        decoders.push_back(OJson{{"type", "Strip"}, {"content", " "}, {"start", 1}, {"stop", 0}});
    }
    head["decoder"] = OJson{{"type", "Sequence"}, {"decoders", std::move(decoders)}};
    OJson model;
    model["type"]                      = "BPE";
    model["dropout"]                   = nullptr;
    model["unk_token"]                 = unk.empty() ? OJson(nullptr) : OJson(unk);
    model["continuing_subword_prefix"] = nullptr;
    model["end_of_word_suffix"]        = nullptr;
    model["fuse_unk"]                  = true;
    model["byte_fallback"]             = true;
    model["ignore_merges"]             = false;
    return tokenizer_text(head, model, vocab, merge_text);
}

std::optional<std::uint64_t> id_of(const GgufFile& file, const char* key, std::size_t count) {
    const auto* value = file.find(std::string("tokenizer.ggml.") + key);
    if (!value || !value->is_number_integer()) { return std::nullopt; }
    const auto id = value->get<std::int64_t>();
    if (id < 0 || static_cast<std::size_t>(id) >= count) { return std::nullopt; }
    return static_cast<std::uint64_t>(id);
}

} // namespace

RebuiltResources gguf_frontend_resources(const GgufFile& file) {
    RebuiltResources out;
    if (const auto* tmpl = file.find("tokenizer.chat_template"); tmpl && tmpl->is_string()) {
        out.files["chat_template.jinja"] = tmpl->get<std::string>();
        out.notes.push_back("chat_template.jinja from tokenizer.chat_template");
    } else {
        out.notes.push_back("chat_template.jinja: the GGUF has no tokenizer.chat_template");
    }
    const auto* model_name = file.find("tokenizer.ggml.model");
    const auto* tokens     = file.find("tokenizer.ggml.tokens");
    if (!model_name || !model_name->is_string() || !tokens || !tokens->is_array()) {
        out.notes.push_back("tokenizer files: the GGUF has no tokenizer.ggml vocabulary");
        return out;
    }
    Vocabulary v;
    v.tokens.reserve(tokens->size());
    for (const auto& token : *tokens) { v.tokens.push_back(token.get<std::string>()); }
    v.types.assign(v.tokens.size(), kNormal);
    if (const auto* types = file.find("tokenizer.ggml.token_type"); types && types->is_array()) {
        for (std::size_t i = 0; i < types->size() && i < v.types.size(); ++i) {
            v.types[i] = (*types)[i].get<int>();
        }
    }
    if (const auto* merges = file.find("tokenizer.ggml.merges"); merges && merges->is_array()) {
        for (const auto& merge : *merges) { v.merges.push_back(merge.get<std::string>()); }
    }
    const auto count  = v.tokens.size();
    const auto token  = [&](std::optional<std::uint64_t> id) {
        return id ? OJson(v.tokens[*id]) : OJson(nullptr);
    };
    const auto bos   = id_of(file, "bos_token_id", count);
    const auto eos   = id_of(file, "eos_token_id", count);
    const auto eot   = id_of(file, "eot_token_id", count);
    const auto eom   = id_of(file, "eom_token_id", count);
    const auto pad   = id_of(file, "padding_token_id", count);
    const auto unk   = id_of(file, "unknown_token_id", count);
    const auto kind  = model_name->get<std::string>();
    const auto* pre  = file.find("tokenizer.ggml.pre");
    const auto pre_s = pre && pre->is_string() ? pre->get<std::string>() : std::string();
    OJson added      = OJson::array();
    const auto* add_bos = file.find("tokenizer.ggml.add_bos_token");
    const auto* gpt2    = kind == "gpt2" ? gpt2_pre(pre_s) : nullptr;
    if (const auto* arch = file.find("general.architecture");
        gpt2 && pre_s == "pixtral" && arch && arch->is_string() &&
        arch->get<std::string>().starts_with("nemotron_h")) {
        gpt2 = gpt2_pre("tekken");
    }
    // Qwen3.5's own fix-ups below follow the model, not the pre-tokenizer name: a qwen35 /
    // qwen35moe file may name llama.cpp's "qwen2" pre-tokenizer (the regex transformers 5
    // re-saves of Qwen3.5+ checkpoints write), which the Qwen3.5 frontend reads as well.
    const auto* arch_v = file.find("general.architecture");
    const auto arch_s  = arch_v && arch_v->is_string() ? arch_v->get<std::string>() : std::string();
    const bool qwen35_model =
        kind == "gpt2" && (pre_s == "qwen35" || arch_s == "qwen35" || arch_s == "qwen35moe");
    const bool declared = add_bos && add_bos->is_boolean();
    // llama.cpp's defaults: per gpt2 pre-tokenizer, and true for SentencePiece.
    const bool adds_bos = declared ? add_bos->get<bool>() : (gpt2 && gpt2->add_bos) || kind == "llama";
    if (gpt2) {
        if (adds_bos && !bos) {
            out.notes.push_back("tokenizer.json: add_bos_token without a bos_token_id is not "
                                "rebuilt; supply the Hugging Face tokenizer files");
            return out;
        }
        out.files["tokenizer.json"] =
            gpt2_tokenizer(v, added, *gpt2, adds_bos ? bos : std::nullopt);
        out.notes.push_back("tokenizer.json rebuilt from tokenizer.ggml (gpt2 BPE, pre " + pre_s +
                            (adds_bos ? ", BOS prepended" : "") + ")");
    } else if (kind == "gemma4") {
        std::uint64_t unresolved = 0;
        out.files["tokenizer.json"] =
            gemma4_tokenizer(v, added, unk ? v.tokens[*unk] : std::string(), unresolved);
        out.notes.push_back("tokenizer.json rebuilt from tokenizer.ggml (gemma4 byte-fallback BPE)" +
                            (unresolved ? ", " + std::to_string(unresolved) +
                                              " merges not resolvable into vocabulary pairs"
                                        : std::string()));
    } else if (kind == "llama") {
        const auto* score_list = file.find("tokenizer.ggml.scores");
        if (!score_list || !score_list->is_array() || score_list->size() != count) {
            out.notes.push_back("tokenizer.json: a SentencePiece vocabulary without one score "
                                "per token is not rebuilt; supply the Hugging Face tokenizer files");
            return out;
        }
        if (adds_bos && !bos) {
            out.notes.push_back("tokenizer.json: add_bos_token without a bos_token_id is not "
                                "rebuilt; supply the Hugging Face tokenizer files");
            return out;
        }
        std::vector<float> scores;
        scores.reserve(count);
        for (const auto& s : *score_list) { scores.push_back(s.get<float>()); }
        if (std::adjacent_find(scores.begin(), scores.end(), std::not_equal_to<>()) ==
            scores.end()) {
            // Converted from a tokenizer.json (TheBloke's TinyLlama): the merge ranks are gone.
            out.notes.push_back("tokenizer.json: a SentencePiece vocabulary whose scores are all "
                                "equal keeps no merge order; supply the Hugging Face tokenizer "
                                "files");
            return out;
        }
        const auto* prefix = file.find("tokenizer.ggml.add_space_prefix");
        const bool prefix_space = !(prefix && prefix->is_boolean()) || prefix->get<bool>();
        out.files["tokenizer.json"] =
            spm_tokenizer(v, scores, added, unk ? v.tokens[*unk] : std::string(),
                          adds_bos ? bos : std::nullopt, prefix_space);
        out.notes.push_back(std::string("tokenizer.json rebuilt from tokenizer.ggml (SentencePiece "
                                        "BPE, merges from the piece scores") +
                            (prefix_space ? ", space prefix" : "") +
                            (adds_bos ? ", BOS prepended" : "") + ")");
    } else {
        out.notes.push_back("tokenizer.json: tokenizer.ggml.model '" + kind + "' (pre '" + pre_s +
                            "') is not rebuilt; supply the Hugging Face tokenizer files");
        return out;
    }
    std::uint64_t specials = 0;
    OJson decoder          = OJson::object();
    for (const auto& entry : added) {
        specials += entry["special"].get<bool>() ? 1 : 0;
        decoder[std::to_string(entry["id"].get<std::uint64_t>())] =
            OJson{{"content", entry["content"]}, {"lstrip", false},       {"normalized", false},
                  {"rstrip", false},             {"single_word", false}, {"special", entry["special"]}};
    }
    OJson config;
    config["added_tokens_decoder"] = std::move(decoder);
    config["bos_token"]            = token(bos);
    config["eos_token"]            = token(eos);
    config["pad_token"]            = token(pad);
    config["unk_token"]            = token(unk);
    if (declared) {
        config["add_bos_token"] = *add_bos;
    } else if (adds_bos) {
        config["add_bos_token"] = true;
        out.notes.push_back("tokenizer_config.json: add_bos_token true (llama.cpp's default for " +
                            (gpt2 ? "pre " + pre_s : std::string("SentencePiece")) +
                            "; the GGUF does not say)");
    } else if (bos && qwen35_model) {
        // A named BOS the GGUF does not ask to prepend (unsloth's Qwen3.8 files carry
        // bos_token_id without add_bos_token): llama.cpp adds none for this pre-tokenizer, and
        // an absent flag next to a bos_token would read as "adds BOS" to the Qwen3.5 frontend.
        config["add_bos_token"] = false;
        out.notes.push_back("tokenizer_config.json: add_bos_token false (llama.cpp's default for "
                            "pre " +
                            pre_s + " of Qwen3.5; the GGUF names a BOS token but does not say)");
    }
    // The rebuilt byte-level BPE pre-tokenizer adds no prefix space (tokenizer.ggml has no key
    // for it; llama.cpp's gpt2 tokenizer adds none either).
    if (kind == "gpt2") { config["add_prefix_space"] = false; }
    // Qwen3.5's official resources pad with <|endoftext|>, which the engine checks; a GGUF
    // publisher's own pad id (unsloth writes <|vision_pad|>) never reaches inference.
    if (qwen35_model) {
        const auto official = std::find(v.tokens.begin(), v.tokens.end(), "<|endoftext|>");
        if (official != v.tokens.end() && config["pad_token"] != OJson("<|endoftext|>")) {
            out.notes.push_back("tokenizer_config.json: pad_token <|endoftext|> (the official "
                                "Qwen3.5 pad) in place of the GGUF's " +
                                (pad ? v.tokens[*pad] : std::string("none")));
            config["pad_token"] = "<|endoftext|>";
        }
    }
    out.files["tokenizer_config.json"] = compact(config);
    out.notes.push_back("tokenizer_config.json rebuilt: " + std::to_string(added.size()) +
                        " added tokens, " + std::to_string(specials) +
                        " special (from CONTROL token types)");
    OJson generation;
    if (bos) { generation["bos_token_id"] = *bos; }
    // The end-of-generation tokens: those the file names (eos, eot, eom) and the stops of the
    // template's chat conventions (src/chat/data/conventions.json, matched by the markers the
    // template contains: Gemma 4 <turn|>, <eos> and <|tool_response>, the tool's turn; harmony
    // <|return|>, <|endoftext|> and <|call|>). A token the conventions place inside a reply is
    // never one, even when the file names it: harmony's <|end|> closes the analysis message
    // (unsloth/gpt-oss-20b-GGUF). A template the table does not know stops on llama.cpp's
    // end-of-generation texts. Control or user-defined tokens only, in id order.
    const auto* template_source             = file.find("tokenizer.chat_template");
    const chat::ChatConventions conventions = chat::resolve_conventions(
        template_source && template_source->is_string() ? template_source->get<std::string>()
                                                        : std::string());
    const std::vector<std::string> wanted =
        conventions.turns ? conventions.generation_stops() : chat::generic_generation_stops();
    OJson stops = OJson::array();
    std::set<std::uint64_t> seen;
    for (const auto& id : {eos, eot, eom}) {
        if (id && !conventions.within_reply(v.tokens[*id]) && seen.insert(*id).second) {
            stops.push_back(*id);
        }
    }
    for (std::size_t id = 0; id < count; ++id) {
        const auto& text = v.tokens[id];
        if ((v.types[id] == kControl || v.types[id] == kUserDefined) &&
            std::find(wanted.begin(), wanted.end(), text) != wanted.end() &&
            seen.insert(id).second) {
            stops.push_back(id);
        }
    }
    if (stops.empty()) {
        out.notes.push_back("generation_config.json: the GGUF names no end-of-sequence token");
        return out;
    }
    generation["eos_token_id"] = stops.size() == 1 ? stops[0] : stops;
    if (pad) { generation["pad_token_id"] = *pad; }
    for (const auto& [key, field] : {std::pair{"general.sampling.top_k", "top_k"},
                                     std::pair{"general.sampling.top_p", "top_p"},
                                     std::pair{"general.sampling.temp", "temperature"}}) {
        if (const auto* value = file.find(key); value && value->is_number()) {
            generation[field]       = value->is_number_integer() ? OJson(value->get<std::int64_t>())
                                                                 : OJson(value->get<double>());
            generation["do_sample"] = true;
        }
    }
    out.files["generation_config.json"] = compact(generation);
    out.notes.push_back("generation_config.json rebuilt from the tokenizer.ggml token ids");
    return out;
}

} // namespace ninferno::convert
