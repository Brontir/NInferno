#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/resources.h"

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/media_cache.h"
#include "models/qwen3_5/frontend/processor.h"
#include "models/qwen3_5/frontend/prompt_layout.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/unicode.h"
#include "support/files.h"
#include "core/sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Frontend        = ninferno::models::qwen3_5::Frontend;
using FrontendFactory = ninferno::models::qwen3_5::FrontendTestAccess;

struct FrontendResources {
    std::string tokenizer_json, tokenizer_config_json, chat_template_jinja, generation_config_json;
    std::string preprocessor_config_json, video_preprocessor_config_json;
};

Frontend make_frontend(const FrontendResources& source,
                       ninferno::models::qwen3_5::FrontendOptions options) {
    ninferno::models::qwen3_5::FrontendResources parsed{
        source.tokenizer_json,           source.tokenizer_config_json,
        source.chat_template_jinja,      source.generation_config_json,
        source.preprocessor_config_json, source.video_preprocessor_config_json};
    parsed.tokenizer = std::make_shared<const ninferno::models::qwen3_5::frontend::Tokenizer>(
        ninferno::models::qwen3_5::frontend::TokenizerResources{
            source.tokenizer_json, source.tokenizer_config_json, source.generation_config_json});
    parsed.public_token_count = static_cast<std::uint32_t>(parsed.tokenizer->vocab_size());
    return ninferno::models::qwen3_5::make_frontend(parsed, options);
}

Frontend make_frontend(const FrontendResources& source, bool vision = true) {
    ninferno::models::qwen3_5::FrontendOptions options;
    options.vision_enabled = vision;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return make_frontend(source, options);
}

using PublishedOutput = ninferno::models::qwen3_5::PublishedOutput;
namespace fi          = ninferno::models::qwen3_5::frontend;

constexpr std::string_view kThinkingControlGuidance =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n";
constexpr std::string_view kThinkingControl =
    "\n\n Considering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n";
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

constexpr ninferno::TokenId kFixtureByteTokenBase = 1'000;

constexpr ninferno::TokenId fixture_byte_token(std::uint8_t byte) {
    // Preserve IDs already used by the output-session fixtures. All other bytes live outside the
    // added-token range so the synthetic tokenizer can encode arbitrary UTF-8 test input.
    switch (byte) {
    case static_cast<std::uint8_t>('x'):
        return 0;
    case 0xe4:
        return 10;
    case 0xb8:
        return 11;
    case 0xad:
        return 12;
    case 0x80:
        return 13;
    case 0xe0:
        return 14;
    case 0xed:
        return 15;
    case 0xa0:
        return 16;
    case 0xf4:
        return 17;
    case 0x90:
        return 18;
    case 0xf5:
        return 19;
    case 0xf0:
        return 20;
    case 0x9f:
        return 21;
    case 0x98:
        return 22;
    case 0xc2:
        return 23;
    case 0xa2:
        return 24;
    default:
        return kFixtureByteTokenBase + byte;
    }
}

constexpr ninferno::TokenId kByte80Token = fixture_byte_token(0x80);
constexpr ninferno::TokenId kByteE0Token = fixture_byte_token(0xe0);
constexpr ninferno::TokenId kByteEDToken = fixture_byte_token(0xed);
constexpr ninferno::TokenId kByteA0Token = fixture_byte_token(0xa0);
constexpr ninferno::TokenId kByteF4Token = fixture_byte_token(0xf4);
constexpr ninferno::TokenId kByte90Token = fixture_byte_token(0x90);
constexpr ninferno::TokenId kByteF5Token = fixture_byte_token(0xf5);
constexpr ninferno::TokenId kByteF0Token = fixture_byte_token(0xf0);
constexpr ninferno::TokenId kByte9FToken = fixture_byte_token(0x9f);
constexpr ninferno::TokenId kByte98Token = fixture_byte_token(0x98);
constexpr ninferno::TokenId kByteC2Token = fixture_byte_token(0xc2);
constexpr ninferno::TokenId kByteA2Token = fixture_byte_token(0xa2);

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

float bf16_value(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

// These integration fixtures explicitly select the maintained template files.
// Rendering semantics belong to those files, not to Frontend construction.
std::string read_template_fixture(const char* path) { return read_file(path); }

const std::string& thinking_toggle_template_source() {
    static const std::string source =
        read_template_fixture(NINFERNO_SOURCE_DIR "/tools/chat_templates/qwen3_6.jinja");
    return source;
}

const std::string& reasoning_effort_template_source() {
    static const std::string source =
        read_template_fixture(NINFERNO_SOURCE_DIR "/tools/chat_templates/qwen3_8.jinja");
    return source;
}

const fi::CompiledChatTemplate& thinking_toggle_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(thinking_toggle_template_source());
    return value;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninferno::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte-level test symbol is outside one byte");
}

FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(8, std::string(kThinkingControlGuidance)), added(30, "user\n"),
         added(31, "assistant\n"), added(32, "\n"), added(248045, "<|im_start|>", true),
         added(248046, "<|im_end|>", true), added(248053, "<|vision_start|>", true),
         added(248054, "<|vision_end|>", true), added(248056, "<|image_pad|>", true),
         added(248044, "<|endoftext|>", true),
         added(248057, "<|video_pad|>", true), added(248068, "<think>"),
         added(248069, "</think>")});
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        const auto byte                = static_cast<std::uint8_t>(value);
        vocab[byte_level_symbol(byte)] = fixture_byte_token(byte);
    }
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"}, {"vocab", std::move(vocab)}, {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

const fi::Tokenizer& fixture_tokenizer() {
    static const FrontendResources fixture = resources();
    static const fi::Tokenizer tokenizer(
        {.tokenizer_json         = fixture.tokenizer_json,
         .tokenizer_config_json  = fixture.tokenizer_config_json,
         .generation_config_json = fixture.generation_config_json});
    return tokenizer;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

std::vector<std::uint8_t> block_ppm(int width, int height, std::uint8_t value) {
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    std::vector<std::uint8_t> ppm;
    ppm.reserve(header.size() +
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    ppm.insert(ppm.end(), static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3,
               value);
    return ppm;
}

ninferno::PromptInput image_text_input(std::vector<std::uint8_t> bytes, std::string text,
                                     std::string source_name) {
    ninferno::MessagePart image;
    image.kind              = ninferno::MessagePartKind::Media;
    image.media.kind        = ninferno::MediaKind::Image;
    image.media.bytes       = std::move(bytes);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = std::move(source_name);

    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(std::move(image));
    if (!text.empty()) {
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
    }
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

ninferno::PromptInput image_input() {
    ninferno::MessagePart image;
    image.kind              = ninferno::MessagePartKind::Media;
    image.media.kind        = ninferno::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(std::move(image));
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

bool near(float actual, float expected) { return std::abs(actual - expected) < 1.0e-6F; }

constexpr std::array<std::uint8_t, 32> kGradientDigest{
    0x1e, 0x8c, 0xd9, 0x22, 0x40, 0xfa, 0x10, 0x62, 0x7b, 0x60, 0x86, 0x8e, 0xe9, 0x66, 0x41, 0xa2,
    0x4d, 0x21, 0xff, 0xc7, 0xe9, 0xa2, 0x2b, 0x34, 0xc0, 0xec, 0x99, 0x84, 0x6c, 0xa9, 0xa4, 0x8a,
};

std::string channel_text(const PublishedOutput& output, ninferno::OutputChannel channel) {
    std::string result;
    for (const ninferno::OutputDelta& delta : output) {
        if (delta.channel == channel) { result += delta.text; }
    }
    return result;
}

fi::ChatMessage chat_message(ninferno::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::RenderedChat render_chat(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return thinking_toggle_template().render(messages, std::move(options));
}

std::string render_chat_text(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return render_chat(std::move(messages), std::move(options)).text;
}

template <class Callable>
bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int test_invalid_public_part_enums(const Frontend& frontend) {
    ninferno::ChatMessage invalid_part_message;
    invalid_part_message.role = ninferno::ChatRole::User;
    invalid_part_message.parts.push_back(ninferno::MessagePart{
        .kind = static_cast<ninferno::MessagePartKind>(255), .text = "invalid", .media = {}});
    ninferno::PromptInput invalid_part;
    invalid_part.messages.push_back(std::move(invalid_part_message));

    ninferno::MessagePart media;
    media.kind       = ninferno::MessagePartKind::Media;
    media.media.kind = static_cast<ninferno::MediaKind>(255);
    media.media.bytes.push_back(0);
    ninferno::ChatMessage invalid_media_message;
    invalid_media_message.role = ninferno::ChatRole::User;
    invalid_media_message.parts.push_back(std::move(media));
    ninferno::PromptInput invalid_media;
    invalid_media.messages.push_back(std::move(invalid_media_message));

    int failures =
        check(throws_invalid_argument([&] { (void)frontend.prepare(std::move(invalid_part)); }),
              "invalid public message-part kind was accepted");
    failures +=
        check(throws_invalid_argument([&] { (void)frontend.prepare(std::move(invalid_media)); }),
              "invalid public media kind was accepted as video");
    return failures;
}

template <class Callable>
bool throws_processor_budget(Callable&& callable) {
    try {
        callable();
    } catch (const fi::ProcessorError& error) {
        return error.kind() == fi::ProcessorErrorKind::BudgetExceeded;
    }
    return false;
}

template <class Callable>
bool throws_context_length(Callable&& callable) {
    try {
        callable();
    } catch (const ninferno::RequestError& error) {
        return error.kind() == ninferno::RequestErrorKind::ContextLengthExceeded;
    }
    return false;
}

// generation_config.json sampling: Gemma 4 recommends top_k 64, the sampler's whole exact domain;
// a top_k above it keeps the domain, and the rest of the recommendation stays as is.
int test_generation_config_sampling() {
    using ninferno::models::qwen3_5::generation_sampling;
    int failures        = 0;
    const auto gemma    = generation_sampling(
        R"({"do_sample": true, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "eos_token_id": [1, 106, 50]})");
    failures += check(gemma && gemma->thinking.top_k == 64 && gemma->non_thinking.top_k == 64 &&
                          gemma->thinking.temperature == 1.0F && gemma->thinking.top_p == 0.95F,
                      "Gemma 4's generation_config top_k 64 changed");
    const auto wide = generation_sampling(R"({"top_k": 100})");
    failures += check(wide && wide->thinking.top_k == ninferno::kMaximumTopK,
                      "a generation_config top_k above the sampler domain did not keep the domain");
    const auto qwen = generation_sampling(R"({"temperature": 0.6, "top_k": 20, "top_p": 0.95})");
    failures += check(qwen && qwen->thinking.top_k == 20 && qwen->thinking.temperature == 0.6F,
                      "a generation_config top_k inside the sampler domain changed");
    const auto narrow = generation_sampling(R"({"top_k": 5})");
    failures += check(narrow && narrow->thinking.top_k == 5, "a narrow top_k changed");
    failures += check(!generation_sampling(R"({"eos_token_id": 1})"),
                      "a generation_config without sampling produced defaults");
    // A checkpoint outside the Qwen 3.5 family whose generation config names no sampling parameter
    // (gpt-oss-20b's) samples with transformers' defaults, not Qwen 3.5's presets (whose presence
    // penalty of 1.5 changed even greedy answers of gpt-oss-20b through ninferno-serve).
    using ninferno::models::qwen3_5::checkpoint_sampling;
    for (const auto& json :
         {std::string(R"({"bos_token_id": 199998, "do_sample": true, "pad_token_id": 199999})"),
          std::string()}) {
        const auto neutral = checkpoint_sampling(json);
        for (const auto& preset : {neutral.thinking, neutral.non_thinking}) {
            failures += check(preset.temperature == 1.0F && preset.top_k == 50 &&
                                  preset.top_p == 1.0F && preset.min_p == 0.0F &&
                                  preset.presence_penalty == 0.0F &&
                                  preset.frequency_penalty == 0.0F,
                              "a generation_config without sampling did not keep transformers' "
                              "defaults");
        }
    }
    const auto recommended = checkpoint_sampling(R"({"temperature": 0.6, "top_p": 0.9})");
    failures += check(recommended.non_thinking.temperature == 0.6F &&
                          recommended.non_thinking.top_p == 0.9F &&
                          recommended.non_thinking.presence_penalty == 0.0F,
                      "a recommended generation_config sampling was not kept");
    return failures;
}

int test_declared_frontend_semantics() {
    auto source                = resources();
    auto tokenizer             = nlohmann::json::parse(source.tokenizer_json);
    tokenizer["normalizer"]    = {{"type", "NFC"}};
    tokenizer["pre_tokenizer"] = nlohmann::json::parse(
        R"json({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]})json");
    tokenizer["decoder"]  = {{"type", "ByteLevel"}};
    source.tokenizer_json = tokenizer.dump();
    const fi::Tokenizer supported(
        {source.tokenizer_json, source.tokenizer_config_json, source.generation_config_json});
    int failures =
        check(supported.encode("ABC e\xcc\x81") ==
                  std::vector<int>{fixture_byte_token('A'), fixture_byte_token('B'),
                                   fixture_byte_token('C'), fixture_byte_token(' '),
                                   fixture_byte_token(0xc3), fixture_byte_token(0xa9)},
              "declared NFC/ByteLevel tokenizer did not preserve case and compose Unicode");
    for (const auto& [path, value] : std::vector<std::pair<const char*, nlohmann::json>>{
             {"/normalizer/type", "Lowercase"},
             {"/pre_tokenizer/pretokenizers/0/pattern/Regex", "\\w+"},
             {"/pre_tokenizer/pretokenizers/1/add_prefix_space", true},
             {"/decoder/type", "WordPiece"},
             {"/model/dropout", 0.1},
             {"/model/ignore_merges", true}}) {
        auto changed                                = tokenizer;
        changed[nlohmann::json::json_pointer(path)] = value;
        source.tokenizer_json                       = changed.dump();
        failures +=
            check(throws_invalid_argument([&] {
                      (void)fi::Tokenizer({source.tokenizer_json, source.tokenizer_config_json,
                                           source.generation_config_json});
                  }),
                  "unsupported declared tokenizer algorithm was accepted");
    }
    source.tokenizer_json = tokenizer.dump();
    for (auto member : {&FrontendResources::preprocessor_config_json,
                        &FrontendResources::video_preprocessor_config_json}) {
        auto original = nlohmann::json::parse(source.*member);
        for (const auto& [field, value] :
             std::vector<std::pair<const char*, nlohmann::json>>{{"do_normalize", false},
                                                                 {"do_rescale", false},
                                                                 {"do_resize", false},
                                                                 {"do_convert_rgb", false},
                                                                 {"resample", 2}}) {
            auto changed   = original;
            changed[field] = value;
            source.*member = changed.dump();
            failures += check(throws_invalid_argument([&] { (void)make_frontend(source); }),
                              "unsupported declared pixel pipeline was accepted");
            // An unselected component does not constrain Text-only startup.
            (void)make_frontend(source, false);
        }
        original["do_normalize"] = original["do_rescale"] = original["do_resize"] = true;
        original["do_convert_rgb"]                                                = true;
        original["resample"]                                                      = 3;
        source.*member                                                            = original.dump();
        (void)make_frontend(source);
    }
    return failures;
}

int test_tokenizer_config_merge() {
    const fi::Tokenizer& tokenizer = fixture_tokenizer();

    constexpr std::array<std::pair<const char*, int>, 7> appended = {{
        {"<|audio_start|>", 248070},
        {"<|audio_end|>", 248071},
        {"<tts_pad>", 248072},
        {"<tts_text_bos>", 248073},
        {"<tts_text_eod>", 248074},
        {"<tts_text_bos_single>", 248075},
        {"<|audio_pad|>", 248076},
    }};
    int failures                                                  = 0;
    for (const auto& [text, id] : appended) {
        const std::vector<int> encoded = tokenizer.encode(text);
        failures += check(encoded == std::vector<int>{id} && tokenizer.is_special_token(id) &&
                              tokenizer.decode_token_bytes(id) == text,
                          "tokenizer_config.json token did not merge exactly");
    }

    FrontendResources conflicting = resources();
    nlohmann::json config         = nlohmann::json::parse(conflicting.tokenizer_config_json);
    config["added_tokens_decoder"]["248045"]["special"] = false;
    conflicting.tokenizer_config_json                   = config.dump();
    failures += check(
        throws_invalid_argument([&] {
            fi::Tokenizer invalid({.tokenizer_json         = conflicting.tokenizer_json,
                                   .tokenizer_config_json  = conflicting.tokenizer_config_json,
                                   .generation_config_json = conflicting.generation_config_json});
        }),
        "conflicting tokenizer/tokenizer_config added-token definitions were accepted");
    return failures;
}

// The transformers-5 layout of the same tokenizer: every added token in tokenizer.json, a slim
// tokenizer_config.json without added_tokens_decoder or add_bos_token (RedHatAI's Qwen3.6
// checkpoints ship it). It must tokenize exactly like the transformers-4 layout.
FrontendResources transformers5_resources() {
    FrontendResources v5            = resources();
    nlohmann::json tokenizer        = nlohmann::json::parse(v5.tokenizer_json);
    nlohmann::json config           = nlohmann::json::parse(v5.tokenizer_config_json);
    std::set<int> in_tokenizer;
    for (const auto& token : tokenizer["added_tokens"]) { in_tokenizer.insert(token["id"].get<int>()); }
    for (const auto& [id, token] : config["added_tokens_decoder"].items()) {
        if (in_tokenizer.contains(std::stoi(id))) { continue; }
        nlohmann::json entry = token;
        entry["id"]          = std::stoi(id);
        tokenizer["added_tokens"].push_back(std::move(entry));
    }
    config.erase("added_tokens_decoder");
    config.erase("add_bos_token");
    config["bos_token"]       = nullptr;
    config["backend"]         = "tokenizers";
    config["tokenizer_class"] = "TokenizersBackend";
    v5.tokenizer_json         = tokenizer.dump();
    v5.tokenizer_config_json  = config.dump();
    return v5;
}

int test_transformers5_tokenizer_layout() {
    const FrontendResources v5 = transformers5_resources();
    const fi::Tokenizer tokenizer({.tokenizer_json         = v5.tokenizer_json,
                                   .tokenizer_config_json  = v5.tokenizer_config_json,
                                   .generation_config_json = v5.generation_config_json});
    const fi::Tokenizer& v4 = fixture_tokenizer();
    int failures            = 0;
    for (const char* text :
         {"<|audio_start|>", "<|im_start|>user\nhello<|im_end|>\n<think>x</think><tts_pad>",
          "plain bytes \xc3\xa9 and <|image_pad|><|video_pad|> tokens"}) {
        const auto ids = tokenizer.encode(text);
        failures += check(ids == v4.encode(text), "transformers-5 layout tokenizes like 4's");
        for (const int id : ids) {
            failures += check(tokenizer.is_special_token(id) == v4.is_special_token(id),
                              "transformers-5 layout keeps the special flags");
        }
    }
    failures += check(tokenizer.vocab_size() == v4.vocab_size(), "same vocabulary size");
    ninferno::PromptInput input;
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back({.text = "hello"});
    input.messages.push_back(std::move(message));
    const auto v5_frontend = make_frontend(v5, false);
    const auto v4_frontend = make_frontend(resources(), false);
    failures += check(FrontendFactory::inspect(v5_frontend.prepare(input)).token_ids ==
                          FrontendFactory::inspect(v4_frontend.prepare(input)).token_ids,
                      "transformers-5 layout prepares the same prompt");
    // A named BOS token with no add_bos_token would be added: still refused.
    FrontendResources with_bos = v5;
    nlohmann::json config      = nlohmann::json::parse(with_bos.tokenizer_config_json);
    config["bos_token"]        = "<|im_start|>";
    with_bos.tokenizer_config_json = config.dump();
    failures += check(throws_invalid_argument([&] { (void)make_frontend(with_bos, false); }),
                      "an implied BOS token was accepted");
    // Without add_prefix_space the tokenizer.json ByteLevel step decides (no prefix space, as the
    // tokenizer requires); a config that asks for one is still refused.
    FrontendResources no_prefix_flag = v5;
    nlohmann::json slim              = nlohmann::json::parse(no_prefix_flag.tokenizer_config_json);
    slim.erase("add_prefix_space");
    no_prefix_flag.tokenizer_config_json = slim.dump();
    bool prepared                        = false;
    try {
        prepared = FrontendFactory::inspect(make_frontend(no_prefix_flag, false).prepare(input))
                       .token_ids == FrontendFactory::inspect(v4_frontend.prepare(input)).token_ids;
    } catch (const std::exception&) {}
    failures += check(prepared, "a tokenizer_config.json without add_prefix_space was refused");
    FrontendResources prefix_space     = v5;
    slim["add_prefix_space"]           = true;
    prefix_space.tokenizer_config_json = slim.dump();
    failures += check(throws_invalid_argument([&] { (void)make_frontend(prefix_space, false); }),
                      "add_prefix_space true was accepted");
    return failures;
}

int test_bpe_merge_order() {
    const std::string tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"a", 0}, {"aa", 1}, {"aaa", 2}, {"b", 3}, {"c", 4}, {"bc", 5}, {"abc", 6}}},
          {"merges",
           nlohmann::json::array(
               {nlohmann::json::array({"a", "a"}), nlohmann::json::array({"aa", "a"}),
                nlohmann::json::array({"b", "c"}), nlohmann::json::array({"a", "bc"})})}}},
        {"added_tokens",
         nlohmann::json::array()}}.dump();
    const std::string tokenizer_config_json =
        nlohmann::json{{"added_tokens_decoder", nlohmann::json::object()}}.dump();
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = tokenizer_config_json,
                                   .generation_config_json = R"({"eos_token_id":0})"});
    return check(tokenizer.encode("aaa") == std::vector<int>{2} &&
                     tokenizer.encode("aaaa") == std::vector<int>({1, 1}) &&
                     tokenizer.encode("abc") == std::vector<int>{6},
                 "priority BPE changed rank or leftmost merge semantics");
}

int test_boundary_aware_tokenization() {
    const std::string tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"a", 0}, {"b", 1}, {"c", 2}, {"bc", 3}, {"ab", 4}}},
          {"merges", nlohmann::json::array(
                         {nlohmann::json::array({"b", "c"}), nlohmann::json::array({"a", "b"})})}}},
        {"added_tokens",
         nlohmann::json::array()}}.dump();
    const std::string tokenizer_config_json =
        nlohmann::json{{"added_tokens_decoder", nlohmann::json::object()}}.dump();
    const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                   .tokenizer_config_json  = tokenizer_config_json,
                                   .generation_config_json = R"({"eos_token_id":0})"});
    constexpr std::array<std::size_t, 5> boundaries{2, 3, 0, 1, 2};
    const fi::BoundaryEncodedText encoded = tokenizer.encode_with_boundaries("abc", boundaries);
    int failures                          = check(
        encoded.input_ids == std::vector<int>({0, 3}) && encoded.boundaries.size() == 5 &&
            !encoded.boundaries[0].exact_frontier && encoded.boundaries[0].stable_frontier == 0 &&
            encoded.boundaries[1].exact_frontier == 2 &&
            encoded.boundaries[2].exact_frontier == 0 &&
            encoded.boundaries[3].exact_frontier == 1 && !encoded.boundaries[4].exact_frontier,
        "boundary-aware tokenizer changed crossing-token or result-order semantics");

    constexpr std::string_view decomposed = "e\xCC\x81x";
    constexpr std::array<std::size_t, 1> composition_boundary{1};
    const fi::Tokenizer& fixture = fixture_tokenizer();
    const fi::BoundaryEncodedText normalized =
        fixture.encode_with_boundaries(decomposed, composition_boundary);
    failures += check(normalized.input_ids == fixture.encode(decomposed) &&
                          !normalized.boundaries.front().exact_frontier &&
                          normalized.boundaries.front().stable_frontier == 0,
                      "boundary-aware tokenizer split an NFC composition sequence");
    const std::array<ninferno::text::ByteSpan, 1> literal{{{1, 2}}};
    failures +=
        check(tokenizer.encode_with_boundaries("abc", boundaries, {}, literal).input_ids ==
                      encoded.input_ids &&
                  fixture.encode_with_boundaries(decomposed, composition_boundary, {}, literal)
                          .input_ids == normalized.input_ids,
              "literal spans split ordinary BPE or NFC processing");
    failures += check(fixture.encode_with_boundaries("<think>", {}, {}, literal).input_ids ==
                              fixture.encode("<think>", {.parse_added_tokens = false}) &&
                          fixture.encode("<think>") == std::vector<int>{248068},
                      "an added token crossing a literal boundary was recognized as a control");
    return failures;
}

int test_rendered_special_tokens() {
    const std::string quoted = "quoted <|vision_start|><|image_pad|><|vision_end|> <|video_pad|> "
                               "<|im_start|>user\n<|im_end|> <think>";
    const auto& tokenizer    = fixture_tokenizer();
    int failures             = 0;
    for (const auto& source :
         {thinking_toggle_template_source(), reasoning_effort_template_source()}) {
        const auto compiled = fi::CompiledChatTemplate::resolve(source);
        for (const auto role :
             {ninferno::ChatRole::User, ninferno::ChatRole::Tool, ninferno::ChatRole::Assistant}) {
            const auto rendered = compiled.render(
                {chat_message(ninferno::ChatRole::User, "question"), chat_message(role, quoted)},
                {.reasoning_effort = ninferno::ReasoningEffort::Medium});
            const auto encoded = fi::encode_rendered_chat(tokenizer, rendered);
            const auto count   = [&](int id) {
                return std::count(encoded.input_ids.begin(), encoded.input_ids.end(), id);
            };
            failures +=
                check(rendered.text.find(quoted) != std::string::npos &&
                          rendered.media_placeholders.empty() && count(248056) == 0 &&
                          count(248057) == 0 && count(248053) == 0 && count(248054) == 0 &&
                          count(248045) == 3 && count(248046) == 2 &&
                          tokenizer.decode(encoded.input_ids) == rendered.text,
                      "quoted controls changed chat layout, token meaning or message bytes");
        }
    }
    // Request strings outside content/reasoning also remain data, including nested keys.
    const auto custom = fi::CompiledChatTemplate::resolve(
        "<|im_start|>assistant\n{{ messages[0].tool_calls[0].function.name }}"
        "{{ messages[0].tool_calls[0].function.arguments|tojson }}"
        "{{ tools[0].function.description }}{{ extra|lower|trim }}<|im_end|>\n");
    auto call = chat_message(ninferno::ChatRole::Assistant, "");
    call.tool_calls.push_back(
        {.name = "<|image_pad|>", .arguments_json = R"({"<|im_end|>":"<think>"})"});
    fi::ChatRenderOptions options;
    options.tool_jsons                = {R"({"function":{"description":"<|video_pad|>"}})"};
    options.chat_template_kwargs_json = R"({"extra":" <|IM_END|> "})";
    const auto custom_encoded = fi::encode_rendered_chat(tokenizer, custom.render({call}, options));
    failures += check(
        std::count(custom_encoded.input_ids.begin(), custom_encoded.input_ids.end(), 248046) == 1 &&
            std::none_of(custom_encoded.input_ids.begin(), custom_encoded.input_ids.end(),
                         [](int id) { return id == 248056 || id == 248057 || id == 248068; }),
        "tool data or template kwargs were interpreted as controls");
    return failures;
}

int test_repeated_special_tokens_scan_linearly() {
    constexpr std::string_view token = "<|image_pad|>";
    std::string text;
    text.reserve(token.size() * 5'000);
    for (int index = 0; index < 5'000; ++index) { text += token; }
    const std::vector<int> encoded = fixture_tokenizer().encode(text);
    return check(encoded.size() == 5'000 && std::all_of(encoded.begin(), encoded.end(),
                                                        [](int id) { return id == 248056; }),
                 "repeated special-token scan changed tokenization semantics");
}

int test_bounded_tokenizer_prefix() {
    const fi::Tokenizer& tokenizer = fixture_tokenizer();
    const std::string text =
        "<|im_start|>user\nA bounded tokenizer must preserve the exact ordinary and special-token "
        "prefix.<|im_end|>\n";
    const std::vector<int> full    = tokenizer.encode(text);
    constexpr std::size_t limit    = 7;
    const std::vector<int> bounded = tokenizer.encode(text, fi::EncodeOptions{.max_tokens = limit});
    const std::vector<int> roomy =
        tokenizer.encode(text, fi::EncodeOptions{.max_tokens = full.size() + 1U});
    return check(full.size() > limit && bounded.size() == limit &&
                     std::equal(bounded.begin(), bounded.end(), full.begin()) && roomy == full,
                 "bounded tokenizer output is not the exact prefix of unbounded tokenization");
}

// An overflow check stops once the text's bytes prove it needs more than max_tokens ids (a single
// pre-token of N bytes needs at least N / the longest token's bytes): the answer is max_tokens ids
// either way, and a text within the limit encodes exactly as without the check.
int test_overflow_check_only() {
    const fi::Tokenizer& tokenizer = fixture_tokenizer();
    const std::string word(1U << 20, 'a');
    const std::vector<int> over =
        tokenizer.encode(word, fi::EncodeOptions{.max_tokens = 1000, .overflow_check_only = true});
    const std::string text      = "<|im_start|>user\nA short prompt within its limit.<|im_end|>\n";
    const std::vector<int> full = tokenizer.encode(text);
    const std::vector<int> checked = tokenizer.encode(
        text, fi::EncodeOptions{.max_tokens = full.size() + 1U, .overflow_check_only = true});
    const std::string medium(5000, 'b');
    const std::size_t bounded =
        tokenizer.encode(medium, fi::EncodeOptions{.max_tokens = 100}).size();
    const std::size_t early =
        tokenizer.encode(medium, fi::EncodeOptions{.max_tokens = 100, .overflow_check_only = true})
            .size();
    return check(over.size() == 1000 && checked == full && bounded == 100 && early == 100,
                 "an overflow check changed the token count or a text within the limit");
}

int test_prompt_continuation() {
    ninferno::PromptInput input;
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = "count to three", .media = {}});
    input.messages.push_back(std::move(message));
    const Frontend frontend = make_frontend(resources(), false);
    const auto prompt       = frontend.prepare(input);
    const auto& source      = FrontendFactory::inspect(prompt);
    const std::vector<ninferno::TokenId> generated{11, 12, 13};
    const auto resumed     = Frontend::continued(prompt, generated);
    const auto& continued  = FrontendFactory::inspect(resumed);
    const std::size_t base = source.token_ids.size();
    const std::size_t all  = base + generated.size();
    bool positions         = continued.positions.size() == all * 3;
    for (std::size_t axis = 0; positions && axis < 3; ++axis) {
        for (std::size_t index = 0; index < all; ++index) {
            const std::int32_t expected =
                index < base ? source.positions[axis * base + index]
                             : static_cast<std::int32_t>(index) + source.rope_delta;
            positions = positions && continued.positions[axis * all + index] == expected;
        }
    }
    int failures = check(Frontend::resumable(prompt), "a text prompt is not resumable");
    failures += check(continued.token_ids.size() == all &&
                          std::equal(source.token_ids.begin(), source.token_ids.end(),
                                     continued.token_ids.begin()) &&
                          std::equal(generated.begin(), generated.end(),
                                     continued.token_ids.begin() +
                                         static_cast<std::ptrdiff_t>(base)) &&
                          continued.token_types.size() == all &&
                          continued.token_types[all - 1] == 0,
                      "a continued prompt does not end with the generated tokens");
    failures += check(positions, "a continued prompt moved positions off decode's");
    failures += check(resumed.summary().prompt_tokens == all,
                      "a continued prompt summarizes the wrong length");
    failures += check(FrontendFactory::inspect(Frontend::copy(prompt)).token_ids == source.token_ids,
                      "a copied prompt differs from its source");
    return failures;
}

int test_context_capacity_guard() {
    ninferno::PromptInput input;
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = "quoted <|image_pad|>", .media = {}});
    input.messages.push_back(std::move(message));

    const Frontend counting         = make_frontend(resources(), false);
    const std::uint32_t exact_count = counting.count_tokens(input);
    ninferno::models::qwen3_5::FrontendOptions exact_options;
    exact_options.vision_enabled = false;
    exact_options.max_context    = exact_count;
    const Frontend exact         = make_frontend(resources(), exact_options);
    int failures = check(exact.prepare(input).summary().prompt_tokens == exact_count,
                         "Frontend rejected a prompt exactly at max_context");

    ninferno::models::qwen3_5::FrontendOptions short_options = exact_options;
    short_options.max_context                              = exact_count - 1U;
    const Frontend short_frontend = make_frontend(resources(), short_options);
    failures += check(short_frontend.count_tokens(input) == exact_count,
                      "exact token counting was incorrectly bounded by max_context");
    failures += check(throws_context_length([&] { (void)short_frontend.prepare(input); }),
                      "Frontend accepted a text prompt at max_context + 1");

    std::vector<ninferno::TokenId> exact_tokens(exact_count, 0);
    failures += check(exact.prepare_tokens(exact_tokens).summary().prompt_tokens == exact_count,
                      "prepare_tokens rejected an exact-capacity token vector");
    exact_tokens.push_back(0);
    failures +=
        check(throws_context_length([&] { (void)exact.prepare_tokens(std::move(exact_tokens)); }),
              "prepare_tokens accepted a token vector at max_context + 1");

    ninferno::models::qwen3_5::FrontendOptions media_options = short_options;
    media_options.vision_enabled                           = true;
    const Frontend media_frontend = make_frontend(resources(), media_options);
    failures += check(throws_context_length([&] {
                          (void)media_frontend.prepare(
                              image_text_input({0}, std::string(64, 'x'), "must-not-decode.bin"));
                      }),
                      "over-capacity media prompt was not rejected before media decoding");

    std::atomic<int> control_checks{0};
    ninferno::PreparationControl cancelled_during_tokenization{
        .deadline     = {},
        .cancellation = ninferno::CancellationView(
            [&control_checks] { return control_checks.fetch_add(1) >= 2; }),
    };
    try {
        (void)media_frontend.prepare(
            image_text_input({0}, std::string(64, 'x'), "cancel-before-oversize.bin"),
            cancelled_during_tokenization);
        failures += check(false, "cancelled over-capacity media prompt completed successfully");
    } catch (const ninferno::RequestError& error) {
        failures += check(
            error.kind() == ninferno::RequestErrorKind::Cancelled && control_checks.load() == 3,
            "media over-capacity result took priority over tokenization cancellation");
    }
    return failures;
}

// A developer message renders as a system message (Codex and the Responses API's instructions send
// one), also through a template that raises on any role it does not know, as the official Qwen
// templates do.
int test_developer_renders_as_system() {
    const std::string strict =
        "{% for m in messages %}"
        "{% if m.role not in ['system', 'user', 'assistant', 'tool'] %}"
        "{{ raise_exception('Unexpected message role.') }}{% endif %}"
        "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    const auto frontend = make_frontend(resources(strict), false);
    const auto prompt   = [](ninferno::ChatRole first) {
        ninferno::PromptInput input;
        for (const auto& [role, text] : {std::pair{first, "Commands run in a sandbox."},
                                         std::pair{ninferno::ChatRole::User, "hi"}}) {
            ninferno::ChatMessage message;
            message.role = role;
            message.parts.push_back({.text = text});
            input.messages.push_back(std::move(message));
        }
        return input;
    };
    // The rendered prompt is encoded region by region, so its ids are compared as decoded text.
    const std::string expected =
        "<|im_start|>system\nCommands run in a sandbox.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n";
    const auto text = [](const auto& ids) {
        const std::vector<int> as_int(ids.begin(), ids.end());
        return fixture_tokenizer().decode(as_int);
    };
    int failures = 0;
    try {
        const auto developer      = frontend.prepare(prompt(ninferno::ChatRole::Developer));
        const auto system         = frontend.prepare(prompt(ninferno::ChatRole::System));
        const auto& developer_ids = FrontendFactory::inspect(developer).token_ids;
        const auto& system_ids    = FrontendFactory::inspect(system).token_ids;
        failures += check(developer_ids == system_ids,
                          "a developer message did not render as the same system message");
        failures += check(text(system_ids) == expected,
                          "the system message did not render through the template as expected");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        failures += check(false, "a developer message was refused by a system-only template");
    }
    return failures;
}

int test_selected_template_instruction_prefix() {
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;

    const std::string leading_developer =
        render_chat_text({chat_message(ninferno::ChatRole::Developer, "policy"),
                          chat_message(ninferno::ChatRole::User, "hi")},
                         no_generation);
    int failures = check(leading_developer == "<|im_start|>system\npolicy<|im_end|>\n"
                                              "<|im_start|>user\nhi<|im_end|>\n",
                         "leading developer did not use the existing Qwen system path");

    const std::string late_system =
        render_chat_text({chat_message(ninferno::ChatRole::User, "hi"),
                          chat_message(ninferno::ChatRole::System, "  current diagnostics  ")},
                         no_generation);
    failures += check(late_system == "<|im_start|>user\nhi<|im_end|>\n"
                                     "<|im_start|>system\ncurrent diagnostics<|im_end|>\n",
                      "late system turn was not rendered at its original position");
    failures += check(
        render_chat_text({chat_message(ninferno::ChatRole::User, "hi"),
                          chat_message(ninferno::ChatRole::Developer, "  current diagnostics  ")},
                         no_generation) == late_system,
        "developer and system did not lower to the same in-place Qwen block");

    const std::string stable_history =
        render_chat_text({chat_message(ninferno::ChatRole::System, "stable policy"),
                          chat_message(ninferno::ChatRole::User, "hi")},
                         no_generation);
    const std::string appended_diagnostics =
        render_chat_text({chat_message(ninferno::ChatRole::System, "stable policy"),
                          chat_message(ninferno::ChatRole::User, "hi"),
                          chat_message(ninferno::ChatRole::System, "current diagnostics")},
                         no_generation);
    failures += check(appended_diagnostics.starts_with(stable_history) &&
                          appended_diagnostics.substr(stable_history.size()) ==
                              "<|im_start|>system\ncurrent diagnostics<|im_end|>\n",
                      "appended diagnostics changed the stable serialized history prefix");
    const std::vector<int> stable_tokens   = fixture_tokenizer().encode(stable_history);
    const std::vector<int> appended_tokens = fixture_tokenizer().encode(appended_diagnostics);
    failures +=
        check(appended_tokens.size() > stable_tokens.size() &&
                  std::equal(stable_tokens.begin(), stable_tokens.end(), appended_tokens.begin()),
              "appended diagnostics changed the stable token prefix");

    fi::ChatRenderOptions tools = no_generation;
    tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}})");
    const std::string tools_with_late_system =
        render_chat_text({chat_message(ninferno::ChatRole::System, "stable policy"),
                          chat_message(ninferno::ChatRole::User, "hi"),
                          chat_message(ninferno::ChatRole::System, "current diagnostics")},
                         tools);
    const std::size_t tools_position  = tools_with_late_system.find("# Tools");
    const std::size_t policy_position = tools_with_late_system.find("stable policy");
    const std::size_t user_position   = tools_with_late_system.find("<|im_start|>user\nhi");
    const std::size_t diagnostics_position =
        tools_with_late_system.find("<|im_start|>system\ncurrent diagnostics");
    failures +=
        check(tools_position != std::string::npos && policy_position != std::string::npos &&
                  user_position != std::string::npos && diagnostics_position != std::string::npos &&
                  tools_with_late_system.find("# Tools", tools_position + 1) == std::string::npos &&
                  tools_position < policy_position && policy_position < user_position &&
                  user_position < diagnostics_position,
              "late system duplicated or moved the leading tools/instruction block");

    const fi::RenderedChat generated =
        render_chat({chat_message(ninferno::ChatRole::User, "hi"),
                     chat_message(ninferno::ChatRole::System, "current diagnostics")});
    const std::string assistant_header = "<|im_start|>assistant\n";
    const std::size_t header           = generated.text.rfind(assistant_header);
    failures += check(header != std::string::npos && generated.rewrite_checkpoint &&
                          generated.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
                          generated.rewrite_checkpoint->offset == header &&
                          generated.text.find("current diagnostics<|im_end|>\n", 0) < header,
                      "late system was not included before the generation rewrite boundary");

    fi::ChatMessage invalid = chat_message(ninferno::ChatRole::System, "diagnostics");
    invalid.tool_calls.push_back({.id = "call", .name = "f", .arguments_json = "{}"});
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninferno::ChatRole::User, "hi"), invalid},
                                            no_generation);
                      }),
                      "system turn carrying assistant tool metadata was accepted");

    fi::ChatMessage media_instruction = chat_message(ninferno::ChatRole::Developer, "diagnostics");
    media_instruction.parts.push_back(fi::ChatPart::image({}));
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat(
                              {chat_message(ninferno::ChatRole::User, "hi"), media_instruction},
                              no_generation);
                      }),
                      "developer turn carrying media was accepted");

    fi::ChatMessage invalid_role = chat_message(ninferno::ChatRole::User, "bad");
    invalid_role.role            = static_cast<ninferno::ChatRole>(255);
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message(ninferno::ChatRole::User, "hi"), invalid_role},
                                    no_generation);
              }),
              "invalid typed chat role was accepted");

    // A replayed call whose arguments are not a JSON object renders its text under "raw": a
    // refusal would fail every later request of the session.
    const auto arguments_template = fi::CompiledChatTemplate::resolve(
        "{% for m in messages %}{% for c in m.tool_calls %}{{ c.function.arguments|tojson }};"
        "{% endfor %}{% endfor %}");
    fi::ChatMessage malformed = chat_message(ninferno::ChatRole::Assistant, "");
    for (const char* text : {"city=Oslo", "{\"city\":", "[1,2]", "\"x\""}) {
        malformed.tool_calls.push_back({.id = "call", .name = "f", .arguments_json = text});
    }
    malformed.tool_calls.push_back({.id = "call", .name = "f", .arguments_json = "{\"a\":1}"});
    std::string rendered_arguments;
    failures += check(!throws_invalid_argument([&] {
        rendered_arguments = arguments_template.render({malformed}, no_generation).text;
    }),
                      "assistant tool call with arguments that are not JSON failed the request");
    std::size_t raw_keys = 0;
    for (std::size_t at = rendered_arguments.find("\"raw\""); at != std::string::npos;
         at             = rendered_arguments.find("\"raw\"", at + 1)) {
        ++raw_keys;
    }
    failures += check(raw_keys == 4 && rendered_arguments.find("city=Oslo") != std::string::npos &&
                          rendered_arguments.find("[1,2]") != std::string::npos &&
                          rendered_arguments.find("\"a\"") != std::string::npos,
                      "malformed history arguments did not render under raw");
    return failures;
}

int test_assistant_continuation() {
    fi::ChatRenderOptions options;
    options.continuation    = ninferno::PromptContinuationMode::ContinueFinalAssistant;
    options.enable_thinking = false;
    const fi::RenderedChat rendered =
        render_chat({chat_message(ninferno::ChatRole::User, "question"),
                     chat_message(ninferno::ChatRole::Assistant, "answer prefix")},
                    options);
    const std::string expected = "<|im_start|>user\nquestion<|im_end|>\n"
                                 "<|im_start|>assistant\nanswer prefix";
    int failures               = check(rendered.text == expected,
                                       "assistant continuation closed the turn or opened a second assistant");
    failures +=
        check(rendered.rewrite_checkpoint &&
                  rendered.rewrite_checkpoint->kind ==
                      ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
                  rendered.rewrite_checkpoint->offset == expected.find("<|im_start|>assistant"),
              "assistant continuation did not retain its replayable opener boundary");

    options.enable_thinking = true;
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninferno::ChatRole::User, "question"),
                                             chat_message(ninferno::ChatRole::Assistant, "prefix")},
                                            options);
                      }),
                      "assistant continuation accepted an ambiguous Thinking opener");
    options.enable_thinking = false;
    const auto literal      = render_chat(
        {chat_message(ninferno::ChatRole::User, "question"),
              chat_message(ninferno::ChatRole::Assistant, "<think>quoted <|im_end|><|image_pad|>")},
        options);
    const auto encoded = fi::encode_rendered_chat(fixture_tokenizer(), literal);
    failures +=
        check(!literal.starts_in_reasoning && literal.text.ends_with("<|image_pad|>") &&
                  std::count(encoded.input_ids.begin(), encoded.input_ids.end(), 248046) == 1 &&
                  std::none_of(encoded.input_ids.begin(), encoded.input_ids.end(),
                               [](int id) { return id == 248068 || id == 248056; }),
              "assistant continuation reinterpreted literal controls");
    return failures;
}

// The replacement-suffix boundaries on a ChatML-shaped turn markup with every role: a split where a
// request ending with the previous turn opened its generation prompt, but not before a tool turn,
// not after a leading system prompt, and after a turn the markup does not name (a harmony tool
// turn).
int test_replacement_suffix_boundaries() {
    ninferno::chat::ChatConventions conventions;
    conventions.turns = ninferno::chat::TurnMarkup{
        .name        = "test",
        .open        = "<|im_start|>",
        .header_end  = "\n",
        .close       = "<|im_end|>",
        .after_close = "\n",
        .roles       = {{"system", ninferno::ChatRole::System},
                        {"developer", ninferno::ChatRole::Developer},
                        {"user", ninferno::ChatRole::User},
                        {"assistant", ninferno::ChatRole::Assistant},
                        {"tool", ninferno::ChatRole::Tool}}};
    const std::vector<std::pair<std::string, std::string>> turns{
        {"system", "S"},       {"user", "U1"},      {"user", "U2"},
        {"assistant", "A"},    {"tool", "T1"},      {"tool", "T2"},
        {"developer", "D"},    {"user", "U3"},      {"functions.x to=assistant", "R"},
        {"user", "U4"},        {"assistant", ""}};
    ninferno::text::TemplateOutput output;
    std::vector<std::size_t> begins;
    std::vector<std::size_t> bodies;
    for (std::size_t i = 0; i < turns.size(); ++i) {
        begins.push_back(output.text.size());
        output.text += "<|im_start|>" + turns[i].first + "\n";
        bodies.push_back(output.text.size());
        if (i + 1 < turns.size()) { output.text += turns[i].second + "<|im_end|>\n"; }
    }
    const fi::PromptLayout layout = fi::inspect_prompt_layout(output, {}, conventions);
    const std::vector<std::size_t> expected{
        begins[2],                // U2 after U1
        begins[3],  bodies[3],    // assistant turn
        begins[6],                // mid-conversation developer message after a tool result
        begins[7],                // U3 after that developer message
        begins[9],                // U4 after the unnamed (tool) turn
        begins[10], bodies[10]};  // generation prompt
    return check(layout.execution_boundaries == expected,
                 "replacement-suffix execution boundaries do not follow the turn roles");
}

// Harmony renders a system then a developer message ahead of the first user turn: both are
// leading instructions, so the first user turn does not split the prefill; a developer message
// after that user turn does.
int test_leading_instruction_run_boundaries() {
    ninferno::chat::ChatConventions conventions;
    conventions.turns = ninferno::chat::TurnMarkup{
        .name        = "test",
        .open        = "<|im_start|>",
        .header_end  = "\n",
        .close       = "<|im_end|>",
        .after_close = "\n",
        .roles       = {{"system", ninferno::ChatRole::System},
                        {"developer", ninferno::ChatRole::Developer},
                        {"user", ninferno::ChatRole::User},
                        {"assistant", ninferno::ChatRole::Assistant}}};
    const std::vector<std::pair<std::string, std::string>> turns{
        {"system", "S"}, {"developer", "D1"}, {"user", "U1"},
        {"developer", "D2"}, {"user", "U2"}, {"assistant", ""}};
    ninferno::text::TemplateOutput output;
    std::vector<std::size_t> begins;
    std::vector<std::size_t> bodies;
    for (std::size_t i = 0; i < turns.size(); ++i) {
        begins.push_back(output.text.size());
        output.text += "<|im_start|>" + turns[i].first + "\n";
        bodies.push_back(output.text.size());
        if (i + 1 < turns.size()) { output.text += turns[i].second + "<|im_end|>\n"; }
    }
    const fi::PromptLayout layout = fi::inspect_prompt_layout(output, {}, conventions);
    const std::vector<std::size_t> expected{
        begins[3],               // D2 after U1 (a mid-conversation instruction)
        begins[4],               // U2 after D2
        begins[5], bodies[5]};   // generation prompt
    return check(layout.execution_boundaries == expected,
                 "leading system and developer messages split the prefill");
}

int test_rewrite_checkpoint_trace() {
    const std::string assistant_header = "<|im_start|>assistant\n";
    fi::ChatMessage first              = chat_message(ninferno::ChatRole::Assistant, "");
    first.reasoning_content            = "first thought";
    first.parts.front().text           = "first answer";
    fi::ChatMessage second             = chat_message(ninferno::ChatRole::Assistant, "");
    second.reasoning_content           = "second thought";
    second.parts.front().text          = "second answer";

    const std::vector<fi::ChatMessage> tool_loop{
        chat_message(ninferno::ChatRole::User, "question"), first,
        chat_message(ninferno::ChatRole::Tool, "result one"), second,
        chat_message(ninferno::ChatRole::Tool, "result two")};
    // Inside a tool loop the next request appends the model's next step, which keeps the loop's
    // reasoning; the boundary is the whole prompt, so every step reuses the steps before it.
    const fi::RenderedChat open       = render_chat(tool_loop);
    const std::size_t open_generation = open.text.rfind(assistant_header);
    int failures          = check(open_generation != std::string::npos && open.rewrite_checkpoint &&
                                      open.rewrite_checkpoint->kind ==
                                          ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
                                      open.rewrite_checkpoint->offset == open_generation,
                                  "tool loop did not retain its whole prefix for the next step");
    fi::ChatMessage third = chat_message(ninferno::ChatRole::Assistant, "");
    third.reasoning_content                = "third thought";
    third.parts.front().text               = "third answer";
    std::vector<fi::ChatMessage> next_step = tool_loop;
    next_step.push_back(third);
    next_step.push_back(chat_message(ninferno::ChatRole::Tool, "result three"));
    const fi::RenderedChat stepped = render_chat(next_step);
    failures +=
        check(open.rewrite_checkpoint &&
                  stepped.text.starts_with(open.text.substr(0, open.rewrite_checkpoint->offset)) &&
                  stepped.rewrite_checkpoint &&
                  stepped.rewrite_checkpoint->offset > open.rewrite_checkpoint->offset,
              "the next tool step did not extend the previous step's boundary");
    const fi::RenderedChat loop_start =
        render_chat({chat_message(ninferno::ChatRole::User, "question")});
    const std::size_t first_header = open.text.find(assistant_header);
    failures += check(loop_start.rewrite_checkpoint &&
                          loop_start.rewrite_checkpoint->offset == first_header,
                      "the loop's first request did not publish the boundary before its first "
                      "assistant turn");

    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking         = true;
    const fi::RenderedChat preserved   = render_chat(tool_loop, preserve);
    const std::size_t preserved_header = preserved.text.rfind(assistant_header);
    failures += check(preserved_header != std::string::npos && preserved.rewrite_checkpoint &&
                          preserved.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
                          preserved.rewrite_checkpoint->offset == preserved_header &&
                          preserved.text.ends_with("<think>\n"),
                      "preserve_thinking did not checkpoint before the generation prologue");

    preserve.enable_thinking             = false;
    const fi::RenderedChat nonthinking   = render_chat(tool_loop, preserve);
    const std::size_t nonthinking_header = nonthinking.text.rfind(assistant_header);
    failures += check(nonthinking_header != std::string::npos && nonthinking.rewrite_checkpoint &&
                          nonthinking.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
                          nonthinking.rewrite_checkpoint->offset == nonthinking_header &&
                          nonthinking.text.ends_with("<think>\n\n</think>\n\n"),
                      "non-thinking response replay did not checkpoint before its generation "
                      "prologue");

    std::vector<fi::ChatMessage> next_turn = tool_loop;
    next_turn.push_back(chat_message(ninferno::ChatRole::User, "next question"));
    const fi::RenderedChat next    = render_chat(next_turn);
    const std::size_t final_header = next.text.rfind(assistant_header);
    failures += check(final_header != std::string::npos && next.rewrite_checkpoint &&
                          next.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
                          next.rewrite_checkpoint->offset == final_header,
                      "new user turn did not move the rewrite boundary before its generation "
                      "opener");

    const fi::RenderedChat branch =
        render_chat({chat_message(ninferno::ChatRole::User, "question"),
                     chat_message(ninferno::ChatRole::User, "summarize the conversation")},
                    preserve);
    const fi::RenderedChat source =
        render_chat({chat_message(ninferno::ChatRole::User, "question")}, preserve);
    failures += check(
        source.rewrite_checkpoint &&
            source.rewrite_checkpoint->kind ==
                ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
            branch.text.starts_with(source.text.substr(0, source.rewrite_checkpoint->offset)) &&
            !branch.text.starts_with(
                source.text.substr(0, source.rewrite_checkpoint->offset + assistant_header.size())),
        "a replacement user suffix lost the stable pre-generation checkpoint");
    // The replacement suffix's prefill splits at the source's checkpoint, with the same execution
    // boundaries before it, so that checkpoint restores it exactly.
    const auto boundaries_through = [](const fi::RenderedChat& rendered, std::size_t offset) {
        std::vector<std::size_t> result;
        for (const std::size_t boundary : rendered.rewrite_execution_boundaries) {
            if (boundary <= offset) { result.push_back(boundary); }
        }
        return result;
    };
    fi::ChatRenderOptions retained;
    retained.preserve_thinking         = true;
    const fi::RenderedChat tool_source = render_chat(tool_loop, retained);
    std::vector<fi::ChatMessage> tool_branch_messages = tool_loop;
    tool_branch_messages.push_back(chat_message(ninferno::ChatRole::User, "summarize the results"));
    const fi::RenderedChat tool_branch = render_chat(tool_branch_messages, retained);
    const std::size_t tool_generation  = tool_source.text.rfind(assistant_header);
    failures += check(
        source.rewrite_checkpoint &&
            boundaries_through(branch, source.rewrite_checkpoint->offset) ==
                boundaries_through(source, source.rewrite_checkpoint->offset) &&
            std::find(branch.rewrite_execution_boundaries.begin(),
                      branch.rewrite_execution_boundaries.end(),
                      source.rewrite_checkpoint->offset) !=
                branch.rewrite_execution_boundaries.end() &&
            std::is_sorted(branch.rewrite_execution_boundaries.begin(),
                           branch.rewrite_execution_boundaries.end()) &&
            tool_generation != std::string::npos &&
            tool_branch.text.starts_with(tool_source.text.substr(0, tool_generation)) &&
            boundaries_through(tool_branch, tool_generation) ==
                boundaries_through(tool_source, tool_generation),
        "a replacement suffix does not split its prefill at the source's generation begin");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat no_assistant =
        render_chat({chat_message(ninferno::ChatRole::User, "question")}, no_generation);
    failures += check(!no_assistant.rewrite_checkpoint,
                      "boundary-less prompt unexpectedly published a rewrite boundary");

    no_generation.preserve_thinking                     = true;
    const fi::RenderedChat preserved_without_generation = render_chat(tool_loop, no_generation);
    failures += check(!preserved_without_generation.rewrite_checkpoint,
                      "response-replay boundary was published without a generation opener");
    no_generation.preserve_thinking = false;

    const fi::RenderedChat wrapped = render_chat(
        {chat_message(ninferno::ChatRole::User, "question"), first,
         chat_message(ninferno::ChatRole::User, "<tool_response>compat result</tool_response>"),
         second},
        no_generation);
    const std::size_t wrapped_first = wrapped.text.find(assistant_header);
    failures += check(wrapped.rewrite_checkpoint &&
                          wrapped.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
                          wrapped.rewrite_checkpoint->offset == wrapped_first,
                      "bare tool-response wrapper incorrectly advanced the real user turn");

    // The checkpoint is a static execution boundary: every prefill of the prompt splits there,
    // whether or not the cache captures it, so a restore there is exactly a cold prefill.
    const auto static_checkpoint = [](const fi::RenderedChat& rendered) {
        return rendered.rewrite_checkpoint &&
               std::find(rendered.rewrite_execution_boundaries.begin(),
                         rendered.rewrite_execution_boundaries.end(),
                         rendered.rewrite_checkpoint->offset) !=
                   rendered.rewrite_execution_boundaries.end() &&
               std::is_sorted(rendered.rewrite_execution_boundaries.begin(),
                              rendered.rewrite_execution_boundaries.end());
    };
    failures += check(static_checkpoint(open) && static_checkpoint(preserved) &&
                          static_checkpoint(nonthinking) && static_checkpoint(next) &&
                          static_checkpoint(source) && static_checkpoint(wrapped),
                      "a rewrite checkpoint is not a static execution boundary of its prompt");
    return failures;
}

int test_selected_template_recovery_boundary() {
    auto assistant              = chat_message(ninferno::ChatRole::Assistant, "answer");
    assistant.reasoning_content = "retained reasoning";
    const std::vector<std::vector<fi::ChatMessage>> histories{
        {chat_message(ninferno::ChatRole::User, "question"), assistant,
         chat_message(ninferno::ChatRole::Tool, "result")},
        {chat_message(ninferno::ChatRole::Tool, "imported result"), assistant}};
    int failures = 0;
    for (const auto& history : histories) {
        const auto qwen36 = thinking_toggle_template().render(history);
        const auto qwen38 =
            fi::CompiledChatTemplate::resolve(reasoning_effort_template_source()).render(history);
        // A history ending with a tool result continues the same turn with the model's next step,
        // which keeps in-loop reasoning in both templates; elsewhere Qwen 3.6 drops it at the next
        // user turn and recovers before the first tail assistant.
        const bool tool_loop              = history.back().role == ninferno::ChatRole::Tool;
        const std::size_t qwen36_boundary = tool_loop ? qwen36.text.rfind("<|im_start|>assistant\n")
                                                      : qwen36.text.find("<|im_start|>assistant\n");
        failures += check(
            qwen36.rewrite_checkpoint && qwen38.rewrite_checkpoint &&
                qwen36.rewrite_checkpoint->offset == qwen36_boundary &&
                qwen38.rewrite_checkpoint->offset == qwen38.text.rfind("<|im_start|>assistant\n"),
            "recovery candidates ignored the selected template's default history policy");
    }
    // Qwen 3.8's template takes low, medium and xhigh: minimal renders as low, high as xhigh (the
    // next stronger level it takes), and the levels it takes are unchanged.
    {
        const auto qwen38 = fi::CompiledChatTemplate::resolve(reasoning_effort_template_source());
        const std::vector<fi::ChatMessage> question{
            chat_message(ninferno::ChatRole::User, "question")};
        const auto at = [&](ninferno::ReasoningEffort effort) {
            try {
                return qwen38.render(question, {.reasoning_effort = effort}).text;
            } catch (const std::exception&) { return std::string("refused"); }
        };
        failures +=
            check(at(ninferno::ReasoningEffort::Minimal) == at(ninferno::ReasoningEffort::Low) &&
                      at(ninferno::ReasoningEffort::High) == at(ninferno::ReasoningEffort::XHigh) &&
                      at(ninferno::ReasoningEffort::Low) != "refused" &&
                      at(ninferno::ReasoningEffort::Medium) != at(ninferno::ReasoningEffort::Low),
                  "an effort the template does not take was not raised to the next level");
    }
    return failures;
}

int test_adjacent_tool_message_boundary() {
    fi::ChatMessage assistant = chat_message(ninferno::ChatRole::Assistant, "");
    assistant.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    const fi::RenderedChat rendered =
        render_chat({chat_message(ninferno::ChatRole::User, "weather?"), std::move(assistant),
                     chat_message(ninferno::ChatRole::Tool, "sunny"),
                     chat_message(ninferno::ChatRole::Tool, "20C")});
    const fi::EncodedChat encoded = fi::encode_rendered_chat(fixture_tokenizer(), rendered);
    return check(rendered.message_boundaries.size() == 5 && rendered.message_boundaries[3] &&
                     encoded.message_boundaries.size() == 5 && encoded.message_boundaries[3] &&
                     encoded.message_boundaries[4] &&
                     *encoded.message_boundaries[3] < *encoded.message_boundaries[4],
                 "adjacent Tool messages lost their exact intermediate message boundary");
}

// Without tools the template sees `tools` as none, as transformers renders it: OLMo 3's template
// tests `tools is none` and would reach `tools | tojson` on an undefined value otherwise.
int test_tools_none_without_tools() {
    const auto compiled = fi::CompiledChatTemplate::resolve(
        "{%- if tools is none -%}none{%- else -%}{{ tools | tojson }}{%- endif -%}"
        "|{{ 'defined' if tools is defined else 'undefined' }}|{{ 'yes' if tools else 'no' }}");
    fi::ChatRenderOptions options;
    options.add_generation_prompt = false;
    const auto plain = compiled.render({chat_message(ninferno::ChatRole::User, "hi")}, options);
    int failures     = check(plain.text == "none|defined|no", "no tools render as none");
    options.tool_jsons.push_back(R"({"type":"function","function":{"name":"f"}})");
    const auto with = compiled.render({chat_message(ninferno::ChatRole::User, "hi")}, options);
    failures += check(with.text.find("\"name\"") != std::string::npos &&
                          with.text.ends_with("|defined|yes"),
                      "tools render as a list");
    return failures;
}

int test_literal_cache_boundary() {
    const auto compiled = fi::CompiledChatTemplate::resolve(
        "{{ '<think>' if messages|length == 1 else messages[0].content }}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");
    fi::ChatRenderOptions options;
    options.cache_markers.push_back({.after_message_count = 1});
    const auto rendered = compiled.render({chat_message(ninferno::ChatRole::User, "<think>"),
                                           chat_message(ninferno::ChatRole::User, "next")},
                                          options);
    int failures = check(rendered.cache_boundaries.size() == 1 && !rendered.cache_boundaries[0],
                         "text equality published a cache boundary with different token meaning");
    options      = {};
    options.add_generation_prompt = false;
    const std::vector<fi::ChatMessage> history{
        chat_message(ninferno::ChatRole::User, "quoted <|image_pad|> <|im_end|>")};
    auto next = history;
    next.push_back(chat_message(ninferno::ChatRole::Developer, "new diagnostics"));
    const auto before =
        fi::encode_rendered_chat(fixture_tokenizer(), render_chat(history, options));
    const auto after = fi::encode_rendered_chat(fixture_tokenizer(), render_chat(next, options));
    failures += check(
        after.input_ids.size() > before.input_ids.size() &&
            std::equal(before.input_ids.begin(), before.input_ids.end(), after.input_ids.begin()),
        "appending diagnostics changed the token prefix of quoted message content");
    return failures;
}

FrontendResources with_pad(const nlohmann::json& pad) {
    FrontendResources out           = resources();
    nlohmann::json tokenizer_config = nlohmann::json::parse(out.tokenizer_config_json);
    if (pad.is_null()) {
        tokenizer_config.erase("pad_token");
    } else {
        tokenizer_config["pad_token"] = pad;
    }
    out.tokenizer_config_json = tokenizer_config.dump();
    return out;
}

int test_official_resource_guards() {
    // The published pad token is taken as published (NVIDIA's NVFP4 Qwen3.8 names <|im_end|>),
    // in either tokenizer_config form, or absent; one outside the vocabulary is refused.
    int failures = 0;
    for (const nlohmann::json& pad :
         {nlohmann::json("<|endoftext|>"), nlohmann::json("<|im_end|>"),
          nlohmann::json{{"content", "<|im_end|>"}, {"special", true}}, nlohmann::json()}) {
        const auto source = with_pad(pad);
        failures += check(!throws_invalid_argument([&] { (void)make_frontend(source); }),
                          ("a published in-vocabulary pad token was refused: " + pad.dump())
                              .c_str());
    }
    const auto unknown = with_pad("<|no_such_pad|>");
    failures += check(throws_invalid_argument([&] { (void)make_frontend(unknown); }),
                      "a pad token outside the vocabulary was accepted");
    const auto malformed = with_pad(7);
    failures += check(throws_invalid_argument([&] { (void)make_frontend(malformed); }),
                      "a malformed pad_token was accepted");

    FrontendResources mismatched       = resources();
    nlohmann::json mismatched_config   = nlohmann::json::parse(mismatched.tokenizer_config_json);
    mismatched_config["chat_template"] = reasoning_effort_template_source();
    mismatched.tokenizer_config_json   = mismatched_config.dump();
    const auto standalone              = make_frontend(mismatched, false);
    ninferno::PromptInput input;
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back({.text = "hello"});
    input.messages.push_back(std::move(message));
    failures += check(standalone.prepare(input).summary().starts_in_reasoning,
                      "tokenizer_config template copy overrode the standalone resource");
    const auto custom   = make_frontend(resources("{{ messages[0].content }}"), false);
    const auto prepared = custom.prepare(input);
    failures += check(!prepared.summary().starts_in_reasoning &&
                          FrontendFactory::inspect(prepared).token_ids ==
                              fixture_tokenizer().encode("hello"),
                      "custom template source was not executed");

    return failures;
}

int test_template_file_execution() {
    struct Temporary {
        std::filesystem::path path =
            std::filesystem::temp_directory_path() /
            ("ninferno-chat-template-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".jinja");

        ~Temporary() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    } file;

    const std::string source = "{% for m in messages %}{{ m.role }}:{{ m.content }}|{% endfor %}"
                               "{{ style|default('plain') }}/{{ extra is none }}"
                               "{{ '<|im_start|>assistant\\n' }}";
    {
        std::ofstream stream(file.path);
        stream << source;
    }
    ninferno::models::qwen3_5::FrontendOptions options;
    options.vision_enabled     = false;
    options.chat_template_path = file.path;
    const auto overridden      = make_frontend(resources("{{ messages[0].content }}"), options);
    {
        std::ofstream stream(file.path);
        stream << "changed after startup";
    }
    ninferno::PromptInput input;
    for (const auto role :
         {ninferno::ChatRole::User, ninferno::ChatRole::System, ninferno::ChatRole::Developer}) {
        ninferno::ChatMessage message;
        message.role = role;
        message.parts.push_back({.text = "text"});
        input.messages.push_back(std::move(message));
    }
    input.options.enable_thinking           = true;
    input.options.chat_template_kwargs_json = R"({"style":"custom","extra":null})";
    const auto prepared                     = overridden.prepare(input);
    const auto expected                     = fixture_tokenizer().encode(
        "user:text|system:text|system:text|custom/True<|im_start|>assistant\n");
    int failures = check(
        FrontendFactory::inspect(prepared).token_ids == expected &&
            !prepared.summary().starts_in_reasoning &&
            overridden.count_tokens(input) == expected.size(),
        "external source, ordered roles, kwargs, or rendered thinking state were not respected");
    const auto embedded = make_frontend(resources(source), false);
    const auto same     = embedded.prepare(input);
    failures += check(FrontendFactory::inspect(same).token_ids == expected,
                      "embedded and external copies of the same template rendered differently");
    // The decision fingerprint's name for the template is the compiled bytes, not the file as it
    // reads after startup; the artifact's own template is "artifact".
    failures += check(overridden.chat_template_source() ==
                              ninferno::named_bytes_sha256(file.path, source) &&
                          embedded.chat_template_source() == "artifact",
                      "the served template is not named by the bytes it compiled");
    const auto rejects_system = make_frontend(
        resources("{% for m in messages %}{% if m.role == 'system' %}{{ raise_exception('template "
                  "declines system') }}{% endif %}{% endfor %}ok"),
        false);
    // The template's own rule holds for a leading instruction; a late one renders as a user
    // message when the template refuses it (below).
    ninferno::PromptInput leading_system;
    for (const auto role : {ninferno::ChatRole::System, ninferno::ChatRole::User}) {
        ninferno::ChatMessage message;
        message.role = role;
        message.parts.push_back({.text = "text"});
        leading_system.messages.push_back(std::move(message));
    }
    failures +=
        check(throws_invalid_argument([&] { (void)rejects_system.prepare(leading_system); }),
              "engine replaced a template's own role rule");
    // A stock Qwen template refuses a system message after the first (Claude Code
    // appends one): the conversation renders with it as a user message instead, the same tokens a
    // client sending it as user gets.
    {
        const auto stock = make_frontend(
            resources("{% for m in messages %}{% if m.role == 'system' and not loop.first %}"
                      "{{ raise_exception('System message must be at the beginning.') }}{% endif %}"
                      "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                      "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"),
            false);
        const auto conversation = [](ninferno::ChatRole late_role) {
            ninferno::PromptInput late;
            for (const auto& [role, text] :
                 std::vector<std::pair<ninferno::ChatRole, std::string>>{
                     {ninferno::ChatRole::User, "question"},
                     {late_role, "reminder"},
                     {ninferno::ChatRole::User, "next"}}) {
                ninferno::ChatMessage message;
                message.role = role;
                message.parts.push_back({.text = text});
                late.messages.push_back(std::move(message));
            }
            late.options.continuation = ninferno::PromptContinuationMode::NewAssistantTurn;
            return late;
        };
        bool rendered = false;
        try {
            const auto demoted = stock.prepare(conversation(ninferno::ChatRole::System));
            const auto as_user = stock.prepare(conversation(ninferno::ChatRole::User));
            rendered           = FrontendFactory::inspect(demoted).token_ids ==
                       FrontendFactory::inspect(as_user).token_ids;
        } catch (const std::exception&) {}
        failures += check(rendered, "a late system message failed a template that refuses one");
        // Two leading system messages: the second renders as a user
        // message, as a later one does.
        const auto opening = [](ninferno::ChatRole second_role) {
            ninferno::PromptInput input;
            for (const auto& [role, text] : std::vector<std::pair<ninferno::ChatRole, std::string>>{
                     {ninferno::ChatRole::System, "rules"},
                     {second_role, "more rules"},
                     {ninferno::ChatRole::User, "question"}}) {
                ninferno::ChatMessage message;
                message.role = role;
                message.parts.push_back({.text = text});
                input.messages.push_back(std::move(message));
            }
            input.options.continuation = ninferno::PromptContinuationMode::NewAssistantTurn;
            return input;
        };
        rendered = false;
        try {
            const auto demoted = stock.prepare(opening(ninferno::ChatRole::System));
            const auto as_user = stock.prepare(opening(ninferno::ChatRole::User));
            rendered           = FrontendFactory::inspect(demoted).token_ids ==
                       FrontendFactory::inspect(as_user).token_ids;
        } catch (const std::exception&) {}
        failures += check(rendered, "two leading system messages failed a template that refuses "
                                    "a second");
    }
    // A template that drops a late system message without raising (one Qwen 3.6 template renders
    // only the leading ones; minefield P-113: the prompt kept its length) renders it as a user
    // message instead of losing it.
    {
        const auto dropping = make_frontend(
            resources("{% for m in messages %}{% if m.role != 'system' or loop.first %}"
                      "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}"
                      "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"),
            false);
        const auto conversation = [](ninferno::ChatRole late_role) {
            ninferno::PromptInput late;
            for (const auto& [role, text] : std::vector<std::pair<ninferno::ChatRole, std::string>>{
                     {ninferno::ChatRole::User, "question"},
                     {late_role, "reminder"},
                     {ninferno::ChatRole::User, "next"}}) {
                ninferno::ChatMessage message;
                message.role = role;
                message.parts.push_back({.text = text});
                late.messages.push_back(std::move(message));
            }
            late.options.continuation = ninferno::PromptContinuationMode::NewAssistantTurn;
            return late;
        };
        bool rendered = false;
        try {
            const auto kept    = dropping.prepare(conversation(ninferno::ChatRole::System));
            const auto as_user = dropping.prepare(conversation(ninferno::ChatRole::User));
            rendered           = FrontendFactory::inspect(kept).token_ids ==
                       FrontendFactory::inspect(as_user).token_ids;
        } catch (const std::exception&) {}
        failures += check(rendered, "a template dropped a late system message silently");
    }
    // A tool declared without a description renders without one; a template that reads it
    // unconditionally (gpt-oss: "// " + tool.description) sees an empty description instead, the
    // same tokens a client declaring "" gets.
    {
        const auto strict = make_frontend(
            // The comment names a tool-call syntax: a template without one cannot take tools.
            resources(
                "{# <tool_call> #}{% for t in tools %}{{ '// ' + t.function.description + ' ' }}"
                "{{ t.function.name }}{% endfor %}{% for m in messages %}<|im_start|>"
                "{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"),
            false);
        const auto with_tool = [](std::string tool) {
            ninferno::PromptInput input;
            ninferno::ChatMessage message;
            message.role = ninferno::ChatRole::User;
            message.parts.push_back({.text = "question"});
            input.messages.push_back(std::move(message));
            input.options.continuation = ninferno::PromptContinuationMode::NewAssistantTurn;
            input.options.tool_jsons.push_back(std::move(tool));
            return input;
        };
        bool rendered = false;
        try {
            const auto bare  = strict.prepare(with_tool(
                R"({"type":"function","function":{"name":"Read","parameters":{"type":"object"}}})"));
            const auto empty = strict.prepare(with_tool(
                R"({"type":"function","function":{"name":"Read","parameters":{"type":"object"},"description":""}})"));
            rendered         = FrontendFactory::inspect(bare).token_ids ==
                       FrontendFactory::inspect(empty).token_ids;
        } catch (const std::exception&) {}
        failures +=
            check(rendered, "a tool without a description failed a template that reads one");
    }
    input.options.chat_template_kwargs_json = R"({"messages":[]})";
    failures += check(throws_invalid_argument([&] { (void)overridden.prepare(input); }),
                      "template kwargs overwrote engine-owned messages");
    {
        std::ofstream stream(file.path);
        stream << "{% if %}";
    }
    failures += check(throws_invalid_argument([&] { (void)make_frontend(resources(), options); }),
                      "malformed external template passed startup parsing");
    return failures;
}

int test_media_token_ids_come_from_tokenizer() {
    auto source    = resources();
    auto tokenizer = nlohmann::json::parse(source.tokenizer_json);
    auto config    = nlohmann::json::parse(source.tokenizer_config_json);
    for (auto& token : tokenizer["added_tokens"]) {
        if (token["content"] == "<|image_pad|>") { token["id"] = 1600; }
    }
    config["added_tokens_decoder"]["1600"] = config["added_tokens_decoder"]["248056"];
    config["added_tokens_decoder"].erase("248056");
    source.tokenizer_json        = tokenizer.dump();
    source.tokenizer_config_json = config.dump();
    const auto frontend          = make_frontend(source);
    const auto prompt            = frontend.prepare(image_input());
    const auto& data             = FrontendFactory::inspect(prompt);
    return check(data.has_media() &&
                     std::count(data.token_ids.begin(), data.token_ids.end(), 1600) > 0 &&
                     std::count(data.token_ids.begin(), data.token_ids.end(), 248056) == 0,
                 "Vision preparation ignored the loaded image token ID");
}

int test_text_and_image_prepare(const Frontend& frontend) {
    ninferno::ChatMessage text_message;
    text_message.role = ninferno::ChatRole::User;
    text_message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput text_input;
    text_input.messages.push_back(std::move(text_message));
    auto text             = frontend.prepare(std::move(text_input));
    const auto& text_data = FrontendFactory::inspect(text);
    // The role is template text (a closed set, not request text), so <|im_start|> + role + newline
    // tokenizes as the rendered string does in transformers: "user\n" is the fixture's added
    // token 30, as "assistant\n" of the generation prompt is 31. Only the content "x" is literal.
    const std::vector<ninferno::TokenId> expected{248045, 30, 0,      248046, 32,
                                                  248045, 31, 248068, 32};
    int failures =
        check(text_data.token_ids == expected, "text frontend did not render/tokenize chat");
    failures += check(text_data.identity.rewrite_checkpoint &&
                          text_data.identity.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
                          text_data.identity.rewrite_checkpoint->frontier == 5 &&
                          text_data.starts_in_reasoning && !text_data.has_media(),
                      "text frontend did not preserve prefix/thinking identity");
    failures +=
        check(text_data.position_axis(0).back() == 8 && text_data.position_axis(1).back() == 8 &&
                  text_data.position_axis(2).back() == 8,
              "text frontend did not construct axis-major positions");

    ninferno::ChatMessage preserved_message;
    preserved_message.role = ninferno::ChatRole::User;
    preserved_message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput preserved_input;
    preserved_input.messages.push_back(std::move(preserved_message));
    preserved_input.options.preserve_thinking = true;
    const auto preserved_prompt               = frontend.prepare(std::move(preserved_input));
    const auto& preserved_data                = FrontendFactory::inspect(preserved_prompt);
    failures += check(preserved_data.identity.rewrite_checkpoint &&
                          preserved_data.identity.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
                          preserved_data.identity.rewrite_checkpoint->frontier == 5 &&
                          preserved_data.identity.rewrite_checkpoint->frontier <
                              preserved_data.token_ids.size(),
                      "preserve-thinking prompt did not publish a pre-generation response "
                      "checkpoint");

    ninferno::ChatMessage nonthinking_message;
    nonthinking_message.role = ninferno::ChatRole::User;
    nonthinking_message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput nonthinking_input;
    nonthinking_input.messages.push_back(std::move(nonthinking_message));
    nonthinking_input.options.preserve_thinking = true;
    nonthinking_input.options.enable_thinking   = false;
    const auto nonthinking_prompt               = frontend.prepare(std::move(nonthinking_input));
    const auto& nonthinking_data                = FrontendFactory::inspect(nonthinking_prompt);
    failures += check(nonthinking_data.identity.rewrite_checkpoint &&
                          nonthinking_data.identity.rewrite_checkpoint->kind ==
                              ninferno::models::qwen3_5::RewriteCheckpointKind::ResponseReplay &&
                          nonthinking_data.identity.rewrite_checkpoint->frontier == 5 &&
                          nonthinking_data.identity.rewrite_checkpoint->frontier <
                              nonthinking_data.token_ids.size() &&
                          !nonthinking_data.starts_in_reasoning,
                      "non-thinking prompt did not publish a pre-generation response checkpoint");

    ninferno::MessagePart image;
    image.kind              = ninferno::MessagePartKind::Media;
    image.media.kind        = ninferno::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninferno::ChatMessage image_message;
    image_message.role = ninferno::ChatRole::User;
    image_message.parts.push_back(std::move(image));
    ninferno::PromptInput image_input;
    image_input.messages.push_back(std::move(image_message));
    image_input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
        .after_message_count      = 1,
        .kind                     = ninferno::PromptCacheMarkerKind::PrivateLongAnchor,
        .location                 = ninferno::PromptCacheMarkerLocation::MessagePartBoundary,
        .after_message_part_count = 1,
    });
    auto prepared             = frontend.prepare(std::move(image_input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    failures += check(prepared_data.has_media() && prepared_data.vision_items.size() == 1,
                      "image frontend did not retain one Vision item");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "image frontend grid/patch/placeholder geometry is incorrect");
        if (!item.token_spans.empty()) {
            const std::size_t span = item.token_spans.front().begin;
            failures += check(
                prepared_data.position_axis(0)[span] == prepared_data.position_axis(1)[span] &&
                    prepared_data.position_axis(1)[span] == prepared_data.position_axis(2)[span] &&
                    prepared_data.position_axis(1)[span + 2] ==
                        prepared_data.position_axis(1)[span] + 1 &&
                    prepared_data.position_axis(2)[span + 1] ==
                        prepared_data.position_axis(2)[span] + 1,
                "image frontend MRoPE positions are incorrect");
        }
    }
    const std::span<const std::uint16_t> image_patches =
        prepared_data.media_payloads.size() == 1 && prepared_data.media_payloads.front()
            ? prepared_data.media_payloads.front()->span()
            : std::span<const std::uint16_t>{};
    failures += check(
        image_patches.size() == 16 * 1536 && prepared_data.prepare.raw_patches == 16 &&
            prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable &&
            prepared_data.identity.rewrite_checkpoint &&
            prepared_data.identity.rewrite_checkpoint->kind ==
                ninferno::models::qwen3_5::RewriteCheckpointKind::TurnClosure &&
            prepared_data.identity.rewrite_checkpoint->frontier < prepared_data.token_ids.size(),
        "image frontend did not own the expected patch payload and identity");
    if (!prepared_data.vision_items.empty() &&
        !prepared_data.vision_items.front().token_spans.empty()) {
        const auto span            = prepared_data.vision_items.front().token_spans.front();
        const auto explicit_marker = std::find_if(
            prepared_data.context_cache.opportunities.begin(),
            prepared_data.context_cache.opportunities.end(), [](const auto& opportunity) {
                return ninferno::has_shared_candidate_evidence(
                    opportunity.evidence, ninferno::SharedCandidateEvidence::ExplicitBoundary);
            });
        failures += check(explicit_marker != prepared_data.context_cache.opportunities.end() &&
                              explicit_marker->frontier >= span.begin + span.count &&
                              explicit_marker->frontier < prepared_data.token_ids.size(),
                          "media expansion did not remap the following message cache boundary");
    }
    if (image_patches.size() == 16 * 1536) {
        failures += check(image_patches[0] == bf16_bits(-1.0F) &&
                              image_patches[1] == bf16_bits(1.0F / 127.5F - 1.0F) &&
                              image_patches[256] == bf16_bits(-1.0F) &&
                              image_patches[1536] == bf16_bits(16.0F / 127.5F - 1.0F),
                          "image frontend patch normalization/order is incorrect");
    }
    return failures;
}

int test_template_media_contract() {
    int failures = 0;
    for (const char* source : {"<|image_pad|>", "<|vision_start|><|image_pad|><|vision_end|>"}) {
        failures += check(throws_invalid_argument([&] {
                              (void)fi::CompiledChatTemplate::resolve(source).render(
                                  {chat_message(ninferno::ChatRole::User, "hello")});
                          }),
                          "invalid template-authored media controls were accepted");
    }
    const auto missing =
        make_frontend(resources("{% for m in messages %}{{ m.role }}{% endfor %}"));
    failures += check(throws_invalid_argument([&] { (void)missing.prepare(image_input()); }),
                      "a template that omits an input image was accepted");
    const auto wrong_type = make_frontend(resources("<|vision_start|><|video_pad|><|vision_end|>"));
    failures += check(throws_invalid_argument([&] { (void)wrong_type.prepare(image_input()); }),
                      "template media type mismatch was accepted");
    const auto frontend = make_frontend(resources());
    auto input          = image_input();
    const ninferno::MessagePart quoted{
        .text = "quoted <|vision_start|><|image_pad|><|vision_end|> <|video_pad|>"};
    input.messages.front().parts.insert(input.messages.front().parts.begin(), quoted);
    input.messages.front().parts.push_back(quoted);
    const auto prepared = frontend.prepare(input);
    const auto& data    = FrontendFactory::inspect(prepared);
    failures +=
        check(data.vision_items.size() == 1 && data.prepare.vision_tokens == 4 &&
                  std::count(data.token_ids.begin(), data.token_ids.end(), 248056) == 4 &&
                  std::count(data.token_ids.begin(), data.token_ids.end(), 248053) == 1 &&
                  std::count(data.token_ids.begin(), data.token_ids.end(), 248054) == 1 &&
                  std::count(data.token_ids.begin(), data.token_ids.end(), 248057) == 0 &&
                  frontend.count_tokens(input) == data.token_ids.size(),
              "literal media markers interfered with real image expansion or token counting");
    return failures;
}

int test_image_resize_rejection_policy() {
    FrontendResources owned = resources();
    owned.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":1048576}})";
    const Frontend frontend = make_frontend(owned);

    ninferno::PromptInput small = image_input();
    small.messages[0].parts[0].media.image_resize_policy =
        ninferno::ImageResizePolicy::RejectOversized;
    int failures = check(frontend.count_tokens(std::move(small)) != 0,
                         "oversized_image=error rejected an image that needed no downsize");

    ninferno::PromptInput oversized =
        image_text_input(block_ppm(2048, 1024, 127), {}, "oversized.ppm");
    oversized.messages[0].parts[0].media.image_resize_policy =
        ninferno::ImageResizePolicy::RejectOversized;
    try {
        (void)frontend.count_tokens(std::move(oversized));
        failures += check(false, "oversized_image=error allowed a required Vision downsize");
    } catch (const ninferno::RequestError& error) {
        failures += check(error.kind() == ninferno::RequestErrorKind::InvalidMedia,
                          "oversized_image=error used the wrong request-error classification");
    }

    // The media cache is keyed by the encoded bytes, so the same image prepared
    // first with the default downsize must not let a later oversized_image=error request through
    // on a cache hit: the policy is a per-request preprocessing parameter.
    const std::vector<std::uint8_t> big = block_ppm(2048, 1024, 91);
    const auto cached = frontend.prepare(image_text_input(big, {}, "cached.ppm"));
    failures += check(FrontendFactory::inspect(cached).has_media(),
                      "the downsized image was not prepared");
    ninferno::PromptInput refused = image_text_input(big, {}, "cached.ppm");
    refused.messages[0].parts[0].media.image_resize_policy =
        ninferno::ImageResizePolicy::RejectOversized;
    try {
        (void)frontend.prepare(std::move(refused));
        failures += check(false, "a media-cache hit bypassed oversized_image=error");
    } catch (const ninferno::RequestError& error) {
        failures += check(error.kind() == ninferno::RequestErrorKind::InvalidMedia,
                          "a media-cache hit changed the oversized_image=error classification");
    }
    return failures;
}

// tool_choice none keeps the declarations: the prompt is the tool-enabled one byte for byte (the
// conversation's cache survives a turn without calls), and no output is read back as a call.
int test_disabled_tool_calls_keep_the_prompt() {
    const Frontend frontend = make_frontend(resources(), false);
    const auto input        = [](bool calls) {
        ninferno::PromptInput result;
        ninferno::ChatMessage user;
        user.role = ninferno::ChatRole::User;
        user.parts.push_back(ninferno::MessagePart{
                   .kind = ninferno::MessagePartKind::Text, .text = "summarize", .media = {}});
        result.messages.push_back(std::move(user));
        result.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}})");
        result.options.tool_calls_enabled = calls;
        return result;
    };
    const auto enabled  = frontend.prepare(input(true));
    const auto disabled = frontend.prepare(input(false));
    return check(FrontendFactory::inspect(enabled).token_ids ==
                         FrontendFactory::inspect(disabled).token_ids &&
                     FrontendFactory::inspect(enabled).tool_call_output != nullptr &&
                     FrontendFactory::inspect(disabled).tool_call_output == nullptr,
                 "disabled tool calls changed the prompt or still read calls back");
}

// OpenAI's automatic boundary after the last message's last part is taken at the closed turn's end,
// an execution frontier of the prompt: the capture adds no prefill split (DECISIONS 63).
int test_last_part_boundary_at_turn_end() {
    const Frontend frontend = make_frontend(resources(), false);
    ninferno::ChatMessage user;
    user.role = ninferno::ChatRole::User;
    user.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = "first part", .media = {}});
    user.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = " and the last part", .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(user));
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    for (const std::uint32_t parts : {1U, 2U}) {
        input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
            .after_message_count      = 1,
            .kind                     = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence                 = ninferno::SharedCandidateEvidence::DefaultAutomatic,
            .location                 = ninferno::PromptCacheMarkerLocation::MessagePartBoundary,
            .after_message_part_count = parts,
        });
    }
    const auto prepared   = frontend.prepare(std::move(input));
    const auto& data      = FrontendFactory::inspect(prepared);
    const auto& static_at = data.identity.rewrite_execution_frontiers;
    const auto& found     = data.context_cache.opportunities;
    const auto at_static  = [&](std::uint32_t frontier) {
        return std::find(static_at.begin(), static_at.end(), frontier) != static_at.end();
    };
    int failures = check(found.size() == 2, "both part boundaries became opportunities");
    if (found.size() == 2) {
        const std::uint32_t middle = std::min(found[0].frontier, found[1].frontier);
        const std::uint32_t last   = std::max(found[0].frontier, found[1].frontier);
        failures += check(at_static(last) && last < data.token_ids.size() &&
                              data.identity.rewrite_checkpoint &&
                              last == data.identity.rewrite_checkpoint->frontier,
                          "the boundary after the last part is the turn's end, the turn closure's "
                          "static frontier");
        failures += check(!at_static(middle) && middle < last,
                          "a boundary between parts stays where its part ends");
    }
    return failures;
}

int test_explicit_leading_instruction_cache_boundary() {
    const Frontend frontend           = make_frontend(resources(), false);
    constexpr std::string_view stable = "stable cache section.";
    ninferno::ChatMessage system;
    system.role = ninferno::ChatRole::System;
    system.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = std::string(stable), .media = {}});
    system.parts.push_back(ninferno::MessagePart{.kind  = ninferno::MessagePartKind::Text,
                                                 .text  = "\ndynamic working directory",
                                                 .media = {}});
    ninferno::ChatMessage user;
    user.role = ninferno::ChatRole::User;
    user.parts.push_back(ninferno::MessagePart{
        .kind = ninferno::MessagePartKind::Text, .text = "question", .media = {}});

    ninferno::PromptInput input;
    input.messages.push_back(std::move(system));
    input.messages.push_back(std::move(user));
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}})");
    input.context_cache.markers.push_back(ninferno::PromptCacheMarker{
        .kind     = ninferno::PromptCacheMarkerKind::SharedStablePrefix,
        .location = ninferno::PromptCacheMarkerLocation::LeadingInstructionBoundary,
        .leading_instruction_bytes = static_cast<std::uint32_t>(stable.size()),
    });

    const auto prepared        = frontend.prepare(std::move(input));
    const auto& data           = FrontendFactory::inspect(prepared);
    const auto explicit_marker = std::find_if(
        data.context_cache.opportunities.begin(), data.context_cache.opportunities.end(),
        [](const auto& opportunity) {
            return ninferno::has_shared_candidate_evidence(
                opportunity.evidence, ninferno::SharedCandidateEvidence::ExplicitBoundary);
        });
    return check(explicit_marker != data.context_cache.opportunities.end() &&
                     explicit_marker->kind == ninferno::PromptCacheMarkerKind::SharedStablePrefix &&
                     explicit_marker->frontier != 0 &&
                     explicit_marker->frontier < data.token_ids.size(),
                 "explicit leading-system cache boundary was lost or shadowed by the automatic "
                 "full-system marker");
}

int test_media_admission_uses_aggregate_resources(const Frontend& frontend) {
    constexpr std::size_t kMediaItems     = 17;
    const std::vector<std::uint8_t> bytes = gradient_ppm();
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    for (std::size_t index = 0; index < kMediaItems; ++index) {
        ninferno::OwnedMedia media;
        media.kind        = ninferno::MediaKind::Image;
        media.bytes       = bytes;
        media.media_type  = "image/x-portable-pixmap";
        media.source_name = "aggregate-" + std::to_string(index) + ".ppm";
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Media, .text = {}, .media = std::move(media)});
    }
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    const auto prepared = frontend.prepare(std::move(input));
    const auto& data    = FrontendFactory::inspect(prepared);
    int failures        = check(data.prepare.media_items == kMediaItems &&
                                    data.prepare.media_bytes == kMediaItems * bytes.size() &&
                                    data.prepare.raw_patches == kMediaItems * 16 &&
                                    data.prepare.vision_tokens == kMediaItems * 4 &&
                                    data.vision_items.size() == kMediaItems,
                                "frontend retained an item-count admission limit");

    fi::ProcessorOptions options;
    options.max_encoded_media_bytes = bytes.size() * 2 - 1;
    auto cache = std::make_shared<fi::MediaPreprocessCache>(ninferno::kDefaultMediaCacheBytes,
                                                            ninferno::kDefaultMediaLiveBytes);
    fi::Processor processor(fixture_tokenizer(), thinking_toggle_template(), options,
                            std::move(cache));
    fi::ChatMessage internal_message;
    internal_message.role = ninferno::ChatRole::User;
    for (std::size_t index = 0; index < 2; ++index) {
        internal_message.parts.push_back(
            fi::ChatPart::image(fi::MediaData{.bytes       = bytes,
                                              .media_type  = "image/x-portable-pixmap",
                                              .source_name = "byte-budget.ppm"}));
    }
    failures += check(throws_processor_budget([&] {
                          (void)processor.process(std::vector<fi::ChatMessage>{internal_message});
                      }),
                      "processor did not enforce the aggregate encoded-media byte budget");
    return failures;
}

int test_multimodal_prompt_over_removed_32k_cap(const Frontend& frontend) {
    const std::string long_text(40'000, 'x');
    const ninferno::MediaCacheSummary before_count = frontend.media_cache_summary();
    const std::uint32_t counted =
        frontend.count_tokens(image_text_input(gradient_ppm(), long_text, "long-context.ppm"));
    const ninferno::MediaCacheSummary after_count = frontend.media_cache_summary();
    const auto prepared =
        frontend.prepare(image_text_input(gradient_ppm(), long_text, "long-context.ppm"));
    const auto& data = FrontendFactory::inspect(prepared);

    int failures = check(counted > 32'768 && data.token_ids.size() == counted,
                         "multimodal prompt retained the removed 32K frontend token cap");
    failures += check(after_count.entries == before_count.entries &&
                          after_count.live_bytes == before_count.live_bytes &&
                          after_count.hits == before_count.hits &&
                          after_count.misses == before_count.misses,
                      "multimodal token counting mutated the prepared-media cache");
    failures += check(data.has_media() && data.vision_items.size() == 1,
                      "long multimodal prompt lost its Vision item");
    return failures;
}

int test_attention_pairs_are_diagnostic(const Frontend& frontend) {
    constexpr std::uint64_t kRemovedAttentionPairLimit = 128ULL * 1024ULL * 1024ULL;
    const auto prepared =
        frontend.prepare(image_text_input(block_ppm(2048, 1536, 127), {}, "large-grid.ppm"));
    const auto& data = FrontendFactory::inspect(prepared);

    int failures = check(data.prepare.attention_pairs > kRemovedAttentionPairLimit,
                         "test image did not exceed the removed attention-pair threshold");
    failures += check(data.prepare.raw_patches == 12'288 && data.prepare.vision_tokens == 3'072 &&
                          data.vision_items.size() == 1,
                      "large image did not retain its expected Vision geometry");
    return failures;
}

int test_video_prepare(const Frontend& frontend) {
    ninferno::MessagePart video;
    video.kind              = ninferno::MessagePartKind::Media;
    video.media.kind        = ninferno::MediaKind::Video;
    video.media.bytes       = gradient_ppm();
    video.media.media_type  = "image/x-portable-pixmap";
    video.media.source_name = "single-frame.ppm";
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(std::move(video));
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));

    const ninferno::MediaCacheSummary before_count = frontend.media_cache_summary();
    const std::uint32_t counted                  = frontend.count_tokens(input);
    const ninferno::MediaCacheSummary after_count  = frontend.media_cache_summary();
    auto prepared                                = frontend.prepare(std::move(input));
    const auto& prepared_data                    = FrontendFactory::inspect(prepared);
    int failures = check(prepared_data.vision_items.size() == 1 && prepared_data.has_media() &&
                             prepared_data.token_ids.size() == counted,
                         "video token counting and preparation geometry diverged");
    failures += check(after_count.entries == before_count.entries &&
                          after_count.live_bytes == before_count.live_bytes &&
                          after_count.hits == before_count.hits &&
                          after_count.misses == before_count.misses,
                      "video token counting mutated the prepared-media cache");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.modality == ninferno::models::qwen3_5::PromptModality::Video &&
                      item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.timestamps.size() == 1 && item.timestamps.front() == 0.0 &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "video frontend temporal/grid/placeholder metadata is incorrect");
    }
    const std::span<const std::uint16_t> video_patches =
        prepared_data.media_payloads.size() == 1 && prepared_data.media_payloads.front()
            ? prepared_data.media_payloads.front()->span()
            : std::span<const std::uint16_t>{};
    failures += check(
        video_patches.size() == 16 * 1536 && video_patches[0] == video_patches[256] &&
            prepared_data.prepare.raw_patches == 16 && prepared_data.prepare.vision_tokens == 4 &&
            prepared_data.prepare.media_cache_misses == 1 &&
            prepared_data.prepare.media_cache_hits == 0 && prepared_data.identity.reusable,
        "video frontend did not duplicate the odd temporal frame correctly");
    return failures;
}

int test_cross_round_stop(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninferno::StopPolicy stop;
    stop.strings.push_back(ninferno::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision = session.preview_model(std::array<ninferno::TokenId, 1>{1}, 2,
                                                      ninferno::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "cross-round stop ended before the stop string was complete");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninferno::OutputChannel::Content) == "hello",
                      "cross-round stop did not retain the ambiguous suffix");

    const auto second_decision = session.preview_model(std::array<ninferno::TokenId, 1>{2}, 1,
                                                       ninferno::FinishReason::OutputLimit);
    failures += check(second_decision.accepted_tokens == 1 &&
                          second_decision.finish_reason == ninferno::FinishReason::StopString,
                      "cross-round stop did not select the exact terminal token prefix");
    const auto second = session.commit_preview();
    failures += check(second.empty(), "stop marker or same-token suffix leaked to output");
    failures += check(session.matched_stop_string() == std::optional<std::string>("STOP"),
                      "terminal output session lost the matched stop declaration");
    return failures;
}

int test_same_token_stop_priority(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninferno::StopPolicy stop;
    stop.strings = {
        ninferno::StopString{.text = "tail", .include_in_output = true},
        ninferno::StopString{.text = "OPtail"},
        ninferno::StopString{.text = "OP", .include_in_output = true},
    };
    auto session        = frontend.make_output_session(prompt, stop);
    const auto decision = session.preview_model(std::array<ninferno::TokenId, 1>{2}, 2,
                                                ninferno::FinishReason::OutputLimit);
    int failures        = check(decision.accepted_tokens == 1 &&
                                    decision.finish_reason == ninferno::FinishReason::StopString,
                                "same-token stop strings did not select a terminal prefix");
    const auto output   = session.commit_preview();
    failures += check(output.empty(),
                      "same-token stops did not prefer the earliest byte and declaration order");
    return failures;
}

int test_terminal_flush(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninferno::StopPolicy stop;
    stop.strings.push_back(ninferno::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision = session.preview_model(std::array<ninferno::TokenId, 1>{1}, 2,
                                                      ninferno::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "terminal flush setup unexpectedly finished");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninferno::OutputChannel::Content) == "hello",
                      "terminal flush setup did not retain the possible stop suffix");

    const auto terminal = session.preview_terminal(ninferno::FinishReason::Cancelled);
    failures += check(terminal.accepted_tokens == 0 &&
                          terminal.finish_reason == ninferno::FinishReason::Cancelled,
                      "between-round terminal preview returned the wrong decision");
    const auto flushed = session.commit_preview();
    failures += check(channel_text(flushed, ninferno::OutputChannel::Content) == "ST",
                      "between-round terminal preview lost the pending stop suffix");
    return failures;
}

int test_structured_tool_output() {
    const Frontend frontend = make_frontend(resources());

    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"TaskUpdate","parameters":{"type":"object","properties":{"taskId":{"type":"string"},"enabled":{"type":"boolean"},"count":{"type":"integer"}},"required":["taskId"]}}})");
    auto prompt = frontend.prepare(std::move(input));
    auto session = frontend.make_output_session(
        prompt, {}, ninferno::OutputOptions{.tool_name_max_length = 64});

    const std::string generated =
        "Calling.  \n<tool_call>\n<function=TaskUpdate>\n<parameter=taskId>\n1\n"
        "</parameter>\n<parameter=enabled>\n</parameter>\n<parameter=count>\nmany\n"
        "</parameter>\n</function>\n</tool_call>";
    const std::vector<ninferno::TokenId> tokens = fixture_tokenizer().encode(generated);
    const auto decision = session.preview_model(tokens, static_cast<std::uint32_t>(tokens.size()),
                                                ninferno::FinishReason::OutputLimit);
    int failures        = check(decision.finish_reason == ninferno::FinishReason::OutputLimit,
                                "tool output did not reach the terminal transaction");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninferno::OutputChannel::Content) == "Calling.",
                      "frontend did not hide the terminal tool-call suffix");
    const std::vector<ninferno::GeneratedToolCall> calls = session.take_tool_calls();
    failures += check(calls.size() == 1 && calls.front().name == "TaskUpdate",
                      "frontend did not publish the structured tool call");
    failures += check(session.tool_call_parse_diagnostics() ==
                          ninferno::ToolCallParseDiagnostics{
                              .marker_seen               = true,
                              .structured_call_count     = 1,
                              .empty_arguments_omitted   = 1,
                              .schema_mismatch_arguments = 1,
                              .fallback_reason = ninferno::ToolCallParseFallbackReason::None,
                          },
                      "frontend did not retain tool-call parse diagnostics");
    if (!calls.empty()) {
        const nlohmann::json arguments = nlohmann::json::parse(calls.front().arguments_json);
        failures += check(arguments.at("taskId").is_string() && arguments.at("taskId") == "1" &&
                              !arguments.contains("enabled") && arguments.at("count") == "many",
                          "frontend did not preserve normalized tool arguments");
    }
    return failures;
}

// Upstream #377: the model ends the turn with its stop token inside a thinking span it never
// closed, and the reasoning ends in a complete call. The call is the turn's call; a call quoted
// before more reasoning, or one the output limit cut off, is not.
int test_tool_call_in_unclosed_reasoning() {
    const Frontend frontend = make_frontend(resources());
    const auto make_prompt  = [&] {
        ninferno::ChatMessage message;
        message.role = ninferno::ChatRole::User;
        message.parts.push_back(ninferno::MessagePart{
            .kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
        ninferno::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.continuation    = ninferno::PromptContinuationMode::NewAssistantTurn;
        input.options.enable_thinking = true;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"TaskUpdate","parameters":{"type":"object","properties":{"taskId":{"type":"string"}},"required":["taskId"]}}})");
        return frontend.prepare(std::move(input));
    };
    const std::string call = "<tool_call>\n<function=TaskUpdate>\n<parameter=taskId>\n1\n"
                             "</parameter>\n</function>\n</tool_call>";

    struct Case {
        std::string generated;
        bool stop_token;
        bool recovered;
        const char* what;
    };

    const std::array<Case, 3> cases{{
        {"I should update it.\n" + call, true, true, "a call ending unclosed reasoning was lost"},
        {"Something like " + call + " might work, but first", true, false,
         "a call quoted inside reasoning was taken"},
        {"I should update it.\n" + call, false, false, "a call cut off by the limit was taken"},
    }};
    int failures = 0;
    for (const Case& c : cases) {
        auto prompt  = make_prompt();
        auto session = frontend.make_output_session(
            prompt, {}, ninferno::OutputOptions{.tool_name_max_length = 64});
        std::vector<ninferno::TokenId> tokens = fixture_tokenizer().encode(c.generated);
        if (c.stop_token) { tokens.push_back(6); }
        const auto budget = static_cast<std::uint32_t>(tokens.size() + (c.stop_token ? 4U : 0U));
        const auto decision =
            session.preview_model(tokens, budget, ninferno::FinishReason::OutputLimit);
        failures +=
            check(decision.finish_reason == (c.stop_token ? ninferno::FinishReason::StopToken
                                                          : ninferno::FinishReason::OutputLimit),
                  "unclosed-reasoning case did not reach its terminal");
        const auto output = session.commit_preview();
        failures += check(channel_text(output, ninferno::OutputChannel::Reasoning) == c.generated &&
                              channel_text(output, ninferno::OutputChannel::Content).empty(),
                          "unclosed reasoning was not published as reasoning");
        const std::vector<ninferno::GeneratedToolCall> calls = session.take_tool_calls();
        if (c.recovered) {
            failures += check(calls.size() == 1 && calls.front().name == "TaskUpdate" &&
                                  calls.front().arguments_json == R"({"taskId":"1"})" &&
                                  session.tool_call_parse_diagnostics().recovered_from_reasoning &&
                                  !session.tool_call_parse_diagnostics().ended_inside_reasoning &&
                                  session.tool_call_parse_diagnostics().structured_call_count == 1,
                              c.what);
        } else {
            failures += check(calls.empty() &&
                                  !session.tool_call_parse_diagnostics().recovered_from_reasoning &&
                                  session.tool_call_parse_diagnostics().ended_inside_reasoning ==
                                      c.stop_token,
                              c.what);
        }
    }
    return failures;
}

// A reasoning close the model quotes while thinking (code or prose about the format) stays
// reasoning when the session reads quoted closes as text (requests without an output constraint):
// a close right after a quote mark or backtick ends the reasoning only when a line break follows it
// or the output ends there. Tokens arrive one at a time, so a quoted close is held
// across rounds until its next byte.
int test_quoted_reasoning_close(const Frontend& frontend) {
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.continuation    = ninferno::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    const auto prompt             = frontend.prepare(std::move(input));

    const auto run = [&](const std::string& text, bool quoted_is_text, bool terminal) {
        auto session = frontend.make_output_session(prompt, {}, {}, {}, quoted_is_text);
        std::pair<std::string, std::string> split;
        const auto take = [&](const PublishedOutput& output) {
            split.first += channel_text(output, ninferno::OutputChannel::Reasoning);
            split.second += channel_text(output, ninferno::OutputChannel::Content);
        };
        // A terminal run ends on its last token through the output budget.
        const auto tokens = fixture_tokenizer().encode(text);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto remaining = terminal ? static_cast<std::uint32_t>(tokens.size() - i) : 4096U;
            (void)session.preview_model(std::array<ninferno::TokenId, 1>{tokens[i]}, remaining,
                                        ninferno::FinishReason::OutputLimit);
            take(session.commit_preview());
        }
        return split;
    };

    const std::string code = "The mock returns \"</think>\\n\\n2\" here.\n</think>\n\nThe answer.";
    const auto quoted      = run(code, true, false);
    int failures =
        check(quoted.first.find("returns \"</think>\\n\\n2\" here.") != std::string::npos &&
                  quoted.second == "The answer.",
              "a close quoted in code ended the reasoning");
    const auto backtick = run("Write `</think>` to end it.\n</think>\n\nDone.", true, false);
    failures += check(backtick.first.find("`</think>` to end it.") != std::string::npos &&
                          backtick.second == "Done.",
                      "a close quoted in backticks ended the reasoning");
    const auto strict = run(code, false, false);
    failures += check(strict.second.find("\\n\\n2\" here.") != std::string::npos,
                      "without the quoted rule the first close did not end the reasoning");
    const auto line_break = run("It is \"</think>\n\nthe answer", true, false);
    failures +=
        check(line_break.second == "the answer",
              "a close after a quote mark and before a line break did not end the reasoning");
    const auto at_end = run("It ends with \"</think>", true, true);
    failures += check(at_end.first == "It ends with \"" && at_end.second.empty(),
                      "a quoted close that ended the output did not end the reasoning");
    return failures;
}

int test_reasoning_split(const Frontend& frontend) {
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.continuation    = ninferno::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    auto prompt                   = frontend.prepare(std::move(input));

    auto canonical_session = frontend.make_output_session(prompt, {});
    const std::vector<ninferno::TokenId> canonical_tokens =
        fixture_tokenizer().encode("thought\n</think>\n\n");
    const auto canonical = canonical_session.preview_model(
        canonical_tokens, static_cast<std::uint32_t>(canonical_tokens.size() + 1U),
        ninferno::FinishReason::OutputLimit);
    int failures =
        check(canonical.accepted_tokens == canonical_tokens.size() && !canonical.finished() &&
                  canonical.prefix_execution_split_after == canonical_tokens.size(),
              "canonical reasoning close did not publish its exact token execution frontier");
    (void)canonical_session.commit_preview();

    auto session = frontend.make_output_session(prompt, {});
    const std::array<ninferno::TokenId, 2> tokens{3, 4};
    const auto decision = session.preview_model(tokens, 2, ninferno::FinishReason::OutputLimit);
    failures += check(decision.accepted_tokens == 2 &&
                          decision.finish_reason == ninferno::FinishReason::OutputLimit &&
                          !decision.prefix_execution_split_after,
                      "reasoning close inside a mixed token produced an execution frontier");
    const auto output = session.commit_preview();
    failures += check(channel_text(output, ninferno::OutputChannel::Reasoning) == "thought",
                      "reasoning channel did not remove the close marker");
    failures += check(channel_text(output, ninferno::OutputChannel::Content) == "answer",
                      "content channel did not strip the post-thinking separator");
    failures += check(session.reasoning_tokens() == 2,
                      "reasoning token usage did not count accepted reasoning tokens exactly");
    // Thinking without a budget still counts its model-origin tokens: the request
    // log reports them, and they match the reasoning usage at a canonical close.
    auto counted = frontend.make_output_session(prompt, {});
    (void)counted.preview_model(canonical_tokens,
                                static_cast<std::uint32_t>(canonical_tokens.size() + 1U),
                                ninferno::FinishReason::OutputLimit);
    (void)counted.commit_preview();
    const ninferno::ThinkingBudgetStats unlimited = counted.thinking_stats();
    failures += check(!unlimited.configured_budget && unlimited.model_thinking_tokens > 0 &&
                          unlimited.model_thinking_tokens == counted.reasoning_tokens() &&
                          unlimited.injected_tokens == 0 && !unlimited.applied,
                      "uncapped thinking did not count its model-origin thinking tokens");
    return failures;
}

// Gemma 4 spells its reasoning open marker as a special token followed by plain text
// (<|channel>thought) and closes it with a special token (<channel|>). Both are hidden special
// tokens of the vocabulary; the reasoning parser must still read them.
int test_special_token_reasoning_markers() {
    const std::string gemma_template =
        "{% for m in messages %}<|turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
        "{{ m.content }}<turn|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|turn>model\n"
        "{% if not enable_thinking %}<|channel>thought\n<channel|>{% endif %}{% endif %}";
    FrontendResources source = resources(gemma_template);
    nlohmann::json tokenizer_json = nlohmann::json::parse(source.tokenizer_json);
    nlohmann::json config         = nlohmann::json::parse(source.tokenizer_config_json);
    const std::array<std::pair<int, const char*>, 4> specials{
        {{100, "<|turn>"}, {101, "<turn|>"}, {102, "<|channel>"}, {103, "<channel|>"}}};
    for (const auto& [id, content] : specials) {
        tokenizer_json["added_tokens"].push_back(added(id, content, true));
        config["added_tokens_decoder"][std::to_string(id)] = decoder_added(content, true);
    }
    source.tokenizer_json        = tokenizer_json.dump();
    source.tokenizer_config_json = config.dump();
    const auto frontend          = make_frontend(source, false);
    const fi::Tokenizer tokenizer({.tokenizer_json         = source.tokenizer_json,
                                   .tokenizer_config_json  = source.tokenizer_config_json,
                                   .generation_config_json = source.generation_config_json});

    ninferno::PromptInput input;
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back({.text = "hi"});
    input.messages.push_back(std::move(message));
    input.options.continuation    = ninferno::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    const auto prompt = frontend.prepare(std::move(input));

    auto session = frontend.make_output_session(prompt, {});
    const std::vector<ninferno::TokenId> tokens =
        tokenizer.encode("<|channel>thought\nweigh it<channel|>answer");
    int failures =
        check(tokens.front() == 102 && std::count(tokens.begin(), tokens.end(), 103) == 1,
              "the fixture did not encode the channel markers as special tokens");
    const auto decision =
        session.preview_model(tokens, static_cast<std::uint32_t>(tokens.size() + 1U),
                              ninferno::FinishReason::OutputLimit);
    failures += check(decision.accepted_tokens == tokens.size(),
                      "channel reasoning output was not accepted whole");
    const auto output = session.commit_preview();
    failures += check(channel_text(output, ninferno::OutputChannel::Reasoning) == "weigh it",
                      "a special-token reasoning open marker did not open the reasoning channel");
    failures += check(channel_text(output, ninferno::OutputChannel::Content) == "answer",
                      "special-token reasoning markers leaked into the content channel");
    return failures;
}

// A harmony template ends every message with <|end|> and the output with <|return|> or
// <|call|>: neither a template that closes an assistant turn with <|end|> (unsloth's
// gpt-oss-20b-GGUF) nor a GGUF's rebuilt configuration naming <|end|> its eos stops the output
// after the analysis channel.
int test_harmony_end_not_a_stop() {
    FrontendResources source =
        resources("{% for m in messages %}<|start|>{{ m.role }}<|channel|>final<|message|>"
                  "{{ m.content }}<|end|>{% endfor %}<|start|>assistant");
    nlohmann::json tokenizer_json = nlohmann::json::parse(source.tokenizer_json);
    nlohmann::json config         = nlohmann::json::parse(source.tokenizer_config_json);
    const std::array<std::pair<int, const char*>, 3> specials{
        {{100, "<|end|>"}, {101, "<|return|>"}, {102, "<|call|>"}}};
    for (const auto& [id, content] : specials) {
        tokenizer_json["added_tokens"].push_back(added(id, content, true));
        config["added_tokens_decoder"][std::to_string(id)] = decoder_added(content, true);
    }
    config["eos_token"]           = "<|end|>";
    source.tokenizer_json         = tokenizer_json.dump();
    source.tokenizer_config_json  = config.dump();
    source.generation_config_json = R"({"eos_token_id":[101,102]})";
    const auto frontend           = make_frontend(source, false);
    const auto& stops             = frontend.default_stop_policy().token_ids;
    const auto has                = [&](int id) {
        return std::find(stops.begin(), stops.end(), id) != stops.end();
    };
    return check(!has(100) && has(101) && has(102),
                 "a harmony <|end|> named the eos stopped the output after its first message");
}

// A Gemma 4 style template: thinking on writes <|think|> into the first turn and leaves the model
// to open its own <|channel>thought; the template defaults enable_thinking to false.
Frontend gemma4_thinking_frontend() {
    const std::string gemma_template =
        "{%- set enable_thinking = enable_thinking | default(false) -%}"
        "{% for m in messages %}<|turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
        "{% if enable_thinking and loop.first %}<|think|>{% endif %}"
        "{% if m.reasoning_content %}<|channel>thought\n{{ m.reasoning_content }}<channel|>{% "
        "endif %}"
        "{{ m.content }}<turn|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}";
    FrontendResources source      = resources(gemma_template);
    nlohmann::json tokenizer_json = nlohmann::json::parse(source.tokenizer_json);
    nlohmann::json config         = nlohmann::json::parse(source.tokenizer_config_json);
    const std::array<std::pair<int, const char*>, 5> specials{{{100, "<|turn>"},
                                                               {101, "<turn|>"},
                                                               {102, "<|channel>"},
                                                               {103, "<channel|>"},
                                                               {104, "<|think|>"}}};
    for (const auto& [id, content] : specials) {
        tokenizer_json["added_tokens"].push_back(added(id, content, true));
        config["added_tokens_decoder"][std::to_string(id)] = decoder_added(content, true);
    }
    source.tokenizer_json        = tokenizer_json.dump();
    source.tokenizer_config_json = config.dump();
    return make_frontend(source, false);
}

ninferno::PromptInput gemma4_prompt(std::optional<bool> thinking) {
    ninferno::PromptInput input;
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back({.text = "hi"});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = thinking;
    return input;
}

// Gemma 4's own template defaults enable_thinking to false and then writes no thought channel
// into the prompt, so the model may still open one itself. A request that turned reasoning off
// (or left the template's default off) must not let the constraint gate admit it: gemma-4-E2B-it
// under a json_schema opened <|channel>thought and reasoned until max_tokens.
int test_reasoning_declined_by_thinking_switch() {
    const auto frontend = gemma4_thinking_frontend();
    const auto summary  = [&](std::optional<bool> thinking) {
        return frontend.prepare(gemma4_prompt(thinking)).summary();
    };
    int failures = check(summary(true).reasoning_open_allowed,
                         "thinking on: the model may not open its reasoning");
    failures += check(!summary(false).reasoning_open_allowed,
                      "thinking off: the constraint gate may still open reasoning");
    failures += check(!summary(std::nullopt).reasoning_open_allowed,
                      "thinking unset (template default off): the gate may still open reasoning");
    failures +=
        check(summary(true).thinking() && !summary(false).thinking(),
              "a request leaving the model to open its reasoning is not a thinking request");
    return failures;
}

// A thinking budget where the model opens its reasoning itself (Gemma 4) counts the thinking
// after the opener, licenses no round past the budget before it, and closes with the control.
int test_thinking_budget_after_model_opener() {
    const auto frontend = gemma4_thinking_frontend();
    const auto prompt   = frontend.prepare(gemma4_prompt(true));
    auto session =
        frontend.make_output_session(prompt, {}, {}, ninferno::ThinkingControlOptions{.budget = 2});
    int failures = check(session.model_token_budget_remaining(20) == 2,
                         "a round before the model's opener was licensed past the budget");
    std::vector<ninferno::TokenId> opener{102};
    for (const char byte : std::string_view("thought")) {
        opener.push_back(fixture_byte_token(static_cast<std::uint8_t>(byte)));
    }
    const auto opened = session.preview_model(opener, 20, ninferno::FinishReason::OutputLimit);
    failures += check(!opened.finished() &&
                          opened.continuation == ninferno::runtime::ContinuationAction::Decode,
                      "the opener's own tokens counted against the thinking budget");
    (void)session.commit_preview();
    const auto remaining = static_cast<std::uint32_t>(20 - opener.size());
    failures += check(session.model_token_budget_remaining(remaining) == 2,
                      "the budget after the model's opener is not the whole budget");
    const std::array<ninferno::TokenId, 2> thinking{fixture_byte_token('\n'), 0};
    const auto boundary =
        session.preview_model(thinking, remaining, ninferno::FinishReason::OutputLimit);
    failures +=
        check(boundary.continuation == ninferno::runtime::ContinuationAction::ApplyTargetControl,
              "the thinking budget after the model's opener did not request control");
    const auto reasoning = session.commit_preview();
    failures += check(channel_text(reasoning, ninferno::OutputChannel::Reasoning) == "x",
                      "the model's thinking after its opener was not published as reasoning");
    const std::span<const ninferno::TokenId> pending = session.pending_control_tokens();
    const std::vector<ninferno::TokenId> control(pending.begin(), pending.end());
    failures += check(control == std::vector<ninferno::TokenId>{103},
                      "the Gemma 4 thinking control is not its channel close");
    (void)session.preview_control(control, remaining - 2U);
    (void)session.commit_preview();
    const ninferno::ThinkingBudgetStats stats = session.thinking_stats();
    failures +=
        check(stats.configured_budget == 2 && stats.model_thinking_tokens == 2 && stats.applied,
              "thinking accounting after the model's opener is incorrect");

    // An answer that begins without the opener is not thinking: no budget limits its rounds.
    auto answer =
        frontend.make_output_session(prompt, {}, {}, ninferno::ThinkingControlOptions{.budget = 2});
    (void)answer.preview_model(std::array<ninferno::TokenId, 1>{0}, 20,
                               ninferno::FinishReason::OutputLimit);
    (void)answer.commit_preview();
    failures += check(answer.model_token_budget_remaining(19) == 19 &&
                          answer.thinking_stats().model_thinking_tokens == 0,
                      "an answer without reasoning stayed under the thinking budget");
    return failures;
}

// Gemma 4 stops a tool call at the tool's turn (<|tool_response>) beside its turn end, whatever
// generation_config.json names: a GGUF's rebuilt configuration may carry <eos> alone.
int test_gemma4_tool_response_stops() {
    const std::string gemma_template =
        "{% for m in messages %}<|turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
        "{{ m.content }}<turn|>\n{% endfor %}"
        "{% if tools %}<|tool_call>call:{}<tool_call|><|tool_response><tool_response|>{% endif %}"
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}";
    FrontendResources source      = resources(gemma_template);
    nlohmann::json tokenizer_json = nlohmann::json::parse(source.tokenizer_json);
    nlohmann::json config         = nlohmann::json::parse(source.tokenizer_config_json);
    const std::array<std::pair<int, const char*>, 6> specials{
        {{100, "<|turn>"},
         {101, "<turn|>"},
         {104, "<|tool_call>"},
         {105, "<tool_call|>"},
         {106, "<|tool_response>"},
         {107, "<tool_response|>"}}};
    for (const auto& [id, content] : specials) {
        tokenizer_json["added_tokens"].push_back(added(id, content, true));
        config["added_tokens_decoder"][std::to_string(id)] = decoder_added(content, true);
    }
    source.tokenizer_json         = tokenizer_json.dump();
    source.tokenizer_config_json  = config.dump();
    source.generation_config_json = R"({"eos_token_id":6})";
    const auto frontend           = make_frontend(source, false);
    const auto& stops             = frontend.default_stop_policy().token_ids;
    const auto has                = [&](int id) {
        return std::find(stops.begin(), stops.end(), id) != stops.end();
    };
    int failures = check(has(6) && has(101) && has(106) && stops.size() == 3,
                         "Gemma 4's stops are not <eos>, the turn end and the tool's turn");
    // A configuration rebuilt from a GGUF that names <turn|> its eos (unsloth's E4B) still
    // stops on the tokenizer's <eos>.
    config["eos_token"]           = "<eos>";
    source.tokenizer_config_json  = config.dump();
    source.generation_config_json = R"({"eos_token_id":101})";
    const auto rebuilt            = make_frontend(source, false);
    const auto& rebuilt_stops     = rebuilt.default_stop_policy().token_ids;
    failures +=
        check(std::find(rebuilt_stops.begin(), rebuilt_stops.end(), 6) != rebuilt_stops.end() &&
                  rebuilt_stops.size() == 3,
              "the tokenizer's <eos> is not a stop when generation_config.json omits it");
    // A tokenizer rebuilt from a GGUF before <|tool_response> was a rebuilt stop keeps it a
    // user-defined (not special) token, and its configuration names <eos> alone.
    for (auto& entry : tokenizer_json["added_tokens"]) {
        if (entry["content"] == "<|tool_response>") { entry["special"] = false; }
    }
    config["added_tokens_decoder"]["106"] = decoder_added("<|tool_response>", false);
    config.erase("eos_token");
    source.tokenizer_json         = tokenizer_json.dump();
    source.tokenizer_config_json  = config.dump();
    source.generation_config_json = R"({"eos_token_id":6})";
    const auto plain              = make_frontend(source, false);
    const auto& plain_stops       = plain.default_stop_policy().token_ids;
    failures += check(std::find(plain_stops.begin(), plain_stops.end(), 106) != plain_stops.end(),
                      "a user-defined <|tool_response> token is not a stop");
    return failures;
}

ninferno::models::qwen3_5::PreparedPrompt thinking_prompt(const Frontend& frontend) {
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.continuation    = ninferno::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = true;
    return frontend.prepare(std::move(input));
}

int test_thinking_budget_control(const Frontend& frontend) {
    auto prompt = thinking_prompt(frontend);
    ninferno::StopPolicy stop;
    stop.strings.push_back(
        ninferno::StopString{.text = "limited", .channel = ninferno::OutputChannel::Reasoning});
    auto session = frontend.make_output_session(prompt, stop, {},
                                                ninferno::ThinkingControlOptions{.budget = 2});
    int failures = check(session.model_token_budget_remaining(20) == 2,
                         "thinking budget did not clamp the model round license");
    // An output limit at or below the budget ends thinking first, so the budget takes no effect
    // and needs no room for control.
    failures += check(!throws_invalid_argument([&] {
        session.validate_generation_capacity(1);
        session.validate_generation_capacity(2);
    }),
                      "planning rejected an output limit that the thinking budget never reaches");

    const std::array<ninferno::TokenId, 2> model_tokens{0, 0};
    const auto boundary =
        session.preview_model(model_tokens, 20, ninferno::FinishReason::OutputLimit);
    failures += check(boundary.accepted_tokens == model_tokens.size() && !boundary.finished() &&
                          boundary.continuation ==
                              ninferno::runtime::ContinuationAction::ApplyTargetControl,
                      "thinking boundary did not request target control");
    const auto model_output = session.commit_preview();
    failures += check(channel_text(model_output, ninferno::OutputChannel::Reasoning) == "xx",
                      "model-origin thinking output was not published before control");

    const std::span<const ninferno::TokenId> pending = session.pending_control_tokens();
    failures += check(pending.size() > 1,
                      "thinking boundary did not expose a multi-token canonical control span");
    const std::vector<ninferno::TokenId> control(pending.begin(), pending.end());
    failures += check(
        throws_invalid_argument([&] {
            session.validate_generation_capacity(static_cast<std::uint32_t>(2 + control.size()));
        }),
        "planning accepted a capacity that cannot fit control plus a post-close model token");
    session.validate_generation_capacity(static_cast<std::uint32_t>(3 + control.size()));

    const auto control_decision = session.preview_control(control, 18);
    failures +=
        check(control_decision.accepted_tokens == control.size() && !control_decision.finished() &&
                  control_decision.prefix_execution_split_after == control.size(),
              "canonical thinking control was not accepted atomically");
    const auto control_output = session.commit_preview();
    failures += check(channel_text(control_output, ninferno::OutputChannel::Reasoning) ==
                              kThinkingControlGuidance &&
                          channel_text(control_output, ninferno::OutputChannel::Content).empty(),
                      "thinking control was truncated by caller stops or published to content");
    const ninferno::ThinkingBudgetStats stats = session.thinking_stats();
    failures += check(stats.configured_budget == 2 && stats.model_thinking_tokens == 2 &&
                          stats.injected_tokens == control.size() && stats.applied &&
                          session.pending_control_tokens().empty() &&
                          session.model_token_budget_remaining(17) == 17,
                      "thinking control accounting or post-close license is incorrect");

    const auto content_decision = session.preview_model(std::array<ninferno::TokenId, 1>{0}, 17,
                                                        ninferno::FinishReason::OutputLimit);
    failures +=
        check(!content_decision.finished(), "post-control model token unexpectedly terminated");
    const auto content_output = session.commit_preview();
    failures += check(channel_text(content_output, ninferno::OutputChannel::Content) == "x",
                      "post-control model output did not enter the content channel");

    auto natural_prompt  = thinking_prompt(frontend);
    auto natural_session = frontend.make_output_session(
        natural_prompt, {}, {}, ninferno::ThinkingControlOptions{.budget = 2});
    const auto natural = natural_session.preview_model(std::array<ninferno::TokenId, 2>{3, 4}, 20,
                                                       ninferno::FinishReason::OutputLimit);
    failures += check(!natural.finished() &&
                          natural.continuation == ninferno::runtime::ContinuationAction::Decode,
                      "natural thinking close at the budget boundary requested control");
    (void)natural_session.commit_preview();
    failures += check(natural_session.pending_control_tokens().empty() &&
                          !natural_session.thinking_stats().applied,
                      "natural thinking close left control armed");

    auto terminal_prompt  = thinking_prompt(frontend);
    auto terminal_session = frontend.make_output_session(
        terminal_prompt, {}, {}, ninferno::ThinkingControlOptions{.budget = 1});
    const auto terminal = terminal_session.preview_model(std::array<ninferno::TokenId, 1>{6}, 10,
                                                         ninferno::FinishReason::OutputLimit);
    failures += check(terminal.finish_reason == ninferno::FinishReason::StopToken &&
                          terminal.continuation == ninferno::runtime::ContinuationAction::Decode,
                      "terminal token at the thinking boundary did not take priority");
    (void)terminal_session.commit_preview();
    failures += check(terminal_session.pending_control_tokens().empty(),
                      "terminal thinking boundary left control pending");

    auto limit_prompt  = thinking_prompt(frontend);
    auto limit_session = frontend.make_output_session(
        limit_prompt, {}, {}, ninferno::ThinkingControlOptions{.budget = 1});
    const auto limited = limit_session.preview_model(std::array<ninferno::TokenId, 1>{0}, 1,
                                                     ninferno::FinishReason::ContextCapacity);
    failures += check(limited.finish_reason == ninferno::FinishReason::ContextCapacity &&
                          limited.continuation == ninferno::runtime::ContinuationAction::Decode,
                      "total capacity did not take priority at the thinking boundary");
    (void)limit_session.commit_preview();

    auto raw_prompt = thinking_prompt(frontend);
    auto raw_session =
        frontend.make_output_session(raw_prompt, {}, ninferno::OutputOptions{.raw = true},
                                     ninferno::ThinkingControlOptions{.budget = 1});
    const auto raw_boundary = raw_session.preview_model(std::array<ninferno::TokenId, 1>{0}, 10,
                                                        ninferno::FinishReason::OutputLimit);
    failures += check(raw_boundary.continuation ==
                          ninferno::runtime::ContinuationAction::ApplyTargetControl,
                      "raw presentation disabled semantic thinking control");
    (void)raw_session.commit_preview();
    const std::vector<ninferno::TokenId> raw_tokens(raw_session.pending_control_tokens().begin(),
                                                  raw_session.pending_control_tokens().end());
    (void)raw_session.preview_control(raw_tokens, 9);
    const auto raw_output = raw_session.commit_preview();
    failures +=
        check(channel_text(raw_output, ninferno::OutputChannel::Content) == kThinkingControl,
              "raw output did not preserve the inserted control representation");
    return failures;
}

int test_utf8_and_hidden_eos(const Frontend& frontend) {
    auto prompt             = frontend.prepare_tokens({0});
    auto session            = frontend.make_output_session(prompt, {});
    int failures            = 0;
    std::uint32_t remaining = 4;
    for (const ninferno::TokenId token : {10, 11}) {
        const auto decision = session.preview_model(std::array<ninferno::TokenId, 1>{token},
                                                    remaining, ninferno::FinishReason::OutputLimit);
        failures += check(decision.accepted_tokens == 1 && !decision.finished(),
                          "partial UTF-8 token unexpectedly ended generation");
        const auto output = session.commit_preview();
        remaining -= decision.accepted_tokens;
        failures += check(output.empty(), "partial UTF-8 codepoint was published");
    }
    const auto complete_decision = session.preview_model(
        std::array<ninferno::TokenId, 1>{12}, remaining, ninferno::FinishReason::OutputLimit);
    failures += check(complete_decision.accepted_tokens == 1 && !complete_decision.finished(),
                      "complete UTF-8 token unexpectedly ended generation");
    const auto complete = session.commit_preview();
    failures += check(channel_text(complete, ninferno::OutputChannel::Content) == "中",
                      "UTF-8 codepoint was not published when complete");

    const auto decode_generated = [&](const std::vector<ninferno::TokenId>& tokens,
                                      bool one_token_per_round) {
        auto generated_prompt  = frontend.prepare_tokens({0});
        auto generated_session = frontend.make_output_session(generated_prompt, {});
        std::string text;
        std::uint32_t budget = static_cast<std::uint32_t>(tokens.size());
        if (one_token_per_round) {
            for (const ninferno::TokenId token : tokens) {
                const auto decision =
                    generated_session.preview_model(std::array<ninferno::TokenId, 1>{token}, budget,
                                                    ninferno::FinishReason::OutputLimit);
                budget -= decision.accepted_tokens;
                text += channel_text(generated_session.commit_preview(),
                                     ninferno::OutputChannel::Content);
            }
        } else {
            (void)generated_session.preview_model(tokens, budget,
                                                  ninferno::FinishReason::OutputLimit);
            text =
                channel_text(generated_session.commit_preview(), ninferno::OutputChannel::Content);
        }
        return text;
    };

    struct Utf8Case {
        std::vector<ninferno::TokenId> tokens;
        std::string expected;
        const char* label;
    };

    const std::string replacement(kUtf8Replacement);
    const std::vector<Utf8Case> utf8_cases = {
        {{10, 1}, replacement + "helloST", "invalid continuation after leading byte"},
        {{10, 11, 1}, replacement + "helloST", "maximal incomplete subpart"},
        {{11, 1}, replacement + "helloST", "isolated continuation byte"},
        {{10, 11}, replacement, "terminal incomplete suffix"},
        {{kByteE0Token, kByte80Token, kByte80Token},
         replacement + replacement + replacement,
         "overlong codepoint"},
        {{kByteEDToken, kByteA0Token, kByte80Token},
         replacement + replacement + replacement,
         "surrogate codepoint"},
        {{kByteF4Token, kByte90Token, kByte80Token, kByte80Token},
         replacement + replacement + replacement + replacement,
         "out-of-range codepoint"},
        {{kByteF5Token, 1}, replacement + "helloST", "invalid leading byte"},
        {{kByteC2Token, kByteA2Token}, "¢", "valid two-byte codepoint"},
        {{kByteF0Token, kByte9FToken, kByte98Token, kByte80Token},
         "😀",
         "valid four-byte codepoint"},
    };
    for (const Utf8Case& test : utf8_cases) {
        const std::string batched = decode_generated(test.tokens, false);
        const std::string split   = decode_generated(test.tokens, true);
        failures += check(batched == test.expected, test.label);
        failures += check(split == test.expected, test.label);
        failures += check(split == batched,
                          "generated UTF-8 recovery changed across decode-round boundaries");
    }

    auto repaired_stop_prompt = frontend.prepare_tokens({0});
    ninferno::StopPolicy repaired_stop;
    repaired_stop.strings.push_back(ninferno::StopString{.text = "STOP"});
    auto repaired_stop_session = frontend.make_output_session(repaired_stop_prompt, repaired_stop);
    const auto repaired_stop_decision = repaired_stop_session.preview_model(
        std::array<ninferno::TokenId, 3>{10, 1, 2}, 3, ninferno::FinishReason::OutputLimit);
    failures += check(repaired_stop_decision.finish_reason == ninferno::FinishReason::StopString,
                      "UTF-8 recovery hid a following stop string");
    const auto repaired_stop_output = repaired_stop_session.commit_preview();
    failures += check(channel_text(repaired_stop_output, ninferno::OutputChannel::Content) ==
                          replacement + "hello",
                      "UTF-8 recovery changed stop-string publication");

    auto repaired_reasoning_prompt  = thinking_prompt(frontend);
    auto repaired_reasoning_session = frontend.make_output_session(repaired_reasoning_prompt, {});
    const auto repaired_reasoning_decision = repaired_reasoning_session.preview_model(
        std::array<ninferno::TokenId, 3>{10, 3, 4}, 3, ninferno::FinishReason::OutputLimit);
    failures +=
        check(repaired_reasoning_decision.finish_reason == ninferno::FinishReason::OutputLimit,
              "UTF-8 recovery changed reasoning termination");
    const auto repaired_reasoning_output = repaired_reasoning_session.commit_preview();
    failures += check(
        channel_text(repaired_reasoning_output, ninferno::OutputChannel::Reasoning) ==
                replacement + "thought" &&
            channel_text(repaired_reasoning_output, ninferno::OutputChannel::Content) == "answer",
        "UTF-8 recovery changed reasoning/content channel routing");

    auto eos_prompt         = frontend.prepare_tokens({0});
    auto eos_session        = frontend.make_output_session(eos_prompt, {});
    const auto eos_decision = eos_session.preview_model(std::array<ninferno::TokenId, 1>{6}, 2,
                                                        ninferno::FinishReason::OutputLimit);
    failures += check(eos_decision.accepted_tokens == 1 &&
                          eos_decision.finish_reason == ninferno::FinishReason::StopToken,
                      "default EOS token did not end generation");
    const auto eos = eos_session.commit_preview();
    failures += check(eos.empty(), "default EOS token was published");

    auto raw_prompt  = frontend.prepare_tokens({0});
    auto raw_session = frontend.make_output_session(
        raw_prompt, {}, ninferno::OutputOptions{.raw = true, .preserve_special_tokens = false});
    const auto raw_eos_decision = raw_session.preview_model(std::array<ninferno::TokenId, 1>{6}, 2,
                                                            ninferno::FinishReason::OutputLimit);
    failures += check(raw_eos_decision.accepted_tokens == 1 &&
                          raw_eos_decision.finish_reason == ninferno::FinishReason::StopToken,
                      "raw EOS token did not end generation");
    const auto raw_eos = raw_session.commit_preview();
    failures += check(channel_text(raw_eos, ninferno::OutputChannel::Content) == "<eos>",
                      "raw output did not preserve the terminal special token");
    return failures;
}

// A completed assistant turn with neither text, media nor calls is not rendered: the
// model would imitate it. Text turns and the final message stay; message-indexed cache markers
// follow the messages they come after.
int test_empty_assistant_turns_are_not_rendered() {
    const Frontend frontend = make_frontend(resources(), false);
    const auto message      = [](ninferno::ChatRole role, std::string text, std::string reasoning) {
        ninferno::ChatMessage result;
        result.role = role;
        if (!text.empty() || role != ninferno::ChatRole::Assistant) {
            result.parts.push_back(ninferno::MessagePart{
                .kind = ninferno::MessagePartKind::Text, .text = std::move(text), .media = {}});
        }
        result.reasoning_content = std::move(reasoning);
        return result;
    };
    const auto tokens = [&](std::vector<ninferno::ChatMessage> messages,
                            std::uint32_t marker_after) {
        ninferno::PromptInput input;
        input.messages = std::move(messages);
        input.context_cache.markers.push_back(
            ninferno::PromptCacheMarker{.after_message_count = marker_after});
        const auto prepared = frontend.prepare(std::move(input));
        const auto& data    = FrontendFactory::inspect(prepared);
        std::vector<std::size_t> frontiers;
        for (const auto& opportunity : data.context_cache.opportunities) {
            frontiers.push_back(opportunity.frontier);
        }
        return std::pair{data.token_ids, frontiers};
    };
    using Role                 = ninferno::ChatRole;
    const auto user            = message(Role::User, "haz un ls", "");
    const auto with_empty      = tokens({user, message(Role::Assistant, "", "Let me run ls."), user,
                                         message(Role::Assistant, " \n", ""), user},
                                        3);
    const auto without         = tokens({user, user, user}, 2);
    const auto with_text       = tokens({user, message(Role::Assistant, "done", ""), user}, 1);
    const auto final_kept      = tokens({user, message(Role::Assistant, "", "thinking")}, 1);
    const auto final_only_user = tokens({user}, 1);
    return check(with_empty == without && with_text.first != tokens({user, user}, 1).first &&
                     final_kept.first != final_only_user.first,
                 "an empty assistant turn was rendered, a text turn or the final message was "
                 "dropped, or a cache marker did not follow its message");
}

int test_disabled_vision() {
    auto source = resources();
    source.preprocessor_config_json.clear();
    source.video_preprocessor_config_json.clear();
    const Frontend frontend = make_frontend(source, false);
    int failures = check(throws_invalid_argument([&] { (void)frontend.prepare(image_input()); }),
                         "Vision-disabled frontend accepted media during prepare");
    failures += check(throws_invalid_argument([&] { (void)frontend.count_tokens(image_input()); }),
                      "Vision-disabled frontend accepted media during token counting");

    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    message.parts.push_back(
        ninferno::MessagePart{.kind = ninferno::MessagePartKind::Text, .text = "x", .media = {}});
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    failures += check(frontend.prepare(input).summary().prompt_tokens != 0,
                      "Vision-disabled frontend rejected a text prompt");

    // A text-only artifact (no Vision component) serves text and refuses media by that reason.
    ninferno::models::qwen3_5::FrontendOptions text_only;
    text_only.vision_enabled   = false;
    text_only.vision_component = false;
    text_only.max_context      = std::numeric_limits<std::uint32_t>::max();
    const Frontend text_frontend = make_frontend(source, text_only);
    std::string refusal;
    try {
        (void)text_frontend.prepare(image_input());
    } catch (const std::invalid_argument& error) { refusal = error.what(); }
    failures += check(refusal.find("no vision component") != std::string::npos,
                      "a text-only artifact refused media without naming the missing component");
    failures += check(text_frontend.prepare(std::move(input)).summary().prompt_tokens != 0,
                      "a text-only frontend rejected a text prompt");
    return failures;
}

int test_invalid_media_classification() {
    const Frontend frontend = make_frontend(resources());
    auto invalid_image      = [] {
        ninferno::PromptInput input                    = image_input();
        input.messages[0].parts[0].media.bytes       = {0x00, 0x01, 0x02};
        input.messages[0].parts[0].media.source_name = "invalid-image.bin";
        return input;
    };
    auto is_invalid_media = [](const auto& operation) {
        try {
            operation();
        } catch (const ninferno::RequestError& error) {
            return error.kind() == ninferno::RequestErrorKind::InvalidMedia;
        }
        return false;
    };
    int failures = check(is_invalid_media([&] { (void)frontend.prepare(invalid_image()); }),
                         "invalid image preparation lost its typed request error");
    failures += check(is_invalid_media([&] { (void)frontend.count_tokens(invalid_image()); }),
                      "invalid image token counting lost its typed request error");
    return failures;
}

int test_media_cache_reuses_immutable_payload() {
    const Frontend frontend = make_frontend(resources());
    auto first              = frontend.prepare(image_input());
    auto second             = frontend.prepare(image_input());
    const auto& first_data  = FrontendFactory::inspect(first);
    const auto& second_data = FrontendFactory::inspect(second);
    int failures            = check(
        first_data.prepare.media_cache_misses == 1 && first_data.prepare.media_cache_hits == 0 &&
            first_data.prepare.built_patch_bytes == 16 * 1536 * sizeof(std::uint16_t),
        "first media preparation did not publish one cache miss");
    failures += check(
        second_data.prepare.media_cache_hits == 1 && second_data.prepare.media_cache_misses == 0 &&
            second_data.prepare.built_patch_bytes == 0 &&
            second_data.prepare.reused_patch_bytes == 16 * 1536 * sizeof(std::uint16_t),
        "second media preparation did not use the prepared-media cache");
    failures +=
        check(first_data.media_payloads.size() == 1 && second_data.media_payloads.size() == 1 &&
                  first_data.media_payloads.front() == second_data.media_payloads.front(),
              "cache hit did not share the immutable per-item patch payload");
    const ninferno::MediaCacheSummary cache = frontend.media_cache_summary();
    failures += check(cache.entries == 1 && cache.misses == 1 && cache.hits == 1 &&
                          cache.retained_bytes == 16 * 1536 * sizeof(std::uint16_t) &&
                          cache.live_bytes == cache.retained_bytes &&
                          cache.preprocess_threads >= 1 && cache.preprocess_threads <= 16,
                      "Frontend media-cache accounting does not describe the retained payload");
    return failures;
}

int test_media_payload_outlives_frontend_cache() {
    ninferno::models::qwen3_5::PreparedPrompt survivor;
    {
        const Frontend frontend = make_frontend(resources());
        survivor                = frontend.prepare(image_input());
    }
    const auto& data = FrontendFactory::inspect(survivor);
    return check(data.media_payloads.size() == 1 && data.media_payloads.front() &&
                     data.media_payloads.front()->patch_elements == 16 * 1536 &&
                     near(bf16_value(data.media_payloads.front()->span().front()), -1.0F),
                 "request-pinned media payload did not survive its Frontend cache owner");
}

int test_media_live_bytes_follow_last_payload_reference() {
    fi::MediaPreprocessCache cache(0, 1ULL << 20, 1);
    auto payload = cache.allocate_payload(1536, {});
    int failures = check(cache.stats().live_bytes == 1536 * sizeof(std::uint16_t),
                         "media live-byte account did not charge the allocated payload");
    payload.reset();
    failures += check(cache.stats().live_bytes == 0,
                      "media live-byte account did not release the final payload reference");
    return failures;
}

int test_media_cache_singleflight() {
    auto cache = std::make_shared<fi::MediaPreprocessCache>(1ULL << 20, 2ULL << 20);
    fi::MediaCacheKey key;
    key.digest.front() = 0x5a;

    std::promise<void> producer_started;
    std::future<void> producer_started_future = producer_started.get_future();
    std::promise<void> release_producer;
    std::shared_future<void> release_future = release_producer.get_future().share();
    std::atomic<int> builders{0};
    std::array<fi::PreparedMedia, 2> results;
    std::array<std::exception_ptr, 2> errors;
    std::array<fi::MediaCacheRequestStats, 2> request_stats;

    const auto builder = [&]() {
        const int count = ++builders;
        if (count == 1) { producer_started.set_value(); }
        release_future.wait();
        auto payload                    = cache->allocate_payload(1536, {});
        payload->mutable_span().front() = 42;
        fi::VisionItem item;
        item.grid = {1, 1, 1};
        return fi::PreparedMedia{std::move(item), std::move(payload)};
    };
    const auto run = [&](std::size_t index) {
        try {
            results[index] = cache->get_or_prepare(key, {}, builder, request_stats[index]);
        } catch (...) { errors[index] = std::current_exception(); }
    };

    std::thread first(run, 0);
    producer_started_future.wait();
    std::thread second(run, 1);
    const auto wait_limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (cache->stats().singleflight_waits == 0 &&
           std::chrono::steady_clock::now() < wait_limit) {
        std::this_thread::yield();
    }
    release_producer.set_value();
    first.join();
    second.join();

    return check(!errors[0] && !errors[1] && builders == 1 && results[0].payload &&
                     results[0].payload == results[1].payload &&
                     cache->stats().singleflight_waits == 1,
                 "concurrent identical media did not collapse into one preprocessing flight");
}

int test_media_cache_runs_independent_misses_in_parallel() {
    auto cache = std::make_shared<fi::MediaPreprocessCache>(1ULL << 20, 2ULL << 20, 4);
    std::array<fi::PendingMedia, 4> pending;
    std::atomic<int> started{0};
    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};

    for (std::size_t index = 0; index < pending.size(); ++index) {
        fi::MediaCacheKey key;
        key.digest.front() = static_cast<std::uint8_t>(index + 1);
        pending[index]     = cache->begin_prepare(key, {}, [&, index] {
            ++started;
            const int now = ++active;
            int maximum   = maximum_active.load();
            while (now > maximum && !maximum_active.compare_exchange_weak(maximum, now)) {}
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (started.load() != static_cast<int>(pending.size()) &&
                   std::chrono::steady_clock::now() < limit) {
                std::this_thread::yield();
            }
            auto payload                    = cache->allocate_payload(1536, {});
            payload->mutable_span().front() = static_cast<std::uint16_t>(index);
            --active;
            fi::VisionItem item;
            item.grid = {1, 1, 1};
            return fi::PreparedMedia{std::move(item), std::move(payload)};
        });
    }

    fi::MediaCacheRequestStats request_stats;
    std::array<fi::PreparedMedia, 4> results;
    for (std::size_t index = 0; index < pending.size(); ++index) {
        results[index] = cache->await(pending[index], {}, request_stats);
    }
    return check(started == 4 && maximum_active == 4 && request_stats.misses == 4 &&
                     cache->stats().preprocess_threads == 4 && results.back().payload &&
                     results.back().payload->span().front() == 3,
                 "independent media misses did not use the bounded preprocessing pool");
}

int test_many_images_prepare_in_one_parallel_batch() {
    const Frontend frontend = make_frontend(resources());
    ninferno::ChatMessage message;
    message.role = ninferno::ChatRole::User;
    for (int index = 0; index < 19; ++index) {
        ninferno::MessagePart image;
        image.kind               = ninferno::MessagePartKind::Media;
        image.media.kind         = ninferno::MediaKind::Image;
        image.media.bytes        = gradient_ppm();
        image.media.bytes.back() = static_cast<std::uint8_t>(index);
        image.media.media_type   = "image/x-portable-pixmap";
        image.media.source_name  = "parallel-" + std::to_string(index) + ".ppm";
        message.parts.push_back(std::move(image));
    }
    ninferno::PromptInput input;
    input.messages.push_back(std::move(message));
    const auto prepared = frontend.prepare(std::move(input));
    const auto& data    = FrontendFactory::inspect(prepared);
    return check(data.vision_items.size() == 19 && data.media_payloads.size() == 19 &&
                     data.prepare.media_cache_misses == 19 && data.prepare.raw_patches == 19 * 16 &&
                     frontend.media_cache_summary().preprocess_threads > 1,
                 "one request with 19 distinct images did not complete as a parallel media batch");
}

int test_media_preparation_cancellation() {
    const Frontend frontend = make_frontend(resources());
    ninferno::PreparationControl control{
        .deadline     = {},
        .cancellation = ninferno::CancellationView([] { return true; }),
    };
    try {
        (void)frontend.prepare(image_input(), control);
    } catch (const ninferno::RequestError& error) {
        return check(error.kind() == ninferno::RequestErrorKind::Cancelled &&
                         frontend.media_cache_summary().entries == 0,
                     "cancelled media preparation published a cache entry");
    }
    return check(false, "cancelled media preparation completed successfully");
}

} // namespace

int main() {
    const FrontendResources owned = resources();
    const Frontend frontend       = make_frontend(owned);
    int failures                  = 0;
    failures += test_generation_config_sampling();
    failures += test_declared_frontend_semantics();
    failures += test_tokenizer_config_merge();
    failures += test_transformers5_tokenizer_layout();
    failures += test_bpe_merge_order();
    failures += test_boundary_aware_tokenization();
    failures += test_rendered_special_tokens();
    failures += test_repeated_special_tokens_scan_linearly();
    failures += test_bounded_tokenizer_prefix();
    failures += test_overflow_check_only();
    failures += test_context_capacity_guard();
    failures += test_prompt_continuation();
    failures += test_developer_renders_as_system();
    failures += test_selected_template_instruction_prefix();
    failures += test_assistant_continuation();
    failures += test_rewrite_checkpoint_trace();
    failures += test_replacement_suffix_boundaries();
    failures += test_leading_instruction_run_boundaries();
    failures += test_adjacent_tool_message_boundary();
    failures += test_literal_cache_boundary();
    failures += test_selected_template_recovery_boundary();
    failures += test_official_resource_guards();
    failures += test_template_file_execution();
    failures += test_invalid_public_part_enums(frontend);
    failures += test_text_and_image_prepare(frontend);
    failures += test_media_token_ids_come_from_tokenizer();
    failures += test_template_media_contract();
    failures += test_image_resize_rejection_policy();
    failures += test_disabled_tool_calls_keep_the_prompt();
    failures += test_explicit_leading_instruction_cache_boundary();
    failures += test_last_part_boundary_at_turn_end();
    failures += test_media_admission_uses_aggregate_resources(frontend);
    failures += test_multimodal_prompt_over_removed_32k_cap(frontend);
    failures += test_attention_pairs_are_diagnostic(frontend);
    failures += test_video_prepare(frontend);
    failures += test_cross_round_stop(frontend);
    failures += test_same_token_stop_priority(frontend);
    failures += test_terminal_flush(frontend);
    failures += test_structured_tool_output();
    failures += test_tool_call_in_unclosed_reasoning();
    failures += test_tools_none_without_tools();
    failures += test_reasoning_split(frontend);
    failures += test_quoted_reasoning_close(frontend);
    failures += test_special_token_reasoning_markers();
    failures += test_reasoning_declined_by_thinking_switch();
    failures += test_thinking_budget_after_model_opener();
    failures += test_gemma4_tool_response_stops();
    failures += test_harmony_end_not_a_stop();
    failures += test_thinking_budget_control(frontend);
    failures += test_utf8_and_hidden_eos(frontend);
    failures += test_media_cache_reuses_immutable_payload();
    failures += test_media_payload_outlives_frontend_cache();
    failures += test_media_live_bytes_follow_last_payload_reference();
    failures += test_media_cache_singleflight();
    failures += test_media_cache_runs_independent_misses_in_parallel();
    failures += test_many_images_prepare_in_one_parallel_batch();
    failures += test_media_preparation_cancellation();
    failures += test_invalid_media_classification();
    failures += test_empty_assistant_turns_are_not_rendered();
    failures += test_disabled_vision();
    return failures == 0 ? 0 : 1;
}
