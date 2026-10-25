// Whether the engine runs a model, from what a repository shows before a download
// (convert/support.h): each rule on the facts that trigger it, and the facts of real hub
// repositories that the plan without weights accepted before (2026-10-25).

#include "convert/fetch.h"
#include "convert/support.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace cv = ninferno::convert;
using Json   = nlohmann::json;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

cv::Support of(const cv::SupportFacts& facts) { return cv::model_support(facts); }

cv::SupportFacts chat(const std::string& architecture) {
    cv::SupportFacts f;
    f.pipeline      = "text-generation";
    f.config        = Json{{"architectures", Json::array({architecture})}};
    f.chat_template = true;
    return f;
}

bool says(const cv::Support& s, cv::Verdict verdict, const std::string& part) {
    return s.verdict == verdict && s.reason.find(part) != std::string::npos;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: ninferno_convert_model_support_test SOURCE_DIR\n";
        return 2;
    }
    using cv::Verdict;
    // The task: autotrust/JEV-9B (a decision model) is tagged text-classification.
    {
        auto f     = chat("Qwen3_5ForCausalLM");
        f.pipeline = "text-classification";
        expect(says(of(f), Verdict::Unsupported, "not a text generation model"), "task");
        f.pipeline.clear();
        expect(of(f).verdict == Verdict::Supported, "an unsaid task is no reason to refuse");
    }
    // The weights' format: MLX by library or by packed U32 tensors, bitsandbytes by its method.
    {
        auto f    = chat("LlamaForCausalLM");
        f.library = "mlx";
        expect(says(of(f), Verdict::Unsupported, "MLX"), "MLX by library");
        f.library       = "transformers";
        f.weight_dtypes = {"U32", "BF16"}; // openbmb/MiniCPM5-2B-MLX
        expect(says(of(f), Verdict::Unsupported, "packed U32"), "MLX by its packing");
        f.config["quantization_config"] = {{"quant_method", "gptq"}, {"bits", 4}};
        expect(of(f).verdict == Verdict::Supported, "U32 under a GPTQ quantization_config");
        // openbmb/MiniCPM5-2B-MLX: MLX's own block, repeated as a quantization_config without a
        // method.
        auto mlx                           = chat("LlamaForCausalLM");
        mlx.config["quantization"]         = {{"group_size", 64}, {"bits", 4}, {"mode", "affine"}};
        mlx.config["quantization_config"]  = mlx.config["quantization"];
        expect(says(of(mlx), Verdict::Unsupported, "MLX"), "MLX by its quantization block");
        mlx.config.erase("quantization");
        mlx.weight_dtypes = {"U32", "BF16"};
        expect(says(of(mlx), Verdict::Unsupported, "packed U32"),
               "U32 under a quantization_config without a method");
        f.config["quantization_config"] = {{"quant_method", "bitsandbytes"}, {"load_in_4bit", true}};
        expect(says(of(f), Verdict::Unsupported, "bitsandbytes"), "bitsandbytes");
        // RadixArk/Muse-Glimmer-NVFP4: Model Optimizer's mixed NVFP4 and MXFP8 layers.
        f.config["quantization_config"] = {
            {"quant_algo", "MIXED_PRECISION"},
            {"quantized_layers",
             {{"model.layers.0.mlp.up_proj", {{"quant_algo", "NVFP4"}, {"group_size", 16}}},
              {"model.layers.0.mlp.down_proj", {{"quant_algo", "MXFP8"}}}}}};
        expect(says(of(f), Verdict::Unsupported, "MXFP8"), "Model Optimizer MXFP8 layers");
        f.config["quantization_config"]["quantized_layers"].erase("model.layers.0.mlp.down_proj");
        expect(of(f).verdict == Verdict::Supported, "Model Optimizer NVFP4 layers");
    }
    // The architecture: the converter's lists, a drafter's kind, a GGUF's general.architecture.
    {
        expect(says(of(chat("GPTNeoXForCausalLM")), Verdict::Unsupported, "GPTNeoXForCausalLM"),
               "another architecture");
        expect(says(of(chat("Qwen3VLForConditionalGeneration")), Verdict::Unsupported,
                    "Qwen3VLForConditionalGeneration"),
               "Qwen3-VL");
        const auto qwen = of(chat("Qwen3ForCausalLM"));
        expect(qwen.verdict == Verdict::Supported && qwen.role == "model" &&
                   qwen.reason.find("not checked") != std::string::npos,
               "a dense decoder from the hub's part of config.json, its fields not checked");
        expect(of(chat("Qwen3_5ForConditionalGeneration")).verdict == Verdict::Supported,
               "Qwen3.5");
        const auto dflash = of(chat("DFlashDraftModel"));
        expect(dflash.verdict == Verdict::Supported && dflash.role == "drafter", "a DFlash drafter");
        expect(cv::serve_refusal(dflash).find("beside its model") != std::string::npos &&
                   cv::serve_refusal(qwen).empty(),
               "a drafter is not served alone, a model is");
        expect(says(of(chat("Eagle3DraftModel")), Verdict::Unsupported, "no engine runs"),
               "an EAGLE-3 drafter");
        // nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark: a drafter, not the model.
        const auto dspark = of(chat("Qwen3DSparkModel"));
        expect(dspark.verdict == Verdict::Unsupported && dspark.role == "drafter" &&
                   dspark.reason.find("no engine runs") != std::string::npos,
               "a DSpark drafter: " + dspark.reason);
        cv::SupportFacts g;
        g.gguf              = true;
        g.gguf_architecture = "lfm2";
        expect(says(of(g), Verdict::Unsupported, "lfm2"), "a GGUF of another architecture");
        g.gguf_architecture = "qwen35";
        expect(of(g).verdict == Verdict::Supported, "a Qwen3.5 GGUF");
        g.gguf_architecture = "clip";
        expect(of(g).verdict == Verdict::Unsupported && of(g).role == "projector", "a projector");
        g.gguf_architecture.clear();
        expect(of(g).verdict == Verdict::Unknown, "a GGUF not read yet");
        g.gguf_error = "output.weight: unknown ggml type 142";
        expect(says(of(g), Verdict::Unsupported, "unknown ggml type"), "an unreadable GGUF");
    }
    // Qwen/Qwen3.6-35B-A3B: a Qwen3.5/3.6 MoE published unquantized has no recipe the engine
    // runs as published; a quantized release of it does, and so does the dense model.
    {
        auto moe          = chat("Qwen3_5MoeForConditionalGeneration");
        moe.config["text_config"] = {{"num_experts", 256}};
        moe.weight_dtypes = {"BF16"};
        expect(says(of(moe), Verdict::Unsupported, "unquantized Qwen3.5/3.6 MoE"),
               "an unquantized Qwen3.6 MoE");
        moe.weight_dtypes = {"BF16", "F8_E4M3", "U8"};
        expect(of(moe).verdict == Verdict::Supported, "an NVFP4 Qwen3.6 MoE");
        moe.weight_dtypes.clear();
        expect(of(moe).verdict == Verdict::Supported, "dtypes not known: not refused");
        auto dense          = chat("Qwen3_5ForConditionalGeneration");
        dense.weight_dtypes = {"BF16"};
        expect(of(dense).verdict == Verdict::Supported, "an unquantized dense Qwen3.5");
    }
    // A gated repository: a token is what is missing, unless the architecture already refuses.
    {
        auto f        = chat("LlamaForCausalLM");
        f.needs_token = true;
        expect(of(f).verdict == Verdict::NeedsToken, "a gated Llama");
        auto other        = chat("GPTNeoXForCausalLM");
        other.needs_token = true;
        expect(of(other).verdict == Verdict::Unsupported, "a gated model of another architecture");
    }
    // The chat template: a base model has none (PowCal/deepseek-math-7b-base-4bit).
    {
        auto f          = chat("LlamaForCausalLM");
        f.chat_template = false;
        expect(says(of(f), Verdict::Unsupported, "no chat template"), "a base model");
        f.chat_template.reset();
        expect(of(f).verdict == Verdict::Supported, "an unknown template is no reason to refuse");
    }
    // The whole config.json: the converter's field checks.
    {
        std::ifstream in(std::string(argv[1]) + "/tests/convert_native/fixtures/dense_rope.json");
        const auto config = Json::parse(in).at("llama32_1b").at("config");
        auto f            = chat("LlamaForCausalLM");
        f.config          = config;
        f.full_config     = true;
        const auto ok     = of(f);
        expect(ok.verdict == Verdict::Supported && ok.reason.empty(),
               "Llama 3.2 1B's whole config.json: " + ok.reason);
        f.config["mlp_bias"] = true;
        expect(says(of(f), Verdict::Unsupported, "mlp_bias"), "a field the converter refuses");
    }
    // A checkpoint folder on disk: its config.json and its template files.
    {
        std::ifstream in(std::string(argv[1]) + "/tests/convert_native/fixtures/dense_rope.json");
        const auto config = Json::parse(in).at("llama32_1b").at("config");
        const auto dir    = std::filesystem::temp_directory_path() / "ninferno_model_support_disk";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::ofstream(dir / "config.json") << config.dump();
        std::ofstream(dir / "tokenizer_config.json") << R"({"bos_token": "<s>"})";
        expect(says(of(cv::local_support_facts(dir)), Verdict::Unsupported, "no chat template"),
               "a folder without a template");
        std::ofstream(dir / "chat_template.jinja") << "{{ messages }}";
        const auto ok = of(cv::local_support_facts(dir));
        expect(ok.verdict == Verdict::Supported && ok.reason.empty(),
               "a folder with config.json and a template: " + ok.reason);
        std::filesystem::remove_all(dir);
    }
    // The token a gated model needs: the hf CLI's file, replaced by a new one; the environment
    // first.
    {
        const auto dir = std::filesystem::temp_directory_path() / "ninferno_model_support_token";
        std::filesystem::remove_all(dir);
        ::unsetenv("HF_TOKEN");
        ::unsetenv("HUGGING_FACE_HUB_TOKEN");
        ::setenv("HF_TOKEN_PATH", (dir / "token").c_str(), 1);
        expect(cv::hub_token().value.empty(), "no token anywhere");
        cv::save_hub_token("hf_first");
        cv::save_hub_token("hf_second");
        const auto saved = cv::hub_token();
        expect(saved.value == "hf_second" && saved.source == (dir / "token").string(),
               "a saved token replaces the one before: " + saved.value);
        ::setenv("HF_TOKEN", "hf_env", 1);
        expect(cv::hub_token().value == "hf_env" && cv::hub_token().source == "HF_TOKEN",
               "HF_TOKEN comes first");
        ::unsetenv("HF_TOKEN");
        cv::save_hub_token("");
        expect(cv::hub_token().value.empty() && !std::filesystem::exists(dir / "token"),
               "an empty token removes the file");
        ::unsetenv("HF_TOKEN_PATH");
        std::filesystem::remove_all(dir);
    }
    // The hub search's answer for one model: its facts.
    {
        const auto facts = cv::search_support_facts(Json::parse(R"({
            "id": "PowCal/deepseek-math-7b-base-4bit", "pipeline_tag": "text-generation",
            "library_name": "transformers",
            "config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                       "quantization_config": {"quant_method": "bitsandbytes"},
                       "tokenizer_config": {"bos_token": "<s>"}},
            "safetensors": {"parameters": {"F32": 1, "BF16": 2, "U8": 3}}})"));
        // The summary shows no template: unknown (the file itself may hold one), not "none".
        expect(!facts.chat_template && facts.weight_dtypes.size() == 3,
               "the search's facts: the template unknown, three dtypes");
        expect(says(of(facts), Verdict::Unsupported, "bitsandbytes"),
               "the search's facts refuse bitsandbytes first");
        const auto gguf = cv::search_support_facts(Json::parse(R"({
            "id": "unsloth/Qwen3.5-9B-GGUF", "pipeline_tag": "image-text-to-text",
            "gguf": {"architecture": "qwen35", "chat_template": "{{ x }}"}})"));
        expect(gguf.gguf && of(gguf).verdict == Verdict::Supported, "a GGUF from the search");
        // Safetensors beside a GGUF: the fetch takes them, so their config decides.
        const auto mixed = cv::search_support_facts(Json::parse(R"({
            "id": "org/mixed", "pipeline_tag": "text-generation",
            "config": {"architectures": ["Qwen3ForCausalLM"],
                       "tokenizer_config": {"chat_template": "{{ x }}"}},
            "gguf": {"architecture": "lfm2", "chat_template": "{{ x }}"},
            "safetensors": {"parameters": {"BF16": 1}}})"));
        expect(!mixed.gguf && of(mixed).verdict == Verdict::Supported,
               "a mixed search result is judged by its safetensors");
    }
    std::cout << (failures == 0 ? "model support: ok\n" : "model support: FAILED\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
