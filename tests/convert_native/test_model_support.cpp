// Whether the engine runs a model, from what a repository shows before a download
// (convert/support.h): each rule on the facts that trigger it, and the facts of real hub
// repositories that the plan without weights accepted before (2026-10-25).

#include "convert/fetch.h"
#include "convert/ggml_quants.h"
#include "convert/support.h"
#include "gguf_writer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

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
        // RadixArk/Muse-Glimmer-NVFP4: Model Optimizer's mixed NVFP4 and MXFP8 layers (its
        // tensors U8 and BF16, not the packed U32 above).
        f.weight_dtypes                 = {"U8", "BF16"};
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
        // cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF: 374 NVFP4 matrices beside Q5_K/Q6_K/Q8_0
        // ones. The official NVFP4 layout (NVFP4 beside unquantized tensors) and a K-quant run.
        g.gguf_nvfp4_matrices = 374;
        g.gguf_block_matrices = 122;
        expect(says(of(g), Verdict::Unsupported, "mixes NVFP4 matrices (374)"),
               "a GGUF mixing NVFP4 and GGML blocks");
        g.gguf_block_matrices = 0;
        expect(of(g).verdict == Verdict::Supported, "an NVFP4 GGUF beside unquantized tensors");
        g.gguf_nvfp4_matrices = 0;
        g.gguf_block_matrices = 400;
        expect(of(g).verdict == Verdict::Supported, "a K-quant GGUF");
        g.gguf_block_matrices = 0;
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
    // GGUF files on disk: a split model's parts count together (NVFP4 in one, Q8_0 in the other),
    // and a plain type (F64) is no block encoding beside NVFP4.
    {
        namespace gg = ninferno::convert::ggml;
        const auto dir = std::filesystem::temp_directory_path() / "ninferno_model_support_gguf";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        const auto blocks = [](std::uint32_t type, std::uint64_t elements) {
            const auto t = gg::traits(type);
            return std::vector<std::uint8_t>(elements / t->block * t->bytes, 0);
        };
        const auto write = [&](const std::string& name, bool head, std::uint32_t type,
                               std::uint64_t columns) {
            ninferno::convert::testing::GgufWriter w;
            if (head) {
                w.kv_str("general.architecture", "qwen35");
                w.kv_str("tokenizer.chat_template", "{{ messages }}");
            }
            w.tensor(name + ".weight", {columns, 2}, type, blocks(type, columns * 2));
            const auto bytes = w.finish();
            std::ofstream(dir / name, std::ios::binary)
                .write(reinterpret_cast<const char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()));
        };
        write("m-00001-of-00002.gguf", true, gg::NVFP4, 64);
        write("m-00002-of-00002.gguf", false, gg::Q8_0, 64);
        const auto split = cv::local_support_facts(dir / "m-00001-of-00002.gguf");
        expect(split.gguf_nvfp4_matrices == 1 && split.gguf_block_matrices == 1 &&
                   says(of(split), Verdict::Unsupported, "mixes NVFP4"),
               "a split GGUF mixing NVFP4 and Q8_0 across its parts");
        write("plain.gguf", true, gg::NVFP4, 64);
        auto plain = cv::local_support_facts(dir / "plain.gguf");
        expect(plain.gguf_block_matrices == 0 && of(plain).verdict == Verdict::Supported,
               "an NVFP4 GGUF alone");
        write("f64.gguf", false, gg::F64, 64);
        const auto f64 = cv::local_support_facts(dir / "f64.gguf");
        expect(f64.gguf_block_matrices == 0, "an F64 matrix is plain, no block encoding");
        std::filesystem::remove_all(dir);
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
    // --quant picks one GGUF model as the fetch does, for the plan without weights too
    // (convert::gguf_group_for_quant, convert::listed_quant_for): "Q4_K_M" takes UD-Q4_K_M where
    // that is the only one, an exact label wins, "4bit" the largest 4-bit model.
    {
        const std::map<std::string, std::uint64_t> groups = {
            {"m-UD-Q4_K_M", 10}, {"m-Q4_K_S", 9}, {"m-UD-Q4_K_XL", 11}, {"m-Q8_0", 20}};
        expect(cv::gguf_group_for_quant(groups, "Q4_K_M", "r") == "m-UD-Q4_K_M",
               "a label inside a longer one takes it");
        expect(cv::gguf_group_for_quant(groups, "4bit", "r") == "m-UD-Q4_K_XL",
               "a width takes the largest of that width");
        bool threw = false;
        try {
            (void)cv::gguf_group_for_quant(groups, "Q4_K", "r");
        } catch (const std::runtime_error&) { threw = true; }
        expect(threw, "a label several models hold, none exactly, names them");
        std::vector<cv::RepoQuant> listing(3);
        listing[0].quant  = "UD-Q4_K_M";
        listing[0].format = "gguf";
        listing[0].bytes  = 10;
        listing[0].files  = {"m-UD-Q4_K_M.gguf"};
        listing[1].quant  = "Q8_0";
        listing[1].format = "gguf";
        listing[1].bytes  = 20;
        listing[1].files  = {"Q8_0/m-Q8_0-00001-of-00002.gguf", "Q8_0/m-Q8_0-00002-of-00002.gguf"};
        listing[2].quant   = "mtp-m";
        listing[2].format  = "gguf";
        listing[2].drafter = true;
        listing[2].files   = {"mtp-m-Q4_K_M.gguf"};
        const auto q4 = cv::listed_quant_for(listing, "Q4_K_M", "r");
        const auto q8 = cv::listed_quant_for(listing, "q8_0", "r");
        expect(q4.quant == "UD-Q4_K_M" && q8.quant == "Q8_0" && q8.files.size() == 2,
               "the listing's quantization for --quant, a drafter beside them not one");
        // A whole file's group is its file name (gguf_group): a width reads its label from it.
        listing[0].files = {"m-UD-Q4_K_M.gguf"};
        expect(cv::listed_quant_for(listing, "4bit", "r").quant == "UD-Q4_K_M",
               "a width over whole files");
        // A split model without a label in its names, listed a part at a time: one entry of both
        // parts, named by its group, as the fetch matches --quant.
        std::vector<cv::RepoQuant> parts(2);
        for (int i = 0; i < 2; ++i) {
            parts[i].quant  = "model-0000" + std::to_string(i + 1) + "-of-00002";
            parts[i].format = "gguf";
            parts[i].bytes  = 7;
            parts[i].files  = {parts[i].quant + ".gguf"};
            parts[i].sha256 = {std::string(64, char('a' + i))};
        }
        const auto whole = cv::listed_quant_for(parts, "model", "r");
        expect(whole.quant == "model" && whole.bytes == 14 && whole.files.size() == 2 &&
                   whole.sha256.size() == 2,
               "an unlabelled split model is one entry of its parts");
        // Two models one label's entry holds (A-Q4_K_M, B-Q4_K_M): each addressable by its own
        // name, a width comparing each model's own bytes.
        std::vector<cv::RepoQuant> two(2);
        two[0].quant  = "Q4_K_M";
        two[0].format = "gguf";
        two[0].files  = {"A-Q4_K_M.gguf", "B-Q4_K_M.gguf"};
        two[0].sha256 = {std::string(64, 'a'), std::string(64, 'b')};
        two[0].sizes  = {5, 9};
        two[0].bytes  = 14;
        two[1].quant  = "Q4_0";
        two[1].format = "gguf";
        two[1].files  = {"A-Q4_0.gguf"};
        two[1].sizes  = {12};
        two[1].bytes  = 12;
        const auto b = cv::listed_quant_for(two, "B-Q4_K_M", "r");
        expect(b.quant == "B-Q4_K_M" && b.bits == 4 &&
                   b.files == std::vector<std::string>{"B-Q4_K_M.gguf"} && b.bytes == 9 &&
                   b.sha256 == std::vector<std::string>{std::string(64, 'b')},
               "a second model of one label's entry, by its name");
        expect(cv::listed_quant_for(two, "4bit", "r").files ==
                   std::vector<std::string>{"A-Q4_0.gguf"},
               "a width compares each model's own bytes (12 over B's 9, not the label's 14)");
        // One name in two folders: the entry keeps the folder, the fetch's second selection (its
        // header read) takes the same group again.
        std::vector<cv::RepoQuant> dirs(1);
        dirs[0].quant  = "Q4_K_M";
        dirs[0].format = "gguf";
        dirs[0].files  = {"A/model-Q4_K_M.gguf", "B/model-Q4_K_M.gguf"};
        dirs[0].sizes  = {5, 9};
        dirs[0].bytes  = 14;
        const auto wide = cv::listed_quant_for(dirs, "4bit", "r");
        const std::map<std::string, std::uint64_t> fetch_groups = {{"A/model-Q4_K_M.gguf", 5},
                                                                   {"B/model-Q4_K_M.gguf", 9}};
        expect(wide.quant == "B/model-Q4_K_M" &&
                   cv::gguf_group_for_quant(fetch_groups, wide.quant, "r") == "B/model-Q4_K_M.gguf",
               "a folder's model keeps its folder in the name the fetch selects again");
        const std::map<std::string, std::uint64_t> files = {{"m-UD-Q4_K_M.gguf", 10},
                                                             {"m-Q8_0.gguf", 20}};
        expect(cv::gguf_group_for_quant(files, "4bit", "r") == "m-UD-Q4_K_M.gguf",
               "a width over the fetch's whole-file groups");
    }
    std::cout << (failures == 0 ? "model support: ok\n" : "model support: FAILED\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
