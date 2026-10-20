// ninferno-convert: converts a checkpoint (Hugging Face safetensors or GGUF; Qwen 3.5, Gemma 4,
// dense and MoE decoders, Nemotron-H, drafters) into a .ninferno artifact, from a local
// directory or file or one fetched from a hub; plans, attaches and relays out artifacts.

#include "convert/acquire.h"
#include "convert/convert.h"
#include "convert/drafters.h"
#include "convert/fetch.h"
#include "convert/relayout.h"
#include "convert/safetensors.h"
#include "core/command_line.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr const char* kUsage =
    "usage: ninferno-convert --version\n"
    "       ninferno-convert (--model PATH | --fetch SPEC) [--recipe NAME] [--out PATH]\n"
    "                      [--name NAME] [--resource ROLE=PATH]... [--source NAME=PATH]...\n"
    "                      [--components text[,vision,mtp,dflash,dflash2]]\n"
    "                      [--proposal] [--proposal-rows N] [--ranking PATH]\n"
    "                      [--device DEV (ignored)] [--rows-per-chunk N] [--max-file-bytes N]\n"
    "                      [--threads N] [--keep-source-cache] [--quiet]\n"
    "                      [--consume-source [--consume-window BYTES]] [--assume-free-bytes N]\n"
    "                      [--maintainer-quantize] [--round-tiny-scales] [--vocab native|q8]\n"
    "                      [--drafter-quant q8|q4|nvfp4]\n"
    "                      [--layout auto|published|frag]\n"
    "  fetching:           [--hub huggingface|modelscope] [--revision REV] [--quant LABEL]\n"
    "                      [--include GLOB]... [--cache DIR] [--streams N] [--max-streams N]\n"
    "                      [--min-speed BYTES_PER_S] [--stall-seconds S] [--retries N]\n"
    "                      [--chunk-bytes N] [--no-verify]\n"
    "                      [--fetch-only] [--no-base-tokenizer]\n"
    "  drafters:           [--no-drafters] [--with-drafters] [--model-id REPO]\n"
    "                      [--drafter-artifact NAME=PATH]...\n"
    "       ninferno-convert --plan-drafters (--model DIR | --target-artifact PATH) [--model-id "
    "REPO]\n"
    "       ninferno-convert --attach-drafters --target-artifact PATH [--model-id REPO]\n"
    "                      [--drafter-artifact NAME=PATH]...\n"
    "       ninferno-convert --relayout auto|published|frag --target-artifact PATH [--out PATH]\n"
    "                      [--max-file-bytes N]\n"
    "--model is a Hugging Face checkpoint directory, a .gguf file (the first part of a split\n"
    "set) or a directory holding one GGUF model. --fetch downloads one instead: org/name[@rev]\n"
    "from the hub (only the files a conversion reads; one GGUF quantization, chosen by --quant\n"
    "or by preference NVFP4, MXFP4, Q8_0, BF16, ...), or an http(s) URL of one file, into the\n"
    "Hugging Face cache layout ($HF_HUB_CACHE, $HF_HOME/hub, $XDG_CACHE_HOME/huggingface/hub,\n"
    "~/.cache/huggingface/hub; --hub also takes hf and ms).\n"
    "$HF_TOKEN authorizes gated repositories. A GGUF without tokenizer files takes them from\n"
    "its base model's repository when it names one, else rebuilds them from its metadata.\n"
    "--model needs --out; with --fetch and no --out the artifact goes to\n"
    "<cache>/models--org--name/ninferno/<commit>/<stem>.<recipe>.ninferno, and a rerun reuses\n"
    "it unless a flag changes what is written (--components, --vocab, --drafter-quant, ...).\n"
    "Without --recipe the\n"
    "official recipe for the checkpoint's architecture is chosen (with its quantized source\n"
    "when it needs one).\n"
    "A model runs as published: a quantized checkpoint keeps its encoding (NVFP4, MXFP4, FP8\n"
    "with per-tensor or per-channel scales), an unquantized one keeps its projections in BF16.\n"
    "A conversion that would dequantize a published quantization (and quantize it again), or\n"
    "quantize the projections of an unquantized checkpoint (dense_q8, dense_fp8, dense_nvfp4,\n"
    "qwen3_6_27b, qwen3_8_27b, qwen3_6_35b_a3b, qwen3_5_q8, moe_q8, or an _nvfp4 recipe on a\n"
    "checkpoint without NVFP4), is refused naming the alternatives; --maintainer-quantize allows\n"
    "it, and the artifact is then NInferno's own quantization.\n"
    "Before the weights are downloaded, config.json alone decides whether the architecture\n"
    "converts, and the cache's free space is checked; $HF_HUB_OFFLINE uses the cached snapshot\n"
    "only.\n"
    "--vocab (dense and MoE recipes) sets the token embedding and output head: native keeps\n"
    "them as the checkpoint publishes them (BF16 stays BF16, a GGUF's Q8_0 stays Q8_0), q8\n"
    "quantizes an unquantized one to q8_g32_fp16. The default is native, and q8 for the recipes\n"
    "that quantize the model themselves (dense_q8, dense_fp8, dense_nvfp4, moe_q8); the choice\n"
    "is recorded in the provenance of <out>.conversion.json.\n"
    "--drafter-quant quantizes a drafter (a Gemma 4 assistant, a DFlash drafter) itself: every\n"
    "projection its recipe leaves in BF16 becomes q8_g32_fp16 (q8), q4_g64_fp16 (q4) or\n"
    "weight-only NVFP4 (nvfp4; q8 where a matrix is not whole 128-row tiles). The target\n"
    "verifies every drafted token, so the output is unchanged; only the speed differs. A\n"
    "published encoding is kept, never quantized again; a target without a companion drafter is\n"
    "refused, unless its separate drafters are attached in the same run (they take it).\n"
    "--consume-source releases the checkpoint's disk space while converting: the checkpoint is\n"
    "destroyed. It journals progress next to --out; rerun the same command to resume.\n"
    "Speculative drafters come with the model: --fetch attaches a checkpoint's (--no-drafters\n"
    "skips it; --with-drafters does it for a GGUF model and for --model too). They are, in\n"
    "order, the MTP layers inside the\n"
    "checkpoint (kept as the mtp component), the drafters the checkpoint itself declares in\n"
    "ninferno_speculation.json, else those the official registry names for the model id or for a\n"
    "base model its model card declares; never a search. Each is checked against the target\n"
    "(vocabulary, geometry, taps) before it is fetched and converted. A separate drafter (Gemma\n"
    "4 assistant, DFlash) becomes its own artifact; a companion (Qwen DFlash/DFlash2) becomes a\n"
    "component of the target. The drafters and the published speculation profiles go to\n"
    "<target>.ninferno.speculation.json, which the engines read at start. --plan-drafters prints\n"
    "the plan and downloads nothing; --attach-drafters adds them to an existing artifact.\n"
    "--drafter-artifact NAME=PATH attaches an already converted drafter instead of fetching it;\n"
    "a NAME the model does not declare adds it (e.g. a NemotronH MTP sidecar: mtp=PATH).\n"
    "--layout sets how the routed-expert banks of GGUF block formats are stored, losslessly\n"
    "either way: auto (the default) keeps a bank's published bytes in ggml_frag_v1, the order\n"
    "the MoE kernel reads them in, for the formats and shapes where that measured faster\n"
    "(ggml_q4_0 at K >= 1024, ggml_q4_k, ggml_iq3_xxs, ggml_iq2_s); published keeps every bank\n"
    "in its published layout (A/B runs); frag takes ggml_frag_v1 wherever a format has it.\n"
    "--relayout moves an existing artifact's banks between the two by the same rule, without\n"
    "the checkpoint: in place under an undo journal (<artifact>.relayout-journal; rerun the\n"
    "same command after an interruption; until then the engines refuse the artifact), or into a\n"
    "new artifact with --out. Stop a server that has the artifact open before relaying it out in\n"
    "place: the bytes it reads change under it. --relayout takes no conversion flag.\n"
    "official recipes: gemma4_published (any Gemma 4 text model as published: its NVFP4 and FP8\n"
    "  projections kept, everything else BF16; the default), gemma4_26b_a4b_nvfp4,\n"
    "  gemma4_31b_nvfp4, gemma4_assistant_published (the assistant as published: BF16 as is,\n"
    "  a GGUF's Q8_0 kept; the default), gemma4_assistant_q8 (NInferno's own Q8, only when\n"
    "  named), gemma4_dflash_published (the DFlash drafter as published, BF16 as is; the\n"
    "  default), gemma4_dflash_q8 (NInferno's own Q8, only when named), dflash_published (a\n"
    "  block DFlash drafter as published; the default for one),\n"
    "  qwen3_6_27b, qwen3_6_27b_nvfp4, qwen3_8_27b, qwen3_8_27b_nvfp4, qwen3_6_35b_a3b,\n"
    "  qwen3_5_q8, qwen3_5_bf16 (any Qwen3.5 Dense size), qwen3_5_published (any Qwen3.5\n"
    "  Dense or MoE size; keeps the checkpoint's encoded projections and experts as published\n"
    "  and quantizes nothing),\n"
    "  qwen3_6_35b_a3b_nvfp4 (the _nvfp4 recipes read --source quantized=PATH),\n"
    "  dense_encoded, dense_published, dense_nvfp4, dense_fp8, dense_q8, dense_bf16 (dense\n"
    "  decoders: Llama, Mistral, Granite, OLMo, ...; dense_published quantizes nothing),\n"
    "  moe_published, moe_q8 (MoE decoders: gpt-oss, Qwen3-MoE; published keeps MXFP4 and\n"
    "  NVFP4 experts as published),\n"
    "  nemotron_h_mtp_published (a NemotronH MTP sidecar as published: a GGUF's Q4_0/Q8_0 blocks\n"
    "  kept word for word, everything else as stored; the default)\n"
    "--override (a Python recipe file) is not accepted here: python3 -m tools.convert runs it.\n"
    "Exit status: 0 done, 2 an argument or conversion error, 1 a failed or incompatible\n"
    "drafter in an attachment (--plan-drafters only prints the plan: 0).\n";

std::uint64_t parse_count(const std::string& flag, const std::string& text,
                          std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max()) {
    const auto invalid = [&] {
        return std::runtime_error(flag + ": invalid integer value '" + text + "'");
    };
    if (text.empty() || text.front() < '0' || text.front() > '9') { throw invalid(); }
    std::size_t used = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &used);
    } catch (const std::logic_error&) { // not a number, or out of range
        throw invalid();
    }
    if (used != text.size() || value > maximum) { throw invalid(); }
    return static_cast<std::uint64_t>(value);
}

unsigned parse_unsigned(const std::string& flag, const std::string& text) {
    return static_cast<unsigned>(parse_count(flag, text, std::numeric_limits<unsigned>::max()));
}

void add_pair(std::vector<std::pair<std::string, std::filesystem::path>>& out,
              const std::string& text, const char* flag) {
    const auto eq   = text.find('=');
    const auto name = eq == std::string::npos ? std::string() : text.substr(0, eq);
    const auto path = eq == std::string::npos ? std::string() : text.substr(eq + 1);
    if (name.empty() || path.empty()) {
        throw std::runtime_error(std::string(flag) + " takes NAME=PATH, not '" + text + "'");
    }
    for (const auto& [existing, _] : out) {
        if (existing == name) {
            throw std::runtime_error(std::string(flag) + " names " + name + " twice");
        }
    }
    out.emplace_back(name, path);
}

void print_converted(const ninferno::convert::ConvertOptions& options,
                     const ninferno::convert::ConvertResult& result) {
    std::printf("wrote %s: %llu objects, %llu files, %.1fs (model %.2fs, plan %.2fs, "
                "weights %.2fs, finish %.2fs)\n",
                options.out.c_str(), static_cast<unsigned long long>(result.objects),
                static_cast<unsigned long long>(result.files), result.seconds,
                result.model_seconds, result.plan_seconds, result.write_seconds,
                result.finish_seconds);
    if (options.consume_source) {
        std::printf("consume-source: %s%llu objects resumed, planned peak %.2f GiB above the "
                    "checkpoint, %.2f GiB of checkpoint punched\n",
                    result.resumed ? "resumed; " : "",
                    static_cast<unsigned long long>(result.resumed_objects),
                    static_cast<double>(result.consume_peak_bound) /
                        static_cast<double>(1ULL << 30),
                    static_cast<double>(result.punched_bytes) / static_cast<double>(1ULL << 30));
    }
}

} // namespace

int main(int argc, char** argv) {
    if (ninferno::version_requested(argc, argv)) {
        std::cout << "NInferno " << NINFERNO_VERSION_STRING << '\n';
        return 0;
    }
    ninferno::convert::ConvertOptions options;
    options.components_auto = true; // until --components names them
    ninferno::convert::FetchOptions fetch;
    std::string fetch_spec;
    bool quiet          = false;
    bool have_model     = false;
    bool have_out       = false;
    bool fetch_only     = false;
    bool base_tokenizer_files = true;
    std::optional<bool> with_drafters; // --with-drafters / --no-drafters
    bool plan_only = false, attach_only = false;
    std::optional<ninferno::convert::ExpertLayout> relayout; // --relayout
    std::vector<std::string> given;                        // every flag, for --relayout's check
    std::string model_id;
    std::filesystem::path target_artifact;
    std::map<std::string, std::filesystem::path> drafter_artifacts;
    bool components_given = false;
    bool arguments_read = false; // after this, an error is the conversion's, not the usage's
    try {
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            std::string value;
            bool inline_value = false;
            if (const auto eq = flag.find('='); flag.starts_with("--") && eq != std::string::npos) {
                value        = flag.substr(eq + 1);
                flag         = flag.substr(0, eq);
                inline_value = true;
            }
            const auto next = [&]() -> std::string {
                if (inline_value) { return value; }
                if (i + 1 >= argc) { throw std::runtime_error(flag + ": expected one argument"); }
                return argv[++i];
            };
            given.push_back(flag);
            // A switch takes no value: --quiet=0 would read as --quiet.
            static const std::set<std::string, std::less<>> kSwitches = {"--no-verify",
                                                                         "--fetch-only",
                                                                         "--no-base-tokenizer",
                                                                         "--with-drafters",
                                                                         "--no-drafters",
                                                                         "--plan-drafters",
                                                                         "--attach-drafters",
                                                                         "--proposal",
                                                                         "--consume-source",
                                                                         "--keep-source-cache",
                                                                         "--round-tiny-scales",
                                                                         "--maintainer-quantize",
                                                                         "--quiet"};
            if (inline_value && kSwitches.contains(flag)) {
                throw std::runtime_error(flag + " takes no value");
            }
            if (flag == "--help" || flag == "-h") {
                std::cout << kUsage;
                return 0;
            } else if (flag == "--model") {
                options.model = next();
                have_model    = true;
            } else if (flag == "--fetch") {
                fetch_spec = next();
            } else if (flag == "--hub") {
                fetch.hub = next();
                if (!ninferno::convert::is_hugging_face_hub(fetch.hub) &&
                    !ninferno::convert::is_modelscope_hub(fetch.hub)) {
                    throw std::runtime_error("--hub must be huggingface (hf) or modelscope (ms)");
                }
            } else if (flag == "--revision") {
                fetch.revision = next();
            } else if (flag == "--quant") {
                fetch.quant = next();
            } else if (flag == "--include") {
                fetch.include.push_back(next());
            } else if (flag == "--cache") {
                fetch.cache = next();
            } else if (flag == "--streams") {
                fetch.streams = parse_unsigned(flag, next());
            } else if (flag == "--max-streams") {
                fetch.max_streams = parse_unsigned(flag, next());
            } else if (flag == "--min-speed") {
                fetch.min_speed = parse_count(flag, next());
            } else if (flag == "--stall-seconds") {
                fetch.stall_seconds = parse_unsigned(flag, next());
            } else if (flag == "--chunk-bytes") {
                fetch.chunk_bytes = parse_count(flag, next());
            } else if (flag == "--retries") {
                fetch.retries = parse_unsigned(flag, next());
            } else if (flag == "--no-verify") {
                fetch.verify = false;
            } else if (flag == "--fetch-only") {
                fetch_only = true;
            } else if (flag == "--no-base-tokenizer") {
                base_tokenizer_files = false;
            } else if (flag == "--recipe") {
                options.recipe = next();
            } else if (flag == "--layout") {
                options.expert_layout = ninferno::convert::parse_expert_layout(next());
            } else if (flag == "--relayout") {
                relayout = ninferno::convert::parse_expert_layout(next());
            } else if (flag == "--drafter-quant") {
                options.drafter_quant = next();
                if (options.drafter_quant != "q8" && options.drafter_quant != "q4" &&
                    options.drafter_quant != "nvfp4") {
                    throw std::runtime_error("--drafter-quant must be q8, q4 or nvfp4");
                }
            } else if (flag == "--vocab") {
                options.vocab = next();
                if (options.vocab != "native" && options.vocab != "q8") {
                    throw std::runtime_error("--vocab must be native or q8");
                }
            } else if (flag == "--out") {
                options.out = next();
                have_out    = true;
            } else if (flag == "--name") {
                options.name = next();
            } else if (flag == "--resource") {
                add_pair(options.resources, next(), "--resource");
            } else if (flag == "--source") {
                add_pair(options.sources, next(), "--source");
            } else if (flag == "--components") {
                options.components      = next();
                options.components_auto = false;
                components_given        = true;
            } else if (flag == "--with-drafters") {
                if (with_drafters == false) {
                    throw std::runtime_error("give one of --with-drafters and --no-drafters");
                }
                with_drafters = true;
            } else if (flag == "--no-drafters") {
                if (with_drafters == true) {
                    throw std::runtime_error("give one of --with-drafters and --no-drafters");
                }
                with_drafters           = false;
                options.components_auto = false;
            } else if (flag == "--plan-drafters") {
                plan_only = true;
            } else if (flag == "--attach-drafters") {
                attach_only = true;
            } else if (flag == "--model-id") {
                model_id = next();
            } else if (flag == "--target-artifact") {
                target_artifact = next();
            } else if (flag == "--drafter-artifact") {
                std::vector<std::pair<std::string, std::filesystem::path>> pair;
                for (const auto& [name, path] : drafter_artifacts) {
                    pair.emplace_back(name, path);
                }
                add_pair(pair, next(), "--drafter-artifact");
                drafter_artifacts[pair.back().first] = pair.back().second;
            } else if (flag == "--override") {
                throw std::runtime_error("--override runs Python and is not available natively");
            } else if (flag == "--proposal") {
                options.proposal = true;
            } else if (flag == "--proposal-rows") {
                options.proposal_rows = parse_count(flag, next());
            } else if (flag == "--ranking") {
                options.ranking = next();
            } else if (flag == "--device") {
                (void)next(); // tools.convert's flag: the native conversion runs on the CPU
            } else if (flag == "--rows-per-chunk") {
                options.rows_per_chunk = parse_count(flag, next());
            } else if (flag == "--max-file-bytes") {
                options.max_file_bytes = parse_count(flag, next());
            } else if (flag == "--threads") {
                options.threads = parse_unsigned(flag, next());
            } else if (flag == "--consume-source") {
                options.consume_source = true;
            } else if (flag == "--consume-window") {
                options.consume_window = parse_count(flag, next());
            } else if (flag == "--assume-free-bytes") {
                options.assume_free_bytes = parse_count(flag, next());
            } else if (flag == "--keep-source-cache") {
                options.keep_source_cache = true;
            } else if (flag == "--round-tiny-scales") {
                options.round_tiny_scales = true;
            } else if (flag == "--maintainer-quantize") {
                options.maintainer_quantize = true;
            } else if (flag == "--quiet") {
                quiet = true;
            } else {
                throw std::runtime_error("unrecognized argument " + flag);
            }
        }
        // The proposal head's settings without the head would do nothing.
        if (!options.proposal) {
            for (const auto& flag : given) {
                if (flag == "--proposal-rows" || flag == "--ranking") {
                    throw std::runtime_error(flag + " applies to --proposal");
                }
            }
        }
        // The flags of a download given without one would do nothing.
        if (fetch_spec.empty()) {
            for (const auto& flag : given) {
                if (flag == "--fetch-only" || flag == "--quant" || flag == "--revision" ||
                    flag == "--include" || flag == "--no-base-tokenizer") {
                    throw std::runtime_error(flag + " applies to --fetch");
                }
            }
        }
        if (int(plan_only) + int(attach_only) + int(relayout.has_value()) > 1) {
            throw std::runtime_error(
                "give one of --plan-drafters, --attach-drafters and --relayout");
        }
        if (!quiet) {
            options.log = [](const std::string& line) { std::printf("%s\n", line.c_str()); };
            fetch.log   = options.log;
        }
        namespace cv = ninferno::convert;
        cv::AttachOptions attach;
        attach.fetch             = fetch;
        attach.log               = options.log;
        attach.repository        = std::filesystem::current_path();
        attach.drafter_artifacts = drafter_artifacts;
        attach.conversion        = options;
        const auto print_plan    = [](const cv::DrafterPlan& plan) {
            std::printf("speculation: source %s%s%s\n", plan.source.c_str(),
                        plan.entry.empty() ? "" : (" (registry entry " + plan.entry + ")").c_str(),
                        plan.matched_by.empty() ? "" : (" by " + plan.matched_by).c_str());
            int failed = 0;
            for (const auto& d : plan.drafters) {
                // No repository: in the checkpoint, or given with --drafter-artifact.
                const bool given  = d.decl.component.empty() && !d.artifact.empty();
                const char* where = !d.decl.repo.empty() ? d.decl.repo.c_str()
                                    : given              ? "(given)"
                                                         : "(in the checkpoint)";
                std::printf("drafter %-10s %-17s %-44s %s\n", d.decl.name.c_str(),
                            d.decl.kind.c_str(), where,
                            d.status.empty() ? (d.compatible ? "ok" : d.reason.c_str())
                                             : d.status.c_str());
                failed += d.status.starts_with("failed") || d.status.starts_with("incompatible");
            }
            for (const auto& note : plan.notes) { std::printf("note: %s\n", note.c_str()); }
            // A target converted wrong takes no drafter: a failure, even with none planned.
            failed += !plan.stale_target.empty();
            return failed;
        };
        if (plan_only) {
            // What would be attached: from the checkpoint and the registry, nothing downloaded.
            auto target = !target_artifact.empty() ? cv::describe_artifact(target_artifact, model_id)
                          : have_model ? cv::describe_checkpoint(options.model, model_id)
                                       : throw std::runtime_error(
                                             "--plan-drafters needs --model DIR or --target-artifact");
            std::printf("%s\n", cv::plan_drafters(std::move(target)).to_json().dump(1).c_str());
            return 0;
        }
        if (relayout) {
            if (target_artifact.empty()) {
                throw std::runtime_error("--relayout needs --target-artifact");
            }
            for (const auto& flag : given) {
                if (flag != "--relayout" && flag != "--target-artifact" && flag != "--out" &&
                    flag != "--max-file-bytes" && flag != "--quiet") {
                    throw std::runtime_error("--relayout takes only --target-artifact, --out, "
                                             "--max-file-bytes and --quiet, not " +
                                             flag);
                }
            }
            cv::RelayoutOptions request;
            request.artifact       = target_artifact;
            request.layout         = *relayout;
            request.max_file_bytes = options.max_file_bytes;
            request.log            = options.log;
            if (have_out) { request.out = options.out; }
            arguments_read  = true;
            const auto done = cv::relayout(request);
            std::printf("relayout %s of %s: %llu banks changed%s, %.1fs\n",
                        std::string(cv::expert_layout_name(*relayout)).c_str(),
                        (have_out ? options.out : target_artifact).c_str(),
                        static_cast<unsigned long long>(done.objects),
                        done.resumed ? " (resumed)" : "", done.seconds);
            return 0;
        }
        if (attach_only) {
            if (target_artifact.empty()) {
                throw std::runtime_error("--attach-drafters needs --target-artifact");
            }
            arguments_read = true;
            auto plan      = cv::plan_drafters(cv::describe_artifact(target_artifact, model_id));
            plan      = cv::attach_drafters(std::move(plan), target_artifact, attach);
            return print_plan(plan) ? 1 : 0;
        }
        if (have_model == !fetch_spec.empty()) {
            throw std::runtime_error("give exactly one of --model and --fetch");
        }
        std::optional<cv::DrafterPlan> drafters;
        ninferno::convert::ConvertResult result;
        if (!fetch_spec.empty()) {
            cv::AcquireOptions acquire;
            acquire.fetch                = fetch;
            acquire.convert              = options;
            acquire.components_given     = components_given;
            acquire.with_drafters        = with_drafters;
            acquire.model_id             = model_id;
            acquire.drafter_artifacts    = drafter_artifacts;
            acquire.repository           = std::filesystem::current_path();
            acquire.base_tokenizer_files = base_tokenizer_files;
            acquire.fetch_only           = fetch_only;
            acquire.log                  = options.log;
            // A rerun without --out finds the artifact the first run converted. The kept
            // artifact is keyed by commit, quant and recipe only, so a flag that changes what
            // is written converts again; an explicit --out is never reused (nor locked).
            static const std::set<std::string, std::less<>> kContentFlags = {
                "--components",
                "--layout",
                "--vocab",
                "--drafter-quant",
                "--round-tiny-scales",
                "--maintainer-quantize",
                "--proposal",
                "--proposal-rows",
                "--ranking",
                "--include",
                "--name",
                "--resource",
                "--source",
                "--no-base-tokenizer",
                "--model-id",
                "--drafter-artifact",
                "--no-drafters",
                "--with-drafters",
                "--max-file-bytes"};
            acquire.reuse = options.out.empty() &&
                            std::none_of(given.begin(), given.end(), [](const std::string& flag) {
                                return kContentFlags.contains(flag);
                            });
            if (options.rows_per_chunk == 0) {
                throw std::runtime_error("--rows-per-chunk must be positive");
            }
            arguments_read = true;
            auto acquired  = cv::acquire_model(fetch_spec, std::move(acquire));
            if (quiet) { // otherwise the acquisition's log said it
                std::printf("fetched %s@%s -> %s\n", acquired.fetched.repo.c_str(),
                            acquired.fetched.commit.substr(0, 12).c_str(),
                            acquired.fetched.model.c_str());
            }
            if (fetch_only) { return 0; }
            options.out    = acquired.artifact;
            options.recipe = acquired.recipe;
            drafters       = std::move(acquired.drafters);
            // An earlier run's artifact (AcquireOptions::reuse) comes with no conversion.
            if (acquired.conversion) {
                result = std::move(*acquired.conversion);
                print_converted(options, result);
            } else if (quiet) { // otherwise the acquisition's log said it
                std::printf("artifact %s (converted earlier)\n",
                            acquired.artifact.string().c_str());
            }
            if (drafters && print_plan(*drafters)) { return 1; }
            return 0;
        }
        // A GGUF model carries no checkpoint to plan from: its drafters are planned for the
        // converted artifact.
        const bool gguf_drafters =
            with_drafters.value_or(false) && !cv::gguf_model_files(options.model).empty();
        // A GGUF target carries no companion: --drafter-quant is its drafters'.
        if (gguf_drafters) { options.drafter_quant.clear(); }
        if (with_drafters.value_or(false) && !gguf_drafters) {
            drafters = cv::plan_checkpoint_drafters(options.model, model_id, components_given,
                                                    options, attach);
        }
        if (!have_out) {
            throw std::runtime_error("--model needs --out (only --fetch has a default path)");
        }
        if (options.rows_per_chunk == 0) {
            throw std::runtime_error("--rows-per-chunk must be positive");
        }
        arguments_read = true;
        if (options.recipe.empty()) {
            auto choice = ninferno::convert::choose_recipe(
                options.model, ninferno::convert::gguf_request_for(options.components));
            options.recipe = choice.recipe;
            for (auto& [name, path] : choice.sources) {
                bool given = false;
                for (const auto& [existing, _] : options.sources) { given = given || existing == name; }
                if (!given) { options.sources.emplace_back(name, path); }
            }
            if (!quiet) {
                std::printf("recipe %s (chosen for the checkpoint)\n", options.recipe.c_str());
                if (!choice.note.empty()) { std::printf("recipe note: %s\n", choice.note.c_str()); }
            }
        }
        result = ninferno::convert::convert(options);
        print_converted(options, result);
        if (gguf_drafters) {
            // Planned and attached once the model is converted: the drafter download is checked
            // against the cache's free space, a failed drafter is listed as failed (a later
            // --attach-drafters tries again), and nothing here undoes the conversion.
            attach.fetch.before_download = [](const cv::FetchPlan& plan) {
                cv::check_free_space(plan);
            };
            try {
                auto target = cv::describe_gguf_artifact(options.out, model_id, options.model);
                const auto done =
                    cv::attach_drafters(cv::plan_drafters(std::move(target)), options.out, attach);
                return print_plan(done) ? 1 : 0;
            } catch (const std::exception& error) {
                std::printf("speculation: drafters not attached: %s\n", error.what());
                return 0;
            }
        }
        if (drafters) {
            const auto done = cv::attach_drafters(std::move(*drafters), options.out, attach);
            if (print_plan(done)) { return 1; }
        } else if (!with_drafters.has_value() && !quiet) {
            // No manifest names the model's registry entry, so ninferno-serve --spec auto finds no
            // profile for it (not even for a drafter converted into the artifact as a component).
            std::printf("speculation: no drafters attached, so --spec auto decodes plainly; "
                        "ninferno-convert --attach-drafters --target-artifact %s --model-id <repo> "
                        "attaches them (or convert with --with-drafters)\n",
                        options.out.string().c_str());
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninferno-convert: error: %s\n%s", error.what(),
                     arguments_read ? "" : kUsage);
        return 2;
    }
}
