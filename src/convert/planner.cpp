#include "convert/planner.h"

#include "artifact/expert_groups.h"
#include "artifact/formats.h"
#include "convert/expert_layout.h"
#include "convert/numerics.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>

namespace ninferno::convert {
namespace {

bool is_direct(QType format) {
    return format == QType::BF16 || format == QType::FP32 || format == QType::INT32;
}

QuantLayout default_layout(QType format) {
    switch (format) {
    case QType::BF16:
    case QType::FP32:
    case QType::INT32:
        return QuantLayout::Contiguous;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
        return QuantLayout::RowScale;
    case QType::FP8_E4M3FN_B128_F32:
        return QuantLayout::BlockScale128F32;
    case QType::NVFP4:
        return QuantLayout::BlockScaleK16M128x4;
    case QType::MXFP4:
        return QuantLayout::BlockScaleK32E8M0;
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
        return QuantLayout::RowGroupZp;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
        return QuantLayout::GgmlBlocks;
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
        return QuantLayout::GgmlPlanes;
    default:
        return QuantLayout::RowSplit;
    }
}

std::string numbered(const char* prefix, std::size_t index) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%s/%06zu", prefix, index);
    return buffer;
}

std::uint64_t align256(std::uint64_t value) { return (value + 255) / 256 * 256; }

struct Fragment {
    std::size_t weight  = 0;
    std::uint64_t first = 0;
    std::uint64_t last  = 0;
};

class Planner {
public:
    Planner(const Model& model, const Recipe& recipe, ExpertLayout expert_layout)
        : model_(model), recipe_(recipe), expert_layout_(expert_layout),
          fragments_(model.parameters.size()), auxiliaries_(model.parameters.size()) {}

    Plan run() {
        warm();
        std::vector<bool> used(model_.parameters.size(), false);
        for (const auto& names : recipe_.explicit_groups) {
            for (const auto name : names) {
                if (used[name] || recipe_.alias[name]) {
                    throw std::runtime_error("explicit groups overlap or include a shared alias");
                }
            }
            emit(names);
            for (const auto name : names) { used[name] = true; }
        }
        for (const auto& names : model_.packing_groups) {
            if (std::any_of(names.begin(), names.end(), [&](std::size_t i) {
                    return used[i] || recipe_.alias[i] || recipe_.separate[i];
                })) {
                continue;
            }
            // The whole group, else its longest leading run of two or more that packs: a GGUF
            // layer stores v in a wider type than q and k in many layers (Q4_K_M, the UD mixes),
            // and q and k still share one parent (one projection instead of two at decode).
            for (std::size_t n = names.size(); n >= 2; --n) {
                const std::vector<std::size_t> run(names.begin(), names.begin() + n);
                if (!packable(run)) { continue; }
                emit(run);
                for (const auto name : run) { used[name] = true; }
                break;
            }
        }
        for (std::size_t i = 0; i < model_.parameters.size(); ++i) {
            if (!used[i] && !recipe_.alias[i]) { emit({i}); }
        }
        return finish();
    }

private:
    // Planning reads every NVFP4 source's divisor scalars and first row: thousands of 4 KiB
    // reads scattered over the checkpoint. Issue them from many threads first so the sequential
    // pass below finds them cached. Selections may share sources (aliases, a proposal head over
    // the output head's source), which resolve once under their own lock; errors are left for
    // the sequential pass to report in order.
    void warm() const {
        std::atomic<std::size_t> next{0};
        const auto work = [&] {
            for (auto i = next.fetch_add(1); i < recipe_.selections.size(); i = next.fetch_add(1)) {
                try {
                    const auto& source = recipe_.selections[i].source;
                    source->resolve();
                    if (recipe_.selections[i].method != Method::ImportEncoded ||
                        !source->is_nvfp4()) {
                        continue;
                    }
                    (void)source->weight_divisor();
                    if (source->provides_input_divisor()) { (void)source->input_divisor(); }
                } catch (...) {}
            }
        };
        std::vector<std::thread> threads;
        const auto count = std::max(1U, std::thread::hardware_concurrency());
        for (unsigned t = 0; t < count; ++t) { threads.emplace_back(work); }
        for (auto& thread : threads) { thread.join(); }
    }

    // FP8 128x128 blocks restart at every source of a parent: all but the last need whole blocks.
    bool whole_blocks(const std::vector<std::size_t>& names) const {
        for (std::size_t i = 0; i + 1 < names.size(); ++i) {
            if (model_.parameters[names[i]].shape.empty() ||
                model_.parameters[names[i]].shape[0] % 128) {
                return false;
            }
        }
        return true;
    }

    // Whether a default packing group may share one parent: one format and method, NVFP4
    // sources agreeing on their divisors, FP8 block grids whole and one column order.
    bool packable(const std::vector<std::size_t>& names) const {
        if (!compatible(names)) { return false; }
        const auto& first = recipe_.selections[names.front()];
        if (first.format == QType::NVFP4 && first.method == Method::ImportEncoded) {
            // Default packing keeps parents apart when their sources disagree on divisors.
            std::optional<std::uint32_t> word;
            for (const auto name : names) {
                const auto& source = recipe_.selections[name].source;
                if (!source->provides_divisors()) { return false; }
                const auto value = source->weight_divisor();
                if (word && *word != value) { return false; }
                word = value;
            }
        }
        if (first.format == QType::FP8_E4M3FN_B128_F32 && !whole_blocks(names)) {
            return false; // each member keeps its own 128-row block grid
        }
        if (first.method == Method::ImportEncoded && !same_orders(names)) {
            return false; // one plane stores one column order: members stay apart
        }
        return true;
    }

    bool same_orders(const std::vector<std::size_t>& names) const {
        const auto order = recipe_.selections[names.front()].source->k_order();
        return std::all_of(names.begin(), names.end(), [&](std::size_t i) {
            return same_order(recipe_.selections[i].source->k_order(), order);
        });
    }

    bool compatible(const std::vector<std::size_t>& names) const {
        const auto& first = recipe_.selections[names.front()];
        return std::all_of(names.begin(), names.end(), [&](std::size_t i) {
            const auto& selection = recipe_.selections[i];
            return selection.format == first.format && selection.method == first.method;
        });
    }

    void emit(const std::vector<std::size_t>& names) {
        std::set<std::string> residency;
        for (const auto name : names) { residency.insert(model_.parameters[name].residency); }
        if (residency.size() > 1) {
            throw std::runtime_error("grouping crosses independently selected private weights");
        }
        if (!compatible(names)) {
            // Name the first member that disagrees with the first, so a refusal says where.
            const auto& first = recipe_.selections[names.front()];
            std::string detail;
            for (const auto name : names) {
                const auto& selection = recipe_.selections[name];
                if (selection.format != first.format || selection.method != first.method) {
                    detail = " (" + model_.parameters[names.front()].name + " " +
                             std::string(artifact::format_name(first.format)) + "/" +
                             std::string(method_name(first.method)) + ", " +
                             model_.parameters[name].name + " " +
                             std::string(artifact::format_name(selection.format)) + "/" +
                             std::string(method_name(selection.method)) + ")";
                    break;
                }
            }
            throw std::runtime_error(
                "explicit grouping has incompatible formats, layouts or methods" + detail);
        }
        const auto& selection = recipe_.selections[names.front()];
        std::vector<SourcePtr> sources;
        for (const auto name : names) {
            sources.push_back(recipe_.selections[name].source);
            sources.back()->resolve();
        }
        WeightJob job;
        job.id         = numbered("weight", jobs_.size());
        job.format     = selection.format;
        job.method     = selection.method;
        job.layout     = default_layout(selection.format);
        job.parameters = names;
        job.sources    = sources;
        job.shape      = parent_shape(sources, selection.format);
        try {
            job.divisors = parent_divisors(sources, selection.format);
        } catch (const std::exception& error) {
            throw std::runtime_error(label(names) + ": " + error.what());
        }
        if (selection.method == Method::ImportEncoded) {
            job.order = sources.front()->k_order();
            if (!same_orders(names)) {
                throw std::runtime_error(label(names) +
                                         ": grouped sources store their columns in different "
                                         "orders, which one plane cannot hold");
            }
            if (job.order) {
                if (job.layout != QuantLayout::RowGroupZp || job.shape.size() != 2 ||
                    job.order->size() != job.shape[1]) {
                    throw std::runtime_error(label(names) +
                                             ": a column order is kept only on a u4 matrix");
                }
                job.layout = QuantLayout::RowGroupZpOrdered;
            }
            // A layer's routed-expert bank of a ggml block format (every parameter one of its
            // experts, as the artifact names them): its published words, in fragment order where
            // the grouped MoE kernel reads that faster (convert/expert_layout.h). Kept losslessly
            // either way, so the fidelity check counts it as kept.
            if (is_ggml_block(job.format) && job.shape.size() == 2 && expert_bank(names)) {
                job.layout =
                    expert_bank_layout(job.format, job.shape[1], expert_layout_, job.layout);
            }
        }
        try {
            job.geometry = weight_geometry(job.format, job.layout, job.shape, job.divisors);
        } catch (const std::exception& error) {
            throw std::runtime_error(label(names) + ": " + error.what());
        }
        try {
            prepare_method(job);
        } catch (const std::exception& error) {
            throw std::runtime_error(label(names) + ": " + error.what());
        }
        std::uint64_t cursor = 0;
        for (std::size_t j = 0; j < names.size(); ++j) {
            const auto count = sources[j]->elements();
            fragments_[names[j]].push_back({jobs_.size(), cursor, cursor + count});
            cursor += count;
        }
        jobs_.push_back(std::move(job));
    }

    // Whether every parameter is a routed expert of one layer (artifact/expert_groups.h).
    bool expert_bank(const std::vector<std::size_t>& names) const {
        std::optional<std::string> first;
        for (const auto name : names) {
            const auto layer = artifact::routed_expert_layer(model_.parameters[name].name);
            if (!layer || (first && *first != *layer)) { return false; }
            first = layer;
        }
        return first.has_value();
    }

    std::string label(const std::vector<std::size_t>& names) const {
        std::string out;
        for (const auto name : names) {
            if (!out.empty()) { out += ","; }
            out += model_.parameters[name].name;
        }
        return out;
    }

    static Shape parent_shape(const std::vector<SourcePtr>& sources, QType format) {
        const auto& first  = sources.front()->shape();
        std::uint64_t size = 0;
        for (const auto& source : sources) { size += source->elements(); }
        if (sources.size() == 1) {
            if (is_direct(format) || first.size() == 2) { return first; }
            if (first.empty()) { return {1, 1}; }
            std::uint64_t leading = 1;
            for (std::size_t i = 0; i + 1 < first.size(); ++i) { leading *= first[i]; }
            return {leading, first.back()};
        }
        if (std::all_of(sources.begin(), sources.end(),
                        [](const SourcePtr& s) { return s->shape().size() == 1; })) {
            return is_direct(format) ? Shape{size} : Shape{1, size};
        }
        if (std::all_of(sources.begin(), sources.end(), [&](const SourcePtr& s) {
                return s->shape().size() == 2 && s->shape()[1] == first[1];
            })) {
            std::uint64_t rows = 0;
            for (const auto& source : sources) { rows += source->shape()[0]; }
            return {rows, first[1]};
        }
        throw std::runtime_error("grouped inputs need compatible rows or an explicit parent shape");
    }

    static std::uint64_t parent_divisors(const std::vector<SourcePtr>& sources, QType format) {
        if (format != QType::NVFP4 || sources.size() < 2) { return 1; }
        std::set<std::uint32_t> words;
        for (const auto& source : sources) {
            if (!source->provides_divisors()) { return 1; }
            words.insert(source->weight_divisor());
        }
        if (words.size() == 1) { return 1; }
        std::set<std::uint64_t> rows;
        for (const auto& source : sources) {
            if (source->shape().size() != 2) { rows.insert(0); }
            rows.insert(source->shape()[0]);
        }
        if (rows.size() != 1) {
            throw std::runtime_error("NVFP4 sources quantised against different divisors must be "
                                     "complete matrices with equal row counts");
        }
        // Each divisor covers its source's rows; one may end inside a 128-row scale tile, as the
        // scale words are per row and each row is divided by its own word. The plane itself is
        // whole tiles, as every NVFP4 operand (validate_nvfp4_weight).
        if (*rows.begin() * sources.size() % 128) {
            throw std::runtime_error("NVFP4 sources quantised against different divisors must "
                                     "stack to whole 128-row scale tiles, not " +
                                     std::to_string(sources.size()) + " x " +
                                     std::to_string(*rows.begin()) + " rows");
        }
        return sources.size();
    }

    void prepare_method(WeightJob& job) {
        switch (job.method) {
        case Method::CastDirect:
            if (!is_direct(job.format)) {
                throw std::runtime_error("cast_direct requires a direct target format");
            }
            return;
        case Method::GroupedAbsmax:
            if (job.layout != QuantLayout::RowSplit || job.shape.size() != 2) {
                throw std::runtime_error("grouped_absmax requires a grouped-integer matrix target");
            }
            return;
        case Method::Fp8RowMaxabs:
            if (job.format != QType::FP8_E4M3FN_ROW_BF16 || job.shape.size() != 2) {
                throw std::runtime_error(
                    "fp8_row_maxabs requires the row-scaled FP8 matrix format");
            }
            return;
        case Method::ImportEncoded:
            break;
        }
        const bool unscaled =
            job.format == QType::FP8_E4M3FN_ROW_BF16 || job.format == QType::FP8_E4M3FN_ROW_F32 ||
            job.format == QType::FP8_E4M3FN_B128_F32 || job.format == QType::MXFP4 ||
            job.format == QType::Q8_G32_FP16 || u4_zp_group(job.format) || q3_group(job.format) ||
            is_ggml_block(job.format);
        if ((job.format != QType::NVFP4 && !unscaled) || job.shape.size() != 2) {
            throw std::runtime_error("import_encoded requires a known encoded matrix target");
        }
        if (job.format == QType::FP8_E4M3FN_B128_F32) {
            for (std::size_t j = 0; j + 1 < job.sources.size(); ++j) {
                if (job.sources[j]->shape()[0] % 128) {
                    throw std::runtime_error(model_.parameters[job.parameters[j]].name +
                                             ": FP8 128x128 blocks of a packed parent need "
                                             "whole 128-row sources");
                }
            }
        }
        if (unscaled) {
            // Row-scaled FP8, MXFP4, groupwise Q8 and U4 carry no divisor and their Uses no
            // auxiliary.
            for (std::size_t j = 0; j < job.sources.size(); ++j) {
                if (job.sources[j]->encoded_format() != job.format) {
                    throw std::runtime_error(model_.parameters[job.parameters[j]].name +
                                             ": source differs from target " +
                                             std::string(artifact::format_name(job.format)));
                }
            }
            return;
        }
        // One activation tensor feeds every source of the plane, so the smallest calibrated
        // activation divisor is kept for all of them (methods.import_encoded). A source whose
        // calibration never measured its input (a dead expert) takes that value and does not
        // set it.
        std::optional<std::uint32_t> activation;
        if (job.sources.size() > 1) {
            for (const auto& source : job.sources) {
                if (!source->provides_input_divisor() || !source->calibrated_input()) { continue; }
                const auto word = source->input_divisor();
                if (!activation || f32_from_bits(word) < f32_from_bits(*activation)) {
                    activation = word;
                }
            }
        }
        for (std::size_t j = 0; j < job.sources.size(); ++j) {
            const auto& source   = job.sources[j];
            const auto parameter = job.parameters[j];
            if (!source->is_nvfp4()) {
                throw std::runtime_error(model_.parameters[parameter].name +
                                         ": source differs from target nvfp4");
            }
            // Python reads row 0 here to check the source's format; the format is known without
            // a read, and every row's scale words are validated when the plane is written.
            const auto& policies = recipe_.policies[parameter];
            auxiliaries_[parameter].assign(policies.size(), std::nullopt);
            for (std::size_t input = 0; input < policies.size(); ++input) {
                if (policies[input] != ActivationPolicy::AllowA4) { continue; }
                if (!source->provides_input_divisor()) {
                    throw std::runtime_error(model_.parameters[parameter].name +
                                             ": supply an activation divisor for AllowA4");
                }
                const auto word = activation ? *activation : source->input_divisor();
                if (!valid_positive_f32_word(word)) {
                    throw std::runtime_error(
                        "activation input divisor must be positive finite FP32");
                }
                auxiliaries_[parameter][input] = word;
            }
        }
        if (job.divisors > 1) {
            for (const auto& source : job.sources) {
                job.divisor_words.push_back(source->weight_divisor());
            }
        } else {
            const auto word = job.sources.front()->weight_divisor();
            for (const auto& source : job.sources) {
                if (source->weight_divisor() != word) {
                    throw std::runtime_error("weight divisor changed between row blocks");
                }
            }
            job.divisor_words.push_back(word);
        }
    }

    Plan finish() {
        Plan plan;
        std::uint64_t offset = 0;
        for (const auto& resource : model_.resources) {
            OJson object       = OJson::object();
            object["id"]       = resource.id;
            object["encoding"] = std::string(artifact::kRawBytesEncoding);
            object["offset"]   = offset;
            object["bytes"]    = resource.bytes.size();
            object["kind"]     = "resource";
            plan.objects.push_back(std::move(object));
            plan.resources.push_back({resource.id, &resource.bytes, offset});
            offset += resource.bytes.size();
        }
        const auto tensor_object = [&](const std::string& id, const Shape& shape, QType format,
                                       QuantLayout layout, std::uint64_t bytes,
                                       std::uint64_t divisors) {
            const auto start = align256(offset);
            OJson object     = OJson::object();
            object["id"]     = id;
            object["shape"]  = shape;
            object["format"] = std::string(artifact::format_name(format));
            object["layout"] = std::string(artifact::layout_name(layout));
            object["offset"] = start;
            object["bytes"]  = bytes;
            if (divisors != 1) { object["divisors"] = divisors; }
            object["kind"] = "tensor";
            plan.objects.push_back(std::move(object));
            offset = start + bytes;
            return start;
        };
        for (auto& job : jobs_) {
            job.offset = tensor_object(job.id, job.shape, job.format, job.layout,
                                       job.geometry.bytes, job.divisors);
        }

        for (std::size_t i = 0; i < model_.parameters.size(); ++i) {
            if (recipe_.alias[i]) { continue; }
            const auto& parameter = model_.parameters[i];
            auto parts            = fragments_[i];
            std::uint64_t count   = 0;
            for (const auto& part : parts) { count += part.last - part.first; }
            if (count != shape_elements(parameter.shape)) {
                throw std::runtime_error(parameter.name + ": logical coverage size is incorrect");
            }
            const auto& first = parts.front();
            const auto& job   = jobs_[first.weight];
            if (parts.size() == 1 && first.first == 0 && job.shape == parameter.shape &&
                first.last == shape_elements(job.shape)) {
                plan.bindings[parameter.name] = OJson{{"object", job.id}};
            } else {
                OJson list = OJson::array();
                for (const auto& part : parts) {
                    OJson item     = OJson::object();
                    item["object"] = jobs_[part.weight].id;
                    item["range"]  = OJson::array({part.first, part.last});
                    list.push_back(std::move(item));
                }
                plan.bindings[parameter.name] = OJson{{"parts", std::move(list)}};
            }
        }
        for (const auto i : recipe_.alias_order) {
            const auto target = recipe_.resolve_alias(i);
            plan.bindings[model_.parameters[i].name] =
                plan.bindings[model_.parameters[target].name];
        }

        std::vector<std::pair<std::string, std::uint32_t>> auxiliary_values;
        for (std::size_t i = 0; i < model_.parameters.size(); ++i) {
            const auto& parameter = model_.parameters[i];
            for (std::size_t input = 0; input < parameter.inputs.size(); ++input) {
                OJson use                = OJson::object();
                use["parameter"]         = parameter.name;
                use["input"]             = parameter.inputs[input];
                use["activation_policy"] = std::string(policy_name(recipe_.policies[i][input]));
                const bool auxiliary     = input < auxiliaries_[i].size() && auxiliaries_[i][input];
                // Explicitly shared weights keep independent Use and calibration records.
                if (recipe_.alias[i] && recipe_.policies[i][input] == ActivationPolicy::AllowA4 &&
                    recipe_.selections[recipe_.resolve_alias(i)].format == QType::NVFP4 &&
                    !auxiliary) {
                    throw std::runtime_error(parameter.name +
                                             ": provide its independent activation divisor");
                }
                if (auxiliary) {
                    const auto id = numbered("auxiliary", auxiliary_values.size());
                    auxiliary_values.emplace_back(id, *auxiliaries_[i][input]);
                    use["auxiliaries"] = OJson{{"activation_input_divisor", OJson{{"object", id}}}};
                }
                plan.uses.push_back(std::move(use));
            }
        }
        for (const auto& [id, word] : auxiliary_values) {
            const auto start = tensor_object(id, {}, QType::FP32, QuantLayout::Contiguous, 4, 1);
            plan.auxiliaries.push_back({id, word, start});
        }
        plan.payload_bytes = offset;
        plan.weights       = std::move(jobs_);
        return plan;
    }

    const Model& model_;
    const Recipe& recipe_;
    ExpertLayout expert_layout_;
    std::vector<WeightJob> jobs_;
    std::vector<std::vector<Fragment>> fragments_;
    std::vector<std::vector<std::optional<std::uint32_t>>> auxiliaries_;
};

} // namespace

Plan prepare(const Model& model, const Recipe& recipe, ExpertLayout expert_layout) {
    return Planner(model, recipe, expert_layout).run();
}

} // namespace ninferno::convert
