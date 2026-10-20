#pragma once

// Logical parameters, packing groups and the recipe's per-parameter choices. These mirror
// tools/convert/model.py and recipe.py as data: architecture adapters fill a Model explicitly,
// recipes fill a Recipe explicitly, and the planner turns both into artifact objects.

#include "artifact/schema.h"
#include "convert/pyjson.h"
#include "convert/sources.h"
#include "core/weight.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninferno::convert {

enum class Method { CastDirect, GroupedAbsmax, Fp8RowMaxabs, ImportEncoded };
[[nodiscard]] std::string_view method_name(Method method) noexcept;

using artifact::ActivationPolicy;
[[nodiscard]] std::string_view policy_name(ActivationPolicy policy) noexcept;

struct Parameter {
    std::string name;
    Shape shape;
    SourcePtr source; // the default source (the Python factory with format None)
    // The adapter's source factory (Python's source_factory): the parameter read from a given
    // checkpoint, in a given encoding (Auto: Python's format None). Empty when the parameter has
    // no checkpoint mapping (gathered heads, id arrays).
    std::function<SourcePtr(const Checkpoint&, MatrixFormat)> encoded_source;
    std::vector<std::string> inputs;
    QType direct          = QType::BF16;
    std::string residency = "text";
    // Checkpoint prefix below the text root, probed for encoded sites; empty if none.
    std::string probe_leaf;
    // Shape of the matrix at probe_leaf when it differs from `shape` (a stacked expert gate_up
    // is probed through its gate half).
    Shape probe_shape;

    [[nodiscard]] bool projection() const noexcept { return !inputs.empty(); }
};

struct Resource {
    std::string id;
    std::string bytes;
};

struct Model {
    OJson components = OJson::object();
    std::vector<Parameter> parameters;
    std::map<std::string, std::size_t, std::less<>> index;
    std::vector<Resource> resources;
    std::vector<std::vector<std::size_t>> packing_groups;
    std::vector<std::size_t> expert_banks; // packing groups that are one layer's expert bank
    std::uint64_t token_count = 0;
    std::vector<std::int64_t> special_token_ids;
    // Resources that did not come from the checkpoint directory (rebuilt from GGUF metadata or
    // derived from config.json), one line each.
    std::vector<std::string> resource_notes;

    [[nodiscard]] OJson& config() { return components["text"]["config"]; }

    [[nodiscard]] const OJson& config() const { return components.at("text").at("config"); }

    std::size_t add(Parameter parameter);
    [[nodiscard]] std::size_t at(std::string_view name) const;
    // Packs the named parameters together; the group's index in packing_groups.
    std::size_t add_packing_group(const std::vector<std::string>& names);
};

struct Selection {
    QType format  = QType::BF16;
    Method method = Method::CastDirect;
    SourcePtr source;
};

struct Recipe {
    std::vector<Selection> selections;                   // per parameter
    std::vector<std::vector<ActivationPolicy>> policies; // per parameter, per input
    std::vector<std::vector<std::size_t>> explicit_groups;
    // Recipe.share: a parameter bound to another's objects, keeping its own Use records. The
    // order is the order Python's aliases dict iterates, which orders their bindings.
    std::vector<std::optional<std::size_t>> alias;
    std::vector<std::size_t> alias_order;
    // Recipe.separate: never packed with the parameters of a model packing group.
    std::vector<bool> separate;

    explicit Recipe(const Model& model);
    // Recipe.add_parameter: a parameter the model gained after the recipe was made.
    void add_parameter(const Model& model, std::size_t parameter);
    // Recipe.assign(name, format=..., method=..., source=...) for one whole parameter.
    void assign(std::size_t parameter, QType format, Method method, SourcePtr source = nullptr);
    // Recipe.assign(name, source=...): only the source changes.
    void assign_source(std::size_t parameter, SourcePtr source);
    void set_policy(std::size_t parameter, ActivationPolicy policy);
    void share(const Model& model, std::size_t parameter, std::size_t target);
    void separate_parameter(std::size_t parameter);
    // The parameter an alias chain ends at.
    [[nodiscard]] std::size_t resolve_alias(std::size_t parameter) const;
};

// A tied output head (tie_word_embeddings) whose selection encodes the same source values the
// same way as the token embedding's is bound to the embedding's objects (Recipe.share) instead of
// being written a second time; the head keeps its own Use records. Nothing is shared when the two
// differ in format, method or source (a scaled embedding, a head the recipe quantizes apart), when
// either is already an alias or a member of a packing group, or when the head admits A4 on NVFP4
// (that use needs its own activation divisor). Returns whether the head was shared.
bool share_tied_head(const Model& model, Recipe& recipe);

// Named checkpoints: the base (--model) and --source NAME=PATH entries, each opened when a
// component or recipe first asks for it (__main__.SourceInputs). Provenance lists the opened
// ones in opening order.
class SourceInputs {
public:
    SourceInputs(const Checkpoint& base,
                 std::vector<std::pair<std::string, std::filesystem::path>> paths);

    [[nodiscard]] const Checkpoint& get(const std::string& name);

    [[nodiscard]] const Checkpoint& base() const noexcept { return base_; }

    [[nodiscard]] OJson provenance() const;
    // Every opened checkpoint, the base first.
    [[nodiscard]] std::vector<const Checkpoint*> opened() const;

private:
    const Checkpoint& base_;
    std::vector<std::pair<std::string, std::filesystem::path>> paths_;
    std::vector<std::pair<std::string, std::unique_ptr<Checkpoint>>> opened_;
};

} // namespace ninferno::convert
