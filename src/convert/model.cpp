#include "convert/model.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace ninferno::convert {

std::string_view method_name(Method method) noexcept {
    switch (method) {
    case Method::CastDirect:
        return "cast_direct";
    case Method::GroupedAbsmax:
        return "grouped_absmax";
    case Method::Fp8RowMaxabs:
        return "fp8_row_maxabs";
    case Method::ImportEncoded:
        return "import_encoded";
    }
    return {};
}

std::string_view policy_name(ActivationPolicy policy) noexcept {
    switch (policy) {
    case ActivationPolicy::A16Only:
        return "A16Only";
    case ActivationPolicy::AllowA8:
        return "AllowA8";
    case ActivationPolicy::AllowA4:
        return "AllowA4";
    }
    return {};
}

std::size_t Model::add(Parameter parameter) {
    if (index.contains(parameter.name)) {
        throw std::runtime_error("duplicate logical parameter " + parameter.name);
    }
    if (parameter.source->shape() != parameter.shape) {
        throw std::runtime_error(parameter.name + ": source shape differs from logical parameter");
    }
    index.emplace(parameter.name, parameters.size());
    parameters.push_back(std::move(parameter));
    return parameters.size() - 1;
}

std::size_t Model::at(std::string_view name) const {
    const auto it = index.find(name);
    if (it == index.end()) {
        throw std::runtime_error("unknown logical parameter " + std::string(name));
    }
    return it->second;
}

std::size_t Model::add_packing_group(const std::vector<std::string>& names) {
    std::vector<std::size_t> group;
    for (const auto& name : names) { group.push_back(at(name)); }
    packing_groups.push_back(std::move(group));
    return packing_groups.size() - 1;
}

Recipe::Recipe(const Model& model) {
    for (std::size_t i = 0; i < model.parameters.size(); ++i) { add_parameter(model, i); }
}

void Recipe::add_parameter(const Model& model, std::size_t parameter) {
    if (parameter != selections.size()) {
        throw std::runtime_error("recipe already contains " + model.parameters.at(parameter).name);
    }
    const auto& item = model.parameters.at(parameter);
    selections.push_back({item.direct, Method::CastDirect, item.source});
    policies.emplace_back(item.inputs.size(), ActivationPolicy::A16Only);
    alias.emplace_back();
    separate.push_back(false);
}

namespace {

void drop_alias(Recipe& recipe, std::size_t parameter) {
    if (!recipe.alias[parameter]) { return; }
    recipe.alias[parameter].reset();
    std::erase(recipe.alias_order, parameter);
}

} // namespace

void Recipe::assign(std::size_t parameter, QType format, Method method, SourcePtr source) {
    auto& selection = selections.at(parameter);
    drop_alias(*this, parameter);
    selection.format = format;
    selection.method = method;
    if (source) {
        if (source->shape() != selection.source->shape()) {
            throw std::runtime_error("override source shape differs from logical parameter");
        }
        selection.source = std::move(source);
    }
}

void Recipe::assign_source(std::size_t parameter, SourcePtr source) {
    auto& selection = selections.at(parameter);
    if (source->shape() != selection.source->shape()) {
        throw std::runtime_error("override source shape differs from logical parameter");
    }
    drop_alias(*this, parameter);
    selection.source = std::move(source);
}

void Recipe::share(const Model& model, std::size_t parameter, std::size_t target) {
    if (parameter >= selections.size() || target >= selections.size()) {
        throw std::runtime_error("shared parameters must exist");
    }
    if (model.parameters[parameter].shape != model.parameters[target].shape) {
        throw std::runtime_error("shared parameters must have equal logical shapes");
    }
    std::set<std::size_t> visited{parameter};
    auto current = target;
    while (alias[current]) {
        if (visited.contains(current)) { throw std::runtime_error("cyclic weight sharing"); }
        visited.insert(current);
        current = *alias[current];
    }
    if (visited.contains(current)) { throw std::runtime_error("cyclic weight sharing"); }
    if (!alias[parameter]) { alias_order.push_back(parameter); }
    alias[parameter]      = target;
    selections[parameter] = selections[target];
}

void Recipe::separate_parameter(std::size_t parameter) { separate.at(parameter) = true; }

std::size_t Recipe::resolve_alias(std::size_t parameter) const {
    while (alias[parameter]) { parameter = *alias[parameter]; }
    return parameter;
}

void Recipe::set_policy(std::size_t parameter, ActivationPolicy policy) {
    for (auto& value : policies.at(parameter)) { value = policy; }
}

bool share_tied_head(const Model& model, Recipe& recipe) {
    if (!model.components.contains("text")) { return false; }
    const auto& config = model.config();
    const auto tied    = config.find("tie_word_embeddings");
    if (tied == config.end() || !tied->is_boolean() || !tied->get<bool>()) { return false; }
    const auto embedding = model.index.find("text/token_embedding");
    const auto head      = model.index.find("text/output_head");
    if (embedding == model.index.end() || head == model.index.end()) { return false; }
    const auto e = embedding->second;
    const auto h = head->second;
    if (recipe.alias[e] || recipe.alias[h]) { return false; }
    const auto& table = recipe.selections[e];
    const auto& rows  = recipe.selections[h];
    if (table.format != rows.format || table.method != rows.method || !table.source ||
        !rows.source) {
        return false;
    }
    if (table.source != rows.source &&
        (table.source->label() != rows.source->label() ||
         table.source->shape() != rows.source->shape() ||
         table.source->encoding() != rows.source->encoding())) {
        return false;
    }
    const auto member = [&](const std::vector<std::size_t>& group) {
        return std::find(group.begin(), group.end(), e) != group.end() ||
               std::find(group.begin(), group.end(), h) != group.end();
    };
    if (std::any_of(model.packing_groups.begin(), model.packing_groups.end(), member) ||
        std::any_of(recipe.explicit_groups.begin(), recipe.explicit_groups.end(), member)) {
        return false;
    }
    if (rows.format == QType::NVFP4 &&
        std::find(recipe.policies[h].begin(), recipe.policies[h].end(),
                  ActivationPolicy::AllowA4) != recipe.policies[h].end()) {
        return false;
    }
    recipe.share(model, h, e);
    return true;
}

SourceInputs::SourceInputs(const Checkpoint& base,
                           std::vector<std::pair<std::string, std::filesystem::path>> paths)
    : base_(base), paths_(std::move(paths)) {
    std::set<std::string> names;
    for (const auto& [name, path] : paths_) {
        if (name.empty() || path.empty() || !names.insert(name).second) {
            throw std::runtime_error("source requires unique NAME=PATH entries");
        }
        if (name == "base") { throw std::runtime_error("select the base source with --model"); }
    }
}

const Checkpoint& SourceInputs::get(const std::string& name) {
    if (name == "base") { return base_; }
    for (const auto& [opened, checkpoint] : opened_) {
        if (opened == name) { return *checkpoint; }
    }
    for (const auto& [known, path] : paths_) {
        if (known == name) {
            opened_.emplace_back(name, std::make_unique<Checkpoint>(path));
            return *opened_.back().second;
        }
    }
    throw std::runtime_error("selected recipe requires source '" + name + "'; provide --source " +
                             name + "=PATH");
}

OJson SourceInputs::provenance() const {
    OJson out   = OJson::object();
    out["base"] = OJson{{"path", base_.path_string()}};
    for (const auto& [name, checkpoint] : opened_) {
        out[name] = OJson{{"path", checkpoint->path_string()}};
    }
    return out;
}

std::vector<const Checkpoint*> SourceInputs::opened() const {
    std::vector<const Checkpoint*> out{&base_};
    for (const auto& [name, checkpoint] : opened_) { out.push_back(checkpoint.get()); }
    return out;
}

} // namespace ninferno::convert
