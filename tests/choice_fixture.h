#pragma once

// The choice fixtures' records as the configuration choice reads them
// (tests/fixtures/choice/*.json): a model, a model at one quantization (its linears split by their
// roles' formats), a card and a quantization. One reader for the tests that price them
// (ninferno_choice_test, ninferno_choice_prefill_routes_test).

#include "product/choice/choice.h"

#include "ninferno/types.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace choice_fixture {

namespace ch = ninferno::product::choice;
using ninferno::KvCacheStorage;
using ninferno::RecurrentStateStorage;
using nlohmann::json;

inline ch::ModelFacts model_of(const json& m) {
    ch::ModelFacts out;
    out.family               = m.at("family").get<std::string>();
    out.params               = m.at("params").get<double>();
    out.active_params        = m.at("active_params").get<double>();
    out.experts              = m.at("experts").get<int>();
    out.experts_per_token    = m.at("experts_per_token").get<int>();
    out.routed_expert_params = m.at("routed_expert_params").get<double>();
    out.input_table_params   = m.at("input_table_params").get<double>();
    const json kv            = m.at("kv_bytes_per_token");
    out.kv_bytes_per_token   = [kv](KvCacheStorage s) {
        return kv.at(std::string(ninferno::kv_cache_storage_name(s))).get<std::uint64_t>();
    };
    const json st   = m.at("state_bytes");
    out.state_bytes = [st](RecurrentStateStorage s) {
        return st.value(std::string(ninferno::recurrent_state_storage_name(s)),
                        st.at("f16").get<std::uint64_t>());
    };
    out.recurrent                = m.value("recurrent", true);
    out.full_attention_width     = m.value("full_attention_width", 0.0);
    out.sliding_attention_width  = m.value("sliding_attention_width", 0.0);
    out.layers                   = m.value("layers", 0U);
    out.hidden                   = m.value("hidden", 0.0);
    out.full_layers              = m.value("full_layers", 0U);
    out.full_head_dim            = m.value("full_head_dim", 0.0);
    out.full_kv_heads            = m.value("full_kv_heads", 0.0);
    out.sliding_layers           = m.value("sliding_layers", 0U);
    out.sliding_head_dim         = m.value("sliding_head_dim", 0.0);
    out.sliding_kv_heads         = m.value("sliding_kv_heads", 0.0);
    out.recurrent_state_elements = m.value("recurrent_state_elements", 0.0);
    if (m.contains("linears")) {
        for (const json& l : m.at("linears")) {
            out.linears.push_back({l.at(0).get<double>(), l.at(1).get<double>(),
                                   l.at(2).get<double>(), false,
                                   l.size() > 3 && l.at(3).get<int>() != 0});
        }
    }
    if (m.contains("ring_bytes_per_token")) {
        const json ring          = m.at("ring_bytes_per_token");
        out.ring_bytes_per_token = [ring](KvCacheStorage s) {
            return ring.at(std::string(ninferno::kv_cache_storage_name(s))).get<std::uint64_t>();
        };
        out.window = m.at("window").get<std::uint32_t>();
    }
    if (!out.recurrent) { out.state_bytes = nullptr; }
    return out;
}

// The model at one quantization: its input table and speculation-only weights as stored.
inline ch::ModelFacts model_at(const json& m, const std::string& quant) {
    ch::ModelFacts out    = model_of(m);
    const json& q         = m.at("quants").at(quant);
    out.input_table_bytes = q.value("input_table_bytes", 0.0);
    // Each linear a format by its role's shares, as plan_facts.cpp splits a GGUF's.
    if (q.contains("role_formats") && m.contains("linear_roles")) {
        std::vector<ch::LinearShape> split;
        for (std::size_t i = 0; i < out.linears.size(); ++i) {
            const json& shares =
                q.at("role_formats").at(m.at("linear_roles").at(i).get<std::string>());
            for (const auto& [format, share] : shares.items()) {
                ch::LinearShape part = out.linears[i];
                part.per_token *= share.get<double>();
                part.format = format;
                split.push_back(part);
            }
        }
        out.linears = std::move(split);
    }
    return out;
}

inline ch::CardFacts card_of(const json& c) {
    ch::CardFacts out;
    out.major            = c.at("major").get<int>();
    out.minor            = c.at("minor").get<int>();
    out.sms              = c.at("sms").get<int>();
    out.clock_hz         = c.at("clock_hz").get<double>();
    out.dram_bytes_per_s = c.at("dram_bytes_per_s").get<double>();
    out.fp4              = c.at("fp4").get<bool>();
    return out;
}

inline ch::QuantOption quant_of(const std::string& label, const json& q) {
    ch::QuantOption out;
    out.label           = label;
    out.bits_per_weight = q.at("bits_per_weight").get<double>();
    // What a decode step reads (product::quant_option_of): the file less its MTP layers and
    // drafter.
    out.weight_bytes = q.at("weight_bytes").get<double>() - q.value("speculation_bytes", 0.0);
    out.weights      = ch::label_class(label);
    // An MoE's routed banks as its GGUF's tensor table stores them (ffn_*_exps).
    out.routed_expert_bytes = q.value("routed_expert_bytes", 0.0);
    return out;
}

} // namespace choice_fixture
