#include "artifact/binder.h"
#include "artifact/file_io.h"
#include "artifact/reader.h"
#include "artifact/fixture.h"
#include "core/weight_view.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::artifact;
using namespace ninferno::test::artifact_fixture;

void file_set_and_bindings() {
    Fixture fixture;
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto bytes = reader.read_object(reader.find("q5"));
    require(bytes.size() == 528 &&
                std::equal(bytes.begin(), bytes.end(), fixture.payload.begin() + 256),
            "cross-file parent bytes changed");
    require(reader.read_range(1344, 0).empty(), "empty terminal range failed");
    rejects([&] { (void)reader.read_range(1344, 1); }, "out-of-range read accepted");
    rejects([&] { (void)reader.geometry(reader.find("unused")); },
            "unknown selected encoding accepted");
    rejects([&] { (void)reader.read_range(1280, 64); }, "required missing part accepted");

    Binder binder(reader);
    const auto full = binder.parameter("matrix", {2, 130});
    const auto row  = binder.parameter("row", {1, 130});
    require(full.binding.parts[0].object == row.binding.parts[0].object, "shared parent lost");
    rejects([&] { (void)binder.parameter("matrix", {1, 260}); },
            "whole-object shape was silently reshaped");
    const auto reshaped = binder.parameter("row", {130});
    require(reshaped.binding.elements == 130, "Part reshape changed logical elements");
    const auto context =
        binder.values(binder.use("row", "context").auxiliaries.at("input_divisor"));
    const auto query = binder.values(binder.use("row", "query").auxiliaries.at("input_divisor"));
    require(std::bit_cast<std::uint32_t>(query.scalar_f32()) == 0x40000000 &&
                std::bit_cast<std::uint32_t>(context.scalar_f32()) == 0x40400000,
            "Use values were mixed");
    require(binder.use("row", "query").activation_policy == ActivationPolicy::AllowA4 &&
                binder.use("row", "context").activation_policy == ActivationPolicy::AllowA8,
            "Use policy changed");
    (void)binder.resource("text", "tokenizer.json");
    auto plan = std::move(binder).finish();
    require(plan.device_objects.size() == 1 && plan.device_capacity_bytes == 528 &&
                plan.host_objects.size() == 1,
            "selected parent deduplication or residency failed");
}

// Strict load: a binding the artifact carries in a component the model loads, but the
// model never binds, is refused by name; a component the model binds nothing of is listed.
void unused_bindings() {
    Fixture fixture;
    fixture.root["bindings"]["text/extra"] = {{"object", "divisors"}};
    fixture.root["bindings"]["text/head"]  = {{"object", "divisors"}};
    fixture.root["bindings"]["mtp/head"]   = {{"object", "divisors"}};
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto bind = [&](std::initializer_list<const char*> names) {
        Binder binder(reader);
        for (const char* name : names) {
            (void)binder.parameter(name,
                                   name == std::string("matrix") ? Shape{2, 130}
                                   : name == std::string("row")  ? Shape{1, 130}
                                                                 : Shape{2},
                                   Residency::Host);
        }
        return binder;
    };
    // Everything the loaded components carry is bound; mtp is not loaded and is listed.
    auto all        = bind({"matrix", "row", "values", "text/extra", "text/head"});
    const auto plan = std::move(all).finish(UnusedBindings::RefuseInLoadedComponents);
    require(plan.unbound_bindings == std::vector<std::string>{"mtp/head"},
            "an unloaded component's bindings were not listed");
    // The control: text is loaded but text/extra is never bound.
    std::string message;
    try {
        auto partial = bind({"matrix", "row", "values", "text/head"});
        (void)std::move(partial).finish(UnusedBindings::RefuseInLoadedComponents);
    } catch (const ArtifactError& error) { message = error.what(); }
    require(message.find("text/extra") != std::string::npos &&
                message.find("text/head") == std::string::npos,
            "an unbound binding of a loaded component was not refused by name");
    // The same partial load under the permissive policy only lists nothing of text.
    auto permissive = bind({"matrix", "row", "values", "text/head"});
    require(std::move(permissive).finish().unbound_bindings == std::vector<std::string>{"mtp/head"},
            "the permissive policy changed what it lists");
    std::cout << "strict load: refused \"" << message << "\"\n";
}

void geometry_and_views() {
    const std::array<std::uint16_t, 12> direct_words{
        0, 0x8000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120};
    const auto* direct_bytes = reinterpret_cast<const std::byte*>(direct_words.data());
    const WeightParent direct_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                                     std::array<std::uint64_t, 3>{2, 2, 3}),
                                     direct_bytes};
    const WeightView direct_view{{2, 3}, {{&direct_parent, 3, 6}, {&direct_parent, 6, 9}}};
    const auto direct = native_weight(direct_view);
    require(direct.n == 2 && direct.k == 3 && direct.padded_shape[1] == 3 &&
                direct.qdata == direct_bytes + 6 && direct.payload == direct_bytes &&
                direct.payload_bytes == sizeof(direct_words),
            "Direct reshape lost its logical shape, element offset or owning parent");
    Fixture fixture;
    WeightParent parent{weight_geometry(QType::Q5_G64_FP16, QuantLayout::RowSplit,
                                        std::array<std::uint64_t, 2>{2, 130}),
                        fixture.payload.data() + 256};
    const WeightView row{{1, 130}, {{&parent, 130, 260}}};
    const auto planes = weight_row_planes(row.parts.front());
    require(parent.geometry.bytes == 528 && planes.codes == parent.data + 128 &&
                planes.high == parent.data + 288 && planes.scales == parent.data + 520,
            "Q5 view was interpreted as a standalone payload");
    const auto weight = native_weight(row);
    rejects<std::invalid_argument>([&] { (void)native_weight(WeightView{{2, 65}, row.parts}); },
                                   "quantized view changed K without preserving encoding groups");
    require(weight.qdata == planes.codes && weight.qhigh == planes.high &&
                weight.scales == planes.scales,
            "native RowSplit view lost its planes");
    require(std::to_integer<int>(planes.codes[0]) == 0x52 &&
                std::to_integer<int>(planes.scales[1]) == 0x40,
            "row view selected the wrong represented values");
    // A row prefix (a drafter reading the leading rows of a ranked head): the same parent, its
    // planes where the whole matrix has them, only the row count cut.
    const WeightView whole{{2, 130}, {{&parent, 0, 260}}};
    const auto first = weight_row_prefix(whole, 1);
    require(first && first->shape == std::vector<std::uint64_t>{1, 130} &&
                first->parts.size() == 1 && first->parts[0].parent == &parent &&
                first->parts[0].begin == 0 && first->parts[0].end == 130,
            "a row prefix is not the leading rows of its parent");
    const auto full_weight = native_weight(whole);
    const auto prefix      = native_weight(*first);
    require(prefix.n == 1 && prefix.k == 130 && prefix.qdata == full_weight.qdata &&
                prefix.qhigh == full_weight.qhigh && prefix.scales == full_weight.scales &&
                prefix.scale_nb[1] == full_weight.scale_nb[1],
            "a RowSplit row prefix moved its planes or strides");
    require(!weight_row_prefix(whole, 2) && !weight_row_prefix(whole, 0),
            "a prefix of all rows or none was made");

    std::vector<std::byte> fp8(16896);
    WeightParent fp8_parent{weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
                                            std::array<std::uint64_t, 2>{256, 64}),
                            fp8.data()};
    WeightView fp8_rows{{128, 64}, {{&fp8_parent, 0, 128 * 64}}};
    require(weight_row_planes(fp8_rows.parts.front()).scales == fp8.data() + 16384,
            "FP8 view recomputed the parent scale base");
    rejects<std::invalid_argument>([&] { (void)native_weight(fp8_rows); },
                                   "complete-parent ABI accepted an FP8 submatrix");
    require(!weight_row_prefix(WeightView{{256, 64}, {{&fp8_parent, 0, 256 * 64}}}, 128),
            "an FP8 row-scaled parent (read whole) gave a row prefix");

    std::vector<std::byte> nvfp4(4612);
    WeightParent nv_parent{weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                           std::array<std::uint64_t, 2>{128, 64}),
                           nvfp4.data(), 2.0F};
    require(weight_scale_offset(nv_parent.geometry, 31, 0) == 4096 + 496 &&
                weight_scale_offset(nv_parent.geometry, 32, 0) == 4096 + 4,
            "NVFP4 parent swizzle origin was lost");
    const auto nv_rows = weight_row_planes({&nv_parent, 31 * 64, 33 * 64});
    require(nv_rows.swizzled_scales && nv_rows.row_begin == 31 &&
                nv_rows.scales == nvfp4.data() + 4096,
            "NVFP4 submatrix lost its parent coordinates");
    const WeightView complete{{128, 64}, {{&nv_parent, 0, 128 * 64}}};
    const auto query   = native_weight(complete, 3.0F);
    const auto context = native_weight(complete, 4.0F);
    require(query.weight_scale_divisor == 2 && query.input_scale_divisor == 3 &&
                context.input_scale_divisor == 4 && nv_parent.weight_scale_divisor == 2,
            "per-use native parameters changed the parent");

    // MXFP4: codes [N,K/2], then at a 256-byte boundary the E8M0 exponents [N,K/32] row-major.
    const auto mx_geometry = weight_geometry(QType::MXFP4, QuantLayout::BlockScaleK32E8M0,
                                             std::array<std::uint64_t, 2>{3, 96});
    require(mx_geometry.code_bytes == 144 && mx_geometry.scale_offset == 256 &&
                mx_geometry.scale_bytes == 9 && mx_geometry.bytes == 265 &&
                mx_geometry.group_size == 32,
            "MXFP4 plane geometry");
    require(weight_scale_offset(mx_geometry, 2, 1) == 256 + 7, "MXFP4 exponent address");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::MXFP4, QuantLayout::BlockScaleK32E8M0,
                                  std::array<std::uint64_t, 2>{3, 80});
        },
        "MXFP4 accepted K off the 32-value exponent group");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK32E8M0,
                                  std::array<std::uint64_t, 2>{128, 64});
        },
        "the MXFP4 layout accepted another format");
    std::vector<std::byte> mx(265);
    WeightParent mx_parent{mx_geometry, mx.data()};
    const WeightView mx_row{{1, 96}, {{&mx_parent, 96, 192}}};
    const auto mx_weight = native_weight(mx_row);
    require(mx_weight.qtype == QType::MXFP4 && mx_weight.qdata == mx.data() + 48 &&
                mx_weight.scales == mx.data() + 256 + 3 && mx_weight.n == 1 &&
                mx_weight.scale_dtype == DType::U8,
            "an MXFP4 row view is not a standalone matrix");

    // FP8 with binary32 row multipliers: row_scale_v1 with four-byte scale words.
    const auto r32_geometry = weight_geometry(QType::FP8_E4M3FN_ROW_F32, QuantLayout::RowScale,
                                              std::array<std::uint64_t, 2>{3, 96});
    require(r32_geometry.code_bytes == 288 && r32_geometry.scale_offset == 512 &&
                r32_geometry.scale_bytes == 12 && r32_geometry.bytes == 524,
            "FP8 row-f32 plane geometry");
    require(weight_scale_offset(r32_geometry, 2, 0) == 512 + 8, "FP8 row-f32 scale address");
    std::vector<std::byte> r32(524);
    WeightParent r32_parent{r32_geometry, r32.data()};
    const auto r32_weight = native_weight(WeightView{{3, 96}, {{&r32_parent, 0, 288}}});
    require(r32_weight.scale_dtype == DType::FP32 && r32_weight.scale_nb[0] == 4 &&
                r32_weight.scales == r32.data() + 512,
            "FP8 row-f32 native weight");

    // FP8 with one binary32 multiplier per 128x128 block: [ceil(N/128), ceil(K/128)] words.
    const auto b128_geometry = weight_geometry(QType::FP8_E4M3FN_B128_F32,
                                               QuantLayout::BlockScale128F32,
                                               std::array<std::uint64_t, 2>{200, 300});
    require(b128_geometry.code_bytes == 60000 && b128_geometry.scale_offset == 60160 &&
                b128_geometry.scale_bytes == 2 * 3 * 4 && b128_geometry.bytes == 60184,
            "FP8 block plane geometry");
    require(weight_scale_offset(b128_geometry, 130, 2) == 60160 + (1 * 3 + 2) * 4,
            "FP8 block multiplier address");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::BlockScale128F32,
                                  std::array<std::uint64_t, 2>{128, 128});
        },
        "the FP8 block layout accepted another format");
    std::vector<std::byte> b128(60184);
    WeightParent b128_parent{b128_geometry, b128.data()};
    rejects<std::invalid_argument>(
        [&] { (void)native_weight(WeightView{{100, 300}, {{&b128_parent, 0, 30000}}}); },
        "an FP8 block row view became a native weight");
    const auto b128_weight = native_weight(WeightView{{200, 300}, {{&b128_parent, 0, 60000}}});
    require(b128_weight.scale_dtype == DType::FP32 && b128_weight.scales == b128.data() + 60160 &&
                b128_weight.scale_ne[0] == 3 && b128_weight.scale_ne[1] == 2,
            "FP8 block native weight");

    // U4 with zero points: codes [N, K/2], zero point bytes [N, K/128], binary16 scales.
    const auto u4_geometry = weight_geometry(QType::U4_G128_ZP_FP16, QuantLayout::RowGroupZp,
                                             std::array<std::uint64_t, 2>{3, 256});
    require(u4_geometry.code_bytes == 384 && u4_geometry.high_offset == 512 &&
                u4_geometry.high_bytes == 6 && u4_geometry.scale_offset == 768 &&
                u4_geometry.scale_bytes == 12 && u4_geometry.bytes == 780,
            "U4 plane geometry");
    require(weight_scale_offset(u4_geometry, 2, 1) == 768 + 2 * 4 + 2, "U4 scale address");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::U4_G128_ZP_FP16, QuantLayout::RowGroupZp,
                                  std::array<std::uint64_t, 2>{3, 192});
        },
        "the U4 layout accepted K that is not whole groups");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::Q4_G64_FP16, QuantLayout::RowGroupZp,
                                  std::array<std::uint64_t, 2>{3, 256});
        },
        "the U4 layout accepted another format");
    // The same planes with 32-value groups: K = 96 is three groups per row.
    const auto u4g32_geometry = weight_geometry(QType::U4_G32_ZP_FP16, QuantLayout::RowGroupZp,
                                                std::array<std::uint64_t, 2>{3, 96});
    require(u4g32_geometry.group_size == 32 && u4g32_geometry.code_bytes == 144 &&
                u4g32_geometry.high_offset == 256 && u4g32_geometry.high_bytes == 9 &&
                u4g32_geometry.scale_offset == 512 && u4g32_geometry.scale_bytes == 18 &&
                u4g32_geometry.bytes == 530,
            "U4 G32 plane geometry");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::U4_G64_ZP_FP16, QuantLayout::RowGroupZp,
                                  std::array<std::uint64_t, 2>{3, 96});
        },
        "the U4 G64 layout accepted K that is not whole groups");
    std::vector<std::byte> u4(780);
    WeightParent u4_parent{u4_geometry, u4.data()};
    const auto u4_weight = native_weight(WeightView{{2, 256}, {{&u4_parent, 256, 768}}});
    require(u4_weight.qtype == QType::U4_G128_ZP_FP16 && u4_weight.n == 2 &&
                u4_weight.qdata == u4.data() + 128 && u4_weight.qhigh == u4.data() + 512 + 2 &&
                u4_weight.scales == u4.data() + 768 + 4 && u4_weight.scale_dtype == DType::FP16 &&
                u4_weight.high_plane_bytes == 6,
            "a U4 row view is not a standalone matrix");
    require(u4_weight.k_order == nullptr, "a natural-order U4 view carries a column order");
    // Act-order: the same planes, then at a 256-byte boundary K int32 column indices, which
    // every row view shares.
    const auto ordered_geometry = weight_geometry(
        QType::U4_G32_ZP_FP16, QuantLayout::RowGroupZpOrdered, std::array<std::uint64_t, 2>{3, 96});
    require(ordered_geometry.scale_offset == 512 && ordered_geometry.scale_bytes == 18 &&
                ordered_geometry.order_offset == 768 && ordered_geometry.bytes == 768 + 4 * 96,
            "ordered U4 plane geometry");
    std::vector<std::byte> ordered(ordered_geometry.bytes);
    WeightParent ordered_parent{ordered_geometry, ordered.data()};
    const auto ordered_weight =
        native_weight(WeightView{{1, 96}, {{&ordered_parent, 96, 192}}});
    require(ordered_weight.layout == QuantLayout::RowGroupZpOrdered && ordered_weight.n == 1 &&
                ordered_weight.qdata == ordered.data() + 48 &&
                ordered_weight.k_order == ordered.data() + 768,
            "an ordered U4 row view lost its column order");

    // ggml K-quants: Q4_K/Q5_K blocks unchanged (one plane); Q6_K as code, scale and d planes.
    const auto q4k_geometry = weight_geometry(QType::GGML_Q4_K, QuantLayout::GgmlBlocks,
                                              std::array<std::uint64_t, 2>{3, 512});
    require(q4k_geometry.code_bytes == 3 * 2 * 144 && q4k_geometry.bytes == 864 &&
                q4k_geometry.scale_bytes == 0 && q4k_geometry.high_bytes == 0,
            "ggml Q4_K plane geometry");
    const auto q6k_geometry = weight_geometry(QType::GGML_Q6_K, QuantLayout::GgmlPlanes,
                                              std::array<std::uint64_t, 2>{3, 512});
    require(q6k_geometry.code_bytes == 1152 && q6k_geometry.high_offset == 1280 &&
                q6k_geometry.high_bytes == 96 && q6k_geometry.scale_offset == 1536 &&
                q6k_geometry.scale_bytes == 12 && q6k_geometry.bytes == 1548,
            "ggml Q6_K plane geometry");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::GGML_Q5_K, QuantLayout::GgmlBlocks,
                                  std::array<std::uint64_t, 2>{3, 384});
        },
        "the ggml block layout accepted K that is not whole superblocks");
    rejects<std::invalid_argument>(
        [] {
            (void)weight_geometry(QType::GGML_Q6_K, QuantLayout::GgmlBlocks,
                                  std::array<std::uint64_t, 2>{3, 256});
        },
        "the ggml block layout accepted Q6_K");
    std::vector<std::byte> q6k(1548);
    WeightParent q6k_parent{q6k_geometry, q6k.data()};
    const auto q6k_weight = native_weight(WeightView{{1, 512}, {{&q6k_parent, 1024, 1536}}});
    require(q6k_weight.qdata == q6k.data() + 2 * 384 &&
                q6k_weight.qhigh == q6k.data() + 1280 + 64 &&
                q6k_weight.scales == q6k.data() + 1536 + 8 && q6k_weight.n == 1,
            "a ggml Q6_K row view is not a standalone matrix");
    std::vector<std::byte> q4k(864);
    WeightParent q4k_parent{q4k_geometry, q4k.data()};
    const auto q4k_weight = native_weight(WeightView{{2, 512}, {{&q4k_parent, 512, 1536}}});
    require(q4k_weight.qdata == q4k.data() + 288 && q4k_weight.scales == nullptr &&
                q4k_weight.qhigh == nullptr,
            "a ggml Q4_K row view is not a standalone matrix");
    // Q3_K: 96 code bytes, 12 scale bytes, d per block; Q2_K: 64, 16, d and dmin.
    const auto q3k_geometry = weight_geometry(QType::GGML_Q3_K, QuantLayout::GgmlPlanes,
                                              std::array<std::uint64_t, 2>{3, 512});
    require(q3k_geometry.code_bytes == 576 && q3k_geometry.high_offset == 768 &&
                q3k_geometry.high_bytes == 72 && q3k_geometry.scale_offset == 1024 &&
                q3k_geometry.scale_bytes == 12 && q3k_geometry.bytes == 1036,
            "ggml Q3_K plane geometry");
    const auto q2k_geometry = weight_geometry(QType::GGML_Q2_K, QuantLayout::GgmlPlanes,
                                              std::array<std::uint64_t, 2>{3, 512});
    require(q2k_geometry.code_bytes == 384 && q2k_geometry.high_offset == 512 &&
                q2k_geometry.high_bytes == 96 && q2k_geometry.scale_offset == 768 &&
                q2k_geometry.scale_bytes == 24 && q2k_geometry.bytes == 792,
            "ggml Q2_K plane geometry");
    std::vector<std::byte> q2k(792);
    WeightParent q2k_parent{q2k_geometry, q2k.data()};
    const auto q2k_weight = native_weight(WeightView{{1, 512}, {{&q2k_parent, 1024, 1536}}});
    require(q2k_weight.qdata == q2k.data() + 2 * 128 && q2k_weight.qhigh == q2k.data() + 512 + 64 &&
                q2k_weight.scales == q2k.data() + 768 + 16 && q2k_weight.n == 1,
            "a ggml Q2_K row view is not a standalone matrix");
    // IQ4_XS: 128 code bytes, 4 scales_l bytes, d and scales_h per 256-value block.
    const auto iq4xs_geometry = weight_geometry(QType::GGML_IQ4_XS, QuantLayout::GgmlPlanes,
                                                std::array<std::uint64_t, 2>{3, 512});
    require(iq4xs_geometry.code_bytes == 768 && iq4xs_geometry.high_offset == 768 &&
                iq4xs_geometry.high_bytes == 24 && iq4xs_geometry.scale_offset == 1024 &&
                iq4xs_geometry.scale_bytes == 24 && iq4xs_geometry.bytes == 1048,
            "ggml IQ4_XS plane geometry");
    // 32-value blocks: Q5_1 16 code bytes, the qh word, d and m; Q4_0 16 code bytes, no high
    // plane, d.
    const auto q51_geometry = weight_geometry(QType::GGML_Q5_1, QuantLayout::GgmlPlanes,
                                              std::array<std::uint64_t, 2>{3, 64});
    require(q51_geometry.code_bytes == 96 && q51_geometry.high_offset == 256 &&
                q51_geometry.high_bytes == 24 && q51_geometry.scale_offset == 512 &&
                q51_geometry.scale_bytes == 24 && q51_geometry.bytes == 536 &&
                q51_geometry.group_size == 32,
            "ggml Q5_1 plane geometry");
    std::vector<std::byte> q51(536);
    WeightParent q51_parent{q51_geometry, q51.data()};
    const auto q51_weight = native_weight(WeightView{{1, 64}, {{&q51_parent, 128, 192}}});
    require(q51_weight.qdata == q51.data() + 2 * 32 && q51_weight.qhigh == q51.data() + 256 + 16 &&
                q51_weight.scales == q51.data() + 512 + 16 && q51_weight.n == 1,
            "a ggml Q5_1 row view is not a standalone matrix");
    const auto q40_geometry = weight_geometry(QType::GGML_Q4_0, QuantLayout::GgmlPlanes,
                                              std::array<std::uint64_t, 2>{3, 64});
    require(q40_geometry.code_bytes == 96 && q40_geometry.high_bytes == 0 &&
                q40_geometry.scale_offset == 256 && q40_geometry.scale_bytes == 12 &&
                q40_geometry.bytes == 268,
            "ggml Q4_0 plane geometry");
}

void invalid_directories() {
    Fixture fixture;
    const auto bad = [&](auto mutate) {
        Json root = fixture.root;
        mutate(root);
        rejects([&] { (void)parse_directory(root, "model.ninferno"); },
                "invalid directory accepted");
    };
    bad([](Json& root) { root["bindings"]["row"]["parts"][0]["range"] = {129, 261}; });
    bad([](Json& root) { root["uses"].push_back(root["uses"][0]); });
    bad([](Json& root) { root["objects"][1]["offset"] = 4; });
    bad([](Json& root) { root["objects"][2]["id"] = "q5"; });
    bad([](Json& root) { root["files"][1]["path"] = "../other"; });
    // A continuation part is one file beside the entry on every platform: no drive or stream, no
    // trailing dot or space Windows drops, no device name, no character Windows refuses.
    for (const char* name : {"C:evil", "a:stream", "a.", "a ", "NUL", "nul.bin", "NUL .bin", "COM1",
                             "CONOUT$", "COM\xC2\xB9", "a<b", "a|b",
                             "a\x7f"
                             "b"}) {
        bad([&](Json& root) { root["files"][1]["path"] = name; });
    }
    bad([](Json& root) { root["files"][1]["payload_bytes"] = 1.5; });
    bad([](Json& root) { root["components"]["text"]["resources"]["tokenizer.json"] = "q5"; });
    // A proposal's drafter rows: indexed only, positive, never past its rows.
    bad([](Json& root) {
        root["components"]["text"]["proposal"] = {{"domain", "full"}, {"drafter_rows", {{"mtp", 1}}}};
    });
    bad([](Json& root) {
        root["components"]["text"]["proposal"] = {
            {"domain", "indexed"}, {"rows", 4}, {"drafter_rows", {{"mtp", 5}}}};
    });
    bad([](Json& root) {
        root["components"]["text"]["proposal"] = {
            {"domain", "indexed"}, {"rows", 4}, {"drafter_rows", {{"mtp", 0}}}};
    });
    bad([](Json& root) {
        root["components"]["text"]["proposal"] = {
            {"domain", "indexed"}, {"rows", 4}, {"drafter_rows", Json::array({2})}};
    });
    {
        Json root = fixture.root;
        root["components"]["text"]["proposal"] = {
            {"domain", "indexed"}, {"rows", 4}, {"drafter_rows", {{"mtp", 2}}}};
        const auto parsed = parse_directory(root, "model.ninferno");
        const auto& proposal = parsed.component("text").proposal;
        require(proposal && proposal->indexed && proposal->rows == 4 &&
                    proposal->drafter_rows.size() == 1 && proposal->drafter_rows.at("mtp") == 2,
                "a proposal's drafter rows were not read");
    }
    rejects([] { (void)parse_json("{\"a\":{\"x\":1,\"x\":2}}", "duplicate"); },
            "duplicate JSON key accepted");
    fixture.write();
    {
        std::fstream file(fixture.entry, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(7);
        file.put(2);
    }
    rejects([&] { Reader reader(fixture.entry); }, "v2 magic accepted");
    fixture.write();
    {
        std::fstream file(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(16);
        file.put(0);
    }
    Reader wrong_part(fixture.entry);
    rejects([&] { (void)wrong_part.read_object(wrong_part.find("q5")); },
            "foreign continuation accepted");
    fixture.write();
    Reader shortened(fixture.entry);
    std::filesystem::resize_file(fixture.entry, std::filesystem::file_size(fixture.entry) - 1);
    rejects([&] { (void)shortened.read_range(499, 1); }, "premature EOF accepted");
}

// Payload reads return the same bytes, up to the file's tail, in every I/O mode --artifact-io
// can select (buffered on network and FUSE file systems, O_DIRECT elsewhere).
void payload_reads_in_every_mode() {
    constexpr std::size_t kBlock = 4096;
    const auto path = std::filesystem::current_path() / "ninferno_payload_read_modes.bin";
    std::vector<char> bytes(3 * kBlock + 100);
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<char>(i * 131 + 7);
    {
        std::ofstream out(path, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    struct Free {
        void operator()(std::byte* p) const { ninferno::test::aligned_free(p); }
    };
    const std::unique_ptr<std::byte, Free> buffer(
        static_cast<std::byte*>(ninferno::test::aligned_alloc(kBlock, 4 * kBlock)));
    for (const ninferno::ArtifactIo mode :
         {ninferno::ArtifactIo::Auto, ninferno::ArtifactIo::Direct, ninferno::ArtifactIo::Buffered}) {
        ninferno::artifact::set_artifact_io(mode);
        std::fill_n(buffer.get(), 4 * kBlock, std::byte{0});
        std::size_t read = 0;
        try {
            const InputFile file(path);
            read = file.read_direct(0, {buffer.get(), 4 * kBlock});
        } catch (const ArtifactError& error) {
            // A file system without O_DIRECT refuses a forced direct read; nothing else may fail.
            if (mode == ninferno::ArtifactIo::Direct) continue;
            throw;
        }
        require(read == bytes.size(), "payload read did not stop at the file's tail");
        require(std::equal(bytes.begin(), bytes.end(), reinterpret_cast<const char*>(buffer.get())),
                "payload read returned different bytes");
    }
    ninferno::artifact::set_artifact_io(ninferno::ArtifactIo::Auto);
    std::filesystem::remove(path);
}

} // namespace

int main(int argc, char** argv) {
    try {
        file_set_and_bindings();
        unused_bindings();
        geometry_and_views();
        invalid_directories();
        payload_reads_in_every_mode();
        // Optional production-writer fixture or explicitly selected real artifact.
        if (argc == 2) {
            Reader reader(argv[1]);
            std::uint64_t bytes = 0;
            for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
                reader.validate_object({i});
                bytes =
                    checked_add(bytes, object_bytes(reader.directory().objects[i]), "object bytes");
            }
            std::cout << "v3 objects=" << reader.directory().objects.size() << " bytes=" << bytes
                      << '\n';
        }
        std::cout << "artifact reader and binding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
