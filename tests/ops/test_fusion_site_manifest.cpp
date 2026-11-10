// The fusion sites' manifest lines (#617 step H, ops/common/fusion_site_manifest.h): a line for
// every site of the registry's generated table; the evaluated sites ask the functions the launch
// asks (launch_overlap, r8_prequantized_takes, ggml_gated_pair_for, rmsnorm_rope_text_takes_width),
// so their lines are checked against known inputs of those functions; a site whose input the query
// lacks is declared with `needs=`; an unknown id is refused. No GPU.
#include "core/device_class.h"
#include "core/units.h"
#include "ops/common/fusion_site_manifest.h"

#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using namespace ninferno;
namespace fusion  = ninferno::ops::detail::fusion;
namespace generic = ninferno::ops::detail::generic;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "fusion site manifest: %s\n", what.c_str());
        ++failures;
    }
}

std::string text(std::string_view id, const fusion::SiteQuery& q) {
    return fusion::text_of(fusion::site_line(id, q));
}

} // namespace

int main() {
    const DeviceClass ada =
        synthetic_device_class(0x2684, 8, 9, 128, 72 * kMiB, 24ULL * 1024 * kMiB);
    const DeviceClass turing =
        synthetic_device_class(0x1E04, 7, 5, 68, 6 * kMiB, 11ULL * 1024 * kMiB);
    const DeviceClass blackwell =
        synthetic_device_class(0x2B85, 12, 0, 170, 96 * kMiB, 32ULL * 1024 * kMiB);

    // A line for every site, in the table's order, each id once; a bare query evaluates none that
    // needs an input.
    fusion::SiteQuery bare;
    bare.device      = &ada;
    bare.tokens      = 4;
    const auto lines = fusion::all_site_lines(bare);
    expect(lines.size() == fusion::kSiteCount && fusion::kSiteCount >= 39, "one line a site");
    std::set<std::string> ids;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        expect(lines[i].id == fusion::kSiteTable[i].id, "registry order: " + lines[i].id);
        expect(ids.insert(lines[i].id).second, "twice: " + lines[i].id);
        expect(fusion::text_of(lines[i]).rfind("site " + lines[i].id + ' ', 0) == 0,
               "line starts with its id");
        expect(lines[i].status == "decided" || lines[i].status == "band" ||
                   lines[i].status == "declared",
               "status of " + lines[i].id);
        expect(lines[i].status != "declared" || lines[i].branch == "per_call",
               "a declared site evaluates nothing: " + lines[i].id);
    }
    expect(text("pdl_overlap", bare) == "site pdl_overlap declared per_call needs=route",
           "pdl_overlap without a route is declared");
    expect(text("r8_prequantized", bare) ==
               "site r8_prequantized declared per_call needs=route,qtype,k",
           "r8_prequantized without its inputs is declared");
    expect(text("qk_norm_rope", bare) ==
               "site qk_norm_rope declared per_call needs=query_heads,key_heads",
           "qk_norm_rope without a geometry is declared");
    expect(text("dense_qkv_prepare", bare) == "site dense_qkv_prepare declared per_call",
           "an always-fused site is declared");

    // pdl_overlap: launch_overlap's answer on a register MMA Store launch.
    const generic::Route rmma{generic::Family::RegisterMma, 0};
    const std::string rmma_name = generic::route_name(rmma);
    fusion::SiteQuery q;
    q.device        = &blackwell;
    q.tokens        = 4;
    q.route         = rmma;
    q.release_asked = true;
    expect(text("pdl_overlap", q) ==
               "site pdl_overlap decided overlap route=" + rmma_name +
                   " join=not_asked release=none prefetch=none pdl=1 tokens=4",
           "register MMA releases and prefetches where PDL exists");
    q.device = &turing;
    expect(text("pdl_overlap", q) ==
               "site pdl_overlap decided stream_order route=" + rmma_name +
                   " join=not_asked release=no_pdl prefetch=no_pdl pdl=0 tokens=4",
           "no PDL: stream order");
    q.device = &blackwell;
    q.tokens = 64;
    expect(text("pdl_overlap", q).find("prefetch=wide") != std::string::npos,
           "a call wider than the prefetch's band does not prefetch");
    q.route      = generic::Route{generic::Family::GgmlMmq, 0};
    q.join_asked = true;
    expect(text("pdl_overlap", q).find("decided stream_order") != std::string::npos &&
               text("pdl_overlap", q).find("join=family release=family prefetch=family") !=
                   std::string::npos,
           "a ggml launch runs in stream order");

    // The gated-pair bands: the vector pair to kGgmlVecMaxTokens, the matrix pair above; a vector
    // band call does not fall through to the matrix pair.
    expect(text("vec_gated_pair", {&ada, 8}) ==
               "site vec_gated_pair band tried tokens=8 vec_max_tokens=8",
           "vec gated pair at the band's edge");
    expect(fusion::site_line("mmq_gated_pair", {&ada, 8}).branch == "not_tried" &&
               fusion::site_line("vec_gated_pair", {&ada, 9}).branch == "not_tried" &&
               fusion::site_line("mmq_gated_pair", {&ada, 9}).branch == "tried",
           "the MMQ pair is tried only above the vec band");

    // R8 pre-quantized: the band and the workspace, then the format's build fact.
    fusion::SiteQuery r8;
    r8.device         = &ada;
    r8.route          = generic::Route{generic::Family::R8Mma, 0};
    r8.qtype          = QType::Q4_G64_FP16;
    r8.k              = 2048;
    r8.tokens         = 1;
    r8.workspace_free = std::size_t{1} << 30;
    expect(fusion::site_line("r8_prequantized", r8).branch == "in_kernel", "one token: in kernel");
    r8.tokens = 9;
    expect(fusion::site_line("r8_prequantized", r8).branch == "in_kernel", "past the verify band");
    r8.tokens         = 4;
    r8.workspace_free = std::nullopt;
    expect(fusion::site_line("r8_prequantized", r8).branch == "in_kernel",
           "no workspace: in kernel");
    r8.workspace_free     = std::size_t{1} << 30;
    const bool expect_pre = generic::r8_verify_compiled(QType::Q4_G64_FP16);
    expect((fusion::site_line("r8_prequantized", r8).branch == "prequantized") == expect_pre,
           "in the band with room: prequantized where the format is compiled for it");

    // qk_norm_rope: the width function; the constant is pending, so 256 on every board.
    fusion::SiteQuery rope;
    rope.device      = &blackwell;
    rope.query_heads = 16;
    rope.key_heads   = 4;
    rope.tokens      = 256;
    expect(text("qk_norm_rope", rope) ==
               "site qk_norm_rope band fused geometry=16+4 tokens=256 max_tokens=256",
           "qk_norm_rope fuses through 256");
    rope.tokens = 257;
    expect(fusion::site_line("qk_norm_rope", rope).branch == "decomposed", "and not above");
    rope.tokens      = 8;
    rope.query_heads = 12;
    rope.key_heads   = 3;
    expect(fusion::site_line("qk_norm_rope", rope).branch == "decomposed",
           "a geometry the kernel is not instantiated for is decomposed");

    bool refused = false;
    try {
        (void)fusion::site_line("no_such_site", bare);
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "an unknown id is refused");

    if (failures == 0) std::printf("fusion site manifest: ok (%zu sites)\n", fusion::kSiteCount);
    return failures == 0 ? 0 : 1;
}
