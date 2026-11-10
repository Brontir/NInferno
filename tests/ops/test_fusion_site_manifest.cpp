// The fusion sites' manifest lines (#617 step H, ops/common/fusion_site_manifest.h): a line for
// every site of the registry's generated table, the evaluated sites' branches on substituted
// boards, the rest declared, an unknown id refused. No GPU.
#include "core/device_class.h"
#include "core/units.h"
#include "ops/common/fusion_site_manifest.h"

#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using namespace ninferno;
namespace fusion = ninferno::ops::detail::fusion;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "fusion site manifest: %s\n", what.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    const DeviceClass ada =
        synthetic_device_class(0x2684, 8, 9, 128, 72 * kMiB, 24ULL * 1024 * kMiB);
    const DeviceClass turing =
        synthetic_device_class(0x1E04, 7, 5, 68, 6 * kMiB, 11ULL * 1024 * kMiB);
    const DeviceClass blackwell =
        synthetic_device_class(0x2B85, 12, 0, 170, 96 * kMiB, 32ULL * 1024 * kMiB);

    // A line for every site, in the table's order, each id once.
    const auto lines = fusion::all_site_lines({&ada, 4});
    expect(lines.size() == fusion::kSiteCount && fusion::kSiteCount >= 39, "one line a site");
    std::set<std::string> ids;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        expect(lines[i].id == fusion::kSiteTable[i].id, "registry order: " + lines[i].id);
        expect(ids.insert(lines[i].id).second, "twice: " + lines[i].id);
        const std::string text = fusion::text_of(lines[i]);
        expect(text.rfind("site " + lines[i].id + ' ', 0) == 0, "line starts with its id");
        expect(lines[i].status == "decided" || lines[i].status == "band" ||
                   lines[i].status == "declared",
               "status of " + lines[i].id);
        expect(lines[i].status != "declared" ||
                   (lines[i].branch == "per_call" && lines[i].inputs.empty()),
               "a declared site evaluates nothing: " + lines[i].id);
    }

    // The evaluated sites.
    expect(fusion::text_of(fusion::site_line("pdl_overlap", {&blackwell, 4})) ==
               "site pdl_overlap decided overlap pdl=1",
           "pdl_overlap on a PDL board");
    expect(fusion::text_of(fusion::site_line("pdl_overlap", {&turing, 4})) ==
               "site pdl_overlap decided stream_order pdl=0",
           "pdl_overlap below PDL");
    expect(fusion::text_of(fusion::site_line("vec_gated_pair", {&ada, 8})) ==
               "site vec_gated_pair band tried tokens=8 vec_max_tokens=8",
           "vec gated pair at the band's edge");
    expect(fusion::site_line("vec_gated_pair", {&ada, 9}).branch == "not_tried" &&
               fusion::site_line("mmq_gated_pair", {&ada, 9}).branch == "first" &&
               fusion::site_line("mmq_gated_pair", {&ada, 8}).branch == "after_vec",
           "the MMQ pair starts where the vec pair's band ends");
    expect(fusion::site_line("r8_prequantized", {&ada, 1}).branch == "in_kernel" &&
               fusion::site_line("r8_prequantized", {&ada, 2}).branch == "prequantized_if_fits" &&
               fusion::site_line("r8_prequantized", {&ada, 8}).branch == "prequantized_if_fits" &&
               fusion::site_line("r8_prequantized", {&ada, 9}).branch == "in_kernel",
           "R8 pre-quantized band is 2..8 tokens");
    expect(fusion::site_line("dense_qkv_prepare", {&ada, 4}).status == "declared",
           "an always-fused site is declared");

    bool refused = false;
    try {
        (void)fusion::site_line("no_such_site", {&ada, 4});
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "an unknown id is refused");

    if (failures == 0) std::printf("fusion site manifest: ok (%zu sites)\n", fusion::kSiteCount);
    return failures == 0 ? 0 : 1;
}
