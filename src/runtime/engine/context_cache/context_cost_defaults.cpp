#include "runtime/engine/context_cache/context_cost.h"

#include "core/measured_boards.h"

#include <array>
#include <vector>

namespace ninferno::runtime {
namespace {

constexpr ContextPrefillCost kGroupwise27bPrefill{
    .chunk_ns              = 40'813'570,
    .token_ns_q32          = 1'012'273'154'411'951,
    .attention_pair_ns_q32 = 30'497'396'515,
    .vision_item_ns        = 5'986'585,
    .vision_patch_ns_q32   = 23'710'212'854'694,
};
// RTX 4090, Qwen3.8 27B UD-Q4_K_XL text-only (CUDA 12.8): the 27B groupwise
// prefill runs at 0.77x the 5090's per token (kGb202Groupwise27bPrefill). Vision is the generic
// Vision cost scaled by that ratio.
constexpr ContextPrefillCost kAd102Groupwise27bPrefill{
    .chunk_ns              = 25'968'054,
    .token_ns_q32          = 1'219'524'115'385'335,
    .attention_pair_ns_q32 = 11'726'266'217,
    .vision_item_ns        = 7'212'268,
    .vision_patch_ns_q32   = 28'564'598'627'547,
};
// RTX 5090, Qwen3.8 27B UD-Q4_K_XL text-only (CUDA 13.1): the current groupwise
// 27B prefill, 7 % below kGroupwise27bPrefill per token. Vision is the generic Vision cost scaled
// by that ratio.
constexpr ContextPrefillCost kGb202Groupwise27bPrefill{
    .chunk_ns              = 26'410'419,
    .token_ns_q32          = 944'785'862'335'153,
    .attention_pair_ns_q32 = 17'562'942'894,
    .vision_item_ns        = 5'587'465,
    .vision_patch_ns_q32   = 22'129'475'429'076,
};
// RTX 5090, Qwen3.8 27B NVFP4 v3 with Vision (CUDA 13.1): 28 % below
// kNvfp4Fp8Prefill per token.
constexpr ContextPrefillCost kGb202Nvfp4Prefill{
    .chunk_ns              = 12'739'204,
    .token_ns_q32          = 271'426'606'345'574,
    .attention_pair_ns_q32 = 13'423'468'496,
    .vision_item_ns        = 2'902'080,
    .vision_patch_ns_q32   = 33'737'882'078'219,
};
// RTX 3090, Qwen3.8 27B UD-Q4_K_XL text-only (CUDA 12.8): 3.1x the 5090's
// per token (kGb202Groupwise27bPrefill). Vision is the generic Vision scaled by that ratio.
constexpr ContextPrefillCost kGa102Groupwise27bPrefill{
    .chunk_ns              = 22'064'137,
    .token_ns_q32          = 2'964'978'981'039'522,
    .attention_pair_ns_q32 = 36'729'540'209,
    .vision_item_ns        = 17'534'890,
    .vision_patch_ns_q32   = 69'447'937'489'738,
};
// RTX 2080 Ti, text-only (CUDA 12.8): Turing runs bf16 MMA through f16, so a
// 27B UD-IQ2_XXS prefills at 21x the 5090's groupwise 27B per token and Qwen3.5 9B Q4_K_M at 2.1x.
// Its attention fit is flat at 27B (the chunk term carries it). Vision is the generic Vision
// scaled by the token ratio.
constexpr ContextPrefillCost kTu102Iq2xxs27bPrefill{
    .chunk_ns              = 60'413'073,
    .token_ns_q32          = 20'155'682'553'363'628,
    .attention_pair_ns_q32 = 0,
    .vision_item_ns        = 119'200'738,
    .vision_patch_ns_q32   = 472'101'350'795'490,
};
constexpr ContextPrefillCost kTu102Q4km9bPrefill{
    .chunk_ns              = 5'399'365,
    .token_ns_q32          = 2'027'672'431'877'711,
    .attention_pair_ns_q32 = 68'182'241'905,
    .vision_item_ns        = 11'991'658,
    .vision_patch_ns_q32   = 47'493'648'082'907,
};
constexpr ContextPrefillCost kNvfp4Fp8Prefill{
    .chunk_ns              = 14'672'989,
    .token_ns_q32          = 375'800'765'711'778,
    .attention_pair_ns_q32 = 8'200'474'657,
    .vision_item_ns        = 5'860'255,
    .vision_patch_ns_q32   = 23'832'529'381'413,
};

} // namespace

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B configuration is the conservative generic estimate. An unknown model
    // still receives continuous recomputation costs instead of falling back to a discrete tuple.
    return kGroupwise27bPrefill;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency. Each
// machine is keyed by the class id of the board it was measured on and records that board: a
// device of another class id takes the nearest one within measured::kCarryDistance, as measured.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .gpu_class = "gb202:170sm",
            // Unified from the measured transfer corpus. A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
            // Existing measurements apply to these Text bindings with Vision disabled/enabled.
            // The signature excludes trained values and the selected speculative backend; it
            // describes the primary reconstruction work priced by ContextPrefillCost.
            .prefill =
                {
                    {"200f57efee7b0fe1172dfd4a06b1e6e0b2dbc36dfcd6a242fea35338bb5ff0d2",
                     kGroupwise27bPrefill},
                    {"badf2271162e72c8c51a91da02cbb4343d4b7574d037a014ca6862ff2862638b",
                     kGroupwise27bPrefill},
                    {"e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
                     kNvfp4Fp8Prefill},
                    {"953e22d9b9a6639c09f7389d49084b059dea07720bc0121dfd9c154002459915",
                     kGb202Nvfp4Prefill},
                    {"77c1f30a6b521bada3dc202ab41cc0b63d058fd91203886955b4ff2c889667aa",
                     kGb202Groupwise27bPrefill},
                },
            // A model of no listed signature: the slower (groupwise 27B) measurement of this card.
            .class_prefill = kGb202Groupwise27bPrefill,
            .board         = measured::kRtx5090,
        },
        ContextCostMachinePreset{
            .gpu_class = "ad102:128sm",
            // Its transfer fit refused the D2D copies; until they fit, the generic transfer.
            .prefill       = {{"77c1f30a6b521bada3dc202ab41cc0b63d058fd91203886955b4ff2c889667aa",
                               kAd102Groupwise27bPrefill}},
            .class_prefill = kAd102Groupwise27bPrefill,
            .board         = measured::kRtx4090,
        },
        ContextCostMachinePreset{
            .gpu_class = "ga102:82sm",
            // Measured (CUDA 12.8 runtime / 13.0 driver, PCIe 4.0 host): the
            // host links run at about half the generic 40 GB/s, so Host KV retention is priced
            // twice as dear as the generic model says.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 27'120, .operation_ns = 2'047, .ns_per_byte_q32 = 210'924'519},
                    ContextTransferCost{
                        .batch_ns = 26'056, .operation_ns = 2'181, .ns_per_byte_q32 = 201'273'943},
                    ContextTransferCost{
                        .batch_ns = 3'740, .operation_ns = 3'283, .ns_per_byte_q32 = 10'943'200},
                },
            .prefill       = {{"77c1f30a6b521bada3dc202ab41cc0b63d058fd91203886955b4ff2c889667aa",
                               kGa102Groupwise27bPrefill}},
            .class_prefill = kGa102Groupwise27bPrefill,
            .board         = measured::kRtx3090,
        },
        ContextCostMachinePreset{
            .gpu_class = "tu102:68sm",
            // Measured (CUDA 12.8, PCIe 3.0 host): the host links at 12 GB/s.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 3'177, .operation_ns = 2'095, .ns_per_byte_q32 = 349'967'190},
                    ContextTransferCost{
                        .batch_ns = 34'834, .operation_ns = 2'351, .ns_per_byte_q32 = 353'655'517},
                    ContextTransferCost{
                        .batch_ns = 4'451, .operation_ns = 2'550, .ns_per_byte_q32 = 17'200'500},
                },
            .prefill =
                {
                    {"29633abbebf56344a253424d60b87feb4822b3126dee6d428f183fd89c0a6ca8",
                     kTu102Iq2xxs27bPrefill},
                    {"ed7fbdac7d2c4155ff33b80b97adfd30519180f94695a70aeedd589b9ad1abfe",
                     kTu102Q4km9bPrefill},
                },
            .class_prefill = kTu102Iq2xxs27bPrefill,
            .board         = measured::kRtx2080Ti,
        },
    };
    return defaults;
}

} // namespace ninferno::runtime
