"""tools/bench/moe_expert_law.py (#619 step 11): the law's pieces against moe.cpp and the committed snapshots."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))

import moe_expert_law as law  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools", "kernel_facts"))
import kernel_facts  # noqa: E402

FACTS = os.path.join(ROOT, "src", "ops", "generic", "data", "kernel_facts", "sm_120a.txt")


def test_r8_lanes_follow_moe_cpp():
    # moe.cpp r8_lanes: loads = K / 32 / per load; <= 24 -> 8, <= 48 -> 16, else 32.
    assert law.r8_lanes("q4_k", 2048) == 16  # 32 loads
    assert law.r8_lanes("q6_k", 2048) == 8   # 16 loads
    assert law.r8_lanes("q4_k", 512) == 8
    assert law.r8_lanes("q4_k", 4096) == 32  # 64 loads


def test_occupancy_takes_the_tightest_limit():
    assert law.occupancy(128, 0, 128, 32, 1536, 89) == 4       # registers: 64 Ki / (128 x 128)
    assert law.occupancy(32, 60 * 1024, 128, 32, 1536, 89) == 1  # shared memory
    assert law.occupancy(16, 0, 512, 32, 1536, 89) == 3          # threads
    # The SM's shared memory and the per-block reserve are the arch's (gpu_facts.json): 100 KiB and
    # 1 KiB a block on Ada, 64 KiB and none on Turing.
    assert law.occupancy(16, 24 * 1024, 128, 32, 1536, 89) == 4  # 100 Ki / (24 Ki + 1 Ki)
    assert law.occupancy(16, 24 * 1024, 128, 16, 1024, 75) == 2  # 64 Ki / 24 Ki


def test_r8_token_limit_follows_the_regime_rules():
    assert law.r8_token_limit("rtx5090", "q4_k") == 8 and law.r8_token_limit("rtx3090", "q4_k") == 4
    assert law.r8_token_limit("rtx4090", "q6_k") == 4
    assert law.r8_token_limit("rtx3060", "q3_k") == 8 and law.r8_token_limit("rtx3060", "q5_k") == 4
    assert not law.admitted("rtx5090", "q6_k", "r8.w4r1|rmma.m1.n1.w1x4", 5)
    assert law.admitted("rtx5090", "q6_k", "rmma.m1.n1.w1x8|rmma.m1.n1.w1x4", 8)


def test_every_plan_is_priced_and_one_pick_a_cell(tmp_path):
    plans = tmp_path / "plans.csv"
    with open(plans, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plan"])
        for gu in ("rmma.m1.n1.w1x8", "r8.w4r1"):
            for dn in ("rmma.m1.n1.w1x4", "r8.w8r1"):
                w.writerow([gu + "|" + dn])
    out = tmp_path / "pred.csv"
    law.BOARD_NAME = "rtx5090"
    law.predict(FACTS, str(plans), (256, 8, 2048, 512), str(out))
    rows = list(csv.DictReader(open(out)))
    # Every plan in every cell, except the R8 stages past the R8 token limit (Q6_K: four, moe_regime_rules.h).
    assert len(rows) == 4 * 8 * 4 - 4 * 3
    assert not [r for r in rows if r["format"] == "q6_k" and int(r["T"]) > 4 and "r8." in r["plan"]]
    for fmt in law.FORMATS:
        for t in range(1, 9):
            cell = [r for r in rows if r["format"] == fmt and int(r["T"]) == t]
            assert sum(int(r["pick"]) for r in cell) == 1
            assert all(float(r["pred_us"]) > 0 for r in cell)


def test_no_price_reads_a_row_ptxas_allocates_either_way():
    # #968: the moegemm rows whose registers ptxas does not reproduce (kernel_facts
    # REGISTER_ALLOCATIONS) are held to a counted set, not one value; a price that read them would
    # rank on the build's allocation. The law prices the GEMM_DEC decoders alone.
    for arch, rows in kernel_facts.REGISTER_ALLOCATIONS["moegemm"].items():
        for key in rows:
            assert key[0] not in law.GEMM_DEC.values(), (arch, key)
