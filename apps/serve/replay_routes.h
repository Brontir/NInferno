#pragma once

namespace ninferno::tools {

// ninferno-serve replay-routes --board BOARD --log FILE (#612 step 10): a start's route lines (the
// last server_start record of a request log, --request-log-jsonl: decision.routes) asked again on
// a CPU against BOARD's class built from its registry row (price_board_class: the row's clocks, no
// GPU is read), under the start's route policy (its canonical text's
// options.maintainer.route_policy). Refused (exit 2): a start whose selection was overridden (taint
// maintainer or user_tune_cache; the other taints, and the start's route table, are printed), a
// start whose build differs from this binary's in a key the selector reads (arch, arch_specific,
// nvfp4, the CUDA architectures: all but version, commit and runtime), a start whose
// facts.pci_device_id is not BOARD's or is missing. Prints every line that moved and the counts.
// Exit 0: none moved; 1: some did; 2: a usage or input error or a refusal.
int replay_routes_main(int argc, char** argv);

} // namespace ninferno::tools
