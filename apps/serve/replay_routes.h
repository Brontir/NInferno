#pragma once

namespace ninferno::tools {

// ninferno-serve replay-routes --board BOARD --log FILE (#612 step 10): a start's route lines (the
// last server_start record of a request log, --request-log-jsonl: decision.routes) asked again on
// a CPU against BOARD's class from the facts table (core board_class: no GPU is read), under the
// start's route policy (its canonical text's options.maintainer.route_policy). Prints every line
// that moved and the counts. Exit 0: none moved; 1: some did; 2: a usage or input error.
int replay_routes_main(int argc, char** argv);

} // namespace ninferno::tools
