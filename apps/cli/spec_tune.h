#pragma once

namespace ninferno::cli {

// `ninferno spec-tune <artifact> ...` (a maintainer tool, see spec_tune.cpp).
int run_spec_tune(int argc, char** argv);

// `ninferno spec-reselect <profile.json> ...`: a saved tune selected again and published (see
// spec_tune.cpp).
int run_spec_reselect(int argc, char** argv);

} // namespace ninferno::cli
