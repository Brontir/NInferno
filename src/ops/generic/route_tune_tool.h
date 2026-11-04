#pragma once

namespace ninferno::tools {

// The generic projection path's route tuner (route_tune_tool.cpp): `ninferno-serve route-tune ...`
// and the bench's ninferno_route_tune. argv[0] is the command's own name, its options follow.
int route_tune_main(int argc, char** argv);

} // namespace ninferno::tools
