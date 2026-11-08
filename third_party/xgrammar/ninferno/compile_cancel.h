#pragma once

// NInferno-owned file (not upstream). Cooperative cancellation of one grammar compile.

#include <functional>
#include <stdexcept>

namespace ninferno_xgrammar {

// Thrown by the compile when its cancel flag was set; the result of such a compile is not wanted.
class CompileCancelled : public std::runtime_error {
public:
    CompileCancelled() : std::runtime_error("grammar compilation was cancelled") {}
};

// Asked at every checkpoint of a compile; true means stop. Called from the compile's worker
// threads, so it must be thread-safe and cheap.
using CompileCancelCheck = std::function<bool()>;

// While an instance is alive, a grammar compile started on the constructing thread asks `check` at
// its checkpoints (between token-mask computations; the grammar and automaton building before
// them has none). Once it answers true, the compile stops and throws CompileCancelled. Scopes do
// not nest.
class CompileCancelScope {
public:
    explicit CompileCancelScope(CompileCancelCheck check);
    ~CompileCancelScope();
    CompileCancelScope(const CompileCancelScope&)            = delete;
    CompileCancelScope& operator=(const CompileCancelScope&) = delete;
};

// The check of the scope active on the calling thread, or an empty function.
[[nodiscard]] CompileCancelCheck current_compile_cancel();

} // namespace ninferno_xgrammar
