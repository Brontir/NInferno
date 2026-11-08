#pragma once

// NInferno-owned file (not upstream). Cooperative cancellation of one grammar compile.

#include <atomic>
#include <stdexcept>

namespace ninferno_xgrammar {

// Thrown by the compile when its cancel flag was set; the result of such a compile is not wanted.
class CompileCancelled : public std::runtime_error {
public:
    CompileCancelled() : std::runtime_error("grammar compilation was cancelled") {}
};

// While an instance is alive, a grammar compile started on the constructing thread watches `flag`:
// once it is set, the compile stops at its next checkpoint (between token-mask computations) and
// throws CompileCancelled. The flag must outlive the scope. Scopes do not nest.
class CompileCancelScope {
public:
    explicit CompileCancelScope(const std::atomic<bool>& flag);
    ~CompileCancelScope();
    CompileCancelScope(const CompileCancelScope&)            = delete;
    CompileCancelScope& operator=(const CompileCancelScope&) = delete;
};

// The flag of the scope active on the calling thread, or null.
[[nodiscard]] const std::atomic<bool>* current_compile_cancel() noexcept;

} // namespace ninferno_xgrammar
