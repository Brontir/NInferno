// NInferno-owned file (not upstream). Cooperative cancellation of one grammar compile.

#include "compile_cancel.h"

#include <stdexcept>

namespace {

ninferno_xgrammar::CompileCancelCheck& active_check() {
    thread_local ninferno_xgrammar::CompileCancelCheck check;
    return check;
}

} // namespace

namespace ninferno_xgrammar {

CompileCancelScope::CompileCancelScope(CompileCancelCheck check) {
    if (active_check()) { throw std::logic_error("XGrammar compile cancel scopes do not nest"); }
    active_check() = std::move(check);
}

CompileCancelScope::~CompileCancelScope() { active_check() = nullptr; }

CompileCancelCheck current_compile_cancel() { return active_check(); }

} // namespace ninferno_xgrammar
