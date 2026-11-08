// NInferno-owned file (not upstream). Cooperative cancellation of one grammar compile.

#include "compile_cancel.h"

#include <stdexcept>

namespace {

const std::atomic<bool>*& active_flag() noexcept {
    thread_local const std::atomic<bool>* flag = nullptr;
    return flag;
}

} // namespace

namespace ninferno_xgrammar {

CompileCancelScope::CompileCancelScope(const std::atomic<bool>& flag) {
    if (active_flag() != nullptr) {
        throw std::logic_error("XGrammar compile cancel scopes do not nest");
    }
    active_flag() = &flag;
}

CompileCancelScope::~CompileCancelScope() { active_flag() = nullptr; }

const std::atomic<bool>* current_compile_cancel() noexcept { return active_flag(); }

} // namespace ninferno_xgrammar
