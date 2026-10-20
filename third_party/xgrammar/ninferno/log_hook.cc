// NInferno-owned file (not upstream). Implements XGrammar's XGRAMMAR_LOG_CUSTOMIZE hook.

#include "log_hook.h"

#include "support/logging.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CaptureState {
    bool active = false;
    std::vector<std::string> warnings;
};

CaptureState& capture_state() {
    thread_local CaptureState state;
    return state;
}

} // namespace

namespace xgrammar {

[[noreturn]] void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
    throw LogFatalError(file, lineno, message);
}

void LogMessageImpl(const std::string&, int, int level, const std::string& message) {
    // Only warnings carry information NInferno acts on. Outside a capture scope (for example a
    // matcher warning on the Engine worker) the message is dropped: the caller already reports
    // the corresponding failure through its own return value.
    if (level != XGRAMMAR_LOG_LEVEL_WARNING) { return; }
    CaptureState& state = capture_state();
    if (state.active) { state.warnings.push_back(message); }
}

} // namespace xgrammar

namespace ninferno_xgrammar {

WarningCapture::WarningCapture() {
    CaptureState& state = capture_state();
    if (state.active) { throw std::logic_error("XGrammar warning capture scopes do not nest"); }
    state.active = true;
    state.warnings.clear();
}

WarningCapture::~WarningCapture() {
    CaptureState& state = capture_state();
    state.active        = false;
}

const std::vector<std::string>& WarningCapture::warnings() const noexcept {
    return capture_state().warnings;
}

std::string plain_error_message(const std::exception& error) {
    if (const auto* fatal = dynamic_cast<const xgrammar::LogFatalError*>(&error)) {
        return fatal->std::runtime_error::what();
    }
    return error.what();
}

} // namespace ninferno_xgrammar
