#pragma once

// NInferno-owned file (not upstream). Implements XGrammar's XGRAMMAR_LOG_CUSTOMIZE hook.

#include <string>
#include <vector>

namespace ninferno_xgrammar {

// While an instance is alive, XGrammar warnings raised on the constructing thread are collected
// instead of printed. Scopes do not nest; construction throws std::logic_error if one is active.
class WarningCapture {
public:
    WarningCapture();
    ~WarningCapture();
    WarningCapture(const WarningCapture&)            = delete;
    WarningCapture& operator=(const WarningCapture&) = delete;

    [[nodiscard]] const std::vector<std::string>& warnings() const noexcept;
};

// XGrammar's LogFatalError::what() prefixes a timestamp and a source location. This returns the
// plain message for client-facing errors.
[[nodiscard]] std::string plain_error_message(const std::exception& error);

} // namespace ninferno_xgrammar
