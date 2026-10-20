#include "convert/pyjson.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace ninferno::convert {
namespace {

void append_string(std::string& out, const std::string& text) {
    static constexpr char kHex[] = "0123456789abcdef";
    out.push_back('"');
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        default:
            if (byte < 0x20) {
                out += "\\u00";
                out.push_back(kHex[byte >> 4]);
                out.push_back(kHex[byte & 15]);
            } else {
                out.push_back(c);
            }
        }
    }
    out.push_back('"');
}

void append_scalar(std::string& out, const OJson& value) {
    switch (value.type()) {
    case OJson::value_t::null:
        out += "null";
        return;
    case OJson::value_t::boolean:
        out += value.get<bool>() ? "true" : "false";
        return;
    case OJson::value_t::number_integer:
        out += std::to_string(value.get<std::int64_t>());
        return;
    case OJson::value_t::number_unsigned:
        out += std::to_string(value.get<std::uint64_t>());
        return;
    case OJson::value_t::number_float:
        out += py_float_repr(value.get<double>());
        return;
    case OJson::value_t::string:
        append_string(out, value.get_ref<const std::string&>());
        return;
    default:
        throw std::invalid_argument("JSON value cannot be written by the Python-compatible writer");
    }
}

void compact(std::string& out, const OJson& value) {
    if (value.is_object()) {
        out.push_back('{');
        bool first = true;
        for (const auto& [key, item] : value.items()) {
            if (!first) { out.push_back(','); }
            first = false;
            append_string(out, key);
            out.push_back(':');
            compact(out, item);
        }
        out.push_back('}');
    } else if (value.is_array()) {
        out.push_back('[');
        bool first = true;
        for (const auto& item : value) {
            if (!first) { out.push_back(','); }
            first = false;
            compact(out, item);
        }
        out.push_back(']');
    } else {
        append_scalar(out, value);
    }
}

void indented(std::string& out, const OJson& value, int level) {
    const auto newline = [&](int depth) {
        out.push_back('\n');
        out.append(static_cast<std::size_t>(2 * depth), ' ');
    };
    if (value.is_object()) {
        if (value.empty()) {
            out += "{}";
            return;
        }
        out.push_back('{');
        bool first = true;
        for (const auto& [key, item] : value.items()) {
            if (!first) { out.push_back(','); }
            first = false;
            newline(level + 1);
            append_string(out, key);
            out += ": ";
            indented(out, item, level + 1);
        }
        newline(level);
        out.push_back('}');
    } else if (value.is_array()) {
        if (value.empty()) {
            out += "[]";
            return;
        }
        out.push_back('[');
        bool first = true;
        for (const auto& item : value) {
            if (!first) { out.push_back(','); }
            first = false;
            newline(level + 1);
            indented(out, item, level + 1);
        }
        newline(level);
        out.push_back(']');
    } else {
        append_scalar(out, value);
    }
}

} // namespace

std::string py_float_repr(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("JSON cannot hold a non-finite float");
    }
    if (value == 0.0) { return std::signbit(value) ? "-0.0" : "0.0"; }
    // Shortest round-trip digits, as CPython's repr takes them from dtoa mode 0.
    char buffer[64];
    const auto [end, error] =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    if (error != std::errc{}) { throw std::runtime_error("float formatting failed"); }
    std::string text(buffer, end);
    std::string sign;
    if (text.front() == '-') {
        sign = "-";
        text.erase(0, 1);
    }
    const auto e = text.find('e');
    std::string digits;
    for (std::size_t i = 0; i < e; ++i) {
        if (text[i] != '.') { digits.push_back(text[i]); }
    }
    const int exponent = std::stoi(text.substr(e + 1));
    const int decpt    = exponent + 1; // value = 0.DIGITS * 10^decpt
    const int ndigits  = static_cast<int>(digits.size());
    std::string out    = sign;
    if (decpt <= -4 || decpt > 16) {
        out.push_back(digits[0]);
        if (ndigits > 1) {
            out.push_back('.');
            out.append(digits, 1, std::string::npos);
        }
        out.push_back('e');
        const int shown = decpt - 1;
        out.push_back(shown < 0 ? '-' : '+');
        const auto magnitude = std::to_string(shown < 0 ? -shown : shown);
        if (magnitude.size() < 2) { out.push_back('0'); }
        out += magnitude;
    } else if (decpt <= 0) {
        out += "0.";
        out.append(static_cast<std::size_t>(-decpt), '0');
        out += digits;
    } else if (decpt >= ndigits) {
        out += digits;
        out.append(static_cast<std::size_t>(decpt - ndigits), '0');
        out += ".0";
    } else {
        out.append(digits, 0, static_cast<std::size_t>(decpt));
        out.push_back('.');
        out.append(digits, static_cast<std::size_t>(decpt), std::string::npos);
    }
    return out;
}

std::string py_format_g(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
    return buffer;
}

std::string py_path_str(const std::string& text) {
    if (text.empty()) { return "."; }
    std::string root;
    if (text.front() == '/') {
        root = (text.size() > 1 && text[1] == '/' && (text.size() == 2 || text[2] != '/')) ? "//"
                                                                                           : "/";
    }
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        auto end = text.find('/', begin);
        if (end == std::string::npos) { end = text.size(); }
        auto part = text.substr(begin, end - begin);
        if (!part.empty() && part != ".") { parts.push_back(std::move(part)); }
        begin = end + 1;
    }
    std::string out = root;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) { out.push_back('/'); }
        out += parts[i];
    }
    return out.empty() ? "." : out;
}

std::string py_dumps_compact(const OJson& value) {
    std::string out;
    compact(out, value);
    return out;
}

std::string py_dumps_indent2(const OJson& value) {
    std::string out;
    indented(out, value, 0);
    return out;
}

} // namespace ninferno::convert
