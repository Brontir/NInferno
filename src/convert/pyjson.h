#pragma once

// JSON text exactly as CPython's json module writes it. The artifact directory and the conversion
// report are compared byte for byte with the Python converter's, so key order (insertion order),
// float spelling (repr) and string escaping (ensure_ascii=False) all follow CPython, not
// nlohmann's own dump().

#include <nlohmann/json.hpp>

#include <string>

namespace ninferno::convert {

using OJson = nlohmann::ordered_json;

// json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
[[nodiscard]] std::string py_dumps_compact(const OJson& value);
// json.dump(value, ensure_ascii=False, allow_nan=False, indent=2)
[[nodiscard]] std::string py_dumps_indent2(const OJson& value);
// repr(float)
[[nodiscard]] std::string py_float_repr(double value);
// format(value, ".{precision}g")
[[nodiscard]] std::string py_format_g(double value, int precision);
// str(pathlib.Path(text)) on POSIX
[[nodiscard]] std::string py_path_str(const std::string& text);

} // namespace ninferno::convert
