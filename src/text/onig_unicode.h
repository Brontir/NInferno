#pragma once

#include <utf8proc/utf8proc.h>

#include <cstddef>

namespace ninferno::text::onig_unicode {

// General category of `cp` as Oniguruma 6.9.x (the engine of Hugging Face tokenizers 0.23)
// sees it: Unicode 16.0.0. utf8proc is Unicode 17.0.0, so the code points whose category
// changed since 16.0.0 (onig_unicode_tables.h) take their 16.0.0 category, which is Cn for
// every one newly assigned.
utf8proc_category_t category(char32_t cp) noexcept;

// Simple case mappings, with the code points that case-fold differently in 16.0.0 left alone:
// a code point added or re-categorised after 16.0.0 has no case variants.
char32_t to_lower(char32_t cp) noexcept;
char32_t to_upper(char32_t cp) noexcept;

// Number of code points whose category differs between the two Unicode versions.
std::size_t differing_code_points() noexcept;

} // namespace ninferno::text::onig_unicode
