#include "text/onig_unicode.h"

#include "text/onig_unicode_tables.h"

#include <algorithm>
#include <iterator>

namespace ninferno::text::onig_unicode {
namespace {

static_assert(UTF8PROC_CATEGORY_CN == 0 && UTF8PROC_CATEGORY_LU == 1 && UTF8PROC_CATEGORY_LL == 2 &&
                  UTF8PROC_CATEGORY_CS == 28 && UTF8PROC_CATEGORY_CO == 29,
              "onig_unicode_tables.h numbers categories like utf8proc_category_t");

const CategoryRange* find_range(char32_t cp) noexcept {
    const auto* first = std::begin(kOnigCategoryRanges);
    const auto* last  = std::end(kOnigCategoryRanges);
    if (cp < first->first || cp > (last - 1)->last) { return nullptr; }
    const auto* it = std::upper_bound(
        first, last, static_cast<std::uint32_t>(cp),
        [](std::uint32_t value, const CategoryRange& range) { return value < range.first; });
    --it;
    return cp <= it->last ? it : nullptr;
}

} // namespace

utf8proc_category_t category(char32_t cp) noexcept {
    if (const CategoryRange* range = find_range(cp)) {
        return static_cast<utf8proc_category_t>(range->category);
    }
    return utf8proc_category(static_cast<utf8proc_int32_t>(cp));
}

const CaseMap* find_case(char32_t cp) noexcept {
    const auto* first = std::begin(kOnigCaseMaps);
    const auto* last  = std::end(kOnigCaseMaps);
    const auto* it    = std::lower_bound(
        first, last, static_cast<std::uint32_t>(cp),
        [](const CaseMap& map, std::uint32_t value) { return map.cp < value; });
    return it != last && it->cp == cp ? it : nullptr;
}

char32_t to_lower(char32_t cp) noexcept {
    if (find_range(cp) != nullptr) { return cp; }
    if (const CaseMap* map = find_case(cp)) { return map->lower; }
    return static_cast<char32_t>(utf8proc_tolower(static_cast<utf8proc_int32_t>(cp)));
}

char32_t to_upper(char32_t cp) noexcept {
    if (find_range(cp) != nullptr) { return cp; }
    if (const CaseMap* map = find_case(cp)) { return map->upper; }
    return static_cast<char32_t>(utf8proc_toupper(static_cast<utf8proc_int32_t>(cp)));
}

std::size_t differing_code_points() noexcept { return kDifferingCodePoints; }

} // namespace ninferno::text::onig_unicode
