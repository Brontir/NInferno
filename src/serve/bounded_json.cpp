#include "serve/bounded_json.h"

#include "serve/request_validation.h"

#include <exception>
#include <string>

namespace ninferno::serve {
namespace {

// The document builder RequestJson::parse uses, which also counts the values it builds and stops
// the parse (returns false from the SAX event) at the limit, before the next one is allocated. A
// callback parser would do the same, but nlohmann's takes quadratic time on a long array. The base
// class is nlohmann's internal one; a single pass costs no more than parse().
template <class InputAdapter>
class BoundedDomBuilder final
    : public nlohmann::detail::json_sax_dom_parser<RequestJson, InputAdapter> {
    using Base = nlohmann::detail::json_sax_dom_parser<RequestJson, InputAdapter>;

public:
    BoundedDomBuilder(RequestJson& root, std::size_t limit) : Base(root), limit_(limit) {}

    [[nodiscard]] bool exceeded() const noexcept { return exceeded_; }

    [[nodiscard]] bool too_deep() const noexcept { return too_deep_; }

    bool null() { return admit() && Base::null(); }

    bool boolean(bool value) { return admit() && Base::boolean(value); }

    bool number_integer(RequestJson::number_integer_t value) {
        return admit() && Base::number_integer(value);
    }

    bool number_unsigned(RequestJson::number_unsigned_t value) {
        return admit() && Base::number_unsigned(value);
    }

    bool number_float(RequestJson::number_float_t value, const std::string& text) {
        return admit() && Base::number_float(value, text);
    }

    bool string(std::string& value) { return admit() && Base::string(value); }

    bool start_object(std::size_t elements) {
        return admit() && enter() && Base::start_object(elements);
    }

    bool end_object() {
        --depth_;
        return Base::end_object();
    }

    bool start_array(std::size_t elements) {
        return admit() && enter() && Base::start_array(elements);
    }

    bool end_array() {
        --depth_;
        return Base::end_array();
    }

    bool key(std::string& name) { return admit() && Base::key(name); }

private:
    // One more value (an object key counts: it is a string of its own).
    bool admit() noexcept {
        exceeded_ = exceeded_ || ++values_ > limit_;
        return !exceeded_;
    }

    // One level deeper; refused past kMaximumRequestJsonDepth, before anything is built there.
    bool enter() noexcept {
        too_deep_ = too_deep_ || ++depth_ > kMaximumRequestJsonDepth;
        return !too_deep_;
    }

    std::size_t limit_;
    std::size_t values_ = 0;
    std::size_t depth_  = 0;
    bool exceeded_      = false;
    bool too_deep_      = false;
};

} // namespace

JsonTextStatus parse_bounded_json(std::string_view text, std::size_t max_values, RequestJson& out) {
    BoundedDomBuilder<decltype(nlohmann::detail::input_adapter(text))> builder(out, max_values);
    bool parsed = false;
    try {
        parsed = RequestJson::sax_parse(text, &builder);
    } catch (const std::exception&) { return JsonTextStatus::Invalid; }
    if (builder.exceeded()) { return JsonTextStatus::TooManyValues; }
    if (builder.too_deep()) { return JsonTextStatus::TooDeep; }
    return parsed ? JsonTextStatus::Ok : JsonTextStatus::Invalid;
}

} // namespace ninferno::serve
