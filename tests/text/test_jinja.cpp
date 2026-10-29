#include "text/jinja.h"

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::ordered_json;
using ninferno::text::JinjaTemplate;

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << message << '\n';
    return 1;
}

int language_semantics() {
    struct Example {
        std::string source;
        Json input;
        std::string expected;
    };

    const Example cases[]{
        {"{{ missing is undefined }}|{{ value is none }}|{{ value|default('fallback') }}|{{ "
         "value|default('fallback', true) }}",
         {{"value", nullptr}},
         "True|True|None|fallback"},
        {"{{ missing }}|{{ [1, none, true, 'x'] }}|{{ 12 ~ none }}", Json::object(),
         "|[1, None, True, 'x']|12None"},
        {"{{ text|length }}|{{ text[-1] }}|{{ text[::-1] }}|{{ text[:-1:-1] }}|{{ text[99:] }}",
         {{"text", "北京🌏"}},
         "3|🌏|🌏京北||"},
        {"{% for c in text %}[{{ c }}]{% endfor %}", {{"text", "北京🌏"}}, "[北][京][🌏]"},
        {"{{ text|upper }}|{{ text|lower }}|{{ text|capitalize }}",
         {{"text", "straße ΟΣ İ"}},
         "STRASSE ΟΣ İ|straße ος i̇|Straße ος i̇"},
        {"{{ text|trim }}|{{ text.split()|tojson }}|{{ text.rsplit(none,1)|tojson }}",
         {{"text", "\u0085北京\u3000two\tend\u001c"}},
         "北京　two\tend|[\"北京\", \"two\", \"end\"]|[\"\u0085北京　two\", \"end\"]"},
        {"{{ text|trim }}|{{ text.strip('🌏x\u3000\u2003') }}|{{ text.lstrip() }}|{{ text.rstrip() "
         "}}|{{ text[2:-2] }}|{{ text[:1] }}|{{ text[-1:] }}|{{ text[5:3] }}",
         {{"text", "\u3000🌏x北 京x🌏\u2003"}},
         "🌏x北 京x🌏|北 京|🌏x北 京x🌏\u2003|\u3000🌏x北 京x🌏|x北 京x|\u3000|\u2003|"},
        {"{{ text.replace('', '-', 2) }}|{{ text.replace('京', 'x', 1) }}",
         {{"text", "北京🌏"}},
         "-北-京🌏|北x🌏"},
        {"{{ value|tojson(ensure_ascii=true,sort_keys=true,separators=(',',':')) }}",
         {{"value", {{"z", "🌏"}, {"a", false}}}},
         "{\"a\":false,\"z\":\"\\ud83c\\udf0f\"}"},
        {"{% macro show(x) %}{{ x|default('local') }}{% endmacro %}{% set ns=namespace(n=0) %}{% "
         "for x in [1,2] %}{% set ns.n=ns.n+x %}{% endfor %}{{ ns.n }}|{{ show() }}",
         Json::object(), "3|local"},
        {"{% set x='outer' %}{% macro show(a='A',b='B') %}{{ x }}:{{ a }}:{{ b }}{% endmacro %}{% "
         "for x in ['inner'] %}{{ show(b='Z') }}{% endfor %}",
         Json::object(), "outer:A:Z"},
        {"{% set n=namespace(f=none) %}{% for x in ['captured'] %}{% macro show() %}{{ x }}{% "
         "endmacro %}{% set n.f=show %}{% endfor %}{{ n.f() }}",
         Json::object(), "captured"},
        {"{% macro wrap() %}[{{ caller('value') }}]{% endmacro %}{% call(x) wrap() %}{{ x }}{% "
         "endcall %}",
         Json::object(), "[value]"},
        {"{{ number }}", {{"number", 1.23456789}}, "1.23456789"},
        {"{% filter upper %}{{ text }}{% endfilter %}", {{"text", "straße"}}, "STRASSE"},
        // llama.cpp #20782 (GHSA-q9j6-4hhc-rq9p, GHSA-2q4c-9gq5-5vfp): comparing containers of
        // different lengths read past the shorter one and could call a prefix equal.
        {"{{ [1, 2] == [1, 2, 3] }}|{{ [1, 2, 3] == [1, 2] }}|{{ {'a': 1} == {'a': 1, 'b': 2} }}|"
         "{{ {'a': 1, 'b': 2} == {'a': 1} }}|{{ [] == [1] }}|"
         "{{ [[1, 2]] == [[1, 2, 3]] }}",
         Json::object(), "False|False|False|False|False|False"},
        // llama.cpp #20913: an empty computed member `a[]` bound a null expression (UB); it is
        // an undefined value, as is a member named by an undefined value.
        {"{{ a[]|default('fallback') }}|{{ a[] is undefined }}|{{ a[undefined] is undefined }}",
         {{"a", {{"name", "Bob"}}}},
         "fallback|True|True"},
        // lstrip_blocks at the start of a template (found by tests/text/jinja_differential.py):
        // the first character survives when only blanks separate it from the first block, a
        // leading newline is not part of the block's line, and pure blanks are still stripped.
        {"a {%- if t %}b{% endif %}", {{"t", true}}, "ab"},
        {"a {% if t -%} b{% endif %}", {{"t", true}}, "a b"},
        {"x {#- c #}y", Json::object(), "xy"},
        {"\n{% if t %}b{% endif %}", {{"t", true}}, "\nb"},
        {"  {% if t %}b{% endif %}", {{"t", true}}, "b"},
        // Jinja2's sort and dictsort fold case unless case_sensitive=true; dictsort yields
        // (key, value) pairs.
        {"{{ ['b', 'C', 'a']|sort|join(',') }}|"
         "{{ ['b', 'C', 'a']|sort(case_sensitive=true)|join(',') }}|"
         "{{ people|sort(attribute='name')|map(attribute='name')|join(',') }}|"
         "{% for k, v in d|dictsort %}{{ k }}{{ v }}{% endfor %}|{{ (d|dictsort)[0][0] }}|"
         "{{ d|dictsort|length }}",
         {{"d", {{"b", 1}, {"C", 2}, {"a", 3}}},
          {"people", Json::array({Json{{"name", "bob"}}, Json{{"name", "Ann"}}})}},
         "a,b,C|C,a,b|Ann,bob|a3b1C2|a|3"},
        // llama.cpp #30533: a {% set %} in a loop body stays in its iteration; Qwen's template
        // sets reasoning_content per message, which leaked into the next message's turn.
        {"{% for i in [1, 2] %}{% if i == 1 %}{% set x = 7 %}{% endif %}{{ x is defined }};"
         "{% endfor %}",
         Json::object(), "True;False;"},
        {"{% set x = 9 %}{% for i in [1, 2] %}{% if i == 1 %}{% set x = 7 %}{% endif %}{{ x }};"
         "{% endfor %}{{ x }}",
         Json::object(), "7;9;9"},
        {"{% for m in messages %}{% if m.reasoning_content is string %}{% set reasoning_content = "
         "m.reasoning_content %}{% elif '</think>' in m.content %}{% set reasoning_content = "
         "m.content.split('</think>')[0].split('<think>')[-1] %}{% endif %}[{{ reasoning_content "
         "is defined }}:{{ reasoning_content if reasoning_content is defined else '' }}]"
         "{% endfor %}",
         {{"messages", Json::array({Json{{"reasoning_content", "first"}, {"content", "answer"}},
                                    Json{{"content", "answer"}},
                                    Json{{"reasoning_content", nullptr}, {"content", "answer"}},
                                    Json{{"content", "<think>embedded</think>answer"}},
                                    Json{{"content", "answer"}}})}},
         "[True:first][False:][False:][True:embedded][False:]"},
        // A loop whose iterations all break or continue still ran: its else block does not.
        {"{% for i in [1, 2] %}{% break %}{% else %}empty{% endfor %}|"
         "{% for i in [1, 2] %}{% continue %}{% else %}empty{% endfor %}|"
         "{% for i in [] %}x{% else %}empty{% endfor %}",
         Json::object(), "||empty"},
        // llama.cpp #30941: a namespace has no dict methods (ns.items is the attribute); a dict
        // keeps them.
        {"{% set ns = namespace(items=[]) %}{{ 'yes' if ns.items else 'no' }}{% set ns.items = "
         "ns.items + [1] %}{{ ns.items | length }}{{ 'yes' if ns.items else 'no' }}|"
         "{% set d = dict(items=1) %}{% for k, v in d.items() %}{{ k }}={{ v }}{% endfor %}",
         Json::object(), "no1yes|items=1"},
        // llama.cpp #31059: select / reject take a test (and its arguments), selectattr /
        // rejectattr an attribute and then a test.
        {"{{ n|select('odd')|join(',') }}|{{ n|reject('odd')|join(',') }}|"
         "{{ v|select|join(',') }}|{{ n|select('divisibleby', 2)|join(',') }}|"
         "{{ n|select('divisibleby', num=2)|join(',') }}|"
         "{% for p in t|selectattr('type', 'defined') %}{{ p.type }} {% endfor %}|"
         "{{ t|rejectattr('type', 'defined')|list|length }}|"
         "{% for p in o|selectattr('odd', 'even') %}{{ p.odd }} {% endfor %}",
         {{"n", Json::array({1, 2, 3, 4, 5})},
          {"v", Json::array({0, 1, "", "a", nullptr, false})},
          {"t", Json::array({Json{{"type", "string"}}, Json{{"enum", Json::array({1})}},
                             Json{{"type", "integer"}}})},
          {"o", Json::array({Json{{"odd", 1}}, Json{{"odd", 2}}, Json{{"odd", 4}}})}},
         "1,3,5|2,4|1,a|2,4|2,4|string integer |1|2 4 "},
        // llama.cpp #30819: unique, first of each value in order; strings fold case unless
        // case_sensitive (as jinja2).
        {"{{ n|unique|join(',') }}|{{ s|unique|join(',') }}|"
         "{{ s|unique(case_sensitive=true)|join(',') }}|{{ s|unique(true)|list|length }}|"
         "{{ r|unique(attribute='id')|join(attribute='v') }}|"
         "{{ r|unique(case_sensitive=true, attribute='id')|join(attribute='v') }}|"
         "{{ m|unique|list|length }}|{{ []|unique|list|length }}|"
         "{{ p|unique(attribute=0)|map(attribute=1)|join(',') }}",
         {{"n", Json::array({3, 1, 3, 2, 1})},
          {"s", Json::array({"a", "A", "b", "B", "a"})},
          {"r", Json::array({Json{{"id", "x"}, {"v", 1}}, Json{{"id", "X"}, {"v", 2}},
                             Json{{"id", "x"}, {"v", 3}}})},
          {"m", Json::array({1, 1.0, 2, "a", "a"})},
          {"p", Json::array({Json::array({"x", 1}), Json::array({"y", 2}),
                             Json::array({"x", 3})})}},
         "3,1,2|a,b|a,A,b,B|4|1|12|3|0|1,2"},
        // llama.cpp #30929: \xHH escapes, two digits, code point HH as UTF-8.
        {"{{ message.content | replace(\"\\x00\", \"\") }}|{{ '\\xE9' }}|{{ '\\x414' }}|"
         "{{ '\\\\xE9' }}",
         {{"message", {{"content", std::string("\0a\0b\0", 5)}}}},
         "ab|\xc3\xa9|A4|\\xE9"},
    };
    for (const char* source : {"{{ '\\x' }}", "{{ '\\x0' }}", "{{ '\\xG1' }}", "{{ '\\x1G' }}",
                               "{{ '\\x", "{{ '\\x0"}) {
        bool rejected = false;
        try {
            (void)JinjaTemplate(source, "hex-test").render(Json::object());
        } catch (const std::exception&) { rejected = true; }
        if (!rejected) {
            std::cerr << "malformed hex escape accepted: " << source << '\n';
            return 1;
        }
    }
    int failures = 0;
    for (const auto& item : cases) {
        const auto actual = JinjaTemplate(item.source, "language-test").render(item.input).text;
        if (actual != item.expected) {
            std::cerr << "Jinja semantic mismatch in " << item.source
                      << "\nexpected: " << Json(item.expected).dump()
                      << "\nactual: " << Json(actual).dump() << '\n';
            ++failures;
        }
    }
    for (const char* source : {"{{ none + 'x' }}", "{{ 'abc'[::0] }}", "{{ missing|tojson }}",
                               "{{ value|unknown_filter }}"}) {
        try {
            (void)JinjaTemplate(source, "error-test").render(Json::object());
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).starts_with("error-test:"),
                              "template error omitted its source name");
            failures +=
                check(std::string(error.what()).find("While executing") != std::string::npos,
                      "template error lost its source location");
        }
    }
    // A template that refuses the request (raise_exception) speaks to the client: its message
    // arrives as written, with no source name and no trace of where the template raised it, also
    // from a macro called inside a loop (each nested statement used to wrap it again).
    for (const char* source :
         {"{% if messages|length == 0 %}{{ raise_exception('No user query found.') }}{% endif %}",
          "{% macro refuse(m) %}{{ raise_exception('No user query found.') }}{% endmacro %}"
          "{% for m in ['a'] %}{% if m %}{{ refuse(m) }}{% endif %}{% endfor %}"}) {
        try {
            (void)JinjaTemplate(source, "raise-test").render(Json{{"messages", Json::array()}});
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()) == "No user query found.",
                              "raise_exception message did not arrive as written");
        }
    }
    return failures;
}

int origins_and_requests() {
    const JinjaTemplate compiled(
        "{% filter upper %}{{ text|trim }}{% endfilter %}|{{ tool|tojson }}", "shared-test");
    const Json context{{"text", " hello "}, {"tool", {{"name", "f"}}}};
    const std::vector<ninferno::text::TemplateInputRegion> regions{{"/text", 7}, {"/tool", 9}};
    ninferno::text::TemplateRenderOptions options{.timestamp = 0, .regions = regions};
    const auto result = compiled.render(context, options);
    int failures = check(result.text == "HELLO|{\"name\": \"f\"}" && result.regions.size() == 2 &&
                             result.regions[0].tag == 7 && result.regions[0].begin == 0 &&
                             result.regions[0].end == 5 && !result.regions[0].source_offset &&
                             result.regions[1].tag == 9 && result.regions[1].begin == 6 &&
                             result.regions[1].end == result.text.size(),
                         "transformed/JSON input regions were lost or incorrectly mapped");
    failures += check(compiled.render(context).regions.empty(),
                      "unannotated render retained request origins");
    const auto clipped =
        JinjaTemplate("{{ text[1:3] }}", "slice-test").render({{"text", "a北京c"}}, options);
    failures += check(clipped.regions.size() == 1 && clipped.regions[0].source_offset == 1 &&
                          clipped.regions[0].end == 6,
                      "Unicode slice lost byte origin mapping");
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.push_back(std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i)
                if (compiled.render(context, options).text != result.text) return false;
            return true;
        }));
    }
    for (auto& worker : workers)
        failures += check(worker.get(), "shared template leaked mutable request state");

    struct Cancelled : std::runtime_error {
        Cancelled() : std::runtime_error("cancelled") {}
    };

    int calls          = 0;
    options.checkpoint = [&] {
        if (++calls > 2) throw Cancelled();
    };
    try {
        (void)JinjaTemplate("{% for i in range(1000000) %}{{ i }}{% endfor %}", "cancel-test")
            .render(Json::object(), options);
        failures += check(false, "template ignored cancellation");
    } catch (const Cancelled&) {}
    options.checkpoint = {};
    const JinjaTemplate clock("{{ strftime_now('%Y') }}", "clock-test");
    failures += check(clock.render(Json::object(), options).text ==
                          clock.render(Json::object(), options).text,
                      "time snapshot was not reused");
    return failures;
}

int literal_content() {
    struct Example {
        const char* source;
        Json context;
        const char* expected;
    };

    const Example cases[]{
        {"{{ data|lower|trim }}", {{"data", " <|IMAGE_PAD|> "}}, "<|image_pad|>"},
        {"{{ data[1:-1][::-1][::-1] }}", {{"data", "x<|image_pad|>x"}}, "<|image_pad|>"},
        {"{{ data.split('|x|')|join('') }}", {{"data", "<|image|x|_pad|>"}}, "<|image_pad|>"},
        {"{{ data|replace('x', '') }}", {{"data", "<|image_xpad|>"}}, "<|image_pad|>"},
        {"{{ data.format('') }}", {{"data", "<|image_{}pad|>"}}, "<|image_pad|>"},
        {"{{ '{}'.format(data) }}", {{"data", "<|image_pad|>"}}, "<|image_pad|>"},
        {"{{ data|indent(2, true)|trim|safe }}", {{"data", "<|image_pad|>"}}, "<|image_pad|>"},
        {"{{ data|indent(2) }}", {{"data", "\n<|image_pad|>\n"}}, "\n  <|image_pad|>\n"},
        {"{% macro show(x) %}{{ x }}{% endmacro %}{% set ns=namespace(x=show(data)) %}{{ ns.x }}",
         {{"data", "<|image_pad|>"}},
         "<|image_pad|>"},
        {"{% set captured %}{{ data }}{% endset %}{{ captured }}",
         {{"data", "<|image_pad|>"}},
         "<|image_pad|>"},
        {"{{ data*2 }}", {{"data", "<|image_pad|>"}}, "<|image_pad|><|image_pad|>"},
        {"{{ data|tojson }}",
         {{"data", {{"<|image_pad|>", "<|image_pad|>"}}}},
         "{\"<|image_pad|>\": \"<|image_pad|>\"}"},
        {"{{ data|string }}", {{"data", Json::array({"<|image_pad|>"})}}, "['<|image_pad|>']"},
        {"{% for k, v in data.items() %}{{ k }}{{ v }}{% endfor %}",
         {{"data", {{"<|image_pad|>", "<|image_pad|>"}}}},
         "<|image_pad|><|image_pad|>"},
        {"{{ data|upper|lower }}", {{"data", "ß<|IMAGE_PAD|>İ"}}, "ss<|image_pad|>i̇"},
    };
    int failures                     = 0;
    constexpr std::string_view start = "<|im_start|>", end = "<|im_end|>", pad = "<|image_pad|>";
    for (const auto& item : cases) {
        const auto output =
            JinjaTemplate(std::string(start) + item.source + std::string(end), "literal-content")
                .render(item.context);
        if (output.text != std::string(start) + item.expected + std::string(end)) {
            std::cerr << "literal-content text mismatch: " << item.source << '\n';
            ++failures;
        }
        failures += check(!ninferno::text::overlaps(output.literal_spans, 0, start.size()) &&
                              !ninferno::text::overlaps(output.literal_spans,
                                                        output.text.size() - end.size(),
                                                        output.text.size()),
                          "input content shielded surrounding template controls");
        for (auto pos = output.text.find(pad); pos != std::string::npos;
             pos      = output.text.find(pad, pos + pad.size())) {
            if (!ninferno::text::overlaps(output.literal_spans, pos, pos + pad.size())) {
                std::cerr << "literal-content marker lost through: " << item.source << '\n';
                ++failures;
            }
        }
    }
    const auto mixed =
        JinjaTemplate("{{ ('<|IM_START|>' ~ data ~ '<|IM_END|>')|lower }}", "mixed-case")
            .render({{"data", "ß<|IMAGE_PAD|>İ"}});
    failures +=
        check(mixed.text == "<|im_start|>ß<|image_pad|>i̇<|im_end|>" &&
                  mixed.literal_spans.size() == 1 && mixed.literal_spans[0].begin == start.size() &&
                  mixed.literal_spans[0].end == mixed.text.size() - end.size(),
              "mixed Unicode case conversion lost content/control boundaries");
    const auto sigma =
        JinjaTemplate("{{ ('Ο' ~ data)|lower }}", "mixed-sigma").render({{"data", "Σ"}});
    failures += check(sigma.text == "ος" && sigma.literal_spans.size() == 1 &&
                          sigma.literal_spans[0].begin == 2 && sigma.literal_spans[0].end == 4,
                      "case conversion lost context across content/control boundaries");
    const auto explicit_control =
        JinjaTemplate("{{ data.replace('x', '<|image_pad|>') }}", "replace")
            .render({{"data", "x"}});
    failures += check(explicit_control.text == pad && explicit_control.literal_spans.empty(),
                      "template-authored replacement lost its control meaning");
    const std::vector<std::string> tokens{"bos_token"};
    const auto variables = JinjaTemplate("{{ bos_token }}{{ data }}", "token-variable")
                               .render({{"bos_token", "<|im_start|>"}, {"data", "<|im_start|>"}},
                                       {.control_variables = tokens});
    failures +=
        check(!ninferno::text::overlaps(variables.literal_spans, 0, start.size()) &&
                  ninferno::text::overlaps(variables.literal_spans, start.size(), 2 * start.size()),
              "engine token variables and ordinary kwargs were not distinguished");
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--render") {
        for (std::string line; std::getline(std::cin, line);) {
            Json result;
            try {
                const auto input = Json::parse(line);
                result["text"]   = JinjaTemplate(input.at("source"), "reference-case")
                                     .render(input.at("context"))
                                     .text;
                result["ok"] = true;
            } catch (const std::exception& error) {
                result["ok"]    = false;
                result["error"] = error.what();
            }
            std::cout << result.dump() << '\n';
        }
        return 0;
    }
    try {
        return language_semantics() + origins_and_requests() + literal_content() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
