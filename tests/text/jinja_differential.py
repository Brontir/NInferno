"""Differential test of NInferno's jinja engine against Python Jinja2.

    python tests/text/jinja_differential.py RENDERER [TEMPLATE.jinja ...]

RENDERER is ninferno_jinja_test (its --render mode; an ASan build works too). The corpus is a table
of expressions and statements chat templates use (filters, tests, string methods, loops, macros,
namespaces, slicing, tojson, whitespace control), each rendered over a fixed context, plus every
TEMPLATE given, rendered over a matrix of dialogs (tools, CJK and emoji, thinking, tool results,
multi-turn, system and developer turns). Jinja2 runs as transformers configures it for chat
templates: ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols, tojson with
ensure_ascii=False, raise_exception.

A case agrees when both render the same text or both refuse it. The report lists every
disagreement with its verdict: "differs" (both render, different text: silent, always a failure),
"gap" (NInferno refuses what Jinja2 renders) or "lenient" (NInferno renders what Jinja2 refuses). Exit
status 1 on any disagreement outside KNOWN_GAPS and KNOWN_LENIENT.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:  # the reference renderer is optional: CTest reports a skip
    print("jinja_differential: Python jinja2 is not installed, skipped")
    sys.exit(77)


class TemplateError(ValueError):
    pass


def fail(message):
    raise TemplateError(message)


def environment():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, **{"ensure_ascii": False, **kw})
    env.globals["raise_exception"] = fail
    return env


CONTEXT = {
    "s": "  Hello, World  ", "cjk": "北京欢迎你🌏", "n": 7, "f": 2.5, "neg": -3, "t": True,
    "none": None, "empty": "", "zero": 0,
    "items": [3, 1, 2], "words": ["b", "a", "c"], "nested": [[1, 2], [3], []],
    "d": {"b": 2, "a": 1, "c": {"x": [1, {"y": "z"}]}},
    "people": [{"name": "Ann", "age": 31, "role": "user"}, {"name": "bob", "age": 25},
               {"name": "Cy", "age": 40, "role": "assistant"}],
    "msg": {"role": "user", "content": "hi", "tool_calls": [
        {"function": {"name": "f", "arguments": {"a": 1, "b": [1, 2]}}}]},
}

EXPRESSIONS = [
    # arithmetic and comparison
    "{{ n + 1 }}|{{ n - 10 }}|{{ n * f }}|{{ n / 2 }}|{{ n // 2 }}|{{ n % 3 }}|{{ 2 ** 10 }}",
    "{{ n > 3 }}|{{ n == 7.0 }}|{{ 'a' < 'b' }}|{{ none is none }}|{{ n != n }}",
    "{{ -n }}|{{ neg|abs }}|{{ f|round }}|{{ 2.675|round(2) }}|{{ f|int }}|{{ '12'|int + 1 }}",
    "{{ '3.5'|float }}|{{ n|string ~ 'x' }}|{{ 1 ~ 2 }}|{{ none ~ 'a' }}|{{ t ~ '' }}",
    "{{ [1, 2] + [3] }}|{{ 'ab' * 3 }}|{{ (1, 2) }}|{{ {'k': 1} }}",
    # strings
    "{{ s|trim }}|{{ s|lower }}|{{ s|upper }}|{{ s|title }}|{{ s|capitalize }}|{{ s|length }}",
    "{{ s.strip() }}|{{ s.lstrip() }}|{{ s.rstrip() }}|{{ s.strip(' H') }}|{{ s.split(',') }}",
    "{{ s.replace('l', 'L') }}|{{ s.replace('l', 'L', 1) }}|{{ s.startswith('  He') }}|"
    "{{ s.endswith('d  ') }}|{{ s.find('W') }}|{{ s.count('l') }}",
    "{{ 'a,b,,c'.split(',') }}|{{ 'a b  c'.split() }}|{{ 'a-b-c'.rsplit('-', 1) }}|"
    "{{ 'x'.join(['1', '2']) }}|{{ words|join(', ') }}",
    "{{ cjk|length }}|{{ cjk[0] }}|{{ cjk[-1] }}|{{ cjk[1:3] }}|{{ cjk|upper }}|{{ cjk|reverse|join }}",
    "{{ 'Hello'[::-1] }}|{{ 'Hello'[1:] }}|{{ 'Hello'[:-2] }}|{{ 'Hello'[::2] }}",
    "{{ '%s=%d' % ('a', 3) }}|{{ '{}-{}'.format(1, 2) }}|{{ 'ab'|center(6) }}|{{ 'x'|indent(2) }}",
    "{{ 'Ab Cd'.lower().title() }}|{{ '  '|trim|length }}|{{ 'a\\nb'|wordcount }}",
    "{{ 'tool_call' in s }}|{{ 'He' in s }}|{{ 'x' not in 'abc' }}",
    # lists
    "{{ items|sort }}|{{ items|sort(reverse=true) }}|{{ words|sort|first }}|{{ items|last }}",
    "{{ items|max }}|{{ items|min }}|{{ items|sum }}|{{ items|length }}|{{ items|list }}",
    "{{ items|reverse|list }}|{{ items[1:] }}|{{ items[-1] }}|{{ items[::-1] }}",
    "{{ nested|length }}|{{ nested[0][1] }}|{{ nested|map('length')|list }}",
    "{{ people|map(attribute='name')|join(',') }}|{{ people|selectattr('role')|list|length }}",
    "{{ people|selectattr('role', 'equalto', 'user')|map(attribute='name')|first }}",
    "{{ people|rejectattr('role')|map(attribute='name')|list }}",
    "{{ items|select('odd')|list }}|{{ items|reject('odd')|list }}|{{ items|select('gt', 1)|list }}",
    "{{ people|sort(attribute='age')|map(attribute='name')|join }}",
    "{{ people|sort(attribute='name')|map(attribute='name')|join }}",
    "{{ range(3)|list }}|{{ range(1, 10, 3)|list }}|{{ range(5)|batch(2)|list }}",
    "{{ items|unique|list }}|{{ [1, 1, 2]|unique|list }}",
    "{{ people|groupby('role')|length }}",
    # dicts
    "{{ d|length }}|{{ d.a }}|{{ d['b'] }}|{{ d.c.x[1].y }}|{{ d.get('zz', 'dflt') }}|{{ d.get('a') }}",
    "{{ d|dictsort }}|{{ d.keys()|list }}|{{ d.values()|list|length }}",
    "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}",
    "{% for k, v in d|dictsort %}{{ k }};{% endfor %}",
    "{{ 'a' in d }}|{{ 'zz' in d }}|{{ d.zz is defined }}|{{ d.zz|default('D') }}",
    # tests
    "{{ n is number }}|{{ s is string }}|{{ d is mapping }}|{{ items is sequence }}|"
    "{{ items is iterable }}|{{ none is none }}|{{ missing is undefined }}|{{ n is odd }}",
    "{{ n is divisibleby 7 }}|{{ 5 is even }}|{{ t is true }}|{{ zero is false }}|"
    "{{ s is lower }}|{{ 'ABC' is upper }}",
    "{{ none is not none }}|{{ empty is defined }}|{{ empty|default('e', true) }}",
    # tojson
    "{{ d|tojson }}",
    "{{ msg.tool_calls[0].function.arguments|tojson }}",
    "{{ cjk|tojson }}|{{ none|tojson }}|{{ t|tojson }}|{{ f|tojson }}|{{ items|tojson }}",
    "{{ d|tojson(indent=2) }}",
    "{{ {'b': 1, 'a': [1, 'x']}|tojson(separators=(',', ':')) }}",
    "{{ 'quote\"and\\\\back'|tojson }}|{{ '<tag>&'|tojson }}",
    # control flow and loops
    "{% for i in items %}{{ loop.index }}{{ loop.index0 }}{{ loop.first }}{{ loop.last }}"
    "{{ loop.length }}{{ loop.revindex }};{% endfor %}",
    "{% for i in items %}{{ loop.cycle('a', 'b') }}{% endfor %}",
    "{% for i in items if i > 1 %}{{ i }}{% else %}none{% endfor %}",
    "{% for i in [] %}x{% else %}empty{% endfor %}",
    "{% for i in items %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endfor %}",
    "{% for i in items %}{% if i == 1 %}{% break %}{% endif %}{{ i }}{% endfor %}",
    "{% for p in people %}{{ loop.previtem.name if loop.previtem else '-' }}{% endfor %}",
    "{% for a, b in [[1, 2], [3, 4]] %}{{ a + b }}{% endfor %}",
    "{% if n > 10 %}big{% elif n > 5 %}mid{% else %}small{% endif %}",
    "{{ 'yes' if t else 'no' }}|{{ 'yes' if none else 'no' }}|{{ n if n > 100 }}",
    "{% set x = 3 %}{% set y = x * 2 %}{{ y }}",
    "{% set ns = namespace(total=0, seen=[]) %}{% for i in items %}{% set ns.total = ns.total + i %}"
    "{% endfor %}{{ ns.total }}",
    "{% set block %}inner {{ n }}{% endset %}[{{ block }}]",
    "{% macro greet(name, punct='!') %}Hi {{ name }}{{ punct }}{% endmacro %}{{ greet('A') }}"
    "{{ greet('B', punct='?') }}",
    "{% macro m() %}{{ varargs|length }}{{ kwargs|length }}{% endmacro %}{{ m(1, 2, k=3) }}",
    "{% raw %}{{ not rendered }}{% endraw %}",
    "{# comment #}after",
    "a {%- if t %} b {% endif -%} c",
    "  {% if t %}\n    line\n  {% endif %}\nend",
    "{% for i in items %}\n  {{ i }}\n{% endfor %}",
    "{{ 'x' }}\n\n{{- 'y' -}}\n\n{{ 'z' }}",
    "{% filter upper %}up {{ s|trim }}{% endfilter %}",
    "{% with a = 1 %}{{ a }}{% endwith %}{{ a is defined }}",
    # errors both must refuse
    "{{ raise_exception('boom') }}",
    "{{ 1 / 0 }}",
    "{{ d.missing.deeper }}",
    "{% for %}",
    "{{ items|nosuchfilter }}",
]

MESSAGE_MATRIX = [
    [{"role": "user", "content": "你好🌏, hello"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
    [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
     {"role": "user", "content": "q2"}],
    [{"role": "user", "content": "weather?"},
     {"role": "assistant", "content": "", "tool_calls": [
         {"id": "c1", "type": "function",
          "function": {"name": "get_weather", "arguments": {"city": "Oslo", "days": 3}}}]},
     {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 7}"},
     {"role": "user", "content": "and tomorrow?"}],
    [{"role": "user", "content": "think"},
     {"role": "assistant", "content": "done", "reasoning_content": "hmm"},
     {"role": "user", "content": "again"}],
    [{"role": "developer", "content": "rules"}, {"role": "user", "content": "go"}],
    [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"}]}],
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"]}}}]


# Constructs NInferno's engine refuses although Jinja2 renders them (feature gaps, reported as
# honest errors rather than wrong text), and the two it renders where Jinja2 raises. Anything not
# listed -- above all a template both render differently -- fails the test.
KNOWN_GAPS = {
    "{{ n + 1 }}|{{ n - 10 }}|{{ n * f }}|{{ n / 2 }}|{{ n // 2 }}|{{ n % 3 }}|{{ 2 ** 10 }}",
    "{{ n // 2 }}",
    "{{ 2 ** 10 }}",
    "{{ n > 3 }}|{{ n == 7.0 }}|{{ 'a' < 'b' }}|{{ none is none }}|{{ n != n }}",
    "{{ 'a' < 'b' }}",
    "{{ -n }}|{{ neg|abs }}|{{ f|round }}|{{ 2.675|round(2) }}|{{ f|int }}|{{ '12'|int + 1 }}",
    "{{ -n }}",
    "{{ f|round }}",
    "{{ 2.675|round(2) }}",
    "{{ s.replace('l', 'L') }}|{{ s.replace('l', 'L', 1) }}|{{ s.startswith('  He') }}|{{ s.endswith('d  ') }}|{{ s.find('W') }}|{{ s.count('l') }}",
    "{{ s.find('W') }}",
    "{{ s.count('l') }}",
    "{{ 'a,b,,c'.split(',') }}|{{ 'a b  c'.split() }}|{{ 'a-b-c'.rsplit('-', 1) }}|{{ 'x'.join(['1', '2']) }}|{{ words|join(', ') }}",
    "{{ 'x'.join(['1', '2']) }}",
    "{{ cjk|length }}|{{ cjk[0] }}|{{ cjk[-1] }}|{{ cjk[1:3] }}|{{ cjk|upper }}|{{ cjk|reverse|join }}",
    "{{ cjk|reverse|join }}",
    "{{ '%s=%d' % ('a', 3) }}|{{ '{}-{}'.format(1, 2) }}|{{ 'ab'|center(6) }}|{{ 'x'|indent(2) }}",
    "{{ '%s=%d' % ('a', 3) }}",
    "{{ 'ab'|center(6) }}",
    "{{ 'Ab Cd'.lower().title() }}|{{ '  '|trim|length }}|{{ 'a\\nb'|wordcount }}",
    "{{ 'a\\nb'|wordcount }}",
    "{{ items|max }}|{{ items|min }}|{{ items|sum }}|{{ items|length }}|{{ items|list }}",
    "{{ items|sum }}",
    "{{ nested|length }}|{{ nested[0][1] }}|{{ nested|map('length')|list }}",
    "{{ nested|map('length')|list }}",
    "{{ items|select('odd')|list }}|{{ items|reject('odd')|list }}|{{ items|select('gt', 1)|list }}",
    "{{ items|select('odd')|list }}",
    "{{ items|reject('odd')|list }}",
    "{{ range(3)|list }}|{{ range(1, 10, 3)|list }}|{{ range(5)|batch(2)|list }}",
    "{{ range(5)|batch(2)|list }}",
    "{{ items|unique|list }}|{{ [1, 1, 2]|unique|list }}",
    "{{ items|unique|list }}",
    "{{ [1, 1, 2]|unique|list }}",
    "{{ n is divisibleby 7 }}|{{ 5 is even }}|{{ t is true }}|{{ zero is false }}|{{ s is lower }}|{{ 'ABC' is upper }}",
    "{{ n is divisibleby 7 }}",
    "{% for i in items %}{{ loop.cycle('a', 'b') }}{% endfor %}",
    "{% macro m() %}{{ varargs|length }}{{ kwargs|length }}{% endmacro %}{{ m(1, 2, k=3) }}",
    "{% raw %}{{ not rendered }}{% endraw %}",
    "{% with a = 1 %}{{ a }}{% endwith %}{{ a is defined }}",
}
KNOWN_LENIENT = {
    "{{ 1 / 0 }}",
    "{{ d.missing.deeper }}",
}


def template_contexts():
    for messages in MESSAGE_MATRIX:
        for tools in (None, TOOLS):
            for thinking in (None, True, False):
                ctx = {"messages": messages, "add_generation_prompt": True}
                if tools:
                    ctx["tools"] = tools
                if thinking is not None:
                    ctx["enable_thinking"] = thinking
                yield ctx


def main() -> int:
    renderer = sys.argv[1]
    templates = {Path(p).name + f"#{i}": Path(p).read_text(encoding="utf-8")
                 for i, p in enumerate(sys.argv[2:])}
    env = environment()
    # A table row of expression tags joined by "|" also runs tag by tag, so one unsupported
    # construct does not hide the others.
    expressions = []
    for source in EXPRESSIONS:
        expressions.append(source)
        parts = source.split("}}|{{")
        if len(parts) > 1 and "{%" not in source:
            expressions += [("" if i == 0 else "{{") + p + ("" if i == len(parts) - 1 else "}}")
                            for i, p in enumerate(parts)]
    cases = [("expr", source, CONTEXT) for source in expressions]
    for name, source in templates.items():
        cases += [(name, source, ctx) for ctx in template_contexts()]
    payload = "".join(json.dumps({"source": source, "context": ctx}, ensure_ascii=False) + "\n"
                      for _, source, ctx in cases)
    run = subprocess.run([renderer, "--render"], input=payload, text=True, capture_output=True,
                         timeout=600)
    if run.returncode != 0:
        print(run.stderr[-2000:], file=sys.stderr)
        return 1
    results = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(results) == len(cases), (len(results), len(cases))
    disagreements = []
    for (kind, source, ctx), got in zip(cases, results):
        try:
            expected = env.from_string(source).render(**ctx)
            ok_py = True
        except Exception as error:  # noqa: BLE001 - any refusal is a refusal
            expected = f"{type(error).__name__}: {error}"
            ok_py = False
        if ok_py and got["ok"] and expected == got["text"]:
            continue
        if ok_py == got["ok"] and not ok_py:
            continue  # both refuse
        # "differs": both render different text (silent, always a failure); "gap": NInferno refuses
        # what Jinja2 renders; "lenient": NInferno renders what Jinja2 refuses.
        verdict = "differs" if ok_py and got["ok"] else ("gap" if ok_py else "lenient")
        known = source in KNOWN_GAPS if verdict == "gap" else source in KNOWN_LENIENT
        disagreements.append({"kind": kind, "verdict": verdict, "known": known,
                              "source": source if kind == "expr" else None,
                              "context": None if kind == "expr" else ctx,
                              "jinja2": expected if ok_py else "REFUSED: " + expected,
                              "ninferno": got["text"] if got["ok"] else
                              "REFUSED: " + got.get("error", "")[:300]})
    unexpected = [d for d in disagreements if not d["known"]]
    print(json.dumps({"cases": len(cases), "disagreements": len(disagreements),
                      "unexpected": len(unexpected), "details": disagreements},
                     ensure_ascii=False, indent=1))
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
