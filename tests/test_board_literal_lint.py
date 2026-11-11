#!/usr/bin/env python3
"""Unit tests of tools/audit/board_literal_lint.py, and the tree is clean."""
import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "board_literal_lint", ROOT / "tools" / "audit" / "board_literal_lint.py")
lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lint)

# Every spelling of a hand-written board the lint must see (one flagged line each).
FLAGGED = {
    "plain": "auto b = Board{12, 0, 170, 96, 1792};",
    "namespaced": "const ninferno::Board b{8, 9, 128, 72, 1008};",
    "global_qualified": "::ninferno::Board b{12, 0, 170, 96, 1792};",
    "global_qualified_named_first": "::ninferno::Board b{kMajor, 0, 170, 96, 1792};",
    "global_qualified_table": "::ninferno::Board k[] = {{8, 6, 82}};",
    "equals": "const Board b = {8, 9, 128, 72, 1008};",
    "space": "x(Board {7, 5, 68, 5.5, 616})",
    "multiline": "inline constexpr Board kMine{\n    12, 0, 170, 96, 1792};",
    "multiline_comment_inside": "Board b{12, /* cc */ 0,\n  170, 96, 1792};",
    "designated": "Board k{.major = 12, .minor = 0, .sms = 170, .l2_mib = 96, .dram_gbps = 1792};",
    "parens": "Board(12, 0, 170, 96, 1792)",
    "parens_decl": "Board b(12, 0, 170, 96, 1792);",
    "array": "Board kBoards[] = {{12, 0, 170, 96, 1792}, {8, 9, 128, 72, 1008}};",
    "std_array": "std::array<Board, 1> k{{{12, 0, 170, 96, 1792}}};",
    "vector": "std::vector<Board> v = {{12, 0, 170, 96, 1792}};",
    "optional": "std::optional<Board> o = Board{12, 0, 170, 96, 1792};",
    "optional_ctor": "std::optional<Board> o{8, 6, 82, 6, 936};",
    "return": "Board f() { return {12, 0, 170, 96, 1792}; }",
    "argument": "g(1, {12, 0, 170, 96, 1792});",
    "assign": "b = {12, 0, 170, 96, 1792};",
    "const_after": "Board const b{12, 0, 170, 96, 1792};",
    "const_ref": "const Board& b = {12, 0, 170, 96, 1792};",
    "alias": "using B = Board;\nB b{12, 0, 170, 96, 1792};",
    "alias_ns": "using B = ninferno::Board;\nB b(12, 0, 170, 96, 1792);",
    "typedef": "typedef Board B;\nB b{12, 0, 170, 96, 1792};",
    "paren_first": "Board b{(12), 0, 170, 96, 1792};",
    "named_first": "Board b{kMajor, 0, 170, 96, 1792};",
    "named_field": "Board b{8, 0, kSms, 96, 1792};",
    "expression_first": "Board b{6 + 6, 0, 170, 96, 1792};",
    "table_row": '{"ga100:108sm", {8, 0, 108, 40, 1555}, kTable},',
    "field": '{.gpu_class = "gb202", .board = {12, 0, 188, 128, 1792}},',
    "field_named": "{.board = {kMajor, 0, 188, 128, 1792}},",
    "field_table": "{.boards = Board{kMajor, 0, 188, 128, 1792}},",
    "float_l2": "x{7, 5, 68, 5.5, 616}",
    "digit_separator": "Board b{12, 0, 170, 96, 1'792};",
    "mma_slow": "Board b{7, 5, 68, 5.5, 616, true};",
    "macro": "#define MINE Board{12, 0, \\\n  170, 96, 1792}",
    "hex": "Board b{0xC, 0, 170, 96, 1792};",
    "after_string_glob": 'const char* kGlob = "/*.gguf";\nBoard b{12, 0, 170, 96, 1792};',
    "after_char_quote": "char c = '\"'; Board b{12, 0, 170, 96, 1792};",
    "after_raw_string": 'auto s = R"x(/* )x";\nBoard b{12, 0, 170, 96, 1792};',
    # Tables of boards: every row of a typed table is tested.
    "table_named_first": "Board kBoards[] = {{kMajor, 0, 170, 96, 1792}};",
    "table_named_field": "Board kBoards[] = {{8, 0, kSms, 96, 1792}};",
    "table_designated": "const Board k[] = {{.major = 8, .minor = 6, .sms = 82, .l2_mib = 6,"
                        " .dram_gbps = 936}};",
    "table_short_rows": "std::vector<Board> v = {{8, 6, 82}, {8, 9, 128}};",
    "table_two_fields": "Board k[] = {{12, 0}};",
    "table_second_row_only": "Board k[] = {measured::kRtx5090, {8, 9, 128, 72, 1008}};",
    # Integer suffixes on a field: the tuple is still a board.
    "tuple_suffixed_hex": "Board b = f({0xCu, 0, 170, 96, 1792});",
    "tuple_suffixed_hex_ull": "x = {12, 0, 0xAAul, 96, 0x700ULL};",
    "tuple_suffixed_decimal": "x = {12u, 0, 170L, 96, 1792ull};",
    # A function returning Board: its braced returns are board literals.
    "return_named_fields": "Board f() { return {kMajor, 0, 170, 96, kBw}; }",
    "return_const_ref_qualified": "const Board& g(int a) const noexcept"
                                  " { return {8, 6, kS, 6, 9}; }",
    # Ref-qualifiers, noexcept(...), attributes and specifiers after the parameters.
    "return_ref_qualified_lvalue": "Board Foo::board() const & { return {kMajor, 0, kSms, kL2, kBw}; }",
    "return_ref_qualified_rvalue": "Board Foo::board() && { return {kMajor, 0, kSms, kL2, kBw}; }",
    "return_ref_qualified_tight": "Board Foo::board() const&& { return {kMajor, 0, kSms, kL2, kBw}; }",
    "return_noexcept_expression": "Board f() noexcept(noexcept(g())) { return {kM, 0, 1, 2, kB}; }",
    "return_noexcept_ref_override": "Board Foo::board() const & noexcept(true) override final"
                                    " { return {kM, 0, 1, 2, kB}; }",
    "return_attribute_after": "Board f() [[deprecated]] { return {kM, 0, 1, 2, kB}; }",
    "return_attribute_before": "[[nodiscard]] static constexpr inline Board f()"
                               " { return {kM, 0, 1, 2, kB}; }",
    "return_trailing_ref_qualified": "auto Foo::board() const & -> Board { return {kM, 0, 1, 2, kB}; }",
    # A trailing return type with a postfix const, a pointer or a reference.
    "return_trailing_postfix_const": "auto f() -> Board const { return {kM, 0, 1, 2, kB}; }",
    "return_trailing_prefix_const": "auto f() -> const Board { return {kM, 0, 1, 2, kB}; }",
    "return_trailing_const_ref": "auto f() const -> Board const& { return {kM, 0, 1, 2, kB}; }",
    # An alias reached through a namespace, however deep.
    "alias_through_namespace": "namespace facts { using GPUBoard = ninferno::Board; }\n"
                               "facts::GPUBoard b{kMajor, 0, kSms, kL2, kBw};",
    "alias_through_nested_namespace": "namespace a::b { using GB = Board; }\n"
                                      "a::b::GB b{kMajor, 0, kSms, kL2, kBw};",
    "alias_through_namespace_tail": "namespace a { namespace b { using GB = Board; } }\n"
                                    "b::GB b{kMajor, 0, kSms, kL2, kBw};",
    "alias_through_global_namespace": "namespace a { using GB = Board; }\n"
                                      "::a::GB b(kMajor, 0, kSms, kL2, kBw);",
    "alias_in_class": "struct S { using GB = Board; };\nS::GB b{kMajor, 0, kSms, kL2, kBw};",
    "alias_template_argument": "namespace a { using GB = Board; }\n"
                               "std::vector<a::GB> v = {{kMajor, 0, 1, 2, 3}};",
    "alias_return_qualified": "namespace a { using GB = Board; }\n"
                              "a::GB f() { return {kM, 0, 1, 2, kB}; }",
    "alias_return_trailing_qualified": "namespace a { using GB = Board; }\n"
                                       "auto f() -> a::GB { return {kM, 0, 1, 2, kB}; }",
    # The same alias basename in two namespaces: the one declared here is flagged.
    "alias_same_basename_declared_here": "namespace facts { using GB = Board; }\n"
                                         "chess::GB c{1, 2}; facts::GB b{kMajor, 0, kSms, kL2, kBw};",
    # An alias defined through another alias, in any order, qualified or not.
    "alias_chain": "using B = Board; using C = B; C b{kMajor, 0, kSms, kL2, kBw};",
    "alias_chain_reversed": "using C = B; using B = Board; C b{kMajor, 0, kSms, kL2, kBw};",
    "alias_chain_three": "using A = Board; using B = A; using C = const B; C b{kMajor, 0, kSms, 1, 2};",
    "alias_chain_qualified": "using B = Board; namespace n { using C = ::x::B; }"
                             " n::C b{kMajor, 0, kSms, kL2, kBw};",
    "alias_chain_typedef": "typedef Board B; typedef B C; C b{kMajor, 0, kSms, kL2, kBw};",
    "alias_chain_mixed": "using B = Board; typedef const B C; using D = C; D b{kMajor, 0, 1, 2, 3};",
    "alias_chain_return": "using B = Board; using C = B; C f() { return {kM, 0, 1, 2, kB}; }",
    # A requires-clause after the parameters (C++20), also with a trailing return.
    "return_requires_name": "template <class T> Board f(T) requires C<T>"
                            " { return {kMajor, 0, kSms, kL2, kBw}; }",
    "return_requires_parens": "template <class T> Board f(T t) requires (sizeof(T) > 1 &&"
                              " std::is_same_v<T, int>) { return {kM, 0, 1, 2, kB}; }",
    "return_requires_after_qualifiers": "template <class T> Board f(T) const noexcept requires C<T>"
                                        " { return {kM, 0, 1, 2, kB}; }",
    "return_requires_trailing": "template <class T> auto f(T) -> Board requires C<T>"
                                " { return {kM, 0, 1, 2, kB}; }",
    "return_requires_leading": "template <class T> requires C<T> Board f(T)"
                               " { return {kM, 0, 1, 2, kB}; }",
    "return_requires_expression": "template <class T> Board f(T) requires requires(T t) { t.x(); }"
                                  " { return {kM, 0, 1, 2, kB}; }",
    "return_through_alias": "using B = Board; B f(int x) { if (x) return {kM, 0, 1, 2, 3};"
                            " return {}; }",
    # A template-id or an operator name for the function.
    "return_template_id": "template <> Board make_board<int>() { return {kMajor, 0, kSms, kL2, kBw}; }",
    "return_template_id_nested": "template <> Board make<std::array<int, 2>>(int)"
                                 " { return {kM, 0, 1, 2, kB}; }",
    "return_qualified_template_id": "Board Foo<int>::make() { return {kM, 0, 1, 2, kB}; }",
    "return_operator_plus": "Board operator+(X, X) { return {kMajor, kMinor, kSms, kL2, kBw}; }",
    "return_operator_call": "Board Foo::operator()(int) const { return {kM, 0, 1, 2, kB}; }",
    "return_operator_subscript": "Board& Foo::operator[](int i) { return {kM, 0, 1, 2, kB}; }",
    "return_operator_shift": "Board operator<<(Board, int) { return {kM, 0, 1, 2, kB}; }",
    # A requires-clause word that is part of a longer name does not hide the function.
    "return_param_ends_with_requires": "Board f(int prerequires) { return {kM, 0, 1, 2, kB}; }",
    "return_param_my_requires": "Board f(int my_requires) { return {kM, 0, 1, 2, kB}; }",
    # Binary literals, and a typed Board of qualified constants only.
    "binary_first": "Board b{0b1100, 0, 170, 96, 1792};",
    "binary_separator_suffix": "Board b{0B1'100u, 0, 170, 96, 1792};",
    "binary_tuple": "x = {0b1100, 0, 170, 96, 1792};",
    "all_qualified_direct": "Board b{facts::kMajor, facts::kMinor, facts::kSms, facts::kL2,"
                            " facts::kBw};",
    "all_qualified_parens": "Board b(facts::kMajor, facts::kMinor, facts::kSms, facts::kL2,"
                            " facts::kBw);",
    "all_qualified_return": "Board f() { return {facts::kMajor, facts::kMinor, facts::kSms,"
                            " facts::kL2, facts::kBw}; }",
    "all_qualified_field": "{.board = {facts::kMajor, facts::kMinor, facts::kSms, facts::kL2,"
                           " facts::kBw}},",
    # A template argument with a call in it.
    "std_array_call_size": "std::array<Board, board_count()> a = {{{kMajor, 0, 170, 96, kBw}}};",
    # Qualified constants and the spellings of an alias.
    "qualified_constant": "Board b{facts::kMajor, 0, 170, 96, kBw};",
    "globally_qualified_constant": "Board b{::ns::facts::kMajor, 0, 170, 96, kBw};",
    "typedef_global_qualified": "typedef ::ninferno::Board B; B b{kMajor, 0, 170, 96, kBw};",
    "using_const_alias": "using B = const Board; B b{kMajor, 0, 170, 96, kBw};",
    "typedef_const_after": "typedef Board const B; B b{kMajor, 0, 170, 96, kBw};",
    # Where the Board function is written: out of class, a trailing return, a braced default.
    "return_member_definition": "Board Foo::bar() { return {kM, 0, 1, 2, 3}; }",
    "return_qualified_member": "ninferno::Board ninferno::Foo::bar() const"
                               " { return {kM, 0, 1, 2, 3}; }",
    "return_lambda_trailing": "auto g = [](int) -> Board { return {kM, 0, 1, 2, 3}; };",
    "return_function_trailing": "auto f() -> Board { return {kM, 0, 1, 2, 3}; }",
    "return_braced_default": "Board f(Board b = {}) { return {kM, 0, 1, 2, 3}; }",
    "return_after_nested_lambda": "Board f() { auto g = [](int a) { return a; };"
                                  " return {kM, 0, 1, 2, 3}; }",
    # A prefixed character or string literal before the board on the same line.
    "after_wide_char": "if (sep == L'/') return Board{kMajor, 0, kSms, kL2, kBw};",
    "after_u16_char": "if (c == u'/') return Board{kMajor, 0, kSms, kL2, kBw};",
    "after_u32_char": "if (c == U'/') return Board{kMajor, 0, kSms, kL2, kBw};",
    "after_u8_char": "if (c == u8'/') return Board{kMajor, 0, kSms, kL2, kBw};",
    "after_wide_string": 'if (s == L"a/b") return Board{kMajor, 0, kSms, kL2, kBw};',
    "after_u8_raw_string": "if (s == u8R\"(a'b)\") return Board{kMajor, 0, kSms, kL2, kBw};",
    "after_digit_separator": "int n = 1'000; Board b{kMajor, 0, kSms, kL2, kBw};",
    "std_array_short_rows": "std::array<Board, 2> a{{{8, 6, 82}, {8, 9, 128}}};",
    "std_array_designated": "std::array<Board, 1> a{{{.major = 8, .minor = 6, .sms = 82}}};",
    "std_array_size_name": "std::array<Board, kN> a = {{{8, 6, 82}}};",
    "table_multiline": "const Board k[] = {\n    {8, 6, 82,\n     6, kBw},\n};",
    "global_table_named": "::ninferno::Board k[] = {{kMajor, 0, 170, 96, 1792}};",
    # A trailing comma, as clang-format writes a long row.
    "tuple_trailing_comma": "{8, 6, 82, 6, 936,}",
    "tuple_trailing_comma_multiline": "push({\n    8, 6, 82, 6, 936,\n});",
    "tuple_bool_trailing_comma": "{7, 5, 68, 5.5, 616, true,}",
    # #711: a table row of qualified constants only, a conversion operator, an attribute in
    # the declarator (after the name or before it).
    "table_all_qualified": "Board k[] = {{facts::kMajor, facts::kMinor, facts::kSms, facts::kL2, facts::kBw}};",
    "conversion_operator": "operator Board() const { return {kMajor, 0, kSms, kL2, kBw}; }",
    "attribute_after_name": "Board b [[maybe_unused]] {kMajor, 0, kSms, kL2, kBw};",
    "attribute_before_name": "Board [[maybe_unused]] b {kMajor, 0, kSms, kL2, kBw};",
}

CLEAN = {
    "declaration": "struct Board {",
    "declaration_body": "struct Board{ int major = 0; };",
    "conversion": "table->board = Board{cc / 10, cc % 10, sms};",
    "registry_list_qualified": "std::vector<Board> v = {measured::kRtx5090, measured::kRtx4090};",
    "registry_table_qualified": "const Board k[] = {measured::kRtx5090, measured::kRtx4090};",
    "registry_std_array_double_brace": "std::array<Board, 2> k = {{measured::kRtx5090, measured::kRtx4090}};",
    "registry_vector_double_brace": "std::vector<Board> v{{measured::kA, measured::kB}};",
    "conversion_operator_fields": "operator Board() const { return {major, minor, sms}; }",
    "row_fields": "return Board{row->major, row->minor};",
    "name_prefix": "FactsBoard{1, 2}",
    "name_prefix_board": "BoardRow{0x2B85, 12, 0, 170, 96}",
    "count": "int BoardCount{12};",
    "index": "std::size_t BoardIndex = {0};",
    "other_namespace": "chess::Board b{8, 8};",
    "other_namespace_global": "::chess::Board b{8, 8};",
    "other_namespace_template": "std::vector<chess::Board> v{{8, 8}};",
    "member": "x.Board{8, 8};",
    "registry": "kRtx5090 = board_registry::board(0x2B85);",
    "table_of_registry_boards": "std::vector<Board> v = {measured::kRtx5090, board_of(1, 2)};",
    "table_of_names": "Board k[] = {{a, b}, {c, d}};",
    "empty_vector": "std::vector<Board> v{};",
    "not_a_board_array": "std::array<int, 2> a{{1, 2}};",
    "list_of_sizes": "for (int t : {8, 256, 4096, 49152, 65536}) {",
    "ids": "std::vector<int> v{1, 2, 46, 47, 768};",
    "pdf_header": "header.insert(header.end(), {8, 2, 0, 0, 0});",
    "sizes_trailing_comma": "for (int t : {4, 2, 16, 64, 256,}) {",
    "return_from_fields": "Board f(int cc) { return {cc / 10, cc % 10, 3}; }",
    "return_of_another_type": "int h() { return {kA, 1}; }",
    "alias_of_another_type": "using B = chess::Board; B b{kMajor, 0, 1, 2, 3};",
    "list_of_qualified_boards": "std::vector<Board> v = {measured::kA, measured::kB,"
                                " measured::kC};",
    "alias_chain_of_another_type": "using B = chess::Board; using C = B; C b{kMajor, 0, 1, 2, 3};",
    "alias_chain_unrelated": "using B = Board; using C = Other; C b{kMajor, 0, 1, 2, 3};",
    "return_requires_other_type": "template <class T> int f(T) requires C<T>"
                                  " { return {kM, 0, 1, 2, kB}; }",
    "return_requires_other_trailing": "template <class T> auto f(T) -> int requires C<T>"
                                      " { return {kM, 0, 1, 2, kB}; }",
    "return_trailing_other_const": "auto f() -> int const { return {kM, 0, 1, 2, kB}; }",
    "alias_of_another_type_qualified": "using GB = chess::Board; a::GB b{kMajor, 0, 1, 2, 3};",
    "alias_of_another_namespace": "namespace facts { using GPUBoard = ninferno::Board; }\n"
                                  "chess::GPUBoard b{kMajor, kMinor};",
    "alias_global_spelled_in_namespace": "using GB = Board; a::GB b{kMajor, 0, 1, 2, 3};",
    "alias_of_class_other_scope": "struct S { using GB = Board; };\nT::GB b{kMajor, 0, 1, 2, 3};",
    "return_other_template_id": "template <> int make<int>() { return {kM, 0, 1, 2, kB}; }",
    "return_other_operator": "int operator+(X, X) { return {kM, 0, 1, 2, kB}; }",
    "list_of_qualified_boards_array": "Board k[] = {measured::kA, measured::kB};",
    "list_of_qualified_boards_std_array": "std::array<Board, 2> k{measured::kA, measured::kB};",
    "member_alias_access": "using GB = Board; x.GB{kMajor, 0, 1, 2, 3};",
    "return_other_ref_qualified": "int Foo::bar() const & { return {kM, 0, 1, 2, kB}; }",
    "return_other_rvalue": "Foo Foo::bar() && { return {kM, 0, 1, 2, kB}; }",
    "return_other_noexcept_expression": "int f() noexcept(noexcept(g())) { return {kM, 0, 1, 2, kB}; }",
    "return_other_attribute": "[[nodiscard]] static constexpr int f() { return {kM, 0, 1, 2, kB}; }",
    "return_other_trailing": "auto Foo::bar() const & -> int { return {kM, 0, 1, 2, kB}; }",
    "return_in_nested_lambda": "Board f() { auto g = []() -> std::pair<int, int>"
                               " { return {1, 2}; }; return measured::kX; }",
    "return_in_lambda_without_type": "Board f() { auto g = [&] { return {kA, kB}; }; return x; }",
    "return_in_local_class": "Board f() { struct P { int a() { return {1, 2}; } }; return x; }",
    "prefixed_char_hides_nothing_else": "auto c = L'\"'; // Board{12, 0, 170, 96, 1792}",
    "comment_line": "// was Board{12, 0, 170, 96, 1792}",
    "comment_block": "/* Board{12, 0, 170}\n still */ int x;",
    "comment_block_tuple": "/* {12, 0, 170, 96, 1792}\n */ int x;",
    "string": 'puts("Board{12, 0, 170, 96, 1792}");',
    "string_tuple": 'puts("{12, 0, 170, 96, 1792}");',
    "string_escaped_quote": 'puts("a\\" {12, 0, 170, 96, 1792}");',
}


def tree(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def pin(text):
    """The sorted literals of `text`, as an ALLOW entry holds them."""
    return tuple(sorted(lit for _n, lit, _l in lint.literals(text)))


class Violations(unittest.TestCase):
    def test_a_long_line_with_no_body_is_read_in_linear_time(self):
        import time
        heads = ["Board f(T) requires " + "(a < b) && " * 1000,
                 "Board f(T) requires " + "C<T> && " * 1250 + ";",
                 "Board f(T) const " + "& " * 5000,
                 "Board f(T) requires (" * 500,
                 "using B = Board; " + "using C = B; " * 800,
                 "Board b " + "[[a]] " * 8000]
        for code in heads:
            start = time.monotonic()
            lint.violations(code)
            self.assertLess(time.monotonic() - start, 3.0, code[:40])

    def test_a_long_requires_clause_is_read_in_bounded_time(self):
        import time
        unit = {"requires_requires": "Board f(T) requires requires { x; } ",
                "requires_requires_open": "Board f(T) requires requires { ",
                "requires_paren_open": "Board f(T) requires (",
                "requires_name": "Board f(T) requires C<T> ",
                "requires_conj": "Board f(T) requires (a) && "}
        for name, u in unit.items():
            code = u * (50000 // len(u))
            start = time.monotonic()
            lint.violations(code)
            self.assertLess(time.monotonic() - start, 1.0, name)

    def test_a_clause_longer_than_the_bound_is_not_followed(self):
        clause = " && ".join(["C<T>"] * 200)
        code = "Board f(T) requires " + clause + " { return {kM, 0, 1, 2, kB}; }"
        self.assertEqual(lint.violations(code), [])

    def test_a_braced_value_in_a_clause_is_taken_for_the_body(self):
        # Documented limit: `requires C<T, int{3}>` ends the clause at its first brace.
        code = "Board f(T) requires C<T, int{3}> { return {kM, 0, 1, 2, kB}; }"
        self.assertEqual(lint.violations(code), [])

    def test_flags_every_spelling(self):
        for name, code in FLAGGED.items():
            with self.subTest(name):
                self.assertEqual(len(lint.violations(code)), 1, code)

    def test_ignores_non_boards(self):
        for name, code in CLEAN.items():
            with self.subTest(name):
                self.assertEqual(lint.violations(code), [], code)

    def test_code_after_a_comment_is_seen(self):
        code = "int x; /* a */ Board{1, 2}"
        self.assertEqual(lint.violations(code), [(1, code)])

    def test_line_numbers(self):
        code = "int a;\n// c\nBoard b{12,\n 0, 170, 96, 1792};\n"
        self.assertEqual([n for n, _ in lint.violations(code)], [3])

    def test_every_row_of_a_table_is_a_literal(self):
        code = "Board k[] = {{8, 6, 82}, {kMajor, 9, 128, 72, 1008}, measured::kX, {12, 0}};"
        self.assertEqual([lit for _n, lit, _l in lint.literals(code)],
                         ["{8,6,82}", "{kMajor,9,128,72,1008}", "{12,0}"])

    def test_a_nested_literal_counts_once(self):
        code = "Board b{12, 0, 170, 96, 1792};\nstd::vector<Board> v = {{8, 6, 82, 6, 936}};"
        self.assertEqual([lit for _n, lit, _l in lint.literals(code)],
                         ["Board b{12,0,170,96,1792}", "{8,6,82,6,936}"])

    def test_a_run_of_ampersands_is_read_in_linear_time(self):
        # Not C++, but a lint must never hang on what a file holds: before `&&|&(?!&)` a run of
        # 40 took about 25 s and one of 10 000 never returned.
        import time
        for code in ("Board f() " + "&" * 10000 + ";", "Board f() const " + "& " * 5000 + ";"):
            start = time.perf_counter()
            lint.violations(code)
            self.assertLess(time.perf_counter() - start, 2.0, code[:30])
        # The ref-qualifiers still read: `&`, `&&`, and `& &&`-free forms as before.
        self.assertEqual(len(lint.violations("Board f() & { return {12, 0, 170, 96, 1792}; }")), 1)
        self.assertEqual(len(lint.violations("Board f() && { return {12, 0, 170, 96, 1792}; }")), 1)

    def test_literal_text_ignores_layout_and_comments(self):
        a = "Board b{12, 0, 170, 96, 1792};"
        b = "Board  b{ 12,   // cc\n 0, 170,\n 96, /* GB/s */ 1792 };"
        self.assertEqual(pin(a), pin(b))


class Scan(unittest.TestCase):
    def test_every_directory_and_suffix_is_scanned(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {
                "src/a.cpp": lit, "tests/b.cpp": lit, "tools/c.cu": lit, "apps/cli/d.cpp": lit,
                "bench/ops/e.cpp": lit, "include/f.h": lit, "examples/g.hpp": lit,
                "src/core/h.inc": lit, "third_party/x/i.cpp": lit, "src/j.txt": lit})
            bad = {r for r, _, _ in lint.scan(root, allow={})}
            self.assertEqual(bad, {"src/a.cpp", "tests/b.cpp", "tools/c.cu", "apps/cli/d.cpp",
                                   "bench/ops/e.cpp", "include/f.h", "examples/g.hpp",
                                   "src/core/h.inc"})

    def test_cpp_templates_are_scanned_by_their_compound_suffix(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {
                "src/a.h.in": lit, "src/b.hpp.in": lit, "src/c.cpp.in": lit, "src/d.cu.in": lit,
                "src/e.cuh.in": lit, "src/f.H.IN": lit, "src/g.in": lit, "src/h.txt.in": lit,
                "src/i.json.in": lit})
            bad = {r for r, _, _ in lint.scan(root, allow={})}
            self.assertEqual(bad, {"src/a.h.in", "src/b.hpp.in", "src/c.cpp.in", "src/d.cu.in",
                                   "src/e.cuh.in", "src/f.H.IN"})

    def test_only_the_third_party_prefixes_are_skipped(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {
                "third_party/spdlog/a.cpp": lit, "src/ui/third_party/x/b.cpp": lit,
                "src/third_party/c.cpp": lit, "src/ops/generic/third_party/d.h": lit,
                "src/my_third_party/e.cpp": lit, "third_party_copy/f.cpp": lit})
            bad = {r for r, _, _ in lint.scan(root, allow={})}
            self.assertEqual(bad, {"src/third_party/c.cpp", "src/ops/generic/third_party/d.h",
                                   "src/my_third_party/e.cpp", "third_party_copy/f.cpp"})

    def test_the_skipped_prefixes_exist_in_the_tree(self):
        self.assertTrue((ROOT / "third_party").is_dir())
        self.assertIn("third_party/", lint.THIRD_PARTY)

    def test_tracked_files_only_in_a_checkout(self):
        lit = "Board b{8, 6, 82, 6, 936};\n"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            tree(root, {"src/tracked.cpp": lit, "src/untracked.cpp": lit})
            subprocess.run(["git", "-C", str(root), "add", "src/tracked.cpp"], check=True)
            self.assertEqual({r for r, _, _ in lint.scan(root, allow={})}, {"src/tracked.cpp"})

    def test_a_checkout_without_git_fails_with_a_message(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / ".git").mkdir()
            with mock.patch.object(lint.subprocess, "run", side_effect=FileNotFoundError("git")):
                with self.assertRaises(SystemExit) as cm:
                    lint.files(root)
            self.assertIn("git command is not installed", str(cm.exception))


class Pin(unittest.TestCase):
    """The allow-list pins the literals of a file, not only how many lines hold one."""
    R8 = ('const Row kRows[] = {\n  {"a100", {8, 0, 108, 40, 1555}, kT},\n'
          '  {"a40", {8, 6, 82, 6, 936}, kU},\n};\n')

    def run_scan(self, text, pinned=None):
        pinned = pin(self.R8) if pinned is None else pinned
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {"src/r8.h": text})
            return lint.scan(root, allow={"src/r8.h": (pinned, "x")})

    def test_unchanged_file_passes_whatever_its_layout(self):
        self.assertEqual(self.run_scan(self.R8), [])
        spaced = self.R8.replace("{8, 0,", "{ 8, // cc\n 0,")
        self.assertEqual(self.run_scan(spaced), [])

    def test_a_replaced_board_fails(self):  # plant 1: the row regenerated from another board
        bad = self.run_scan(self.R8.replace("{8, 0, 108, 40, 1555}", "{8, 0, 108, 80, 2039}"))
        self.assertEqual(len(bad), 2)
        self.assertEqual({n > 0 for _r, n, _t in bad}, {True, False})

    def test_a_second_board_on_a_flagged_line_fails(self):  # plant 2
        bad = self.run_scan(self.R8.replace(
            '{"a100", {8, 0, 108, 40, 1555}, kT},',
            '{"a100", {8, 0, 108, 40, 1555}, kT}, {"b", {8, 9, 128, 72, 1008}, kT},'))
        self.assertEqual([(r, n > 0) for r, n, _t in bad], [("src/r8.h", True)])

    def test_a_hypothetical_swapped_for_a_real_board_fails(self):  # plant 3
        hypo = "Board h{12, 0, 170, 0, 1792};\n"
        pinned = pin(hypo)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {"tests/x.cpp": "Board h{8, 6, 82, 6, 936};\n"})
            bad = lint.scan(root, allow={"tests/x.cpp": (pinned, "x")})
        self.assertEqual(sorted(n > 0 for _r, n, _t in bad), [False, True])

    def test_a_removed_literal_and_an_added_one_elsewhere_fail(self):  # plant 4: same count
        text = self.R8.replace('  {"a40", {8, 6, 82, 6, 936}, kU},\n', "") + "Board b{8, 9, 34, 32, 288};\n"
        bad = self.run_scan(text)
        self.assertEqual(sorted(n > 0 for _r, n, _t in bad), [False, True])

    def test_a_removed_literal_fails(self):
        bad = self.run_scan(self.R8.replace('  {"a40", {8, 6, 82, 6, 936}, kU},\n', ""))
        self.assertEqual([(r, n) for r, n, _t in bad], [("src/r8.h", 0)])
        self.assertIn("{8,6,82,6,936}", bad[0][2])

    def test_a_new_line_fails_with_its_line_number(self):
        bad = self.run_scan(self.R8 + "Board c{12, 0, 170, 96, 1792};\n")
        self.assertEqual([(r, n) for r, n, _t in bad], [("src/r8.h", 5)])

    def test_a_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, {"src/other.cpp": "int x;\n"})
            bad = lint.scan(root, allow={"src/missing.h": (("{8,6,82,6,936}",), "x")})
        self.assertEqual([(r, n) for r, n, _t in bad], [("src/missing.h", 0)])

    def test_a_duplicate_literal_is_pinned_twice(self):
        two = "Board a{8, 6, 82, 6, 936};\nBoard a{8, 6, 82, 6, 936};\n"
        self.assertEqual(len(pin(two)), 2)
        self.assertEqual(self.run_scan(two, pin(two)), [])
        self.assertEqual(len(self.run_scan(two, pin(two)[:1])), 1)


class Main(unittest.TestCase):
    def run_main(self, files, allow=None):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            tree(root, files)
            with contextlib.redirect_stdout(out):
                code = lint.main(root, {} if allow is None else allow)
        return code, out.getvalue()

    def test_exit_0_on_a_clean_tree(self):
        code, out = self.run_main({"src/a.cpp": "int x;\n"})
        self.assertEqual((code, out), (0, ""))

    def test_exit_1_on_a_new_literal(self):
        code, out = self.run_main({"src/a.cpp": "Board b{8, 6, 82, 6, 936};\n"})
        self.assertEqual(code, 1)
        self.assertIn("src/a.cpp:1: board literal outside the registry", out)
        self.assertIn("name the board by PCI id", out)
        self.assertNotIn("pin is stale", out)

    def test_exit_1_on_a_stale_pin_with_its_own_message(self):
        code, out = self.run_main({"src/a.cpp": "int x;\n"},
                                  {"src/a.cpp": (("{8,6,82,6,936}",), "reason of the pin")})
        self.assertEqual(code, 1)
        self.assertIn("pin is stale", out)
        self.assertNotIn("name the board by PCI id", out)


class AllowList(unittest.TestCase):
    def test_entries_exist_with_literals_and_a_reason(self):
        for rel, (lits, reason) in lint.ALLOW.items():
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertGreater(len(lits), 0, rel)
            self.assertEqual(list(lits), sorted(lits), rel)
            self.assertGreater(len(reason), 20, rel)

    def test_tree_is_clean(self):
        bad = lint.scan(ROOT)
        self.assertEqual(bad, [], "\n".join(f"{r}:{n}: {l}" for r, n, l in bad))


if __name__ == "__main__":
    unittest.main()
