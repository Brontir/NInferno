"""The validation protocol of user measurements (docs/maintainer/validation-protocol.md), no GPU:
tools/kit/validate.py reads a submission as JSON only, under a size cap, refuses an unknown schema
version or key and a build that is not a release's, and judges the switch-factor and KLD runs;
every file under docs/performance/user-runs/ passes it."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402
import releases  # noqa: E402
import validate  # noqa: E402
from test_kit_validate import doc, one_pass  # noqa: E402

COMMIT, ENGINE = "a" * 40, "b" * 64
RELEASES = [{"version": "0.1.0", "commit": COMMIT, "engines": {"linux-x86_64": {"89": ENGINE, "120a": "c" * 64}}}]


def released(**changes):
    """A plain kit file (schema 1.4) measured with release 0.1.0's sm_89 engine."""
    d = doc()
    d["schema"] = "ninferno-user-run/1.4"
    d["ninferno"] = {"version": "NInferno 0.1.0 (built for sm_89, CUDA 13.4)",
                     "build": {"version": "0.1.0", "git_commit": COMMIT, "arch": "89", "arch_specific": "0"},
                     "engine": {"tag": "89", "sha256": ENGINE}}
    for path, value in changes.items():
        *parents, key = path.split(".")
        node = d
        for p in parents:
            node = node[p]
        node[key] = value
    return d


def cli(*args):
    """validate.py's exit code and output for `args`."""
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["validate.py", *args]), contextlib.redirect_stdout(out):
        code = validate.main()
    return code, out.getvalue()


class Provenance(unittest.TestCase):
    def test_a_release_build_is_accepted(self):
        r = validate.validate(released(), releases=RELEASES)
        self.assertEqual(r["file"], "accepted")
        self.assertTrue(r["runs"][0]["metrics"])

    def test_a_wrong_engine_hash_is_refused(self):
        r = validate.validate(released(**{"ninferno.engine.sha256": "d" * 64}), releases=RELEASES)
        self.assertTrue(r["file"].startswith("rejected: P1 the sm_89 engine's SHA-256"))
        self.assertEqual(r["runs"], [])

    def test_another_releases_engine_or_tag_is_refused(self):
        # 120a's engine hash under the tag 89: each tag's own hash only.
        r = validate.validate(released(**{"ninferno.engine.sha256": "c" * 64}), releases=RELEASES)
        self.assertTrue(r["file"].startswith("rejected: P1"))

    def test_a_wrong_commit_or_version_is_refused(self):
        r = validate.validate(released(**{"ninferno.build.git_commit": "e" * 40}), releases=RELEASES)
        self.assertTrue(r["file"].startswith("rejected: P1 the build's commit"))
        dirty = validate.validate(released(**{"ninferno.build.git_commit": COMMIT + "-dirty"}), releases=RELEASES)
        self.assertTrue(dirty["file"].startswith("rejected: P1") and "local or modified" in dirty["file"])
        r = validate.validate(released(**{"ninferno.build.version": "0.2.0"}), releases=RELEASES)
        self.assertTrue(r["file"].startswith("rejected: P1 version"))

    def test_a_file_without_the_build_identity_is_refused(self):
        older = doc()  # schema 1.0: no build recorded
        self.assertTrue(validate.validate(older, releases=RELEASES)["file"].startswith("rejected: P1 no build"))
        # Without a manifest (the kit's own report, a lab's build) P1 is not applied.
        self.assertEqual(validate.validate(older)["file"], "accepted")

    def test_the_in_tree_manifest_loads(self):
        self.assertIsInstance(validate.load_releases(), list)

    def test_a_malformed_manifest_stops_the_validator(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "releases.json"
            bad.write_text(json.dumps({"schema": validate.RELEASES_SCHEMA,
                                       "releases": [{"version": "1", "commit": "x", "engines": {}}]}))
            with self.assertRaises(ValueError):
                validate.load_releases(bad)


class Shape(unittest.TestCase):
    def test_an_unknown_schema_version_is_refused(self):
        for schema in ("ninferno-user-run/1.9", "ninferno-user-run/2.0", "ninferno-user-run/1",
                       "ninferno-user-run/1.4x", "ninferno-user-run/1.04", "", None):
            with self.subTest(schema=schema):
                d = released()
                d["schema"] = schema
                self.assertTrue(validate.validate(d, releases=RELEASES)["file"].startswith("rejected: R0"))
        d = released()
        d["schema"] = "ninferno-user-run/1.5"
        self.assertIn("newer than this validator knows", validate.validate(d)["file"])

    def test_unknown_keys_are_refused(self):
        for where in ("top", "kit", "run", "mode"):
            with self.subTest(where=where):
                d = released()
                d["kit"] = {"version": "3"}
                if where == "top":
                    d["script"] = "import os"
                elif where == "kit":
                    d["kit"]["hook"] = 1
                elif where == "run":
                    d["runs"][0]["shapes"] = []  # a routes run's key on a plain run
                else:
                    d["runs"][0]["mode"] = "eval"
                r = validate.validate(d, releases=RELEASES)
                self.assertTrue(r["file"].startswith("rejected: R0 unknown keys"), r["file"])

    def test_not_an_object(self):
        self.assertTrue(validate.validate([1, 2])["file"].startswith("rejected: R0"))


class Reading(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.manifest = Path(self.dir.name) / "releases.json"
        self.manifest.write_text(json.dumps({"schema": validate.RELEASES_SCHEMA, "releases": RELEASES}))

    def write(self, name, text):
        path = Path(self.dir.name) / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_accepted_file_through_the_cli(self):
        good = self.write("good.json", json.dumps(released()))
        code, out = cli(good, "--releases", str(self.manifest))
        self.assertEqual(code, 0, out)
        self.assertIn("accepted", out.splitlines()[0])

    def test_wrong_hash_through_the_cli(self):
        bad = self.write("bad.json", json.dumps(released(**{"ninferno.engine.sha256": "0" * 64})))
        good = self.write("good.json", json.dumps(released()))
        code, out = cli(good, bad, "--releases", str(self.manifest), "--json")
        self.assertEqual(code, 1)
        reports = json.loads(out)
        self.assertEqual(reports["good.json"]["file"], "accepted")
        self.assertTrue(reports["bad.json"]["file"].startswith("rejected: P1"))

    def test_oversize_file_is_refused_unread(self):
        path = self.write("big.json", json.dumps(released()))
        with self.assertRaises(validate.Refused) as e:
            validate.load_run(path, max_bytes=100)
        self.assertIn("size cap", str(e.exception))
        with mock.patch.object(validate, "MAX_FILE_BYTES", 100):
            code, out = cli(path, "--releases", str(self.manifest))
        self.assertEqual(code, 1)
        self.assertIn("size cap", out)

    def test_malformed_json_is_refused(self):
        for name, text in (("trunc.json", '{"schema": "ninferno-user-run/1.4", "runs": ['),
                           ("nan.json", '{"schema": NaN}'),
                           ("inf.json", '{"x": -Infinity}'),
                           ("twice.json", '{"schema": "a", "schema": "b"}'),
                           ("deep.json", "[" * 100000 + "]" * 100000)):
            with self.subTest(name=name):
                with self.assertRaises(validate.Refused):
                    validate.load_run(self.write(name, text))
                code, out = cli(str(Path(self.dir.name) / name), "--releases", str(self.manifest))
                self.assertEqual(code, 1)
                self.assertIn("rejected: R0", out)
        latin = Path(self.dir.name) / "latin.json"
        latin.write_bytes(b'{"x": "\xe9"}')
        with self.assertRaises(validate.Refused):
            validate.load_run(str(latin))

    def test_a_number_too_large_is_refused(self):
        # json reads 1e400 as inf without calling parse_constant.
        with self.assertRaises(validate.Refused) as e:
            validate.load_run(self.write("huge.json", '{"x": 1e400}'))
        self.assertIn("not finite", str(e.exception))

    def test_an_overflow_rejects_that_file_and_the_next_is_read(self):
        bad = self.write("a.json", json.dumps(released()))
        good = self.write("b.json", json.dumps(released()))
        real = validate.validate

        def overflow_once(doc, *args, **kw):
            if not getattr(overflow_once, "done", False):
                overflow_once.done = True
                raise OverflowError("int too large to convert to float")
            return real(doc, *args, **kw)
        with mock.patch.object(validate, "validate", side_effect=overflow_once):
            code, out = cli(bad, good, "--releases", str(self.manifest), "--json")
        reports = json.loads(out)
        self.assertEqual(code, 1)
        self.assertTrue(reports["a.json"]["file"].startswith("rejected: R0 a malformed value (OverflowError)"))
        self.assertEqual(reports["b.json"]["file"], "accepted")

    def test_no_submitted_string_starts_a_line(self):
        # A CI runner reads "::command::" at a line's start: every printed string stays on its line.
        payload = "x\n::error::pwned\r\n::add-mask::y ::warning::z"
        d = released()
        d["runs"][0]["id"] = payload
        d["runs"].append({"id": payload, "repo": "o/m", "quant": "Q", "mode": "plain", "error": payload})
        code, out = cli(self.write("inj.json", json.dumps(d)), "--releases", str(self.manifest))
        self.assertEqual(code, 0)
        self.assertFalse([line for line in out.splitlines() if line.lstrip().startswith("::")], out)
        self.assertIn("x\\n::error::pwned", out)
        bad = released()
        bad["schema"] = payload
        code, out = cli(self.write("inj2.json", json.dumps(bad)), "--releases", str(self.manifest))
        self.assertEqual(code, 1)
        self.assertFalse([line for line in out.splitlines() if line.lstrip().startswith("::")], out)

    def test_a_value_of_another_type_is_refused_not_crashed(self):
        d = released()
        d["runs"][0]["passes"] = "not a list of passes"
        code, out = cli(self.write("odd.json", json.dumps(d)), "--releases", str(self.manifest))
        self.assertEqual(code, 1)
        self.assertIn("rejected: R0", out)

    def test_a_directory_is_not_a_file(self):
        with self.assertRaises(validate.Refused):
            validate.load_run(self.dir.name)

    def test_the_file_is_only_parsed(self):
        # Text that would run if evaluated stays data.
        marker = Path(self.dir.name) / "ran"
        d = released()
        d["runs"][0]["id"] = f"__import__('pathlib').Path({str(marker)!r}).touch()"
        cli(self.write("code.json", json.dumps(d)), "--releases", str(self.manifest))
        self.assertFalse(marker.exists())


class ExistingUserRuns(unittest.TestCase):
    def test_every_submitted_file_is_accepted(self):
        folder = ROOT / "docs" / "performance" / "user-runs"
        manifest = validate.load_releases()
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            with self.subTest(file=path.name):
                report = validate.validate(validate.load_run(path), releases=manifest)
                self.assertEqual(report["file"], "accepted")


def switch_run(kv, activations, decode, predicted, **extra):
    run = {"id": "q9", "repo": "org/m", "quant": "Q4_K_M", "mode": "switch", "kv": kv,
           "activations": activations, "context": 90112, "flags": [],
           "plan": {"chosen": {"decode_tok_s_at_depth": predicted, "prefill_tok_s": 3000.0, "depth": 8192,
                               "kv": kv, "activations": activations}, "prompt": 2048},
           "passes": []}
    for i in range(2):
        p = one_pass(decode=decode * (1 + 0.005 * i))
        p["served"]["choice"] = f"KV {kv}, activations {activations} | ..."
        run["passes"].append(p)
    run.update(extra)
    return run


def switch_doc(*runs):
    d = released()
    d["runs"] = list(runs)
    return d


def factor(report, metric):
    rows = next(r for r in report["runs"] if r["mode"] == "switch factors")["metrics"]
    return next(m for m in rows if m["metric"].startswith(metric))


class SwitchFactors(unittest.TestCase):
    def test_ratio_confirms_and_contradicts(self):
        d = switch_doc(switch_run("int8", "fast", 50.0, 48.0),
                       switch_run("bf16", "fast", 45.0, 44.0),     # 0.900 against 0.917: confirms
                       switch_run("nvfp4", "fast", 30.0, 40.0))    # 0.600 against 0.833: contradicts
        r = validate.validate(d, releases=RELEASES)
        self.assertEqual(r["file"], "accepted")
        bf16 = factor(r, "KV bf16")
        self.assertEqual((bf16["predicted"], bf16["status"]), (0.917, "accepted"))
        self.assertTrue(bf16["verdict"].startswith("confirms the KV read factor of bf16"))
        self.assertTrue(factor(r, "KV nvfp4")["verdict"].startswith("contradicts the KV read factor"))
        self.assertIn("switch factors", validate.render(r))

    def test_exact_tests_the_exact_route_factor(self):
        d = switch_doc(switch_run("int8", "fast", 50.0, 48.0), switch_run("int8", "exact", 40.0, 38.0))
        self.assertIn("exact route's compute factor", factor(validate.validate(d), "KV int8, activations exact")["verdict"])

    def test_a_rejected_side_gives_no_ratio(self):
        base = switch_run("int8", "fast", 50.0, 48.0)
        base["passes"][1]["measured"]["decode"]["median"] = 60.0  # R5: passes 20 % apart
        d = switch_doc(base, switch_run("fp8", "fast", 45.0, 44.0))
        self.assertTrue(factor(validate.validate(d), "KV fp8")["status"].startswith("rejected: W1"))

    def test_no_base_no_ratio(self):
        d = switch_doc(switch_run("fp8", "fast", 45.0, 44.0))
        notes = next(r for r in validate.validate(d)["runs"] if r["mode"] == "switch factors")["notes"]
        self.assertTrue(notes[0].startswith("W1 no base"))

    def test_a_capped_card_is_not_used(self):
        d = switch_doc(switch_run("int8", "fast", 50.0, 48.0), switch_run("fp8", "fast", 45.0, 44.0))
        d["card"]["smi"]["power.limit"] = 120.0
        self.assertTrue(factor(validate.validate(d), "KV fp8")["verdict"].startswith("not used"))


# The lab's corpus as ninferno-perplexity names it in report.json: the manifest's own corpus_id.
CORPUS_ID = json.loads((ROOT / "eval/corpora/perplexity-1m/manifest.json").read_text(encoding="utf-8"))["corpus_id"]


def kld_run(terms, **reference):
    ref = {"activations": "exact", "kv": "bf16", "state": "f32", "context": 4096, "stride": 2048,
           "corpus": f"{CORPUS_ID}/quick"}
    ref.update(reference)
    return {"id": "q9", "repo": "org/m", "quant": "Q4_K_M", "mode": "kld", "reference": ref, "terms": terms}


def term(axis, value, kld, lab=None, measured=True, **extra):
    t = {"axis": axis, "value": value, "kld": kld, "top1": 0.98,
         "plan": {"value": value, "kld": lab, "measured": measured, "transferred": False}}
    t.update(extra)
    return t


class Kld(unittest.TestCase):
    def rows(self, run):
        d = released()
        d["runs"] = [run]
        return validate.validate(d, releases=RELEASES)["runs"][0]["metrics"]

    def test_terms_confirm_contradict_or_are_candidates(self):
        rows = self.rows(kld_run([term("kv", "fp8", 0.0030, lab=0.0028),          # x1.07: confirms
                                  term("kv", "nvfp4", 0.0150, lab=0.00982),       # x1.53: contradicts
                                  term("activations", "fast", 0.004, lab=0.004, measured=False)]))
        self.assertTrue(rows[0]["verdict"].startswith("confirms the kv fp8 term"))
        self.assertTrue(rows[1]["verdict"].startswith("contradicts"))
        self.assertTrue(rows[2]["verdict"].startswith("no lab term"))
        self.assertIn("candidate model constant", rows[2]["verdict"])
        # Another family's term, borrowed for the estimate: no lab term of this family.
        borrowed = term("kv", "fp8", 0.009, lab=0.0028)
        borrowed["plan"]["transferred"] = True
        row = self.rows(kld_run([borrowed]))[0]
        self.assertTrue(row["verdict"].startswith("no lab term") and "borrows" in row["verdict"])
        self.assertEqual(row["ratio"], 3.214)

    def test_another_reference_or_corpus_is_refused(self):
        self.assertTrue(self.rows(kld_run([term("kv", "fp8", 0.003, 0.0028)], kv="fp8"))[0]["status"]
                        .startswith("rejected: K1 the reference"))
        self.assertTrue(self.rows(kld_run([term("kv", "fp8", 0.003, 0.0028)], corpus="own"))[0]["status"]
                        .startswith("rejected: K1 the corpus"))

    def test_the_kits_own_kld_run_passes_k1(self):
        # measure.kld_run with ninferno-perplexity's report shape (corpus.id from the manifest,
        # corpus.mode): what the kit writes is what the validator expects.
        def report(*args, kld_out=None, kld=None, **kw):
            out = {"corpus": {"id": CORPUS_ID, "mode": "quick"}, "overall": {"perplexity": 5.0}}
            if kld:
                out["kld"] = {"overall": {"mean_kld": 0.0029, "top1_agreement": 0.986, "p99_kld": 0.05}}
            return out
        plan = {"value": "fp8", "kld": 0.0028, "measured": True, "transferred": False}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(measure, "prepare", return_value=("m.ninferno", None)), \
                mock.patch.object(measure, "perplexity", side_effect=report), \
                mock.patch.object(measure, "plan_term", return_value=plan), \
                contextlib.redirect_stdout(io.StringIO()):
            run = measure.kld_run("serve", "ppl", {"id": "q9", "repo": "org/m", "quant": "Q4_K_M"}, 0, Path(d))
        self.assertNotIn("error", run)
        self.assertEqual(run["reference"]["corpus"], validate.kld_corpus())
        rows = self.rows(run)
        self.assertEqual(len(rows), len(measure.KLD_SWITCHES))
        self.assertTrue(all(r["status"] == "accepted" for r in rows), rows)
        self.assertTrue(rows[1]["verdict"].startswith("confirms"))

    def test_a_failed_or_out_of_range_term(self):
        rows = self.rows(kld_run([term("kv", "nvfp4", None, error="refused"),
                                  term("kv", "fp8", -1.0, lab=0.0028),
                                  term("kv", "k8v4", 0.005, lab=0.005, top1=1.5)]))
        self.assertEqual([r["status"][:11] for r in rows], ["rejected: K"] * 3)


class Measure(unittest.TestCase):
    def test_engine_tag(self):
        self.assertEqual(measure.engine_tag({"arch": "89", "arch_specific": "0"}), "89")
        self.assertEqual(measure.engine_tag({"arch": "120", "arch_specific": "1"}), "120a")
        self.assertIsNone(measure.engine_tag({"arch": "unknown"}))

    def test_build_identity_hashes_the_release_engine(self):
        with tempfile.TemporaryDirectory() as d:
            exe = ".exe" if measure.os.name == "nt" else ""
            serve, engine = Path(d) / f"ninferno-serve{exe}", Path(d) / f"ninferno-serve-sm89{exe}"
            serve.write_bytes(b"dispatcher")
            engine.write_bytes(b"engine")
            out = mock.Mock(stdout=json.dumps({"build": {"version": "0.1.0", "git_commit": COMMIT, "arch": 89,
                                                         "arch_specific": "0", "path": "/home/u"}}))
            with mock.patch.object(measure.subprocess, "run", return_value=out):
                build, ident = measure.build_identity(str(serve), 0)
            self.assertEqual(build, {"version": "0.1.0", "git_commit": COMMIT, "arch": "89", "arch_specific": "0"})
            self.assertEqual(ident, {"tag": "89", "sha256": measure.sha256_of(engine)})
            engine.unlink()  # a build: ninferno-serve is the engine
            with mock.patch.object(measure.subprocess, "run", return_value=out):
                self.assertEqual(measure.build_identity(str(serve), 0)[1]["sha256"], measure.sha256_of(serve))

    def test_switch_flags(self):
        flags = measure.run_flags({"context": 32768}, 1, context=90112, kv="fp8", activations="exact")
        self.assertEqual(flags[:2], ["--max-context", "90112"])
        self.assertEqual(flags[-4:], ["--kv-dtype", "fp8", "--activations", "exact"])
        self.assertEqual(measure.SWITCH_CONTEXT // 4, measure.SWITCH_DEPTH)
        self.assertEqual(measure.SWITCH_CONFIGS[0], validate.SWITCH_BASE)

    def test_a_kit_file_has_only_keys_the_validator_knows(self):
        # The keys measure.py writes, as validate.py lists them: a new one must be added to both.
        d = released()
        d["kit"] = {"version": measure.KIT_VERSION, "models_version": 2, "models_sha256": "0" * 16,
                    "models_file": None}
        d["created"], d["host"], d["budget"] = "2026-11-12T00:00:00Z", {"on_battery": False}, {}
        self.assertEqual(validate.unknown_keys(d), [])
        self.assertEqual(validate.schema_minor(measure.SCHEMA), max(validate.SCHEMA_MINORS))


class Releases(unittest.TestCase):
    def test_add_hashes_the_engines_once(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("ninferno-serve-sm89", "ninferno-serve-sm120a", "ninferno-serve", "ninferno-ui"):
                (Path(d) / name).write_bytes(name.encode())
            engines = releases.engines_of(d)
            self.assertEqual(sorted(engines), ["120a", "89"])
            manifest = {"schema": validate.RELEASES_SCHEMA, "releases": []}
            releases.add(manifest, "0.1.0", COMMIT, "linux-x86_64", engines)
            releases.add(manifest, "0.1.0", COMMIT, "windows-x86_64", {"89": "f" * 64})
            with self.assertRaises(ValueError):  # a published release never changes
                releases.add(manifest, "0.1.0", COMMIT, "linux-x86_64", engines)
            with self.assertRaises(ValueError):
                releases.add(manifest, "0.1.0", "e" * 40, "macos", engines)
            with self.assertRaises(ValueError):
                releases.add(manifest, "0.2.0", "short", "linux-x86_64", engines)
            path = Path(d) / "releases.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(validate.load_releases(path)[0]["engines"]["linux-x86_64"]["89"],
                             measure.sha256_of(Path(d) / "ninferno-serve-sm89"))


if __name__ == "__main__":
    unittest.main()
