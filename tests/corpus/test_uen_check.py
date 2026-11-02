"""Tests for the build-time UEN collision check (task 28.1).

Feature: synthetic-corpus-generator, Requirement 20 (see the design section
"Build-time UEN collision check").

They fail with ``ImportError`` until task 28.2 creates ``fraud_detect/corpus/uen_check.py``.

Interface assumed (from the design):

* ``uen_check.UenCheckError``, ``uen_check.load_snapshot(path) -> frozenset[str]``
* ``uen_check.UenChecker(snapshot=frozenset(), *, live=False, fetch=None, snapshot_file=None)``
  with ``is_taken(uen)``, ``describe()`` and public counters ``checked`` and ``redraws``
* ``make_company(seed, company_id, is_taken=None)``
* ``BuildConfig(uen_snapshot=None, check_live_uen=False)``
* ``CorpusBuilder(cfg, uen_fetch=None)``: ``uen_fetch`` is only for tests (a fake JSON fetcher)
* ``build.py``: ``--uen-snapshot PATH`` and ``--check-live-uen``; ``UenCheckError`` gives exit code 2

No test here uses the network: ``urllib.request.urlopen`` is replaced by a function that
records the attempt and raises, and every test must end with no attempt recorded unless it
cleared the list on purpose (the live-failure tests do).
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from fraud_detect import registry
from fraud_detect.corpus import company as company_mod
from fraud_detect.corpus import uen_check
from fraud_detect.corpus.build import _parser, main
from fraud_detect.corpus.builder import CorpusBuilder
from fraud_detect.corpus.company import make_company
from fraud_detect.corpus.schema import BuildConfig
from fraud_detect.corpus.seeding import rng
from fraud_detect.corpus.uen_check import UenCheckError, UenChecker, load_snapshot
from fraud_detect.reference_data import ACRA_REGISTRY

UEN_RE = re.compile(r"\d{8,9}[A-Z]")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Record (and refuse) any attempt to open a URL; the test must not make one."""
    attempts: list[str] = []

    def refuse(*args, **kwargs):
        attempts.append("urlopen")
        raise OSError("network is switched off in tests")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    registry.clear_cache()
    yield attempts
    registry.clear_cache()
    assert attempts == [], "a test tried to use the network"


def make_fetch(real_uens=(), calls=None):
    """A fake data.gov.sg fetcher: knows only ``real_uens``; collection metadata is absent."""
    real = {u.upper() for u in real_uens}

    def fetch(url: str) -> dict:
        if calls is not None:
            calls.append(url)
        if "datastore_search" not in url:
            raise ValueError("no metadata in the fake")
        found = re.search(r"%22uen%22%3A\+%22(\w+)%22", url)
        assert found, url
        uen = found.group(1)
        records = ([{"uen": uen, "entity_name": "FAKE REAL CO", "uen_status_desc": "Registered"}]
                   if uen in real else [])
        return {"success": True, "result": {"records": records}}

    return fetch


def failing_fetch(url: str) -> dict:
    if "datastore_search" not in url:
        raise ValueError("no metadata in the fake")
    raise OSError("service down")


# ------------------------------------------------------------------- snapshot loader
def test_load_text_snapshot_skips_blank_and_comment_lines_and_upper_cases(tmp_path):
    f = tmp_path / "uens.txt"
    f.write_text("# header comment\n\n  201111111a  \n201222222B\n   \n# another\n", encoding="utf-8")
    assert load_snapshot(f) == frozenset({"201111111A", "201222222B"})


def test_load_snapshot_with_unknown_extension_is_text(tmp_path):
    f = tmp_path / "uens.list"
    f.write_text("201111111A\n201222222B\n", encoding="utf-8")
    assert load_snapshot(str(f)) == frozenset({"201111111A", "201222222B"})


@pytest.mark.parametrize("header", ["uen", "UEN", "Uen"])
def test_load_csv_snapshot_with_uen_column_in_any_case(tmp_path, header):
    f = tmp_path / "uens.csv"
    f.write_text(f"entity_name,{header},status\nX,201111111a,R\nY,201222222B,R\n", encoding="utf-8")
    assert load_snapshot(f) == frozenset({"201111111A", "201222222B"})


def test_load_csv_without_uen_column_is_an_error(tmp_path):
    f = tmp_path / "uens.csv"
    f.write_text("entity_name,status\nX,R\n", encoding="utf-8")
    with pytest.raises(UenCheckError) as exc:
        load_snapshot(f)
    assert "uens.csv" in str(exc.value)


def test_load_json_list_of_strings(tmp_path):
    f = tmp_path / "uens.json"
    f.write_text(json.dumps(["201111111a", "201222222B"]), encoding="utf-8")
    assert load_snapshot(f) == frozenset({"201111111A", "201222222B"})


def test_load_json_list_of_objects_with_uen_key(tmp_path):
    f = tmp_path / "uens.json"
    f.write_text(json.dumps([{"uen": "201111111a", "entity_name": "X"}, {"uen": "201222222B"}]),
                 encoding="utf-8")
    assert load_snapshot(f) == frozenset({"201111111A", "201222222B"})


@pytest.mark.parametrize("payload", [{"uen": "201111111A"}, "just text", 5, [1, 2], [{"name": "x"}]])
def test_load_json_of_an_unknown_shape_is_an_error(tmp_path, payload):
    f = tmp_path / "uens.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UenCheckError):
        load_snapshot(f)


def test_load_missing_file_names_the_file(tmp_path):
    with pytest.raises(UenCheckError) as exc:
        load_snapshot(tmp_path / "nope.txt")
    assert "nope.txt" in str(exc.value)


@pytest.mark.parametrize("name,body", [("a.txt", ""), ("b.txt", "# only a comment\n\n"),
                                       ("c.csv", ""), ("d.json", "[]")])
def test_load_empty_snapshot_is_an_error(tmp_path, name, body):
    f = tmp_path / name
    f.write_text(body, encoding="utf-8")
    with pytest.raises(UenCheckError) as exc:
        load_snapshot(f)
    assert name in str(exc.value)


def test_load_unreadable_bytes_is_an_error(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_bytes(b"\xff\xfe\x00\x80\x81")
    with pytest.raises(UenCheckError):
        load_snapshot(f)


# ----------------------------------------------------------------------- UenChecker
def test_is_taken_uses_the_snapshot_case_insensitively():
    checker = UenChecker(frozenset({"201111111A"}))
    assert checker.is_taken("201111111A") is True
    assert checker.is_taken("201111111a") is True
    assert checker.is_taken("201999999Z") is False


def test_is_taken_knows_the_in_repo_registry():
    uen = next(iter(ACRA_REGISTRY))
    checker = UenChecker()
    assert checker.is_taken(uen) is True
    assert checker.is_taken(uen.lower()) is True


def test_is_taken_without_live_never_calls_fetch():
    calls: list[str] = []
    checker = UenChecker(frozenset(), live=False, fetch=make_fetch(["201999999Z"], calls))
    assert checker.is_taken("201999999Z") is False
    assert calls == []


def test_live_fetch_found_means_taken():
    checker = UenChecker(frozenset(), live=True, fetch=make_fetch(["201999999Z"]))
    assert checker.is_taken("201999999Z") is True


def test_live_fetch_empty_result_means_free():
    checker = UenChecker(frozenset(), live=True, fetch=make_fetch([]))
    assert checker.is_taken("201999999Z") is False


def test_live_fetch_failure_raises_and_never_means_free():
    checker = UenChecker(frozenset(), live=True, fetch=failing_fetch)
    with pytest.raises(UenCheckError) as exc:
        checker.is_taken("201999999Z")
    assert "live UEN check could not be completed" in str(exc.value)


def test_live_without_fetch_uses_the_registry_and_failure_raises(no_network):
    checker = UenChecker(frozenset(), live=True)
    with pytest.raises(UenCheckError):
        checker.is_taken("201999999Z")
    assert no_network, "expected an attempt that the test fixture refused"
    no_network.clear()


def test_describe_has_the_documented_keys_and_no_real_name(tmp_path):
    snap = tmp_path / "snap.txt"
    snap.write_text("201111111A\n201222222B\n", encoding="utf-8")
    checker = UenChecker(load_snapshot(snap), live=True, fetch=make_fetch(["201999999Z"]),
                         snapshot_file=snap)
    checker.is_taken("201111111A")
    checker.is_taken("201999999Z")           # found by the fake live source (name FAKE REAL CO)
    checker.is_taken(next(iter(ACRA_REGISTRY)))
    info = checker.describe()
    assert {"source", "snapshot_file", "snapshot_sha256", "snapshot_uens", "uens_checked",
            "redraws", "real_registry_checked"} <= set(info)
    assert info["source"] == "snapshot and live"
    assert info["snapshot_file"] == "snap.txt"
    assert info["snapshot_sha256"] == hashlib.sha256(snap.read_bytes()).hexdigest()
    assert info["snapshot_uens"] == 2
    assert info["uens_checked"] == 3
    assert info["real_registry_checked"] is True
    dumped = json.dumps(info)
    assert "FAKE REAL CO" not in dumped
    assert not any(name in dumped for name in ACRA_REGISTRY.values())


@pytest.mark.parametrize("snapshot,live,source,checked", [
    (False, False, "in-repo registry only", False),
    (True, False, "snapshot", True),
    (False, True, "live", True),
    (True, True, "snapshot and live", True),
])
def test_describe_source_labels(tmp_path, snapshot, live, source, checked):
    snap = tmp_path / "snap.txt"
    snap.write_text("201111111A\n", encoding="utf-8")
    checker = UenChecker(load_snapshot(snap) if snapshot else frozenset(), live=live,
                         fetch=make_fetch([]) if live else None,
                         snapshot_file=snap if snapshot else None)
    info = checker.describe()
    assert info["source"] == source
    assert info["real_registry_checked"] is checked
    if not snapshot:
        assert info["snapshot_file"] is None and info["snapshot_sha256"] is None
        assert info["snapshot_uens"] == 0


# ------------------------------------------------------------------------ make_company
PINNED_SEED, PINNED_ID = 12345, "C1G0001"
PINNED_UEN = "22791655L"
PINNED_NAME = "DRIFTMARK KITCHENS PTE. LTD."
PINNED_DIGEST = "3d6cfe2004dc23f2ee6bb06aa39edb2ec656b9191f1913ef1f98d27abbfa1933"


def _digest(profile) -> str:
    text = json.dumps(dataclasses.asdict(profile), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def test_make_company_without_callback_is_the_old_profile():
    profile = make_company(PINNED_SEED, PINNED_ID)
    assert (profile.uen, profile.name) == (PINNED_UEN, PINNED_NAME)
    assert _digest(profile) == PINNED_DIGEST
    assert make_company(PINNED_SEED, PINNED_ID, is_taken=None) == profile


def test_make_company_with_a_callback_that_never_fires_is_the_old_profile():
    seen: list[str] = []

    def free(uen: str) -> bool:
        seen.append(uen)
        return False

    assert _digest(make_company(PINNED_SEED, PINNED_ID, is_taken=free)) == PINNED_DIGEST
    assert seen == [PINNED_UEN]


def test_forced_clash_changes_only_the_uen():
    base = make_company(PINNED_SEED, PINNED_ID)
    got = make_company(PINNED_SEED, PINNED_ID, is_taken=lambda u: u == base.uen)
    assert got.uen != base.uen and UEN_RE.fullmatch(got.uen)
    assert dataclasses.replace(got, uen=base.uen) == base


def test_redraw_comes_from_the_dedicated_child_stream():
    base = make_company(PINNED_SEED, PINNED_ID)
    got = make_company(PINNED_SEED, PINNED_ID, is_taken=lambda u: u == base.uen)
    expected = company_mod._uen(rng(PINNED_SEED, "company", PINNED_ID, "uen-redraw"))
    assert got.uen == expected


def test_redraw_is_the_same_on_every_call():
    base = make_company(PINNED_SEED, PINNED_ID)
    taken = lambda u: u == base.uen  # noqa: E731
    first = make_company(PINNED_SEED, PINNED_ID, is_taken=taken)
    assert all(make_company(PINNED_SEED, PINNED_ID, is_taken=taken) == first for _ in range(3))


def test_two_clashes_in_a_row_draw_the_next_uens_of_the_same_stream():
    base = make_company(PINNED_SEED, PINNED_ID)
    stream = rng(PINNED_SEED, "company", PINNED_ID, "uen-redraw")
    r1, r2 = company_mod._uen(stream), company_mod._uen(stream)
    got = make_company(PINNED_SEED, PINNED_ID, is_taken=lambda u: u in {base.uen, r1})
    assert got.uen == r2
    assert dataclasses.replace(got, uen=base.uen) == base


def test_always_taken_ends_in_uen_check_error_after_1000_tries():
    calls: list[str] = []

    def always(uen: str) -> bool:
        calls.append(uen)
        return True

    with pytest.raises(UenCheckError):
        make_company(PINNED_SEED, PINNED_ID, is_taken=always)
    assert len(calls) == 1001            # the first draw and 1000 redraws


# --------------------------------------------------------------------------- builder
def _cfg(out: Path, **extra) -> BuildConfig:
    return BuildConfig(out_dir=out, master_seed=7, per_layout_genuine=1, per_layout_tampered=0,
                       months_min=1, months_max=1, **extra)


def _companies(out: Path) -> list[dict]:
    data = json.loads((out / "companies.json").read_text(encoding="utf-8"))
    return data["companies"] if isinstance(data, dict) else data


def _manifest(out: Path) -> dict:
    return json.loads((out / "corpus_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """One default build (no snapshot, no live check) shared by the read-only tests."""
    out = tmp_path_factory.mktemp("uen_baseline")
    summary = CorpusBuilder(_cfg(out)).build()
    return out, summary


def test_default_build_stores_and_returns_the_not_checked_warning(baseline):
    out, summary = baseline
    assert any("not checked against the real registry" in w for w in summary.warnings)
    assert any("not checked against the real registry" in w for w in _manifest(out)["warnings"])
    assert "not checked against the real registry" in (out / "README.txt").read_text(encoding="utf-8")


def test_default_build_manifest_and_readme_show_source_and_counts(baseline):
    out, _ = baseline
    info = _manifest(out)["uen_check"]
    assert info["source"] == "in-repo registry only"
    assert info["real_registry_checked"] is False
    assert info["snapshot_file"] is None and info["snapshot_sha256"] is None
    assert info["snapshot_uens"] == 0
    assert info["uens_checked"] == 6 and info["redraws"] == 0
    readme = (out / "README.txt").read_text(encoding="utf-8")
    line = next(ln for ln in readme.splitlines() if "UEN check" in ln)
    assert "in-repo registry only" in line and "6" in line


def test_default_build_uens_are_unique_and_not_in_the_repo_registry(baseline):
    uens = [c["uen"] for c in _companies(baseline[0])]
    assert len(uens) == 6 and len(set(uens)) == 6
    assert not set(uens) & set(ACRA_REGISTRY)


def test_default_build_makes_no_network_request(baseline, no_network):
    assert no_network == []


def test_same_seed_and_same_snapshot_give_equal_companies_json(tmp_path):
    snap = tmp_path / "snap.txt"
    snap.write_text("201111111A\n", encoding="utf-8")
    a, b = tmp_path / "a", tmp_path / "b"
    CorpusBuilder(_cfg(a, uen_snapshot=snap)).build()
    CorpusBuilder(_cfg(b, uen_snapshot=snap)).build()
    assert (a / "companies.json").read_bytes() == (b / "companies.json").read_bytes()


def test_snapshot_with_one_first_uen_changes_only_that_company(baseline, tmp_path):
    base_out, _ = baseline
    base = {c["company_id"]: c for c in _companies(base_out)}
    victim = sorted(base)[2]
    snap = tmp_path / "snap.csv"
    snap.write_text(f"uen\n{base[victim]['uen'].lower()}\n", encoding="utf-8")
    out = tmp_path / "out"
    CorpusBuilder(_cfg(out, uen_snapshot=snap)).build()
    got = {c["company_id"]: c for c in _companies(out)}
    assert set(got) == set(base)
    for cid, rec in base.items():
        if cid == victim:
            assert got[cid]["uen"] != rec["uen"] and UEN_RE.fullmatch(got[cid]["uen"])
            assert {k: v for k, v in got[cid].items() if k != "uen"} == \
                   {k: v for k, v in rec.items() if k != "uen"}
        else:
            assert got[cid] == rec
    info = _manifest(out)["uen_check"]
    assert info["source"] == "snapshot" and info["real_registry_checked"] is True
    assert info["snapshot_file"] == "snap.csv"
    assert info["snapshot_sha256"] == hashlib.sha256(snap.read_bytes()).hexdigest()
    assert info["snapshot_uens"] == 1 and info["redraws"] == 1
    assert info["uens_checked"] == 7          # six first draws and one redraw
    assert not any("not checked against the real registry" in w for w in _manifest(out)["warnings"])
    readme = (out / "README.txt").read_text(encoding="utf-8")
    assert "snapshot" in next(ln for ln in readme.splitlines() if "UEN check" in ln)
    assert "not checked against the real registry" not in readme


def test_the_uen_of_a_victim_is_written_nowhere_in_the_snapshot_build(baseline, tmp_path):
    base = _companies(baseline[0])
    snap = tmp_path / "snap.txt"
    snap.write_text(base[0]["uen"] + "\n", encoding="utf-8")
    out = tmp_path / "out"
    CorpusBuilder(_cfg(out, uen_snapshot=snap)).build()
    for path in out.rglob("*"):
        if path.suffix in {".json", ".csv", ".txt"}:
            assert base[0]["uen"] not in path.read_text(encoding="utf-8"), path.name


def test_two_companies_forced_onto_the_same_first_uen_get_different_uens(monkeypatch, tmp_path):
    real = company_mod._uen
    state = {"n": 0}

    def forced(r):
        state["n"] += 1
        return "111111111A" if state["n"] <= 2 else real(r)

    monkeypatch.setattr(company_mod, "_uen", forced)
    out = tmp_path / "out"
    CorpusBuilder(_cfg(out)).build()
    uens = [c["uen"] for c in _companies(out)]
    assert len(uens) == 6 and len(set(uens)) == 6
    assert uens.count("111111111A") == 1
    assert _manifest(out)["uen_check"]["redraws"] == 1


def test_live_check_with_fake_fetch_records_live_source(tmp_path):
    base_out = tmp_path / "base"
    CorpusBuilder(_cfg(base_out)).build()
    first = _companies(base_out)[0]
    out = tmp_path / "live"
    summary = CorpusBuilder(_cfg(out, check_live_uen=True),
                            uen_fetch=make_fetch([first["uen"]])).build()
    info = _manifest(out)["uen_check"]
    assert info["source"] == "live" and info["real_registry_checked"] is True
    assert info["redraws"] == 1 and info["uens_checked"] == 7
    assert not any("not checked against the real registry" in w for w in summary.warnings)
    new = {c["company_id"]: c for c in _companies(out)}[first["company_id"]]
    assert new["uen"] != first["uen"]


def test_bad_snapshot_stops_the_build_and_writes_nothing(corpus_dir):
    cfg = _cfg(corpus_dir, uen_snapshot=corpus_dir.parent / "no_such_snapshot.txt")
    with pytest.raises(UenCheckError):
        CorpusBuilder(cfg).build()
    assert list(corpus_dir.iterdir()) == []


def test_failing_live_fetch_stops_the_build_and_writes_nothing(corpus_dir):
    cfg = _cfg(corpus_dir, check_live_uen=True)
    with pytest.raises(UenCheckError) as exc:
        CorpusBuilder(cfg, uen_fetch=failing_fetch).build()
    assert "live UEN check could not be completed" in str(exc.value)
    assert list(corpus_dir.iterdir()) == []


def test_builder_does_not_create_the_output_folder_when_the_check_fails(tmp_path):
    out = tmp_path / "not_yet"
    with pytest.raises(UenCheckError):
        CorpusBuilder(_cfg(out, check_live_uen=True), uen_fetch=failing_fetch).build()
    assert not out.exists()


# ------------------------------------------------------------------------ build.py
def test_cli_defaults_and_parsing(tmp_path):
    args = _parser().parse_args([])
    assert args.uen_snapshot is None and args.check_live_uen is False
    args = _parser().parse_args(["--uen-snapshot", str(tmp_path / "s.txt"), "--check-live-uen"])
    assert Path(args.uen_snapshot) == tmp_path / "s.txt" and args.check_live_uen is True


def test_cli_help_mentions_both_options(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "--uen-snapshot" in text and "--check-live-uen" in text


TINY = ["--per-layout-genuine", "1", "--per-layout-tampered", "0",
        "--months-min", "1", "--months-max", "1", "--seed", "7"]


def test_cli_default_run_prints_the_warning(tmp_path, capsys):
    assert main(["--out", str(tmp_path / "c"), *TINY]) == 0
    assert "not checked against the real registry" in capsys.readouterr().out


def test_cli_bad_snapshot_exits_two_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "c"
    code = main(["--out", str(out), *TINY, "--uen-snapshot", str(tmp_path / "missing.txt")])
    assert code == 2
    captured = capsys.readouterr()
    assert "missing.txt" in captured.out + captured.err
    assert not out.exists() or list(out.iterdir()) == []


def test_cli_empty_snapshot_exits_two(tmp_path):
    snap = tmp_path / "empty.txt"
    snap.write_text("", encoding="utf-8")
    out = tmp_path / "c"
    assert main(["--out", str(out), *TINY, "--uen-snapshot", str(snap)]) == 2
    assert not out.exists() or list(out.iterdir()) == []


def test_cli_live_check_that_cannot_answer_exits_two(tmp_path, capsys, no_network):
    out = tmp_path / "c"
    assert main(["--out", str(out), *TINY, "--check-live-uen"]) == 2
    assert "live UEN check could not be completed" in capsys.readouterr().err
    assert no_network, "the live check should have tried (and been refused by the fixture)"
    no_network.clear()
    assert not out.exists() or list(out.iterdir()) == []


def test_cli_with_a_snapshot_builds_and_does_not_warn(tmp_path, capsys):
    snap = tmp_path / "snap.txt"
    snap.write_text("201111111A\n", encoding="utf-8")
    assert main(["--out", str(tmp_path / "c"), *TINY, "--uen-snapshot", str(snap)]) == 0
    assert "not checked against the real registry" not in capsys.readouterr().out


# -------------------------------------------------------------------------- hygiene
FORBIDDEN_IMPORTS = {"urllib", "socket", "requests"}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
    return roots


def test_uen_check_and_its_tests_import_no_network_module_directly():
    assert _imported_roots(Path(uen_check.__file__)) & FORBIDDEN_IMPORTS == set()
    assert _imported_roots(Path(__file__)) & FORBIDDEN_IMPORTS == set()


def test_uen_check_uses_only_the_public_registry_function():
    tree = ast.parse(Path(uen_check.__file__).read_text(encoding="utf-8"))
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "registry"}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and (n.module or "").endswith("registry") for a in n.names}
    private = {name for name in used | imported if name.startswith("_")}
    assert private == set()
    assert "lookup_entity" in used | imported
    assert "ACRA_REGISTRY" not in used | imported


def test_registry_reference_data_and_nodes_are_unchanged_in_git():
    import subprocess
    repo = Path(__file__).resolve().parents[2]
    res = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "fraud_detect/registry.py",
         "fraud_detect/reference_data.py", "fraud_detect/nodes"],
        cwd=repo, capture_output=True, text=True)
    assert res.returncode == 0
    assert res.stdout.strip() == ""
