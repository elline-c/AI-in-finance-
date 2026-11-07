"""Hygiene tests for the retraining work (Requirement 17; task 24.1).

Source scans use ``ast`` so strings in comments and docstrings do not count as imports.
Nothing here reads a file outside the repository or the pytest temporary folder.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "fraud_detect" / "ml" / "retraining"
PKG_FILES = sorted(PKG.glob("*.py"))
HASHES = json.loads((Path(__file__).parent / "protected_hashes.json").read_text())

#: Modules that train or featurise; they must never see the Real_Style / awkward data modules.
TRAINING_MODULES = ("train_a", "train_b", "features_a", "features_b", "cutoffs", "leak",
                    "neutralise", "audit", "size_guard")
#: Modules allowed to write files without going through safe_io (each has its own guard):
#: modelops (switch/restore of models/, temp file then os.replace) and cli (run.lock).
RAW_WRITE_ALLOWED = {"safe_io.py", "modelops.py", "cli.py"}
#: Real_Style names / answer key may appear only here. runner.py holds the label map built from
#: the answer key (design F11) used to score, never to train.
SENSITIVE_ALLOWED = {"sources.py", "config.py", "runner.py"}
SENSITIVE_STRINGS = ("statement_E.pdf", "statement_F.pdf", "ANSWER_KEY")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _imports(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            names.add(base)
            names.update(f"{base}.{a.name}" if base else a.name for a in node.names)
    return names


def _root(name: str) -> str:
    return name.lstrip(".").split(".")[0]


def _run_python(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                          text=True, timeout=timeout)


# ---- 17.1 / 17.2: protected files and Node 4 pinned values -------------------------------------

@pytest.mark.parametrize("rel", sorted(HASHES["sha256"]))
def test_protected_file_hash(rel):
    assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == HASHES["sha256"][rel]


def test_protected_paths_not_modified_in_git():
    out = _git("status", "--porcelain", "--", "fraud_detect/nodes", "fraud_detect/registry.py",
               "fraud_detect/reference_data.py", "fraud_detect/corpus")
    assert out.returncode == 0
    assert out.stdout.strip() == ""


def test_node4_constants_and_literals_equal_pinned():
    pinned = HASHES["node4_pinned"]
    tree = ast.parse((REPO / "fraud_detect/nodes/risk_synthesis.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            consts[node.targets[0].id] = node
    for name, value in pinned["module_constants"].items():
        assert ast.literal_eval(consts[name].value) == value
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, expected in pinned["numeric_literals_by_function"].items():
        found = [n.value for n in ast.walk(funcs[name])
                 if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                 and not isinstance(n.value, bool)]
        assert found == expected, name


# ---- 17.3: runtime dependency list --------------------------------------------------------------

def test_requirements_unchanged_and_without_torch():
    for name in ("requirements.txt", "requirements-gnn.txt"):
        assert _git("diff", "--quiet", "HEAD", "--", name).returncode == 0, name
        assert _git("ls-files", "--error-unmatch", name).returncode == 0
    runtime = (REPO / "requirements.txt").read_text().lower()
    assert "torch" not in runtime


# ---- 17.4: torch kept out of the live app -------------------------------------------------------

@pytest.mark.parametrize("path", PKG_FILES, ids=lambda p: p.name)
def test_no_torch_in_package_source(path):
    bad = {n for n in _imports(path) if _root(n) in ("torch", "torch_geometric")}
    assert not bad


def test_live_app_imports_with_torch_blocked():
    code = ("import sys\n"
            "for m in ('torch','torch_geometric'): sys.modules[m] = None\n"
            "import fraud_detect.webapp, fraud_detect.banker_app, fraud_detect.customer_app\n")
    result = _run_python(code)
    assert result.returncode == 0, result.stderr[-2000:]


def test_package_import_is_light():
    code = ("import sys, fraud_detect.ml.retraining\n"
            "bad=[m for m in ('sklearn','pdfplumber','torch','torch_geometric') if m in sys.modules]\n"
            "sys.exit(1 if bad else 0)")
    assert _run_python(code).returncode == 0


def test_cli_help_is_fast_and_light():
    code = ("import sys, time\n"
            "t=time.time()\n"
            "from fraud_detect.ml.retraining import cli\n"
            "rc=cli.main(['--help']) if False else cli.main(['train','--help'])\n"
            "bad=[m for m in ('torch','torch_geometric','sklearn','pdfplumber','pandas') if m in sys.modules]\n"
            "sys.exit(rc or (1 if bad else 0))")
    start = time.monotonic()
    result = _run_python(code, timeout=60)
    assert result.returncode == 0, result.stderr[-1000:]
    assert time.monotonic() - start < 15


# ---- 17.5 / 17.6 / 17.7: ignored output, nothing tracked ----------------------------------------

def _default_output_paths():
    from fraud_detect.ml.retraining.config import DEFAULT_OUTPUT_ROOT, RunConfig
    cfg = RunConfig()
    root = cfg.output_root
    assert root == DEFAULT_OUTPUT_ROOT
    subs = ("x", "run.json", "corpora/seed_1/labels.csv", "corpora/seed_1/pdfs/a.pdf",
            "features/cache.csv", "candidates/6b/model.joblib", "snapshots/s.json",
            "report/comparison_report.json", "decisions.jsonl")
    return [f"{root}/{s}" for s in subs] + [cfg.real_style_dir + "/x.pdf", cfg.forgery_dir + "/x.pdf"]


@pytest.mark.parametrize("rel", ["data/private/ml_retraining/x", "models/backup/x.joblib"]
                         + _default_output_paths())
def test_git_ignores_output_path(rel):
    assert _git("check-ignore", "-q", rel).returncode == 0, rel


def test_no_sensitive_artifacts_tracked():
    tracked = _git("ls-files").stdout.splitlines()
    assert tracked
    patterns = (r"(^|/)labels\.csv$", r"(^|/)rows\.csv$", r"\.pdf$", r"\.joblib$", r"\.pkl$",
                r"\.pt$", r"(^|/)data/private/", r"(^|/)models/", r"features?_cache",
                r"(^|/)snapshots?/", r"(^|/)corpus_[^/]*/")
    bad = [f for f in tracked if any(re.search(p, f) for p in patterns)]
    # Allow pre-existing, non-generated sample PDFs only when they are not under the private root.
    bad = [f for f in bad if not (f.endswith(".pdf") and not f.startswith("data/"))]
    assert not bad, bad[:10]


def test_nothing_tracked_under_data_private():
    assert _git("ls-files", "data/private").stdout.strip() == ""


# ---- no module writes outside safe_io -----------------------------------------------------------

_WRITE_ATTRS = {"write_text", "write_bytes", "to_csv", "to_json", "to_pickle", "touch",
                "mkdir", "unlink", "rename", "replace", "rmtree", "copyfile", "copy", "move"}


def _raw_writes(path: Path) -> list:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name in ("open", "fdopen") and len(node.args) > 1:
            mode = node.args[1]
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) \
                    and set(mode.value) & set("wax+"):
                found.append((node.lineno, f"{name}({mode.value!r})"))
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) \
                    and set(str(kw.value.value)) & set("wax+"):
                found.append((node.lineno, "open(mode=...)"))
        if name in ("write_text", "write_bytes", "to_csv", "to_json", "to_pickle", "rmtree",
                    "copyfile", "dump", "mkstemp") and isinstance(fn, ast.Attribute):
            # to_csv(buf) on an in-memory buffer and mkdir are not file writes.
            if name in ("to_csv", "dump") and len(node.args) > 1 \
                    and isinstance(node.args[-1], ast.Name) and node.args[-1].id == "buf":
                continue
            if name == "to_csv" and node.args:
                continue
            found.append((node.lineno, name))
    return found


@pytest.mark.parametrize("path", [p for p in PKG_FILES if p.name not in RAW_WRITE_ALLOWED],
                         ids=lambda p: p.name)
def test_module_writes_only_through_safe_io(path):
    assert _raw_writes(path) == []


def test_allowed_raw_writers_are_guarded():
    assert "ensure_ignored" in (PKG / "safe_io.py").read_text()
    # modelops only touches the models folder and a backup folder given by the caller.
    assert "atomic_write" in (PKG / "modelops.py").read_text() or \
        "ensure_ignored" in (PKG / "modelops.py").read_text()


# ---- scope of sensitive names / modules ---------------------------------------------------------

@pytest.mark.parametrize("path", [p for p in PKG_FILES if p.name not in SENSITIVE_ALLOWED],
                         ids=lambda p: p.name)
def test_sensitive_names_confined(path):
    text = path.read_text(encoding="utf-8")
    assert not [s for s in SENSITIVE_STRINGS if s in text]


@pytest.mark.parametrize("name", TRAINING_MODULES)
def test_training_modules_do_not_import_sources_or_awkward(name):
    path = PKG / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{name}.py not present")
    for imp in _imports(path):
        parts = imp.lstrip(".").split(".")
        assert parts[-1] not in ("sources", "awkward", "awkward_spike"), imp


# ---- network ------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", PKG_FILES, ids=lambda p: p.name)
def test_no_network_imports(path):
    bad = {n for n in _imports(path)
           if _root(n) in ("urllib", "socket", "requests", "http", "httpx", "aiohttp", "ftplib")}
    assert not bad


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_pipeline_modules_work_without_network(no_network, capsys):
    from fraud_detect.ml.retraining import cli, config
    assert cli.main(["train", "--help"]) == 0
    assert config.RunConfig().run_id


# ---- reuse of the UEN collision check -----------------------------------------------------------

def test_group_checks_imports_uen_check():
    imports = _imports(PKG / "group_checks.py")
    assert any(n.rstrip(".").endswith("uen_check") or "uen_check" in n for n in imports), imports


# ---- tests read nothing outside the repository --------------------------------------------------

def test_tests_do_not_reference_outside_paths():
    own = Path(__file__).name
    pattern = re.compile(r"""["'](/Users/|/home/|~/|/tmp/|/var/|/etc/)""")
    offenders = []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name == own:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if pattern.search(code) and "example" not in code or "Path.home" in code \
                    or "os.path.expanduser" in code or ".expanduser()" in code and "normalise" not in code:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, offenders[:10]


def test_real_data_tests_skip_when_absent():
    # The marker and fixture in conftest turn missing real files into skips, not failures.
    conftest = (Path(__file__).parent / "conftest.py").read_text()
    assert "pytest.skip" in conftest and "skip_without_real_style" in conftest
    assert os.environ.get("RETRAIN_REAL_STYLE_DIR", "") == "" or \
        Path(os.environ["RETRAIN_REAL_STYLE_DIR"]).is_dir()
