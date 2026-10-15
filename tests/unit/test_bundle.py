"""D-026: committed single-file bundles must match src/dce exactly, and must run."""

import subprocess
import sys

from dce.remote.bundle import BUNDLE_DIR, entry_of, render


def bundles():
    return sorted(p for p in BUNDLE_DIR.glob("*.py") if p.name != "__init__.py")


def test_every_bundle_is_up_to_date():
    assert bundles(), "no bundle committed"
    for path in bundles():
        assert path.read_text() == render(entry_of(path)), f"{path.name} is stale: rebuild it"


def test_render_is_deterministic():
    assert render("dce.remote.bundle_smoke") == render("dce.remote.bundle_smoke")


def test_bundle_runs_outside_the_repo(tmp_path):
    src = BUNDLE_DIR / "bundle_smoke.py"
    run = subprocess.run([sys.executable, "-I", str(src)], cwd=tmp_path, capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert '"found": 8' in run.stdout
    assert '"from_bundle": true' in run.stdout  # imports resolve to the extracted copy, not the repo
