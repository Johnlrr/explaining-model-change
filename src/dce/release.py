"""Build an anonymous review copy of the repository (code, configs, tests, spec + amendments, protocol log, run evidence).

Only git-tracked files under INCLUDE are copied; EXCLUDE drops harness and private working files. Text files are
rewritten with the replacements of a patterns file kept OUTSIDE the release (one `regex<TAB>replacement` per line),
so the identifying strings themselves are never published; the copy is then scanned and the build fails if any
pattern still matches. Binary files (.npz, .pt, .pdf, .png) are copied unchanged and listed in the manifest.

    python -m dce.release <out_dir> <patterns_file> [--with-checkpoints]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

INCLUDE = ("src/", "configs/", "tests/", "docs/", "runs/", "research/DECISIONS.md", "research/HYPOTHESES.md",
           "pyproject.toml", "uv.lock")
EXCLUDE = ("tools/", "AGENTS.md", "project rules file", ".local/", "research/STATE.md", "research/NOTES.md",
           "research/LITERATURE.md", "research/extended_version.md", "paper/")
BINARY = {".npz", ".pt", ".pdf", ".png", ".zip"}

README = """# Review copy

Anonymous copy for review. Layout:

- `src/dce/` — all research code (data renderer, models, solvers, baselines, analysis, CLI stages).
- `configs/` — tier parameters and one config per experiment (`exp_NNNN`) or smoke run.
- `tests/unit/` — unit tests (`uv run pytest tests/unit`).
- `docs/` — the evaluation protocol; `docs/source/` holds the specification and its amendments v1.1–v1.10
  (partly in Vietnamese), each committed before the test data it concerns.
- `research/DECISIONS.md` — the protocol log (every decision, with before/after-results status);
  `research/HYPOTHESES.md` — every hypothesis with its outcome.
- `runs/` — run evidence: configs, metadata (commit, platform), per-pair outputs and results.

Kaggle kernel identifiers and local paths are replaced by placeholders.
"""


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    files = [f for f in out.splitlines() if f.startswith(INCLUDE) and not f.startswith(EXCLUDE)]
    return files + (["uv.lock"] if (root / "uv.lock").exists() and "uv.lock" not in files else [])   # lock: untracked


def scrub(text: str, patterns: list[tuple[re.Pattern, str]]) -> str:
    for rx, rep in patterns:
        text = rx.sub(rep, text)
    return text


def load_patterns(path: Path) -> list[tuple[re.Pattern, str]]:
    pats = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rx, _, rep = line.partition("\t")
            pats.append((re.compile(rx, re.IGNORECASE), rep))
    return pats


def build(root: Path, out: Path, patterns: list[tuple[re.Pattern, str]], with_checkpoints: bool) -> dict:
    if out.exists():
        shutil.rmtree(out)
    files, manifest, rewritten = tracked(root), [], 0
    for rel in files:
        src, dst = root / rel, out / scrub(rel, patterns)       # directory names can carry an account name
        if src.suffix == ".pt" and not with_checkpoints:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in BINARY:
            shutil.copy2(src, dst)
        else:
            text = src.read_text(errors="strict")
            new = scrub(text, patterns)
            rewritten += new != text
            dst.write_text(new)
        manifest.append({"path": str(dst.relative_to(out)), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
    (out / "README_REVIEW.md").write_text(README)
    leaks = [(m["path"], rx.pattern) for m in manifest for rx, _ in patterns
             if rx.search(m["path"]) or (Path(m["path"]).suffix not in BINARY and rx.search((out / m["path"]).read_text()))]
    if leaks:
        raise SystemExit(f"identifying strings left: {leaks[:10]}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return {"files": len(manifest), "rewritten": rewritten, "with_checkpoints": with_checkpoints,
            "bytes": sum((out / m["path"]).stat().st_size for m in manifest)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("patterns", type=Path)
    ap.add_argument("--with-checkpoints", action="store_true")
    args = ap.parse_args(argv)
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())
    print(json.dumps(build(root, args.out.resolve(), load_patterns(args.patterns), args.with_checkpoints)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
