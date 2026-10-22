"""v1.10 T-C (exp_0034, D-043 §4) on a Kaggle CPU kernel (D-044).

The kernel runs the local code path unchanged: `dce.cli.v110_tc tc-run` on the pairs named in the parameters,
with the committed configs and the hash-verified checkpoints of the `dce-v110-tc-inputs` dataset. To keep
floating-point results close to the local runs it first installs the local library versions (torch CPU wheel,
numpy, scipy; Internet ON), fixes the thread count, and selects the same ATen CPU kernels as the local machine
when the CPU supports them. Every choice and the observed CPU are written to result.json.

Parameters (harness): config, pairs, threads, deadline_h, pins {torch, numpy, scipy}, cpu_capability,
inputs_sha256 (of v110_tc_inputs.json), optional resume (a dataset folder with pair_<id>.partial.* files to continue from).
Outputs (flat in /kaggle/working, Kaggle lists ≤ 20 files): pair_<id>.json/.npz for finished pairs,
pair_<id>.partial.json/.npz for a pair stopped by the deadline, metadata.json, result.json, artifact_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_info() -> dict:
    text = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("model name")), platform.processor())
    flags = set(next((ln.split(":", 1)[1].split() for ln in text.splitlines() if ln.startswith("flags")), []))
    return {"model": model, "logical_cpus": os.cpu_count(), "avx2": "avx2" in flags, "avx512f": "avx512f" in flags}


def install(pins: dict) -> None:
    """Install the local versions before torch is imported (nothing in this process has imported it)."""
    if "torch" in sys.modules:
        raise RuntimeError("torch imported before the pinned install")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--extra-index-url", TORCH_INDEX,
           f"torch=={pins['torch']}", f"numpy=={pins['numpy']}", f"scipy=={pins['scipy']}"]
    subprocess.run(cmd, check=True)


def find(name: str) -> list[Path]:
    """Under /kaggle/input, or ./in for the local smoke."""
    root = next((r for r in (Path("/kaggle/input"), Path("in").resolve()) if r.exists()), None)
    return sorted(root.rglob(name)) if root else []


def stage_inputs(listing_sha256: str, work: Path) -> Path:
    """Kaggle unpacks the uploaded zip into a v110_tc_inputs/ folder next to v110_tc_inputs.json. The listing must
    have the expected hash, and every listed file (repository path -> sha256) is checked before it is copied."""
    hits = find("v110_tc_inputs.json")
    if not hits:
        raise FileNotFoundError("v110_tc_inputs.json not found")
    listing = hits[0]
    if sha256(listing) != listing_sha256:
        raise RuntimeError(f"inputs listing hash mismatch: {sha256(listing)} != {listing_sha256}")
    base = listing.parent / "v110_tc_inputs"
    for name, digest in json.loads(listing.read_text())["files"].items():
        src = base / name
        if sha256(src) != digest:
            raise RuntimeError(f"input hash mismatch: {name}")
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, work / name)
    return listing


def main() -> None:
    p = json.loads(os.environ.get("RESEARCH_HARNESS_PARAMETERS", "{}"))
    start = time.time()
    working = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out").resolve()
    working.mkdir(parents=True, exist_ok=True)
    cpu = cpu_info()
    threads = str(p["threads"])
    os.environ.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    # same ATen kernels as the local runs (AVX512 there); only when this CPU has them
    if p.get("cpu_capability") == "avx512" and cpu["avx512f"]:
        os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
    if p.get("pins"):
        install(p["pins"])

    work = Path(tempfile.mkdtemp(prefix="dce_work_"))
    listing = stage_inputs(p["inputs_sha256"], work)
    out = work / "out"
    (out / "pairs").mkdir(parents=True)
    pairs = [int(x) for x in p["pairs"]]
    resumed = []
    for f in (f for d in find(p["resume"]) for f in d.glob("pair_*.partial.*")) if p.get("resume") else ():
        if int(f.name.split("_")[1].split(".")[0]) in pairs:
            shutil.copy(f, out / "pairs" / f.name)
            resumed.append(f.name)
    os.chdir(work)

    import importlib

    import numpy
    import scipy
    import torch

    module = p.get("module", "dce.cli.v110_tc")          # D-044 executor: T-C by default; any `run_pairs` stage
    run_main = importlib.import_module(module).main
    if module == "dce.cli.v110_tc":
        argv = ["tc-run", "--config", p["config"], "--out", str(out), "--shard", "dyn", "--threads", threads,
                "--pairs", ",".join(map(str, pairs))]
        if p.get("deadline_h"):
            argv += ["--deadline-h", str(p["deadline_h"])]
    else:
        argv = [*p["argv"], "--config", p["config"], "--out", str(out)]
    rc = run_main(argv)

    for f in sorted((out / "pairs").glob("pair_*")):
        if not f.name.endswith(".claim"):
            shutil.copy(f, working / f.name)
    meta = sorted(out.glob("metadata_*.json"))
    if meta:
        shutil.copy(meta[0], working / "metadata.json")
    done = [q for q in pairs if (working / f"pair_{q}.json").exists() and (working / f"pair_{q}.npz").exists()]
    result = {
        "experiment": p.get("experiment", "exp_0034"), "module": module, "decision": "D-044", "observed_hardware": f"CPU: {cpu['model']}", "rc": rc, "pairs": pairs, "pairs_done": done,
        "pairs_partial": [q for q in pairs if (working / f"pair_{q}.partial.json").exists()], "resumed_files": resumed, "inputs_listing": str(listing),
        "config": p["config"], "config_sha256": sha256(work / p["config"]), "inputs_listing_sha256": p["inputs_sha256"],
        "bundle_tree_hash": os.environ.get("DCE_BUNDLE_TREE_HASH"),
        "env": {"cpu": cpu, "python": sys.version.split()[0], "platform": platform.platform(),
                "torch": torch.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
                "aten_cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "aten_cpu_capability_env": os.environ.get("ATEN_CPU_CAPABILITY"),
                "torch_threads": torch.get_num_threads(), "mkl": torch.backends.mkl.is_available(),
                "mkldnn": torch.backends.mkldnn.is_available()},
        "wall_s": round(time.time() - start, 1),
        "work": {"processed": len(done)},
    }
    (working / "result.json").write_text(json.dumps(result, indent=2))
    files = sorted(f for f in working.iterdir() if f.is_file() and f.name != "artifact_manifest.json")
    manifest = [{"name": f.name, "path": f.name, "sha256": sha256(f), "bytes": f.stat().st_size} for f in files]
    (working / "artifact_manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": manifest}, indent=2))
    print(json.dumps({k: result[k] for k in ("rc", "pairs_done", "pairs_partial", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()
