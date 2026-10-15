"""Infrastructure smoke, preparation kernel (Internet ON).

Downloads MNIST digits 4 and 9 and writes them to the kernel output so that the
offline kernels can attach them through `kernel_sources`. Not a scientific run.
"""

import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("out")
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def version(name):
    try:
        return getattr(importlib.import_module(name), "__version__", "present")
    except Exception as exc:  # noqa: BLE001 - report any import problem
        return f"missing: {type(exc).__name__}"


def main():
    start = time.time()
    import numpy as np
    from torchvision import datasets

    root = OUT / "_download"
    arrays = {}
    for split in ("train", "test"):
        data = datasets.MNIST(str(root), train=(split == "train"), download=True)
        images = data.data.numpy()
        labels = data.targets.numpy()
        keep = (labels == 4) | (labels == 9)
        arrays[f"{split}_x"] = images[keep]
        arrays[f"{split}_y"] = labels[keep]
    target = OUT / "mnist_4_9.npz"
    np.savez_compressed(target, **arrays)
    import shutil

    shutil.rmtree(root, ignore_errors=True)

    result = {
        "schema_version": 1,
        "kind": "smoke_prep",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {n: version(n) for n in ("torch", "torchvision", "numpy", "scipy", "skimage", "yaml")},
        "mnist_4_9": {k: list(v.shape) for k, v in arrays.items()},
        "label_values": sorted({int(v) for v in arrays["train_y"]}),
        "internet_download_ok": True,
        "seconds": round(time.time() - start, 2),
        "work": {"processed": int(arrays["train_x"].shape[0] + arrays["test_x"].shape[0])},
    }
    manifest = {
        "schema_version": 1,
        "artifacts": [
            {
                "name": "mnist_4_9",
                "kind": "dataset",
                "path": "mnist_4_9.npz",
                "sha256": sha256(target),
            }
        ],
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
