"""Infrastructure smoke, offline GPU kernel.

Reads the preparation kernel output through `kernel_sources`, verifies its
SHA256, records the observed accelerator, and runs one tiny training step on the
GPU. Not a scientific run.
"""

import hashlib
import json
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


def find_prep():
    for root in (Path("/kaggle/input"), Path("in")):
        if root.exists():
            hits = sorted(root.rglob("mnist_4_9.npz"))
            if hits:
                return hits[0]
    raise FileNotFoundError("mnist_4_9.npz not found under /kaggle/input")


def main():
    start = time.time()
    import numpy as np
    import torch
    from torch import nn

    prep = find_prep()
    manifest_path = prep.parent / "artifact_manifest.json"
    expected = {
        a["path"]: a["sha256"] for a in json.loads(manifest_path.read_text())["artifacts"]
    }["mnist_4_9.npz"]
    actual = sha256(prep)
    if actual != expected:
        raise RuntimeError(f"prep artifact hash mismatch: {actual} != {expected}")

    data = np.load(prep)
    x = torch.tensor(data["train_x"][:512], dtype=torch.float32).div(255).unsqueeze(1)
    y = torch.tensor((data["train_y"][:512] == 9).astype("int64"))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2)
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
    loss.backward()
    optimiser.step()

    try:
        import scipy
        import skimage

        sk = {"skimage": skimage.__version__, "scipy": scipy.__version__}
    except Exception as exc:  # noqa: BLE001
        sk = {"error": f"{type(exc).__name__}: {exc}"}

    result = {
        "schema_version": 1,
        "kind": "smoke_gpu",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "gpus": gpus,
        "observed_hardware": gpus or ["cpu"],
        "prep_artifact_sha256": actual,
        "prep_hash_verified": True,
        "mnist_train_rows_used": int(x.shape[0]),
        "loss": float(loss.item()),
        "image_libs": sk,
        "seconds": round(time.time() - start, 2),
        "work": {"processed": int(x.shape[0])},
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": []}, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
