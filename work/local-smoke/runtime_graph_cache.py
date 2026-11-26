"""Reuse the official ONNX execution graph from a separate process."""

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "upstream" / "HoloMotion"))

from holoboxing.evaluation.runtime import ordered_topk
from holoboxing.evaluation.runtime.model_contract import sha256


def build(specialized, cached):
    specialized = Path(specialized).resolve(strict=True)
    cached = Path(cached).resolve()
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        raise FileExistsError(cached)
    partial = cached.with_suffix(".partial")
    digest = sha256(specialized)
    print("Preparing the official execution graph", digest, flush=True)
    source = ordered_topk.execution_source(specialized)
    with partial.open("xb") as output:
        if isinstance(source, bytes):
            output.write(source)
        else:
            with Path(source).open("rb") as original:
                shutil.copyfileobj(original, output)
    record = {
        "specialized_sha256": digest,
        "execution_sha256": sha256(partial),
        "execution_bytes": partial.stat().st_size,
        "converter_sha256": sha256(Path(ordered_topk.__file__)),
        "official_conversion_returned_bytes": isinstance(source, bytes),
    }
    partial.rename(cached)
    cached.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print("Execution graph cache ready", json.dumps(record), flush=True)


def train(cached, argv):
    from holoboxing.training import train_rl

    cached = Path(cached).resolve(strict=True)
    record = json.loads(cached.with_suffix(".json").read_text())
    if record["converter_sha256"] != sha256(Path(ordered_topk.__file__)):
        raise ValueError("The official graph converter differs from the cache source")
    if record["execution_bytes"] != cached.stat().st_size or record["execution_sha256"] != sha256(cached):
        raise ValueError("The execution graph cache failed its integrity check")
    output = Path(argv[argv.index("--output") + 1]).resolve()
    expected = output / ".runtime" / "lower_batch_4.onnx"
    default_source = ordered_topk.execution_source

    def execution_source(path):
        candidate = Path(path).resolve()
        if candidate != expected:
            return default_source(path)
        if record["specialized_sha256"] != sha256(candidate):
            raise ValueError("The current specialized graph differs from the cache input")
        print("Loading cached official execution graph", record["execution_sha256"], flush=True)
        return str(cached)

    ordered_topk.execution_source = execution_source
    train_rl.main(argv)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:])
    elif sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit("Expected build or train")
