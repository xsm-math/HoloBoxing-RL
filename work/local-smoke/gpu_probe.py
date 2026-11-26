"""Exercise the pinned CUDA runtimes and the official frozen ONNX graph."""

import importlib.metadata
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[2] / "upstream" / "HoloMotion"
sys.path.insert(0, str(SOURCE))

from holoboxing.evaluation.models import verify_bundle
from holoboxing.evaluation.runtime.onnx_device import create_device_session, verify_cuda_execution


def stack():
    import mujoco
    import mujoco_warp
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime
    import warp as wp

    for name in ("torch", "warp-lang", "mujoco", "mujoco-warp", "onnx", "onnxruntime-gpu"):
        print(name, importlib.metadata.version(name), flush=True)
    assert torch.version.cuda == "12.8"
    value = torch.ones((16, 16), dtype=torch.float32, device="cuda:0")
    assert (value @ value).sum().item() == 4096
    print("torch_cuda_matmul PASS", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    wp.init()

    @wp.kernel
    def add_one(values: wp.array(dtype=wp.float32)):
        index = wp.tid()
        values[index] = values[index] + 1.0

    values = wp.zeros(4, dtype=wp.float32, device="cuda:0")
    wp.launch(add_one, dim=4, inputs=[values], device="cuda:0")
    wp.synchronize()
    assert values.numpy().tolist() == [1.0] * 4
    print("warp_cuda_kernel PASS", wp.get_device("cuda:0"), flush=True)
    box = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 2])
    node = helper.make_node("MatMul", ["matrix", "matrix"], ["result"])
    graph = helper.make_graph([node], "cuda_matmul", [box("matrix")], [box("result")])
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_operatorsetid("", 17)])
    with TemporaryDirectory() as temp:
        file = Path(temp) / "test.onnx"
        onnx.save(model, file)
        session = create_device_session(file)
        (product,) = session.run(None, {"matrix": np.ones((2, 2), dtype=np.float32)})
        np.testing.assert_array_equal(product, np.full((2, 2), 2.0, dtype=np.float32))
        verify_cuda_execution(session)
        print("onnxruntime_cuda_matmul PASS", session.get_providers(), flush=True)


def lower(directory):
    model = verify_bundle(directory) / "exported" / "model_10500.onnx"
    print("Frozen bundle SHA-256 verified", model, flush=True)
    session = create_device_session(model)
    expected = {"obs": [1, 1378], "past_key_values": [1, 2, 1, 32, 4, 64], "step_idx": [1]}
    assert {value.name: value.shape for value in session.get_inputs()} == expected
    print("Frozen model CUDA session created", session.get_providers(), flush=True)
    inputs = {
        name: np.zeros(shape, dtype=np.int64 if name == "step_idx" else np.float32)
        for name, shape in expected.items()
    }
    (action,) = session.run(["actions"], inputs)
    assert action.shape == (1, 29) and np.isfinite(action).all()
    verify_cuda_execution(session)
    print("frozen_lower_one_forward PASS", action.shape, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "stack":
        stack()
    elif sys.argv[1] == "lower":
        lower(sys.argv[2])
    else:
        raise SystemExit("Expected stack or lower")
