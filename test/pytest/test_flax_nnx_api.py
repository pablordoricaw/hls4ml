from pathlib import Path

import pytest

import numpy as np
from flax import nnx

import hls4ml

test_root_path = Path(__file__).parent


@pytest.mark.parametrize("backend", ["Vivado", "Vitis", "Quartus", "oneAPI"])
@pytest.mark.parametrize("io_type", ["io_parallel", "io_stream"])
def test_linear_model(backend, io_type):
    class ReLU(nnx.Module):
        def __call__(self, x):
            return nnx.relu(x)

    class Sigmoid(nnx.Module):
        def __call__(self, x):
            return nnx.sigmoid(x)

    model = nnx.Sequential(
        nnx.Linear(1, 2, rngs=nnx.Rngs(0)),
        ReLU(),
        nnx.Linear(2, 1, rngs=nnx.Rngs(0)),
        Sigmoid(),
    )

    X_input = np.random.rand(100, 1)

    flax_prediction = model(X_input)

    output_dir = str(test_root_path / f"hls4mlprj_flax_api_linear_{backend}_{io_type}")

    config = None
    hls_model = hls4ml.converters.convert_from_flax_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        io_type=io_type,
    )
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, flax_prediction, rtol=1e-2, atol=0.01)
