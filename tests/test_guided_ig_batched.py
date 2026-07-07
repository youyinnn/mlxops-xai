"""The batched guided_ig must match the per-image reference numerically."""

import pytest
import torch

from mlxops_xai import guided_ig

from conftest import make_batch


@pytest.mark.parametrize("direction", ["both", "positive", "negative", "abs"])
def test_batched_matches_per_image(model, device, direction):
    images, targets = make_batch(device, n=6, size=32)

    batched = guided_ig(
        model, images, targets, num_samples=12, max_dist=0.1,
        direction=direction, aggregation="mean", batched=True,
    )
    per_image = guided_ig(
        model, images, targets, num_samples=12, max_dist=0.1,
        direction=direction, aggregation="mean", batched=False,
    )

    assert batched.shape == per_image.shape
    assert torch.allclose(batched, per_image, atol=1e-4), (
        f"max diff {(batched - per_image).abs().max().item():.3e}"
    )


def test_batched_is_faster(model, device):
    """Sanity check that batching actually reduces wall time (not a strict SLA)."""
    import time

    images, targets = make_batch(device, n=8, size=32)

    def run(batched):
        return guided_ig(
            model, images, targets, num_samples=10, max_dist=0.1,
            batched=batched,
        )

    # warmup
    run(True)
    run(False)

    def timed(batched):
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(batched)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    t_batched = timed(True)
    t_per_image = timed(False)

    assert t_batched < t_per_image
