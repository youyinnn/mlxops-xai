"""Speed benchmark for guided_ig.

Run with:
    pytest tests/test_benchmark_guided_ig.py -s -v

Override the device with XAI_TEST_DEVICE=cpu|mps|cuda.
Scale the workload with XAI_BENCH_N / XAI_BENCH_STEPS / XAI_BENCH_SIZE.
"""

import os
import time

import pytest
import torch

from mlxops_xai import guided_ig

from conftest import make_batch


def _sync(device: torch.device):
    """Block until queued device work is done so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _bench(fn, device, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


@pytest.mark.parametrize("n", [int(os.environ.get("XAI_BENCH_N", "8"))])
@pytest.mark.parametrize("steps", [int(os.environ.get("XAI_BENCH_STEPS", "15"))])
def test_benchmark_guided_ig(model, device, n, steps):
    size = int(os.environ.get("XAI_BENCH_SIZE", "32"))
    images, targets = make_batch(device, n=n, size=size)

    def run():
        return guided_ig(
            model, images, targets, num_samples=steps, max_dist=0.1
        )

    best, avg = _bench(run, device, warmup=1, repeat=3)

    per_image = best / n
    print(
        f"\n[guided_ig] device={device.type} n={n} steps={steps} size={size} "
        f"| best={best*1e3:.1f}ms avg={avg*1e3:.1f}ms per_image={per_image*1e3:.1f}ms"
    )

    # Guard against a hang / catastrophic regression, not a strict SLA.
    # Baseline (MPS, TinyCNN): ~110ms/image. CPU is slower, so keep headroom.
    assert per_image < 5.0
