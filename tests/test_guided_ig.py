import torch

from mlxops_xai import guided_ig

from conftest import make_batch


def test_output_shape_and_range(model, device):
    """guided_ig returns one (H, W) map per image, min-max normalized."""
    images, targets = make_batch(device, n=2, size=32)

    out = guided_ig(model, images, targets, num_samples=3, max_dist=0.1)

    assert out.shape == (images.shape[0], images.shape[2], images.shape[3])
    assert torch.isfinite(out).all()
    # min_max_normalize keeps values in [0, 1]
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_targets_none_uses_argmax(model, device):
    """Passing targets=None should not error (targets inferred from model)."""
    images, _ = make_batch(device, n=1, size=32)

    out = guided_ig(model, images, None, num_samples=2, max_dist=0.1)

    assert out.shape == (1, 32, 32)
    assert torch.isfinite(out).all()


def test_deterministic(model, device):
    """Same input + zero baseline => identical output across runs."""
    images, targets = make_batch(device, n=1, size=32)

    a = guided_ig(model, images, targets, num_samples=3, max_dist=0.1)
    b = guided_ig(model, images, targets, num_samples=3, max_dist=0.1)

    assert torch.allclose(a, b, atol=1e-5)
