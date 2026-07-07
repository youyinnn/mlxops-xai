import os

import pytest
import torch
import torch.nn as nn


def _resolve_device() -> torch.device:
    """Pick a device. Override with XAI_TEST_DEVICE=cpu|mps|cuda."""
    forced = os.environ.get("XAI_TEST_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return _resolve_device()


class TinyCNN(nn.Module):
    """Small conv classifier, big enough to exercise the backward hooks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@pytest.fixture(scope="session")
def model(device) -> nn.Module:
    torch.manual_seed(0)
    m = TinyCNN().to(device).eval()
    return m


def make_batch(device, n: int = 2, size: int = 32, seed: int = 0):
    """Deterministic image batch + targets."""
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, size, size, generator=g).to(device)
    targets = torch.randint(0, 10, (n,), generator=g).to(device)
    return images, targets


@pytest.fixture
def batch(device):
    return make_batch(device, n=2, size=32)
