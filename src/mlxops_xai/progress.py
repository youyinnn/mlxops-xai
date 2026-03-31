from __future__ import annotations

from dataclasses import dataclass
from torchmetrics import Metric


@dataclass
class XAIProgress:
    """Represents a single progress update from RCAP or AUC."""
    source: str   # 'RCAP' or 'AUC'
    desc: str     # description of the current stage
    current: int  # number of items completed so far
    total: int    # total number of items

    @property
    def fraction(self) -> float:
        return self.current / self.total if self.total > 0 else 0.0


class XAIMetric(Metric):
    """
    Base class for XAI evaluators (RCAP, AUC).

    Provides shared configuration:
        debug:        If True, enable verbose debug output and visualizations.
        tqdm_verbose: If True, show internal tqdm progress bars.
        on_progress:  Optional callable(XAIProgress) invoked on each progress step.
    """

    def __init__(
        self,
        debug: bool = False,
        tqdm_verbose: bool = False,
        on_progress: callable = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.debug = debug
        self.tqdm_verbose = tqdm_verbose
        self.on_progress = on_progress
