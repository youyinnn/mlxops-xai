# Enable PEP 563 postponed evaluation of annotations for Python < 3.10 compatibility
# (allows `list[float] | None` syntax without importing from typing)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from tqdm import tqdm
from mlxops_utils import plot_hor, clp
from mlxops_xai.progress import XAIProgress, XAIMetric


class AUC(XAIMetric):
    """
    Deletion/Insertion AUC evaluator for XAI saliency maps.

    Measures saliency map quality by:
    - Deletion: progressively zeroing out the most salient pixels and tracking confidence drop.
    - Insertion: progressively revealing the most salient pixels from a blurred background
      and tracking confidence rise.

    A good saliency map should cause a sharp confidence drop under deletion (low DAUC)
    and a rapid confidence rise under insertion (high IAUC).

    Follows the torchmetrics Metric API: call update() per batch, compute() at epoch end.
    State is automatically reset after compute(), or manually via reset().

    Usage:
        auc = AUC(percentages=[1, 0.8, 0.6, 0.4, 0.2])
        for batch, saliency_maps in dataloader:
            auc.update(model, batch, saliency_maps)
        result = auc.compute()
        print(result['DAUC'], result['IAUC'])

    Args:
        percentages: Pixel-retention thresholds for deletion/insertion steps (default [1, 0.8, 0.6, 0.4, 0.2]).
        sigma: Gaussian blur kernel sigma used for insertion background (default 16).
        batch_size: Batch size for model inference (default 128).
        debug: If True, visualize deletion/insertion images and print probabilities.
        tqdm_verbose: If True, show internal tqdm progress bars.
        on_progress: Optional callable(XAIProgress) invoked on each sample processed.
        kwargs: Passed to torchmetrics.Metric (e.g. compute_on_cpu, dist_sync_on_step).
    """

    d_probs: list
    i_probs: list

    def __init__(
        self,
        percentages: list[float] | None = None,
        sigma: int = 16,
        batch_size: int = 128,
        debug: bool = False,
        tqdm_verbose: bool = False,
        on_progress: callable = None,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug, tqdm_verbose=tqdm_verbose, on_progress=on_progress, **kwargs)
        self.percentages = percentages or [1, 0.8, 0.6, 0.4, 0.2]
        self.sigma = sigma
        self.batch_size = batch_size
        self._delete_percentage = torch.tensor(self.percentages)
        self._insert_percentage = self._delete_percentage.flip(0)

        self.add_state('d_probs', default=[], dist_reduce_fx='cat')
        self.add_state('i_probs', default=[], dist_reduce_fx='cat')

    def update(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> None:
        """
        Process a batch and accumulate deletion/insertion probabilities.

        Args:
            model: Trained classification model.
            batch: Tuple of (images, targets) — normalized input tensor (N, C, H, W) and class indices (N,).
            saliency_maps: Saliency maps (N, H, W), values in [0, 1].
        """
        d_prob, i_prob = self.get_input(model, batch, saliency_maps)
        self.d_probs.append(d_prob)
        self.i_probs.append(i_prob)

    def compute(self) -> dict:
        """
        Aggregate all accumulated batches and compute AUC scores over the full dataset.

        Returns:
            Dict with DAUC, IAUC, AUC_Percentage, DAUC_arr, IAUC_arr.
        """
        d_pred_prob = torch.cat(self.d_probs, dim=0)
        i_pred_prob = torch.cat(self.i_probs, dim=0)
        return self.score(d_pred_prob, i_pred_prob)

    def evaluate(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> dict:
        """
        Compute AUC scores for a single batch of images in one shot.

        For multi-batch evaluation over a full dataset, use update() + compute() instead.
        """
        self.update(model, batch, saliency_maps)
        result = self.compute()
        self.reset()
        return result

    def get_input(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build deletion/insertion image variants and run model inference.

        Args:
            model: Trained classification model.
            batch: Tuple of (images, targets) — normalized input tensor (N, C, H, W) and class indices (N,).
            saliency_maps: Saliency maps (N, H, W), values in [0, 1].

        Returns:
            d_pred_prob: Model confidence on deleted images (N, num_steps).
            i_pred_prob: Model confidence on inserted images (N, num_steps).
        """
        original_images, targets = batch
        n = original_images.shape[0]
        device = original_images.device
        num_aug = len(self._delete_percentage)

        all_deleted, all_inserted = self._build_variants(original_images, saliency_maps)

        with torch.no_grad():
            d_pred_prob = self._get_pred(
                model,
                DataLoader(TensorDataset(all_deleted), batch_size=self.batch_size, shuffle=False),
                device, targets, n, num_aug, 'deletion',
            )
            i_pred_prob = self._get_pred(
                model,
                DataLoader(TensorDataset(all_inserted), batch_size=self.batch_size, shuffle=False),
                device, targets, n, num_aug, 'insertion',
            )

        if self.debug:
            print(f'D Prob: {d_pred_prob.cpu().numpy()}')
            print(f'I Prob: {i_pred_prob.cpu().numpy()}')

        return d_pred_prob.cpu(), i_pred_prob.cpu()

    def score(
        self,
        d_pred_prob: torch.Tensor,
        i_pred_prob: torch.Tensor,
    ) -> dict:
        """
        Compute DAUC, IAUC, and Overall AUC from model confidence arrays.

        Args:
            d_pred_prob: Deletion probabilities (N, num_steps).
            i_pred_prob: Insertion probabilities (N, num_steps).

        Returns:
            Dict with DAUC, IAUC, AUC_Percentage, DAUC_arr, IAUC_arr.
        """
        dauc = metrics.auc(self._delete_percentage.flip(0).numpy(), d_pred_prob.mean(0).numpy())
        iauc = metrics.auc(self._insert_percentage.numpy(), i_pred_prob.mean(0).numpy())
        return {
            'DAUC': dauc,
            'IAUC': iauc,
            'AUC_Percentage': self._insert_percentage,
            'DAUC_arr': d_pred_prob,
            'IAUC_arr': i_pred_prob,
        }

    def _build_variants(
        self,
        original_images: torch.Tensor,
        saliency_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(original_images)
        device = original_images.device
        blurrer = v2.GaussianBlur(kernel_size=self.sigma * 2 + 1, sigma=self.sigma)

        if self.on_progress is not None:
            self.on_progress(XAIProgress(source='AUC', desc='blurring', current=0, total=n))

        blurred = blurrer(original_images).unsqueeze(1)

        if self.on_progress is not None:
            self.on_progress(XAIProgress(source='AUC', desc='building variants', current=0, total=n))

        # Compute per-image quantile thresholds for all images at once
        sorted_sal, _ = torch.sort(saliency_maps.flatten(1), dim=1)  # (N, H*W)
        num_pixels = sorted_sal.shape[1]
        d_idx = torch.clamp((self._delete_percentage.to(device) * num_pixels).long() - 1, min=0)  # (num_steps,)
        i_idx = torch.clamp((self._insert_percentage.to(device) * num_pixels).long() - 1, min=0)

        # (N, num_steps); deletion needs low→high order (flip), insertion is already low→high
        d_thresholds = sorted_sal[:, d_idx].flip(1)  # (N, num_steps)
        i_thresholds = sorted_sal[:, i_idx]           # (N, num_steps)

        # Vectorized mask + image generation: (N, num_steps, H, W)
        sal = saliency_maps.unsqueeze(1)              # (N, 1, H, W)
        img = original_images.unsqueeze(1)            # (N, 1, C, H, W)

        d_masks = sal > d_thresholds.view(n, -1, 1, 1)  # (N, num_steps, H, W)
        i_masks = sal > i_thresholds.view(n, -1, 1, 1)

        all_deleted = torch.where(d_masks.unsqueeze(2), torch.zeros_like(img), img)   # (N, num_steps, C, H, W)
        all_inserted = torch.where(i_masks.unsqueeze(2), img, blurred)

        if self.on_progress is not None:
            self.on_progress(XAIProgress(source='AUC', desc='building variants', current=n, total=n))

        if self.debug:
            num_steps = len(self._delete_percentage)
            for i in range(n):
                plot_hor([clp(all_deleted[i, s].cpu()) for s in range(num_steps)])
                plot_hor([clp(all_inserted[i, s].cpu()) for s in range(num_steps)])

        # flatten to (N*num_steps, C, H, W) for DataLoader
        return all_deleted.flatten(0, 1), all_inserted.flatten(0, 1)

    def _get_pred(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        targets: torch.Tensor,
        n: int,
        num_aug: int,
        phase: str = 'inference',
    ) -> torch.Tensor:
        batches = []
        total_batches = len(dataloader)
        desc = f'AUC: {phase} inference'
        for batch_idx, d in enumerate(tqdm(dataloader, desc=desc, disable=not self.tqdm_verbose)):
            batches.append(model(d[0].to(device)))
            if self.on_progress is not None:
                self.on_progress(XAIProgress(
                    source='AUC',
                    desc=f'{phase} inference',
                    current=batch_idx + 1,
                    total=total_batches,
                ))

        prediction = torch.cat(batches, dim=0)  # (N*num_aug, num_classes)
        num_classes = prediction.shape[1]
        prediction = prediction.reshape(n, num_aug, num_classes)
        sm = F.softmax(prediction, dim=2)       # (n, num_aug, num_classes)
        # gather target-class prob for each image — no Python loop
        targets_exp = targets.to(device)[:, None, None].expand(n, num_aug, 1)
        return sm.gather(2, targets_exp).squeeze(2).cpu()  # (n, num_aug)

    @staticmethod
    def _get_quantiles(
        saliency_map: torch.Tensor,
        percentage: torch.Tensor,
        remove_head: bool = False,
    ) -> torch.Tensor:
        v, _ = torch.sort(saliency_map.flatten())
        q_idx = torch.clamp((percentage * v.shape[0]).long() - 1, min=0)
        q = v[q_idx].flip(0)
        if remove_head and (len(q) == len(q.unique())) and (len(q) > 1):
            if q[0] == 1:
                q = q[1:]
        return q


# ---------------------------------------------------------------------------
# Module-level convenience functions — thin wrappers around AUC class
# ---------------------------------------------------------------------------

def get_auc_input(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    saliency_maps: torch.Tensor,
    sigma: int | None = None,
    debug: bool = False,
    tqdm_verbose: bool = False,
    on_progress: callable = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around AUC.get_input(). See AUC for full documentation."""
    return AUC(sigma=sigma or 16, debug=debug, tqdm_verbose=tqdm_verbose, on_progress=on_progress).get_input(model, batch, saliency_maps)


def get_auc_score(
    d_pred_prob: torch.Tensor,
    i_pred_prob: torch.Tensor,
) -> dict:
    """Convenience wrapper around AUC.score(). See AUC for full documentation."""
    return AUC().score(d_pred_prob, i_pred_prob)
