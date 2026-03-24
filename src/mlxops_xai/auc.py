# Enable PEP 563 postponed evaluation of annotations for Python < 3.10 compatibility
# (allows `list[float] | None` syntax without importing from typing)
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from mlxops_utils import plot_hor, clp


class AUC:
    """
    Deletion/Insertion AUC evaluator for XAI saliency maps.

    Measures saliency map quality by:
    - Deletion: progressively zeroing out the most salient pixels and tracking confidence drop.
    - Insertion: progressively revealing the most salient pixels from a blurred background
      and tracking confidence rise.

    A good saliency map should cause a sharp confidence drop under deletion (low DAUC)
    and a rapid confidence rise under insertion (high IAUC).

    Usage:
        auc = AUC(percentages=[1, 0.8, 0.6, 0.4, 0.2])
        d_prob, i_prob = auc.get_input(model, batch, saliency_maps)
        result = auc.score(d_prob, i_prob)
        print(result['Overall_AUC'])

    Args:
        percentages: Pixel-retention thresholds for deletion/insertion steps (default [1, 0.8, 0.6, 0.4, 0.2]).
        sigma: Gaussian blur kernel sigma used for insertion background (default 16).
        batch_size: Batch size for model inference (default 128).
        debug: If True, visualize deletion/insertion images and print probabilities.
    """

    def __init__(
        self,
        percentages: list[float] | None = None,
        sigma: int = 16,
        batch_size: int = 128,
        debug: bool = False,
    ) -> None:
        self.percentages = percentages or [1, 0.8, 0.6, 0.4, 0.2]
        self.sigma = sigma
        self.batch_size = batch_size
        self.debug = debug
        self._delete_percentage = self.percentages
        self._insert_percentage = np.flip(self.percentages)

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
                device, targets, n, num_aug,
            )
            i_pred_prob = self._get_pred(
                model,
                DataLoader(TensorDataset(all_inserted), batch_size=self.batch_size, shuffle=False),
                device, targets, n, num_aug,
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
            d_pred_prob: Output of get_input() deletion probabilities (N, num_steps).
            i_pred_prob: Output of get_input() insertion probabilities (N, num_steps).

        Returns:
            Dict with DAUC, IAUC, Overall_AUC, AUC_Percentage, DAUC_arr, IAUC_arr.
        """
        dauc = metrics.auc(np.flip(self._delete_percentage), d_pred_prob.mean(0))
        iauc = metrics.auc(self._insert_percentage, i_pred_prob.mean(0))
        return {
            'DAUC': dauc,
            'IAUC': iauc,
            'AUC_Percentage': self._insert_percentage,
            'DAUC_arr': d_pred_prob,
            'IAUC_arr': i_pred_prob,
        }

    def evaluate(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> dict:
        """Convenience method: get_input + score in one call."""
        d_prob, i_prob = self.get_input(model, batch, saliency_maps)
        return self.score(d_prob, i_prob)

    def _build_variants(
        self,
        original_images: torch.Tensor,
        saliency_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_deleted = []
        all_inserted = []
        blurrer = v2.GaussianBlur(kernel_size=self.sigma * 2 + 1, sigma=self.sigma)

        for i in range(len(original_images)):
            image = original_images[i]
            saliency_map = saliency_maps[i]

            delete_quantiles = self._get_quantiles(saliency_map, self._delete_percentage)
            insert_quantiles = self._get_quantiles(saliency_map, self._insert_percentage, remove_head=False)

            deleted = [torch.where(saliency_map > q, 0, image) for q in np.flip(delete_quantiles)]
            blurred = blurrer(image)
            inserted = [torch.where(saliency_map > q, image, blurred) for q in insert_quantiles]

            if self.debug:
                plot_hor([clp(k.cpu()) for k in deleted])
                plot_hor([clp(k.cpu()) for k in inserted])

            all_deleted.extend(deleted)
            all_inserted.extend(inserted)

        return torch.stack(all_deleted), torch.stack(all_inserted)

    @staticmethod
    def _get_pred(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        targets: torch.Tensor,
        n: int,
        num_aug: int,
    ) -> torch.Tensor:
        prediction = []
        for d in dataloader:
            prediction.extend(model(d[0].to(device)))

        prediction = torch.stack(prediction).to(device)
        num_classes = prediction.shape[1]
        prediction = prediction.reshape(n, num_aug, num_classes)
        sm = F.softmax(prediction, dim=2)
        pred_prob = []
        for i, smm in enumerate(sm):
            pred_prob.extend(smm[:, targets[i]])
        return torch.tensor(pred_prob, device=device).reshape(n, num_aug)

    @staticmethod
    def _get_quantiles(
        saliency_map: torch.Tensor,
        percentage: list[float],
        remove_head: bool = False,
    ) -> np.ndarray:
        v, _ = torch.sort(saliency_map.flatten())
        q_idx = [max(int(v.shape[0] * rate) - 1, 0) for rate in percentage]
        q = np.flip(np.array([v[i].item() for i in q_idx]))
        if remove_head and (len(q) == len(set(q))) and (len(q) > 1):
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around AUC.get_input(). See AUC for full documentation."""
    return AUC(sigma=sigma or 16, debug=debug).get_input(model, batch, saliency_maps)


def get_auc_score(
    d_pred_prob: torch.Tensor,
    i_pred_prob: torch.Tensor,
) -> dict:
    """Convenience wrapper around AUC.score(). See AUC for full documentation."""
    return AUC().score(d_pred_prob, i_pred_prob)
