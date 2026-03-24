# Enable PEP 563 postponed evaluation of annotations for Python < 3.10 compatibility
# (allows `list[float] | None` syntax without importing from typing)
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlxops_utils import data_utils, plotting_utils


class RCAP:
    """
    RCAP (Recovered image Confidence during Progressive image recovery) evaluator.

    Measures XAI saliency map quality by progressively recovering an image from
    black starting from the most salient regions, and tracking model confidence
    at each stage. A good saliency map should restore model confidence quickly
    using only a small fraction of the image.

    Usage:
        rcap = RCAP(lower_bound=0.5, recover_interval=0.1)
        result = rcap.evaluate(model, batch, saliency_func, saliency_func_kwargs)
        print(result['overall_rcap']['RCAP'])

    Args:
        lower_bound: Lowest saliency quantile to start recovery from (default 0.7).
        recover_interval: Quantile step between recovery stages (default 0.3).
        debug: If True, visualize recovered images at each stage.
    """

    def __init__(
        self,
        lower_bound: float = 0.7,
        recover_interval: float = 0.3,
        debug: bool = False,
    ) -> None:
        self.lower_bound = lower_bound
        self.recover_interval = recover_interval
        self.debug = debug

    def evaluate(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Compute RCAP scores for a batch of images.

        Args:
            model: Trained classification model.
            batch: Tuple of (images, targets) — normalized input tensor and class indices.
            saliency_maps: Pre-computed saliency maps (N, H, W), values in [0, 1].

        Returns:
            Dict with prediction scores, saliency statistics, and overall RCAP scores.
        """
        model = model.eval()
        images, targets = batch

        saliency_maps = saliency_maps.to(images.device)

        recovery_outputs = self._compute_recovery_scores(
            model, images, targets, saliency_maps)

        original_pred_score, recovered_pred_score, \
            original_pred_prob, recovered_pred_prob, \
            local_heat_mean, local_heat_sum, \
            overall_heat_mean, overall_heat_sum, \
            original_pred_prob_full, recovered_pred_prob_full, \
            recovered_imgs = recovery_outputs

        score_inputs = (
            np.array(original_pred_score), np.array(recovered_pred_score),
            np.array(original_pred_prob), np.array(recovered_pred_prob),
            np.array(local_heat_mean), np.array(local_heat_sum),
            np.array(overall_heat_mean), np.array(overall_heat_sum),
            None,
        )

        return {
            'original_pred_score':       score_inputs[0],
            'recovered_pred_score':      score_inputs[1],
            'original_pred_prob':        score_inputs[2],
            'recovered_pred_prob':       score_inputs[3],
            'local_heat_mean':           score_inputs[4],
            'local_heat_sum':            score_inputs[5],
            'overall_heat_mean':         score_inputs[6],
            'overall_heat_sum':          score_inputs[7],
            'all_original_pred_prob_full':  np.array(original_pred_prob_full),
            'all_recovered_pred_prob_full': np.array(recovered_pred_prob_full),
            'overall_rcap':              self._compute_rcap_score(score_inputs),
        }

    @staticmethod
    def compute_score(recovered_pred: tuple, debug: bool = False) -> dict[str, np.ndarray]:
        """
        Compute the final RCAP metric from pre-computed recovery statistics.

        Useful when recovery outputs are already available and only the score
        computation needs to be re-run (e.g. with different weighting).

        Notation (following the paper):
            M_{p_k}        : cumulative saliency sum of revealed region at stage k  (local_heat_sum)
            f(I_{p_k})     : model logit on recovered image at stage k              (recovered_pred_score)
            sigma(f(...))  : model softmax prob on recovered image                  (recovered_pred_prob)

        visual_noise_level : cumulative revealed saliency / total saliency per stage
                             = local_heat_sum / overall_heat_sum
                             Measures what fraction of the total explanation has been exposed.

        RCAP = mean_k( visual_noise_level_k * sigma(f(I_{p_k})) )
               High RCAP → model confidence restored with small revealed saliency → sharp localization.
        """
        _original_pred_score, _recovered_pred_score, \
            _original_pred_prob, recovered_pred_prob, \
            local_heat_mean, local_heat_sum, \
            _overall_heat_mean, overall_heat_sum, \
            _recovered_imgs = recovered_pred

        # visual_noise_level[i, k] = local_heat_sum[i, k] / overall_heat_sum[i]
        # overall_heat_sum[:, None] broadcasts (n,) → (n, 1) so each row divides by its own total
        visual_noise_level: np.ndarray = local_heat_sum / \
            overall_heat_sum[:, None]

        rcap_with_heat_weight: np.ndarray = \
            (local_heat_mean * visual_noise_level * recovered_pred_prob).mean(-1)
        rcap_score: np.ndarray = \
            (visual_noise_level * recovered_pred_prob).mean(-1)

        if debug:
            print('\r\n3-- visual_noise_level = local_heat_sum / overall_heat_sum')
            print(visual_noise_level)
            print('\r\n4-- recovered pred prob')
            print(recovered_pred_prob, np.mean(recovered_pred_prob, axis=1))
            print('\r\n6-- RCAP variants')
            print('local_heat_mean * visual_noise_level * recovered_pred_prob',
                  rcap_with_heat_weight)
            print('visual_noise_level * recovered_pred_prob', rcap_score)

        return {
            'visual_noise_level': visual_noise_level,
            'localization':       recovered_pred_prob,
            'RCAP':               rcap_score,
        }

    def _compute_rcap_score(self, recovered_pred: tuple) -> dict[str, np.ndarray]:
        return self.compute_score(recovered_pred, debug=self.debug)

    def _build_recovered_image(
        self,
        img: torch.Tensor,
        saliency_map: torch.Tensor,
    ) -> tuple[torch.Tensor, list[float], list[float]]:
        """
        Build a stack of progressively recovered images (I_{p_k} in the paper).

        The saliency map is divided into quantile bins. Starting from the most
        salient bin, pixels are revealed one bin at a time into a blank canvas,
        producing one recovered image per stage.

        Args:
            img: Denormalized image tensor (C, H, W).
            saliency_map: 2-D saliency map tensor (H, W), values in [0, 1].

        Returns:
            recovered_stack: Stacked recovered images (N_stages, C, H, W).
            local_heat_mean: Cumulative mean saliency of revealed region at each stage.
            local_heat_sum:  Cumulative sum saliency of revealed region at each stage.

        Note on local_heat vs current_bin_mask:
            - current_bin_mask uses <= saliency_upper_bound to select only the *current*
              bin's pixels for incremental image reveal (avoids re-filling already-revealed pixels).
            - local_heat uses all pixels > saliency_lower_bound to accumulate saliency over
              *all revealed bins so far*, matching the cumulative nature of the recovered image.
        """
        recovered_stack: list[torch.Tensor] = []
        sorted_saliency, _ = torch.sort(saliency_map.flatten())

        # Compute quantile boundary indices and look up saliency thresholds in one shot
        # (avoids Python loop + .item() calls for each quantile)
        quantiles = [round(r, 2) for r in np.arange(
            self.lower_bound, 0.999999, self.recover_interval)]
        quantile_idx = torch.tensor(
            [int(sorted_saliency.shape[0] * q) - 1 for q in quantiles],
            dtype=torch.long,
        )
        # flip to high→low order so we reveal most-salient regions first
        quantile_thresholds = sorted_saliency[quantile_idx].flip(0)

        local_heat_mean: list[float] = []
        local_heat_sum: list[float] = []
        saliency_upper_bound: float = 1.0
        canvas = torch.zeros_like(img)  # start from a blank canvas

        for threshold in quantile_thresholds:
            t = threshold.item()
            saliency_lower_bound = t if t < 1 else 0.0

            # Mask for the current bin only — used to incrementally reveal pixels
            current_bin_mask = (saliency_map > saliency_lower_bound) & (
                saliency_map <= saliency_upper_bound)

            # Cumulative saliency stats over all revealed regions (> saliency_lower_bound)
            # saliency_map is already in [0,1] so <= 1 is always true — skip that condition
            revealed = saliency_map[saliency_map > saliency_lower_bound]
            local_heat_mean.append(revealed.mean().item())
            local_heat_sum.append(revealed.sum().item())

            canvas = torch.where(current_bin_mask, img, canvas)
            recovered_stack.append(canvas.reshape(1, *canvas.shape))
            saliency_upper_bound = saliency_lower_bound

        return torch.vstack(recovered_stack), local_heat_mean, local_heat_sum

    def _compute_recovery_scores(
        self,
        model: nn.Module,
        original_images: torch.Tensor,
        targets: torch.Tensor | None,
        saliency_maps: torch.Tensor,
    ) -> tuple:
        """
        Build recovered images for all images in the batch and measure model confidence.

        For each image, constructs N_stages recovered versions (most salient first)
        plus the full original, batches them all through the model, then splits
        predictions into recovered vs. original.

        Layout of all_inputs_batch (flattened batch sent to model):
            [img0_stage0, img0_stage1, ..., img0_full,
             img1_stage0, img1_stage1, ..., img1_full, ...]
        After reshape: (n, n_bins, num_classes), where the last bin is the full image.
        """
        device = original_images.device
        n = original_images.shape[0]
        all_inputs_batch: list[torch.Tensor] = []
        n_bins: int | None = None

        if targets is None:
            with torch.no_grad():
                targets = torch.argmax(model(original_images), dim=1)

        original_images = data_utils.denormalize(original_images)
        local_heat_mean: list[list[float]] = []
        local_heat_sum: list[list[float]] = []
        recovered_imgs: list[np.ndarray] = []

        for i in range(n):
            img = original_images[i]
            saliency_map = saliency_maps[i]
            recovered_stack, stage_heat_mean, stage_heat_sum = self._build_recovered_image(
                img, saliency_map)
            recovered_imgs.append(recovered_stack.cpu().detach().numpy())
            local_heat_mean.append(stage_heat_mean)
            local_heat_sum.append(stage_heat_sum)
            # n_bins = N_stages + 1 (the +1 is the full original image appended below)
            n_bins = recovered_stack.shape[0] + 1
            recovered_stack = torch.vstack(
                [recovered_stack, img.reshape(1, *img.shape)])
            if self.debug:
                plotting_utils.plot_hor([np.transpose(stage.cpu().detach().numpy(), (1, 2, 0))
                                         for stage in recovered_stack])
            all_inputs_batch.append(recovered_stack)

        # Stack all images into one batch and re-normalize for model inference
        all_inputs_batch_tensor = torch.vstack(all_inputs_batch).to(device)
        all_inputs_batch_tensor = data_utils.normalize(all_inputs_batch_tensor)

        local_heat_mean_t = torch.tensor(local_heat_mean, device=device)
        local_heat_sum_t = torch.tensor(local_heat_sum, device=device)
        overall_heat_mean = saliency_maps.mean(dim=(1, 2))
        overall_heat_sum = saliency_maps.sum(dim=(1, 2))

        # Run all recovered + original images through the model in one forward pass
        with torch.no_grad():
            prediction = model(all_inputs_batch_tensor)
            # Reshape to (n, n_bins, num_classes) for per-image indexing
            prediction = prediction.reshape(n, n_bins, prediction.shape[1])

            # Extract target-class logit and softmax prob using tensor gather — no CPU/numpy roundtrip
            # targets: (n,) → (n, n_bins) for gather along class dim
            # .to(device) ensures targets are on the same device as prediction
            targets_expanded = targets.to(device)[:, None].expand(n, n_bins)
            target_logits_t = prediction.gather(
                2, targets_expanded.unsqueeze(-1)).squeeze(-1)
            # last bin = full image
            original_pred_score = target_logits_t[:, -1:]
            # all recovery stages
            recovered_pred_score = target_logits_t[:, :-1]

            softmax_probs = F.softmax(prediction, dim=2)
            target_probs_t = softmax_probs.gather(
                2, targets_expanded.unsqueeze(-1)).squeeze(-1)
            original_pred_prob = target_probs_t[:, -1:]
            recovered_pred_prob = target_probs_t[:, :-1]
            # full softmax dist on original / recovered
            original_pred_prob_full = softmax_probs[:, -1:, :]
            recovered_pred_prob_full = softmax_probs[:, :-1, :]

        return (
            original_pred_score.cpu().detach().numpy(),
            recovered_pred_score.cpu().detach().numpy(),
            original_pred_prob.cpu().detach().numpy(),
            recovered_pred_prob.cpu().detach().numpy(),
            local_heat_mean_t.cpu().detach().numpy(),
            local_heat_sum_t.cpu().detach().numpy(),
            overall_heat_mean.cpu().detach().numpy(),
            overall_heat_sum.cpu().detach().numpy(),
            original_pred_prob_full.cpu().detach().numpy(),
            recovered_pred_prob_full.cpu().detach().numpy(),
            np.array(recovered_imgs),
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions — thin wrappers around RCAP class
# ---------------------------------------------------------------------------

def batch_rcap(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    saliency_maps: torch.Tensor,
    lower_bound: float = 0.7,
    recover_interval: float = 0.3,
    debug: bool = False,
) -> dict[str, np.ndarray]:
    """Convenience wrapper around RCAP.evaluate(). See RCAP for full documentation."""
    return RCAP(lower_bound, recover_interval, debug).evaluate(model, batch, saliency_maps)


def get_rcap_score(recovered_pred: tuple, debug: bool = False) -> dict[str, np.ndarray]:
    """Convenience wrapper around RCAP.compute_score(). See RCAP for full documentation."""
    return RCAP.compute_score(recovered_pred, debug)
