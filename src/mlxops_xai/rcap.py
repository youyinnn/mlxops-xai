# Enable PEP 563 postponed evaluation of annotations for Python < 3.10 compatibility
# (allows `list[float] | None` syntax without importing from typing)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mlxops_utils import data_utils, plotting_utils
from mlxops_xai.progress import XAIProgress, XAIMetric


class RCAP(XAIMetric):
    """
    RCAP (Recovered image Confidence during Progressive image recovery) evaluator.

    Measures XAI saliency map quality by progressively recovering an image from
    black starting from the most salient regions, and tracking model confidence
    at each stage. A good saliency map should restore model confidence quickly
    using only a small fraction of the image.

    Follows the torchmetrics Metric API: call update() per batch, compute() at epoch end.
    State is automatically reset after compute(), or manually via reset().

    Usage:
        rcap = RCAP(lower_bound=0.5, recover_interval=0.1)
        for batch, saliency_maps in dataloader:
            rcap.update(model, batch, saliency_maps)
        result = rcap.compute()
        print(result['overall_rcap']['RCAP'])

    Args:
        lower_bound: Lowest saliency quantile to start recovery from (default 0.7).
        recover_interval: Quantile step between recovery stages (default 0.3).
        debug: If True, visualize recovered images at each stage.
        tqdm_verbose: If True, show internal tqdm progress bars.
        on_progress: Optional callable(XAIProgress) invoked on each sample processed.
        kwargs: Passed to torchmetrics.Metric (e.g. compute_on_cpu, dist_sync_on_step).
    """

    # Accumulated per-sample arrays — stored as list states so torchmetrics
    # can cat them across distributed workers and reset them automatically.
    original_pred_score: list
    recovered_pred_score: list
    original_pred_prob: list
    recovered_pred_prob: list
    local_heat_mean: list
    local_heat_sum: list
    overall_heat_mean: list
    overall_heat_sum: list
    all_original_pred_prob_full: list
    all_recovered_pred_prob_full: list

    def __init__(
        self,
        lower_bound: float = 0.7,
        recover_interval: float = 0.3,
        debug: bool = False,
        tqdm_verbose: bool = False,
        on_progress: callable = None,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug, tqdm_verbose=tqdm_verbose, on_progress=on_progress, **kwargs)
        self.lower_bound = lower_bound
        self.recover_interval = recover_interval

        for name in (
            'original_pred_score', 'recovered_pred_score',
            'original_pred_prob', 'recovered_pred_prob',
            'local_heat_mean', 'local_heat_sum',
            'overall_heat_mean', 'overall_heat_sum',
            'all_original_pred_prob_full', 'all_recovered_pred_prob_full',
        ):
            self.add_state(name, default=[], dist_reduce_fx='cat')

    def update(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> None:
        """
        Process a batch and accumulate recovery statistics.

        Args:
            model: Trained classification model.
            batch: Tuple of (images, targets) — normalized input tensor and class indices.
            saliency_maps: Pre-computed saliency maps (N, H, W), values in [0, 1].
        """
        model = model.eval()
        images, targets = batch
        saliency_maps = saliency_maps.to(images.device)

        (
            original_pred_score, recovered_pred_score,
            original_pred_prob, recovered_pred_prob,
            local_heat_mean, local_heat_sum,
            overall_heat_mean, overall_heat_sum,
            original_pred_prob_full, recovered_pred_prob_full,
            _recovered_imgs,
        ) = self._compute_recovery_scores(model, images, targets, saliency_maps)

        self.original_pred_score.append(original_pred_score)
        self.recovered_pred_score.append(recovered_pred_score)
        self.original_pred_prob.append(original_pred_prob)
        self.recovered_pred_prob.append(recovered_pred_prob)
        self.local_heat_mean.append(local_heat_mean)
        self.local_heat_sum.append(local_heat_sum)
        self.overall_heat_mean.append(overall_heat_mean)
        self.overall_heat_sum.append(overall_heat_sum)
        self.all_original_pred_prob_full.append(original_pred_prob_full)
        self.all_recovered_pred_prob_full.append(recovered_pred_prob_full)

    def compute(self) -> dict[str, torch.Tensor | dict]:
        """
        Aggregate all accumulated batches and compute RCAP scores over the full dataset.

        Returns:
            Dict with prediction scores, saliency statistics, and overall RCAP scores.
        """
        def _cat(tensors: list[torch.Tensor]) -> torch.Tensor:
            return torch.cat(tensors, dim=0).cpu()

        original_pred_score   = _cat(self.original_pred_score)
        recovered_pred_score  = _cat(self.recovered_pred_score)
        original_pred_prob    = _cat(self.original_pred_prob)
        recovered_pred_prob   = _cat(self.recovered_pred_prob)
        local_heat_mean       = _cat(self.local_heat_mean)
        local_heat_sum        = _cat(self.local_heat_sum)
        overall_heat_mean     = _cat(self.overall_heat_mean)
        overall_heat_sum      = _cat(self.overall_heat_sum)
        original_pred_prob_full  = _cat(self.all_original_pred_prob_full)
        recovered_pred_prob_full = _cat(self.all_recovered_pred_prob_full)

        score_inputs = (
            original_pred_score, recovered_pred_score,
            original_pred_prob, recovered_pred_prob,
            local_heat_mean, local_heat_sum,
            overall_heat_mean, overall_heat_sum,
            None,
        )

        return {
            'original_pred_score':          original_pred_score,
            'recovered_pred_score':         recovered_pred_score,
            'original_pred_prob':           original_pred_prob,
            'recovered_pred_prob':          recovered_pred_prob,
            'local_heat_mean':              local_heat_mean,
            'local_heat_sum':               local_heat_sum,
            'overall_heat_mean':            overall_heat_mean,
            'overall_heat_sum':             overall_heat_sum,
            'all_original_pred_prob_full':  original_pred_prob_full,
            'all_recovered_pred_prob_full': recovered_pred_prob_full,
            'overall_rcap':                 self._compute_rcap_score(score_inputs),
        }

    def evaluate(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        saliency_maps: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Compute RCAP scores for a single batch of images in one shot.

        For multi-batch evaluation over a full dataset, use update() + compute() instead.

        Args:
            model: Trained classification model.
            batch: Tuple of (images, targets) — normalized input tensor and class indices.
            saliency_maps: Pre-computed saliency maps (N, H, W), values in [0, 1].

        Returns:
            Dict with prediction scores, saliency statistics, and overall RCAP scores.
        """
        self.update(model, batch, saliency_maps)
        result = self.compute()
        self.reset()
        return result

    @staticmethod
    def compute_score(recovered_pred: tuple, debug: bool = False) -> dict[str, torch.Tensor]:
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
        visual_noise_level: torch.Tensor = local_heat_sum / overall_heat_sum[:, None]

        rcap_with_heat_weight: torch.Tensor = \
            (local_heat_mean * visual_noise_level * recovered_pred_prob).mean(-1)
        rcap_score: torch.Tensor = \
            (visual_noise_level * recovered_pred_prob).mean(-1)

        if debug:
            print('\r\n3-- visual_noise_level = local_heat_sum / overall_heat_sum')
            print(visual_noise_level)
            print('\r\n4-- recovered pred prob')
            print(recovered_pred_prob, recovered_pred_prob.mean(dim=1))
            print('\r\n6-- RCAP variants')
            print('local_heat_mean * visual_noise_level * recovered_pred_prob',
                  rcap_with_heat_weight)
            print('visual_noise_level * recovered_pred_prob', rcap_score)

        return {
            'visual_noise_level': visual_noise_level,
            'localization':       recovered_pred_prob,
            'RCAP':               rcap_score,
        }

    def _compute_rcap_score(self, recovered_pred: tuple) -> dict[str, torch.Tensor]:
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
        """
        sorted_saliency, _ = torch.sort(saliency_map.flatten())
        num_pixels = sorted_saliency.shape[0]

        # Compute quantile boundary indices and look up saliency thresholds — pure tensor ops
        quantile_vals = torch.arange(
            self.lower_bound, 0.999999, self.recover_interval,
            device=saliency_map.device,
        )
        quantile_idx = torch.clamp((quantile_vals * num_pixels).long() - 1, min=0)
        # flip to high→low order so we reveal most-salient regions first
        quantile_thresholds = sorted_saliency[quantile_idx].flip(0)  # (n_stages,)

        # Vectorized: build all cumulative masks at once — (n_stages, H, W)
        # stage k reveals all pixels with saliency > thresholds[k]
        thresholds_clamped = torch.where(
            quantile_thresholds < 1.0, quantile_thresholds, torch.zeros_like(quantile_thresholds))
        cumulative_masks = saliency_map.unsqueeze(0) > thresholds_clamped.view(-1, 1, 1)

        # Build all recovered images in one broadcast where — (n_stages, C, H, W)
        recovered_stack = torch.where(
            cumulative_masks.unsqueeze(1),
            img.unsqueeze(0).expand(len(quantile_thresholds), -1, -1, -1),
            torch.zeros_like(img).unsqueeze(0),
        )

        # Cumulative saliency stats: sum/mean over revealed pixels per stage
        cumulative_masks_f = cumulative_masks.float()
        masked_sal = saliency_map.unsqueeze(0) * cumulative_masks_f          # (n_stages, H, W)
        n_revealed = cumulative_masks_f.sum(dim=(1, 2))                       # (n_stages,)
        local_heat_sum_t = masked_sal.sum(dim=(1, 2))                         # (n_stages,)
        local_heat_mean_t = local_heat_sum_t / n_revealed.clamp(min=1)

        return recovered_stack, local_heat_mean_t, local_heat_sum_t

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

        if targets is None:
            with torch.no_grad():
                targets = torch.argmax(model(original_images), dim=1)

        original_images = data_utils.denormalize(original_images)

        if self.on_progress is not None:
            self.on_progress(XAIProgress(source='RCAP', desc='building recovered images', current=0, total=n))

        # Build recovered stacks for all images — still per-image because each image has its
        # own saliency distribution, but _build_recovered_image is fully vectorized internally
        recovered_stacks: list[torch.Tensor] = []
        heat_means: list[torch.Tensor] = []
        heat_sums: list[torch.Tensor] = []
        for i in tqdm(range(n), desc='RCAP: building recovered images', disable=not self.tqdm_verbose):
            recovered_stack, stage_heat_mean, stage_heat_sum = self._build_recovered_image(
                original_images[i], saliency_maps[i])
            recovered_stacks.append(recovered_stack)
            heat_means.append(stage_heat_mean)
            heat_sums.append(stage_heat_sum)
            if self.on_progress is not None:
                self.on_progress(XAIProgress(source='RCAP', desc='building recovered images', current=i + 1, total=n))

        n_stages = recovered_stacks[0].shape[0]
        n_bins = n_stages + 1  # stages + full original

        # Append full original to each stack, then flatten into one batch — (n*n_bins, C, H, W)
        # original_images: (n, C, H, W) → unsqueeze(1) → cat with stacks along dim=1
        all_recovered = torch.stack(recovered_stacks, dim=0)        # (n, n_stages, C, H, W)
        all_inputs = torch.cat(
            [all_recovered, original_images.unsqueeze(1)], dim=1,   # (n, n_bins, C, H, W)
        ).flatten(0, 1)                                              # (n*n_bins, C, H, W)

        if self.debug:
            for i in range(n):
                plotting_utils.plot_hor([
                    all_inputs[i * n_bins + s].cpu().detach().permute(1, 2, 0).numpy()
                    for s in range(n_bins)
                ])

        # Stack all images into one batch and re-normalize for model inference
        all_inputs_batch_tensor = data_utils.normalize(all_inputs.to(device))

        local_heat_mean_t = torch.stack(heat_means).to(device)   # (n, n_stages)
        local_heat_sum_t  = torch.stack(heat_sums).to(device)    # (n, n_stages)
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
            original_pred_score.cpu().detach(),
            recovered_pred_score.cpu().detach(),
            original_pred_prob.cpu().detach(),
            recovered_pred_prob.cpu().detach(),
            local_heat_mean_t.cpu().detach(),
            local_heat_sum_t.cpu().detach(),
            overall_heat_mean.cpu().detach(),
            overall_heat_sum.cpu().detach(),
            original_pred_prob_full.cpu().detach(),
            recovered_pred_prob_full.cpu().detach(),
            torch.stack(recovered_stacks) if self.debug else torch.empty(0),
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
    tqdm_verbose: bool = False,
    on_progress: callable = None,
) -> dict[str, torch.Tensor | dict]:
    """Convenience wrapper around RCAP.evaluate(). See RCAP for full documentation."""
    return RCAP(lower_bound, recover_interval, debug, tqdm_verbose, on_progress).evaluate(model, batch, saliency_maps)


def get_rcap_score(recovered_pred: tuple, debug: bool = False) -> dict[str, torch.Tensor]:
    """Convenience wrapper around RCAP.compute_score(). See RCAP for full documentation."""
    return RCAP.compute_score(recovered_pred, debug)
