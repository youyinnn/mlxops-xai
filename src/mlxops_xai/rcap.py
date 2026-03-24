from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlxops_utils import data_utils, plotting_utils


def batch_rcap(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    saliency_func: Callable[..., torch.Tensor],
    saliency_func_kwargs: dict,
    lower_bound: float = 0.7,
    # recover_interval=0.05,
    recover_interval: float = 0.3,
    debug: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute RCAP scores for a batch of images.

    Progressive recovery: the image is revealed region-by-region in descending
    saliency order (most salient first). Model confidence is recorded at each
    stage. A good explanation should yield high confidence even when only the
    most salient regions are revealed.

    Args:
        model: Trained classification model.
        batch: Tuple of (images, targets) — normalized input tensor and class indices.
        saliency_func: Callable that returns a saliency map tensor, e.g. vanilla_gradient.
        saliency_func_kwargs: Extra kwargs forwarded to saliency_func.
        lower_bound: Lowest saliency quantile to start recovery from (default 0.7).
        recover_interval: Quantile step between recovery stages (default 0.3).
        debug: If True, visualize recovered images at each stage.

    Returns:
        Dict with prediction scores, saliency statistics, and overall RCAP scores.
    """
    model = model.eval()

    images, targets = batch
    saliency_maps = saliency_func(
        model, images, targets, **saliency_func_kwargs)
    saliency_maps = saliency_maps.to(images.device)

    recovery_outputs = get_visulalization_and_localization_score(
        model, images, targets, saliency_maps,
        recover_interval=recover_interval, lower_bound=lower_bound, debug=debug)
    original_pred_score, recovered_pred_score, \
        original_pred_prob, recovered_pred_prob, \
        local_heat_mean, local_heat_sum, \
        overall_heat_mean, overall_heat_sum, \
        original_pred_prob_full, recovered_pred_prob_full, \
        recovered_imgs = recovery_outputs

    all_original_pred_score = np.array(original_pred_score)
    all_recovered_pred_score = np.array(recovered_pred_score)
    all_original_pred_prob = np.array(original_pred_prob)
    all_recovered_pred_prob = np.array(recovered_pred_prob)

    all_local_heat_mean = np.array(local_heat_mean)
    all_local_heat_sum = np.array(local_heat_sum)
    all_overall_heat_mean = np.array(overall_heat_mean)
    all_overall_heat_sum = np.array(overall_heat_sum)

    all_original_pred_prob_full = np.array(original_pred_prob_full)
    all_recovered_pred_prob_full = np.array(recovered_pred_prob_full)

    score_inputs = (
        all_original_pred_score, all_recovered_pred_score,
        all_original_pred_prob, all_recovered_pred_prob,
        all_local_heat_mean, all_local_heat_sum,
        all_overall_heat_mean, all_overall_heat_sum,
        None
    )

    rcap_scores = get_rcap_score(score_inputs)

    return {
        'original_pred_score': all_original_pred_score,
        'recovered_pred_score': all_recovered_pred_score,
        'original_pred_prob': all_original_pred_prob,
        'recovered_pred_prob': all_recovered_pred_prob,

        'local_heat_mean': all_local_heat_mean,
        'local_heat_sum': all_local_heat_sum,
        'overall_heat_mean': all_overall_heat_mean,
        'overall_heat_sum': all_overall_heat_sum,

        'all_original_pred_prob_full': all_original_pred_prob_full,
        'all_recovered_pred_prob_full': all_recovered_pred_prob_full,

        'overall_rcap': rcap_scores
    }


def get_recovered_image(
    img: torch.Tensor,
    saliency_map: torch.Tensor,
    lower_bound: float,
    recover_interval: float,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """
    Build a stack of progressively recovered images (I_{p_k} in the paper).

    The saliency map is divided into quantile bins. Starting from the most
    salient bin, pixels are revealed one bin at a time into a blank canvas,
    producing one recovered image per stage.

    Args:
        img: Denormalized image tensor (C, H, W).
        saliency_map: 2-D saliency map tensor (H, W), values in [0, 1].
        lower_bound: Lowest quantile threshold to include (e.g. 0.7 → top 30%).
        recover_interval: Quantile step size between bins (e.g. 0.1 → 10% bins).

    Returns:
        recovered_stack: Stacked recovered images (N_stages, C, H, W).
        local_heat_mean: Cumulative mean saliency of revealed region at each stage.
        local_heat_sum:  Cumulative sum saliency of revealed region at each stage.

    Note on local_heat vs current_bin_mask:
        - current_bin_mask uses <= saliency_upper_bound to select only the *current*
          bin's pixels for incremental image reveal (avoids re-filling already-revealed pixels).
        - local_heat uses <= 1 to accumulate saliency over *all revealed bins so far*,
          matching the cumulative nature of the recovered image at each stage.
    """
    recovered_stack: list[torch.Tensor] = []
    sorted_saliency, _ = torch.sort(saliency_map.flatten())

    # Compute saliency thresholds at each quantile boundary
    quantiles = [round(rate, 2).item() for rate in np.arange(
        lower_bound, 0.999999, recover_interval)]
    quantile_idx = [int(sorted_saliency.shape[0] * quantile) - 1
                    for quantile in quantiles]

    # Map quantile indices back to actual saliency values; flip to high→low order
    quantile_thresholds = np.array(
        [sorted_saliency[i].item() for i in quantile_idx])
    quantile_thresholds = np.flip(quantile_thresholds)

    local_heat_mean: list[float] = []
    local_heat_sum: list[float] = []
    saliency_upper_bound: float = 1.0
    canvas = torch.zeros_like(img)  # start from a blank canvas

    for threshold in quantile_thresholds:
        saliency_lower_bound = float(threshold) if threshold < 1 else 0.0

        # Mask for the current bin only — used to incrementally reveal pixels
        current_bin_mask = (saliency_map > saliency_lower_bound) & (
            saliency_map <= saliency_upper_bound)

        # Cumulative saliency stats over all revealed regions (> lower_bound, <= 1)
        local_heat_mean.append(
            saliency_map[(saliency_map > saliency_lower_bound)
                         & (saliency_map <= 1)]
            .mean()
            .item()
        )
        local_heat_sum.append(
            saliency_map[(saliency_map > saliency_lower_bound) &
                         (saliency_map <= 1)].sum().item()
        )

        canvas = torch.where(current_bin_mask, img, canvas)
        recovered_stack.append(canvas.reshape(1, *canvas.shape))
        saliency_upper_bound = saliency_lower_bound

    return torch.vstack(recovered_stack), local_heat_mean, local_heat_sum


def get_visulalization_and_localization_score(
    model: nn.Module,
    original_images: torch.Tensor,
    targets: torch.Tensor | None,
    saliency_maps: torch.Tensor,
    recover_interval: float,
    lower_bound: float,
    debug: bool = False,
) -> tuple:
    """
    Core RCAP computation: build recovered images and measure model confidence.

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
            orig_pred = model(original_images)
            targets = torch.argmax(orig_pred, dim=1)

    original_images = data_utils.denormalize(original_images)
    local_heat_mean: list[list[float]] = []
    local_heat_sum: list[list[float]] = []
    recovered_imgs: list[np.ndarray] = []

    for i in range(n):
        img = original_images[i]
        saliency_map = saliency_maps[i]
        recovered_stack, stage_heat_mean, stage_heat_sum = get_recovered_image(
            img, saliency_map, recover_interval=recover_interval, lower_bound=lower_bound)
        recovered_imgs.append(recovered_stack.cpu().detach().numpy())
        local_heat_mean.append(stage_heat_mean)
        local_heat_sum.append(stage_heat_sum)
        # n_bins = N_stages + 1 (the +1 is the full original image appended below)
        n_bins = recovered_stack.shape[0] + 1
        recovered_stack = torch.vstack(
            [recovered_stack, img.reshape(1, *img.shape)])
        if debug:
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

        # Extract raw logit for the target class at each recovery stage
        target_logits: list[float] = []
        for i, logits_per_stage in enumerate(prediction.cpu().detach().numpy()):
            target_logits.extend(logits_per_stage[:, targets[i]])
        target_logits_t = torch.tensor(
            target_logits, device=device).reshape(n, n_bins)
        original_pred_score = target_logits_t[:, -1:]   # last bin = full image
        recovered_pred_score = target_logits_t[:, :-1]  # all recovery stages

        # Extract softmax probability for the target class
        softmax_probs = F.softmax(prediction, dim=2)
        target_probs: list[float] = []
        for i, probs_per_stage in enumerate(softmax_probs.cpu().detach().numpy()):
            target_probs.extend(probs_per_stage[:, targets[i]])
        target_probs_t = torch.tensor(
            target_probs, device=device).reshape(n, n_bins)
        original_pred_prob = target_probs_t[:, -1:]
        recovered_pred_prob = target_probs_t[:, :-1]
        # full softmax dist on original
        original_pred_prob_full = softmax_probs[:, -1:, :]
        # full softmax dist on recovered
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


def get_rcap_score(
    recovered_pred: tuple,
    debug: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute the final RCAP metric from recovery statistics.

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
    visual_noise_level: np.ndarray = local_heat_sum / overall_heat_sum[:, None]

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
        "visual_noise_level": visual_noise_level,
        "localization": recovered_pred_prob,
        'RCAP': rcap_score,
    }
