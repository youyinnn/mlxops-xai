from torchvision.transforms import v2

import math
import torch
import torch.nn.functional as F
import time

import numpy as np

from mlxops_utils import data_utils


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def get_gradients(
    model,
    images: torch.Tensor,
    targets: torch.Tensor,
    loss=False,
    task="classification",
    pred_fn=None,
    outputs_agg_fn=None,
):
    """
    Based function for gradient calculation
    """
    original_model_mode_on_train = model.training
    model.eval()

    images = images.clone().detach()
    images = images.requires_grad_(True)
    if task == "detection":
        # model = model.train()
        # outputs = model(images, targets)
        pass
    else:
        outputs = model(images) if pred_fn is None else pred_fn(model, images)

    if task == "detection":
        assert targets is not None
        assert outputs_agg_fn is not None
        # outputs = model(images, targets)
        # agg = sum(outputs.values())

        outputs = pred_fn(model, images)
        agg = outputs_agg_fn(outputs)

    elif task == "classification" or task is None:
        if targets is None:
            # targets = (outputs.data.max(1, keepdim=True)[1]).flatten()
            targets = torch.argmax(outputs, dim=1)
        if loss:
            outputs = torch.log_softmax(outputs, 1)
            agg = F.nll_loss(outputs, targets, reduction="sum")
        else:
            agg = -1.0 * F.nll_loss(outputs, targets, reduction="sum")

    model.zero_grad()
    # Gradients w.r.t. input and features
    gradients = torch.autograd.grad(
        # outputs=agg, inputs=images, retain_graph=False)[0]
        outputs=agg,
        inputs=images,
        retain_graph=False,
        # outputs=outputs.max(), inputs=images, retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]
    del agg, outputs, images
    if original_model_mode_on_train:
        model = model.train()
    return gradients


def vanilla_gradient(model, images, targets, loss=False, **kwargs):

    model.eval()
    input_grad = get_gradients(model, images, targets, loss=loss, **kwargs)
    # input_grad = torch.minimum(input_grad, torch.zeros_like(input_grad))
    return data_utils.min_max_normalize(input_grad.abs().sum(1))


def smooth_grad(
    model,
    images,
    targets,
    loss=False,
    num_samples=10,
    std_spread=0.15,
    ifabs=False,
    ifsquare=False,
    **kwargs
):
    """
    https://github.com/idiap/fullgrad-saliency/blob/master/saliency/smoothgrad.py
    """
    model.eval()
    std_dev = std_spread * (images.max().item() - images.min().item())

    cam = torch.zeros_like(images).to(images.device)
    for i in range(num_samples):
        noise_p = torch.normal(
            mean=torch.zeros_like(images).to(images.device), std=std_dev
        )
        mixed = images + noise_p
        ng = get_gradients(model, mixed, targets, loss=loss, **kwargs)
        if ifabs:
            ng = ng.abs()

        if ifsquare:
            ng *= ng

        cam += (ng) / num_samples

    return data_utils.min_max_normalize(cam.abs().sum(1))


def integrated_gradients_impl(
    model,
    images,
    targets,
    baseline,
    steps=50,
    batch=10,
    direction="both",
    ifabs=False,
    **kwargs
):
    """
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
    """
    if baseline is None:
        baseline = torch.zeros_like(images)

    n, _, w, h = images.shape
    true_steps = steps + 1
    scaled_inputs = torch.vstack(
        [
            baseline + (float(i) / steps) * (images - baseline)
            for i in range(0, true_steps)
        ]
    ).to(images.device)

    scaled_inputs = scaled_inputs.reshape(true_steps, n, 3, w, h)
    scaled_inputs = torch.transpose(scaled_inputs, dim0=1, dim1=0)
    all_mean_gradients = []
    targets = targets.repeat(true_steps, 1).T.flatten()
    scaled_inputs = scaled_inputs.reshape(n * true_steps, _, w, h)

    n_scaled_inputs = scaled_inputs.shape[0]
    s = 0
    gradient_list = []
    while s < n_scaled_inputs:
        e = s + batch if s + batch < n_scaled_inputs else n_scaled_inputs
        gradients = get_gradients(
            model, scaled_inputs[s:e], targets[s:e], loss=True, **kwargs
        )
        gradient_list.extend(gradients)
        s += batch
    gradients = torch.stack(gradient_list)
    # gradients.shape = (n, steps, channel, w, h)
    gradients = gradients.reshape(n, true_steps, _, w, h)

    for gradients_one in gradients:
        # gradients_one.shape = (steps, channel, w, h)
        if ifabs or direction == "abs":
            gradients_one = gradients_one.abs()
        elif direction == "both":
            pass
        elif direction == "positive":
            gradients_one = torch.clamp(gradients_one, min=0)
        elif direction == "negative":
            gradients_one = torch.clamp(gradients_one, max=0)

        # gradients_one.mean(dim=0, keepdim=True).shape
        #                                           = (1, channel, w, h)
        all_mean_gradients.append(gradients_one.mean(dim=0, keepdim=True))

    # all_mean_gradients.shape = (n, channel, w, h)
    all_mean_gradients = torch.vstack(all_mean_gradients)

    return (images - baseline) * all_mean_gradients


def integrated_gradients(
    model,
    images,
    targets,
    direction="both",
    aggregation=None,
    th=0,
    steps=10,
    trials=3,
    ifabs=False,
):
    # all_intgrads = None
    all_intgrads = None if aggregation is None else []
    model.eval()
    d = torch.distributions.uniform.Uniform(
        images.min().item(), images.max().item())
    baselines = d.sample(sample_shape=(
        trials, *images.shape)).to(images.device)

    if targets is None:
        targets = torch.argmax(model(images), dim=1)

    for i in range(trials):
        intgrads = integrated_gradients_impl(
            model,
            images,
            targets,
            direction=direction,
            baseline=baselines[i],
            steps=steps - 1,
            ifabs=ifabs,
        )
        if aggregation is None:
            if all_intgrads is None:
                all_intgrads = intgrads
            else:
                all_intgrads += intgrads
        else:
            all_intgrads.append(intgrads)

    if aggregation is None:
        return data_utils.min_max_normalize(
            (all_intgrads / trials).sum(1).abs().detach()
        )
    else:
        return aggregate_saliency_maps(all_intgrads, ifabs, th, aggregation)


def guided_back_propagation(
    model,
    images,
    targets=None,
    ifabs=False,
    aggregation="sum",
    direction="positive",
    iteration=1,
    **kwargs
):
    """
    https://github.com/vectorgrp/coderskitchen-xai/blob/main/part2_Torch_guided_backprop.ipynb
    https://www.coderskitchen.com/guided-backpropagation-with-pytorch-and-tensorflow/
    """
    model.eval()
    handles = []
    try:
        for i, module in enumerate(model.modules()):
            if isinstance(module, torch.nn.ReLU):
                setattr(module, "inplace", False)
                # original guidance
                if direction == "positive" and not ifabs:
                    # print(1)
                    handles.append(
                        module.register_full_backward_hook(
                            lambda m, grad_in, grad_out: (
                                torch.clamp(grad_in[0], min=0.0),
                            )
                        )
                    )
                if direction == "negative" and not ifabs:
                    # print(2)
                    handles.append(
                        module.register_full_backward_hook(
                            lambda m, grad_in, grad_out: (
                                torch.clamp(grad_in[0], max=0.0),
                            )
                        )
                    )
                if direction == "both" and not ifabs:
                    # print(3)
                    # same as vanilla graident
                    handles.append(
                        module.register_full_backward_hook(
                            lambda m, grad_in, grad_out: (grad_in[0],)
                        )
                    )
                if direction == "abs" or ifabs:
                    # print(4)
                    # this will ruin the propagations
                    handles.append(
                        module.register_full_backward_hook(
                            lambda m, grad_in, grad_out: (grad_in[0].abs(),)
                        )
                    )

        images = images.clone().detach()
        images.requires_grad_(True)

        grads_l = []
        for i in range(iteration):
            grads = get_gradients(model, images, targets, **kwargs)
            # if ifabs:
            #     grads_l.append(grads.abs())
            # else:
            grads_l.append(grads)

        grads = torch.stack(grads_l).sum(0)
        model.zero_grad()
    finally:
        for h in handles:
            h.remove()
    if aggregation == "sum":
        return data_utils.min_max_normalize(grads.abs().sum(1))
    else:
        return data_utils.min_max_normalize(grads.abs().mean(1))


def aggregate_saliency_maps(cams, ifabs=False, th=None, aggregation="mean"):
    """
    Aggregate the saliency maps by leveraging the magnitude of the gradient

    cams: keeps the saliency maps from multiple noise introductions
    th: is the p parameter in the paper

    """
    cams_k = torch.stack(cams, dim=4).detach()
    n, c, w, h, num_samples = cams_k.shape
    if aggregation == "variance":
        return data_utils.min_max_normalize(cams_k.var(dim=4).sum(1))

    final_cam = torch.zeros(size=(n, c, w, h)).to(cams_k.device)
    # AbsoluteGrad
    if aggregation == "mean":
        for cam in cams:
            if ifabs:
                final_cam += cam.abs()
            else:
                final_cam += cam

    if aggregation == "guided":
        if th is None:
            th = 0.7

        if th > 0:
            # equation 5 and 6
            var = cams_k.var(4)

            # this keeps the guide for differenct samples
            q = torch.quantile(var.reshape(n, c, w * h),
                               th, dim=2, keepdim=True)
            vs = torch.where(
                var > q.reshape(n, c, 1, 1).repeat(1, 1, w, h), 1, var
            ).reshape(n, c, w, h)
        else:
            vs = torch.full(size=(n, 3, w, h), fill_value=1,
                            device=cams_k.device)

        for i, cam in enumerate(cams):
            if ifabs:
                final_cam += cam.abs() * vs
            else:
                final_cam += cam * vs

    # line 16 of Algorithm 1
    return data_utils.min_max_normalize(final_cam.abs().sum(1))


def guided_absolute_grad(
    model,
    images,
    targets,
    loss=False,
    num_samples=10,
    th=0,
    ifabs=True,
    ifsquare=False,
    aggregation="guided",
    std_spread=0.15,
    **kwargs
):

    model.eval()
    cams = []
    std_dev = std_spread * (images.max().item() - images.min().item())

    # similar to SmoothGrad
    for i in range(num_samples):
        noise_p = torch.normal(
            mean=torch.zeros_like(images).to(images.device), std=std_dev
        )
        mixed = images + noise_p
        ng = get_gradients(model, mixed, targets, loss=loss, **kwargs)

        if ifsquare:
            ng *= ng
        cams.append(ng)

    # aggregate the saliency maps by leveraging the magnitude of the gradient
    return aggregate_saliency_maps(cams, ifabs, th, aggregation)


def l1_distance(x1, x2):
    return torch.abs(x1 - x2).sum()


def translate_x_to_alpha(x, x_input, x_baseline):
    return torch.where(
        x_input - x_baseline != 0, (x - x_baseline) /
        (x_input - x_baseline), torch.nan
    )


def translate_alpha_to_x(alpha, x_input, x_baseline):
    assert 0 <= alpha <= 1.0
    return x_baseline + (x_input - x_baseline) * alpha


def guided_ig_impl(
    model,
    images: torch.Tensor,
    targets,
    x_baseline=None,
    loss=False,
    steps=10,
    fraction=0.25,
    ifabs=False,
    direction="both",
    max_dist=0.02,
    debug=False,
    plot=False,
    **kwargs
):
    """
    https://github.com/PAIR-code/saliency/blob/master/saliency/core/guided_ig.py
    """

    # A very small number for comparing floating point values.
    EPSILON = 1e-9
    if x_baseline is None:
        x_baseline = torch.zeros_like(images)
    x = x_baseline.clone()
    l1_total = l1_distance(images, x_baseline)
    # attr = torch.zeros_like(images)
    attr = []

    total_diff = images - x_baseline
    if torch.abs(total_diff).sum() == 0:
        return torch.zeros_like(images)
    ss = [0, 0, 0, 0, 0]
    # Iterate through every step.
    for step in range(steps):
        st0 = time.time()
        # Calculate gradients and make a copy.
        grad_actual = get_gradients(
            model, x[None, :, :, :], targets, loss=loss, **kwargs
        )[0]

        ss[0] += time.time() - st0

        st1 = time.time()
        grad = grad_actual.clone()
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = translate_alpha_to_x(alpha_min, images, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, images, x_baseline)
        l1_target = l1_total * (1 - (step + 1) / steps)
        ss[1] += time.time() - st1

        if debug and plot:
            print("===")
        gamma = torch.inf

        st2 = time.time()
        while gamma > 1.0:
            x_old = x.clone()
            x_alpha = translate_x_to_alpha(x, images, x_baseline)
            x_alpha[torch.isnan(x_alpha)] = alpha_max
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            l1_current = l1_distance(x, images)
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual
                break
            grad[x == x_max] = torch.inf

            threshold = torch.quantile(
                torch.abs(grad), fraction, interpolation="lower")
            s = torch.logical_and(
                torch.abs(grad) <= threshold, grad != torch.inf)

            l1_s = (torch.abs(x - x_max) * s).sum()

            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                break
                # gramma = torch.inf

            if gamma > 1.0:
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]

            rs = (x - x_old) * grad_actual

            if ifabs or direction == "abs":
                rs = rs.abs()
            elif direction == "both":
                pass
            elif direction == "positive":
                rs = torch.clamp(rs, min=0)
            elif direction == "negative":
                rs = torch.clamp(rs, max=0)

            # attr += rs
            attr.append(rs)

        ss[2] += time.time() - st2

    if debug:
        print(ss)
    return attr


def guided_ig(
    model,
    images,
    targets,
    loss=False,
    num_samples=5,
    direction="both",
    fraction=0.25,
    max_dist=0.1,
    debug=False,
    ifabs=False,
    aggregation="mean",
    th=0,
    **kwargs
):

    if targets is None:
        targets = torch.argmax(model(images), dim=1)

    us = []
    for i, image in enumerate(images):
        # can't do batch, do it one by one
        u = guided_ig_impl(
            model,
            image,
            targets[None, i],
            None,
            loss=loss,
            steps=num_samples,
            fraction=fraction,
            direction=direction,
            max_dist=max_dist,
            debug=debug,
            ifabs=ifabs,
            **kwargs
        )

        us.append([uu[None, :, :, :] for uu in u])

    uss = [
        aggregate_saliency_maps(uu, ifabs=ifabs, th=th,
                                aggregation=aggregation)
        for uu in us
    ]
    final = torch.stack(uss)
    n, _, w, h = final.shape
    return final.reshape(n, w, h)
    # return data_utils.min_max_normalize(us.abs().sum(1))


def torch_gaussian_blur(image, sigma, radius):
    blurrer = v2.GaussianBlur(2 * radius + 1, sigma=sigma)
    return blurrer(image)


def blur_integrated_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    num_samples=30,
    radius=20,
    loss=False,
    max_sigma=50,
    grad_step=0.01,
    aggregation=None,
    th=0,
    ifabs=False,
    direction=None,
    ifsqrt=False,
    **kwargs
):
    """
    https://github.com/PAIR-code/saliency/blob/master/saliency/core/blur_ig.py
    """
    if ifsqrt:
        sigmas = [
            math.sqrt(float(i) * max_sigma / float(num_samples))
            for i in range(0, num_samples + 1)
        ]
    else:
        sigmas = [
            float(i) * max_sigma / float(num_samples) for i in range(0, num_samples + 1)
        ]
    step_vector_diff = torch.tensor(
        [sigmas[i + 1] - sigmas[i] for i in range(0, num_samples)], device=images.device
    )

    total_gradients = (
        torch.zeros(size=images.shape, device=images.device)
        if aggregation is None
        else []
    )

    et1 = 0
    et2 = 0
    for i in range(num_samples):
        st1 = time.time()
        if sigmas[i] == 0:
            x_step = images.detach().clone()
        else:
            x_step = torch_gaussian_blur(images, sigmas[i], radius)
        gaussian_gradient = (
            torch_gaussian_blur(images, sigmas[i] + grad_step, radius) - x_step
        ) / grad_step

        et1 += time.time() - st1

        st2 = time.time()
        ng = get_gradients(
            model,
            torch.tensor(x_step, device=images.device),
            targets,
            loss=loss,
            **kwargs
        )

        if ifabs:
            ng = ng.abs()
        elif direction in ["both", None]:
            pass
        elif direction == "positive":
            ng = torch.clamp(ng, min=0)
        elif direction == "negative":
            ng = torch.clamp(ng, max=0).abs()

        tmp = step_vector_diff[i] * (gaussian_gradient * ng)

        if aggregation is None:
            total_gradients += tmp
        else:
            total_gradients.append(tmp)
        et2 += time.time() - st2

    if aggregation is None:
        return data_utils.min_max_normalize(total_gradients.abs().sum(1))
    else:
        return aggregate_saliency_maps(total_gradients, ifabs, th, aggregation)
