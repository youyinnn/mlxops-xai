import torch
from mlxops_utils import data_utils
import torch.nn.functional as F
import numpy as np


def batch_rcap(
    model,
    batch,
    saliency_func,
    saliency_func_kwargs,
    lower_bound=0.7,
    # recover_interval=0.05,
    recover_interval=0.3,
    dataloader_kwargs={},
):
    """
    Get and save rcap input
    """
    model = model.eval()
    rcap_scores = []

    images, targets = batch
    saliency_maps = saliency_func(
        model, images, targets, ** saliency_func_kwargs)
    saliency_maps = saliency_maps.to(images.device)

    recovered_pred = get_visulalization_and_localization_score(
        model, images, targets, saliency_maps,
        recover_interval=recover_interval, lower_bound=lower_bound)
    original_pred_score, recovered_pred_score, \
        original_pred_prob, recovered_pred_prob, \
        local_heat_mean, local_heat_sum, \
        overall_heat_mean, overall_heat_sum, \
        original_pred_prob_full, recovered_pred_prob_full, \
        recovered_imgs = recovered_pred

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

    rcap_get_visulalization_and_localization_score = (
        all_original_pred_score, all_recovered_pred_score,
        all_original_pred_prob, all_recovered_pred_prob,
        all_local_heat_mean, all_local_heat_sum,
        all_overall_heat_mean, all_overall_heat_sum,
        None
    )

    rcap_scores = get_rcap_score(
        rcap_get_visulalization_and_localization_score)

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


def get_recovered_image(img, saliency_map, lower_bound, recover_interval):
    """
    Get the multiple recovered image,
    This is referred to I_{p_k}
    """
    rs = []
    ury_f = saliency_map.flatten()
    v, _ = torch.sort(ury_f)
    rr = [round(rate, 2) for rate in np.arange(
        lower_bound, 0.999999, recover_interval)]
    q_idx = [int(v.shape[0] * rate) - 1 for rate in rr]

    """
    Different partitions
    """
    q = np.array([v[i].item() for i in q_idx])
    imgg = img
    local_heat_mean = []
    local_heat_sum = []
    q = np.flip(q)
    # if q[0] == 1:
    #     q = q[1:]
    upper_rate = 1
    imgg = torch.zeros_like(img)

    """
    Recover based on partitions
    """
    for i in range(len(q)):
        lower_rate = q[i] if q[i] < 1 else 0
        mc = (saliency_map > lower_rate) & (saliency_map <= upper_rate)
        local_heat_mean.append(
            saliency_map[(saliency_map > lower_rate) & (saliency_map <= 1)]
            .mean()
            .item()
        )
        local_heat_sum.append(
            saliency_map[(saliency_map > lower_rate) &
                         (saliency_map <= 1)].sum().item()
        )
        imgg = torch.where(mc, img, imgg)
        rs.append(imgg.reshape(1, *imgg.shape))
        upper_rate = lower_rate
    return torch.vstack(rs), local_heat_mean, local_heat_sum


def get_visulalization_and_localization_score(
    model, original_images, targets, saliency_maps,
    recover_interval, lower_bound, debug=False
):
    """
    RCAP core code, recover and predict
    """
    device = original_images.device
    n = original_images.shape[0]
    rss = []
    number_bin = None
    original_images = data_utils.denormm_i_t(original_images)
    local_heat_mean = []
    local_heat_sum = []
    recovered_imgs = []
    for i in range(n):
        img = original_images[i]
        saliency_map = saliency_maps[i]
        rs, lhm, lhs = get_recovered_image(
            img, saliency_map, recover_interval=recover_interval, lower_bound=lower_bound)
        recovered_imgs.append(rs.cpu().detach().numpy())
        local_heat_mean.append(lhm)
        local_heat_sum.append(lhs)
        number_bin = rs.shape[0] + 1
        rs = torch.vstack([rs, img.reshape(1, *img.shape)])
        if debug:
            ...
        rss.append(rs)
    rss = torch.vstack(rss).to(device)
    rss = data_utils.normm_i_t(rss)

    local_heat_mean = torch.tensor(local_heat_mean, device=device)
    local_heat_sum = torch.tensor(local_heat_sum, device=device)
    overall_heat_mean = saliency_maps.mean(dim=(1, 2))
    overall_heat_sum = saliency_maps.sum(dim=(1, 2))

    """
    Predict on recovered images
    """
    with torch.no_grad():
        prediction = model(rss)
        if targets is None:
            targets = torch.argmax(prediction, dim=1)
        prediction = prediction.reshape(n, number_bin, prediction.shape[1])
        pred_score = []
        for i, pp in enumerate(prediction.cpu().detach().numpy()):
            pred_score.extend(pp[:, targets[i]])
        pred_score = torch.tensor(
            pred_score, device=device).reshape(n, number_bin)
        original_pred_score = pred_score[:, -1:]
        recovered_pred_score = pred_score[:, :-1]
        sm = F.softmax(prediction, dim=2)
        pred_prob = []
        for i, smm in enumerate(sm.cpu().detach().numpy()):
            pred_prob.extend(smm[:, targets[i]])
        pred_prob = torch.tensor(
            pred_prob, device=device).reshape(n, number_bin)
        original_pred_prob = pred_prob[:, -1:]
        recovered_pred_prob = pred_prob[:, :-1]
        original_pred_prob_full = sm[:, -1:, :]
        recovered_pred_prob_full = sm[:, :-1, :]

    """
    Get all the scores: original score, recovered scores;
    Get all the saliency mean values;
    """
    return (
        original_pred_score.cpu().detach().numpy(),
        recovered_pred_score.cpu().detach().numpy(),
        original_pred_prob.cpu().detach().numpy(),
        recovered_pred_prob.cpu().detach().numpy(),
        local_heat_mean.cpu().detach().numpy(),
        local_heat_sum.cpu().detach().numpy(),
        overall_heat_mean.cpu().detach().numpy(),
        overall_heat_sum.cpu().detach().numpy(),
        original_pred_prob_full.cpu().detach().numpy(),
        recovered_pred_prob_full.cpu().detach().numpy(),
        np.array(recovered_imgs),
    )


def get_rcap_score(recovered_pred, debug=False):
    """
    Calculate RCAP
    local_heat_mean refers to mean of the M_{p_k}
    recovered_pred_score refers to mean of the f(I_{p_k})
    recovered_pred_prob refers to mean of the sigma(f(I_{p_k}))
    """
    original_pred_score, recovered_pred_score, \
        original_pred_prob, recovered_pred_prob, \
        local_heat_mean, local_heat_sum, \
        overall_heat_mean, overall_heat_sum, \
        recovered_imgs = recovered_pred

    visual_noise_level = (
        local_heat_sum / overall_heat_sum.repeat(local_heat_mean.shape[-1]).reshape(*recovered_pred_score.shape))

    score_lhm_hr_rpp = \
        (local_heat_mean * visual_noise_level * recovered_pred_score).mean(-1)
    score_lhm_rpp = \
        (local_heat_mean * recovered_pred_score).mean(-1)

    prob_lhm_hr_rpp = \
        (local_heat_mean * visual_noise_level * recovered_pred_prob).mean(-1)
    prob_hr_rpp = \
        (visual_noise_level * recovered_pred_prob).mean(-1)

    prob_hr_rpp2 = \
        (visual_noise_level + recovered_pred_prob).mean(-1)
    # prob_lhm_rpp = \
    #     (local_heat_mean * recovered_pred_prob).mean(-1)

    if debug:
        # print('1-- local heat mean')
        # print(local_heat_mean)
        # print(local_heat_sum)

        # print('2-- overall heat sum')
        # print(overall_heat_sum)

        print('\r\n3-- visual_noise_level = local_heat_sum / overall_heat_sum')
        print(visual_noise_level)

        # print('\r\n4-- removed pred score')
        # print(recovered_pred_score)
        print('\r\n4-- removed pred prob')
        print(recovered_pred_prob, np.mean(recovered_pred_prob, axis=1))

        # print('\r\n5-- pre rs 1: local_heat_mean * visual_noise_level * recovered_pred_score')
        # print(local_heat_mean * visual_noise_level * recovered_pred_score)
        # print('\r\n5-- pre rs 2: local_heat_mean * recovered_pred_score')
        # print(local_heat_mean * recovered_pred_score)

        print('\r\n6-- rs')
        # print('local_heat_mean * visual_noise_level * recovered_pred_score', score_lhm_hr_rpp)
        # print('local_heat_mean * recovered_pred_score', score_lhm_rpp)
        print('local_heat_mean * visual_noise_level * recovered_pred_prob',
              prob_lhm_hr_rpp)
        print('visual_noise_level * recovered_pred_prob', prob_hr_rpp)
        print('visual_noise_level + recovered_pred_prob', prob_hr_rpp2)
        # print('local_heat_mean * recovered_pred_prob', prob_lhm_rpp)

    # eval_rs['Score: M1'] = score_lhm_hr_rpp
    # eval_rs['Score: M2'] = score_lhm_rpp
    # eval_rs['Prob: M1'] = prob_lhm_hr_rpp

    return {
        "visual_noise_level": visual_noise_level,
        "localization": recovered_pred_prob,
        'RCAP': prob_hr_rpp,
    }
