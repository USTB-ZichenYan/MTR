# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch 


def nll_loss_gmm_direct(pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                        timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
    """
    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi 

    Args:
        pred_scores (batch_size, num_modes):
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    # Validate the dimensionality of pred_trajs based on the use_square_gmm flag
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3 
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_scores.shape[0]

    # Determine the index of the nearest mode
    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        # Calculate the distance between predicted trajectories and ground truth trajectories
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1) 
        # Adjust the distance calculation by the validity mask of the ground truth
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1) 

        # Find the index of the nearest mode
        nearest_mode_idxs = distance.argmin(dim=-1)
        # # 计算各模式下轨迹与真实轨迹的距离，选择距离最小的模式
        # distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)  # (B, M, T)
        # distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)  # 仅有效时间步求和，形状：(B, M)
        # nearest_mode_idxs = distance.argmin(dim=-1)  # 每个样本的最优模式索引，形状：(B,)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size,)

    # Extract the nearest trajectories and calculate the residuals
    nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    # Calculate the standard deviation and correlation coefficient based on the GMM model used
    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        std1 = std2 = torch.exp(log_std1)   # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
        log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    # Prepare the validity mask for loss calculation
    gt_valid_mask = gt_valid_mask.type_as(pred_scores)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # Calculate the GMM loss components
    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * ((dx**2) / (std1**2) + (dy**2) / (std2**2) - 2 * rho * dx * dy / (std1 * std2))  # (batch_size, num_timestamps)

    # Calculate the final regression loss
    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    # Return the calculated loss and the index of the nearest mode
    return reg_loss, nearest_mode_idxs