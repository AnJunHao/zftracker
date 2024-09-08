import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def local_mse_loss(preds, gts, reduction='mean'):
    """
    Args:
        preds (torch.Tensor): Predictions from the model.
        gts (torch.Tensor): Ground truth coordinates.
        reduction (str): Reduction method. Default: ``'mean'``.
    Returns:
        torch.Tensor: The loss.
    """
    mse = F.mse_loss(preds, gts, reduction='none')
    mask = (gts != 0).float()  # Only calculate loss for non-zero coordinates
    loss = mse * mask
    if reduction == 'mean':
        return loss.sum() / mask.sum()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    
def local_mae_loss(preds, gts, reduction='mean'):
    """
    Args:
        preds (torch.Tensor): Predictions from the model.
        gts (torch.Tensor): Ground truth coordinates.
        reduction (str): Reduction method. Default: ``'mean'``.
    Returns:
        torch.Tensor: The loss.
    """
    mae = F.l1_loss(preds, gts, reduction='none')
    mask = (gts != 0).float()  # Only calculate loss for non-zero coordinates
    loss = mae * mask
    if reduction == 'mean':
        return loss.sum() / mask.sum()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")

def local_bce_loss(preds, gts, reduction='mean'):
    """
    Args:
        preds (torch.Tensor): Predictions from the model.
        gts (torch.Tensor): Ground truth coordinates.
        reduction (str): Reduction method. Default: ``'mean'``.
    Returns:
        torch.Tensor: The loss.
    """
    clamped_gts = torch.clamp(gts, 0, 1)
    bce = F.binary_cross_entropy_with_logits(preds, clamped_gts, reduction='none')
    mask = (gts != -1).float()  # Only calculate loss for coordinates with ground truth value = 0 or 1
    loss = bce * mask
    if reduction == 'mean':
        return loss.sum() / mask.sum()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")

def local_focal_loss(
    preds: torch.Tensor,
    gts: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    apply_sigmoid: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    if apply_sigmoid:
        p = torch.sigmoid(preds)
    else:
        p = preds
    
    clamped_gts = torch.clamp(gts, 0, 1)
    ce_loss = F.binary_cross_entropy(p, clamped_gts, reduction="none")
    mask = (gts != -1).float()  # Only calculate loss for coordinates with ground truth value = 0 or 1
    
    p_t = p * gts + (1 - p) * (1 - gts) # p_t will be larger if the predicted p is closer to the target (0 or 1)
    loss = ce_loss * ((1 - p_t) ** gamma) * mask

    if alpha >= 0:
        alpha_t = alpha * gts + (1 - alpha) * (1 - gts)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

class HeatmapFocalLoss(nn.Module):
    """
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.
    """

    def __init__(self, alpha=2, beta=4, epsilon=1e-6, apply_sigmoid=False):
        """
        Args:
            alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``2``.
            beta (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``4``.
            epsilon (float): for numerical stability when dividing. Default: ``1e-6``.
            apply_sigmoid (bool): Whether to apply sigmoid on the input. Default: ``False``.
        """
        super(HeatmapFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.apply_sigmoid = apply_sigmoid

    def forward(self, pred, gt):
        """
        Args:
            pred (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            gt (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor with the reduction option applied.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        neg_weights = torch.pow(1 - gt, self.beta)

        pred = torch.clamp(pred, self.epsilon, 1. - self.epsilon)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                                   self.alpha) * neg_weights * neg_inds

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        pos_loss /= pos_inds.sum() + self.epsilon
        neg_loss /= neg_inds.sum() + self.epsilon

        return -(pos_loss + neg_loss)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
    apply_sigmoid=False
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    if apply_sigmoid:
        p = torch.sigmoid(inputs)
    else:
        p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def regression_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2,
    reduction: str = "mean",
    apply_sigmoid=False
) -> torch.Tensor:
    """
    Our custom implementation of focal loss for heatmap regression tasks.
    The objective is to directly generate the heatmap with keypoints represented as 2D gaussian distributions.
    To balance the loss between positive and negative examples
    and the loss between easy and hard examples, we introduce a modulating factor
    calculated from simple linear formula.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0, 1). A higher alpha puts more 
                emphasis on correctly classifying the positive class. Default: ``0.5``.
        gamma (float): Exponent of the modulating factor to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """

    if apply_sigmoid:
        prob = torch.sigmoid(inputs)
    else:
        prob = inputs
    ce_loss = F.binary_cross_entropy(prob, targets, reduction="none")
    k = alpha ** (1 / gamma) / 2
    factor = ((1 - k) * torch.abs(targets - prob) +
              k * targets + k * prob) ** gamma
    factor = factor.detach()
    loss = ce_loss * factor

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

def masked_regression_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2,
    reduction: str = "mean",
    apply_sigmoid=False
) -> torch.Tensor:
    """
    Our custom implementation of focal loss for heatmap regression tasks.
    The objective is to directly generate the heatmap with keypoints represented as 2D gaussian distributions.
    To balance the loss between positive and negative examples
    and the loss between easy and hard examples, we introduce a modulating factor
    calculated from simple linear formula.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0, 1). A higher alpha puts more 
                emphasis on correctly classifying the positive class. Default: ``0.5``.
        gamma (float): Exponent of the modulating factor to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """

    if apply_sigmoid:
        prob = torch.sigmoid(inputs)
    else:
        prob = inputs
    clamped_targets = torch.clamp(targets, 0, 1)
    ce_loss = F.binary_cross_entropy(prob, clamped_targets, reduction="none")
    # Only calculate loss for coordinates with ground truth value >= 0
    ce_loss *= (targets >= 0).float()
    k = alpha ** (1 / gamma) / 2
    factor = ((1 - k) * torch.abs(targets - prob) +
              k * targets + k * prob) ** gamma
    factor = factor.detach()
    loss = ce_loss * factor

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss