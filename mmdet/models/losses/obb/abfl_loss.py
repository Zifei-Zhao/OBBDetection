import torch
import torch.nn as nn

from mmdet.ops import convex_sort
from mmdet.core import bbox2type, get_bbox_areas
from mmdet.models.builder import LOSSES
from ..utils import weighted_loss
import math
from scipy.special import i0

import pdb


@weighted_loss
def abfl_loss(pred, target, asp_ratio=None, as_thr=1.3, kappa=3., alpha=0.66, eps=1e-6):
    # 0.8/0.31, 0.9/0.33/,  1/0.35,  2/0.52  
    #   3/0.66,   4/0.77,   5/0.87,  6/0.96, 6.5/1,  
    #  10/1.3,   20/1.8,   30/2.2,  50/2.9,  
    # 100/4.0,  200/5.7,  300/7.0, 500/9.0

    angle_diff = pred-target
    angle_diff_ = angle_diff + math.pi / 2

    radii = 1.0 - torch.exp(kappa*torch.cos(2.0*(angle_diff)))/(2.0*math.pi*i0(kappa)) / alpha
    radii_ = radii - torch.exp(kappa*torch.cos(2.0*(angle_diff_)))/(2.0*math.pi*i0(kappa)) / alpha
    loss_azimuth_vml = torch.where(asp_ratio.unsqueeze(1)>as_thr, radii, radii_).to(torch.float32)

    loss_l1 = (2.0*pred/math.pi).abs().clamp(1.0)

    loss_azimuth = torch.where(pred.abs()<=math.pi/2.0, loss_azimuth_vml, loss_l1)

    return loss_azimuth


@LOSSES.register_module()
class ABFLLoss(nn.Module):

    def __init__(self,
                 as_thr=1.1,
                 kappa=6.5,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(ABFLLoss, self).__init__()
        self.as_thr = as_thr
        self.kappa = kappa
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                asp_ratio,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * abfl_loss(
            pred,
            target,
            weight,
            asp_ratio=asp_ratio,
            as_thr=self.as_thr,
            kappa=self.kappa,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
