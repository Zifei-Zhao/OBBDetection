import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import (build_bbox_coder, distance2obb, force_fp32, multi_apply, multiclass_arb_nms,
                        mintheta_obb)
from mmdet.models.builder import HEADS, build_loss
from .obb_anchor_free_head import OBBAnchorFreeHead
import numpy as np
INF = 1e8


@HEADS.register_module()
class OBBFCOSOASLHead(OBBAnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.
    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        separate_theta (bool): If true, theta prediction is separated from
            bbox regression loss. Default: False.
        scale_theta (bool): If true, add scale to theta pred branch. Default: True.
        h_bbox_coder (dict): Config of horzional bbox coder, only used when separate_theta is True.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_theta (dict): Config of theta loss, only used when separate_theta is True.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
    Example:
        >>> self = OBBFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, theta_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 scale_theta=True,
                 square_like=False,
                 h_bbox_coder=dict(type='DistancePointBBoxCoder'),
                 lamda=0.0,
                 gating_factor=20.0,
                 shift_factor=0.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_refine=dict(type='IoULoss', loss_weight=1.0),
                 loss_theta=dict(
                     type='ABFLLoss', 
                     as_thr=1.30,
                     kappa=10.0, 
                     alpha=1.3,
                     loss_weight=0.2),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.scale_theta = scale_theta
        self.square_like = square_like
        self.lamda = lamda
        self.gating_factor = gating_factor
        self.shift_factor = shift_factor
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self.gradient_mul = 0.1
        super().__init__(
            num_classes,
            in_channels,
            bbox_type='obb',
            reg_dim=4,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_cls = build_loss(loss_cls)
        self.loss_theta = build_loss(loss_theta)
        self.h_bbox_coder = build_bbox_coder(h_bbox_coder)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_theta = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.scale_theta:
            self.scale_t = Scale(1.0)

        self.azinet_reg_base_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels//2,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.azinet_cls_base_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels//2,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)

        self.azinet_reg_refine_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels//2,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.azinet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.azinet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels//2,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.azinet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_theta, std=0.01)
        normal_init(self.azinet_reg_base_dconv, std=0.01)
        normal_init(self.azinet_cls_base_dconv, std=0.01)
        normal_init(self.azinet_reg_refine_dconv, std=0.01)
        normal_init(self.azinet_reg_refine, std=0.01)
        normal_init(self.azinet_cls_dconv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.azinet_cls, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                centernesses (list[Tensor]): Centerness for each scale level,
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.scales_refine, self.strides)

    def forward_single(self, x, scale, scale_refine, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.
        Returns:
            tuple: scores for each class, bbox predictions and centerness
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        theta_pred = self.conv_theta(reg_feat)
        if self.scale_theta:
            theta_pred = self.scale_t(theta_pred)

        dcn_b_offset = self.theta_dcn_base_offset(
            theta_pred).to(reg_feat.dtype).contiguous()
        reg_feat_l = self.azinet_reg_base_dconv(reg_feat, dcn_b_offset).clamp(min=0)
        cls_feat_l = self.azinet_cls_base_dconv(cls_feat, dcn_b_offset).clamp(min=0)

        dcn_offset = self.theta_dcn_offset(
            bbox_pred, 
            theta_pred, 
            self.gradient_mul,
            stride).to(reg_feat.dtype).contiguous()
        reg_feat_g = self.azinet_reg_refine_dconv(reg_feat, dcn_offset).clamp(min=0)
        reg_feat = torch.cat([reg_feat_l, reg_feat_g], dim=1)

        bbox_pred_refine = scale_refine(self.azinet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        cls_feat_g = self.azinet_cls_dconv(cls_feat, dcn_offset).clamp(min=0)
        cls_feat = torch.cat([cls_feat_l, cls_feat_g], dim=1)
        cls_score = self.azinet_cls(cls_feat)

        return cls_score, bbox_pred, bbox_pred_refine, theta_pred

    def theta_dcn_offset(self, bbox_pred, theta_pred, gradient_mul, stride):
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        l = bbox_pred_grad_mul[:, 0, :, :]
        t = bbox_pred_grad_mul[:, 1, :, :]
        r = bbox_pred_grad_mul[:, 2, :, :]
        b = bbox_pred_grad_mul[:, 3, :, :]

        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)

        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * t    # -t
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * l    # -l
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * t    # t -ftc
        # bbox_pred_grad_mul_offset[:, 3, :, :] = 0 * l/r   # 0 -ftc
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * t    # -t
        bbox_pred_grad_mul_offset[:, 5, :, :] = r           # r
        # bbox_pred_grad_mul_offset[:, 6, :, :] =  * t/b    # 0 ltc
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * l    # -l ltc
        # bbox_pred_grad_mul_offset[:, 8, :, :] = 0 * t/b   # 0 ctr_x
        # bbox_pred_grad_mul_offset[:, 9, :, :] = 0 * l/r   # 0 ctr_y
        # bbox_pred_grad_mul_offset[:, 10, :, :] = 0 * t/b  # 0 rtc
        bbox_pred_grad_mul_offset[:, 11, :, :] = r          # r rtc
        bbox_pred_grad_mul_offset[:, 12, :, :] = b          # b
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * l   # -l
        bbox_pred_grad_mul_offset[:, 14, :, :] = b          # b btc
        # bbox_pred_grad_mul_offset[:, 15, :, :] = 0 * l/r  # 0 btc
        bbox_pred_grad_mul_offset[:, 16, :, :] = b          # b
        bbox_pred_grad_mul_offset[:, 17, :, :] = r          # r
        
        # dcn_offset = bbox_pred_grad_mul_offset
        
        # theta_pred's shape: [N, 1, H, W]
        # print(f'shape: {theta_pred.shape}, {dcn_offset.shape}')
        theta_pred = theta_pred.permute(0, 2, 3, 1)
        bbox_pred_grad_mul_offset = bbox_pred_grad_mul_offset.permute(0, 2, 3, 1)
        # print(f'shape: {theta_pred.shape}, {dcn_offset.shape}')
        Cos, Sin = torch.cos(-theta_pred), torch.sin(-theta_pred)
        # print(f'shape: {Cos.shape}, {Sin.shape}')
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=-1).reshape(
                N, H, W, 2, 2)
        bbox_pred_grad_mul_offset = torch.matmul(
            bbox_pred_grad_mul_offset.reshape(N, H, W, 9, 2), 
            Matrix).reshape(N, H, W, 18).permute(0, 3, 1, 2)
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset


    def theta_dcn_base_offset(self, theta_pred):
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(theta_pred)
        N, C, H, W = theta_pred.size()

        theta_pred = theta_pred.permute(0, 2, 3, 1)
        dcn_base_offset = dcn_base_offset.permute(0, 2, 3, 1).expand(N, H, W, -1)
        Cos, Sin = torch.cos(-theta_pred), torch.sin(-theta_pred)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=-1).reshape(
            N, H, W, 2, 2)
        dcn_base_offset = torch.matmul(
            dcn_base_offset.reshape(N, H, W, 9, 2), 
            Matrix).reshape(N, H, W, 18).permute(0, 3, 1, 2)

        return dcn_base_offset

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine', 'theta_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,
             theta_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]   # bbox_pred = (N, 4*NP, H, W) => (N, H, W, 4*NP) => (NHWNP, 4)
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_theta_preds = [
            theta_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for theta_pred in theta_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_theta_preds = torch.cat(flatten_theta_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_theta_preds = flatten_theta_preds[pos_inds]
        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_bbox_targets, pos_theta_targets = pos_bbox_targets.split([4, 1], dim=1)
            pos_points = flatten_points[pos_inds]
            bbox_coder = self.h_bbox_coder
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_bbox_preds_refine = bbox_coder.decode(pos_points, pos_bbox_preds_refine)
            pos_decoded_bbox_targets = bbox_coder.decode(pos_points, pos_bbox_targets)

            pair_wise_ious = self.loss_bbox(
                pos_decoded_bbox_preds.detach(),
                pos_decoded_bbox_targets.detach(),
                reduction_override='none').clamp(min=1e-8)
            pair_wise_ious = pair_wise_ious.clone().detach()

            if self.square_like:
                W = pos_bbox_preds[:, 0] + pos_bbox_preds[:, 2]
                H = pos_bbox_preds[:, 1] + pos_bbox_preds[:, 3]
                W_regular = torch.where(W > H, W, H)
                H_regular = torch.where(W > H, H, W)
                asp_ratio = W_regular / H_regular
                pair_wise_azi = self.loss_theta(
                    pos_theta_preds.detach(), 
                    pos_theta_targets.detach(), 
                    asp_ratio=asp_ratio,
                    reduction_override='none').clamp(min=1e-8).squeeze(1)
            else:
                pair_wise_azi = self.loss_theta(
                    pos_theta_preds.detach(),
                    pos_theta_targets.detach(),
                    reduction_override='none').clamp(min=1e-8).squeeze(1)
            pair_wise_azi = pair_wise_azi.clone().detach() / self.loss_theta.loss_weight

            pair_wise_cls, _ = self.loss_cls(
                flatten_cls_scores[pos_inds].detach(),
                flatten_labels[pos_inds].detach(),
                reduction_override='none').min(1)

            pair_wise_cls = pair_wise_cls.clone().detach()
            # cost = (
            #     pair_wise_cls
            #     + 3.0 * pair_wise_ious
            #     + float(1e6) * (~pos_inds)
            # )

            loss_pred = torch.stack([pair_wise_cls, pair_wise_ious, pair_wise_azi], dim=1)
            loss_expe = torch.ones(loss_pred.shape).to(loss_pred.device)

            # cos ∈ (0.5774, 1]
            cos_sim_weight = (F.cosine_similarity(loss_pred, loss_expe, dim=1) \
                - self.lamda)/(1 - self.lamda) * \
                gating_func(
                    loss_pred.mean(-1), 
                    self.gating_factor).clamp(min=1e-8)

            pair_wise_ious_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine.detach(),
                pos_decoded_bbox_targets.detach(),
                reduction_override='none').clamp(min=1e-8)
            pair_wise_ious_refine = pair_wise_ious_refine.clone().detach()

            loss_pred_refine = torch.stack([pair_wise_cls, pair_wise_ious_refine, pair_wise_azi], dim=1)
            cos_sim_weight_refine = (F.cosine_similarity(loss_pred_refine, loss_expe, dim=1) \
                - self.lamda)/(1 - self.lamda) * \
                gating_func(
                    loss_pred.mean(-1), 
                    self.gating_factor).clamp(min=1e-8)

            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets,
                weight=cos_sim_weight,
                avg_factor=cos_sim_weight.sum())
            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_bbox_targets,
                weight=cos_sim_weight_refine,
                avg_factor=cos_sim_weight_refine.sum())
            if self.square_like:
                loss_theta = self.loss_theta(
                    pos_theta_preds, 
                    pos_theta_targets,
                    asp_ratio=asp_ratio,
                    avg_factor=num_pos)
            else:
                loss_theta = self.loss_theta(
                    pos_theta_preds, 
                    pos_theta_targets, 
                    avg_factor=num_pos)
            
            if self.square_like:
                del W, H, W_regular, H_regular, asp_ratio
            
            del pair_wise_ious, pair_wise_azi, pair_wise_cls, \
                loss_pred, loss_expe, cos_sim_weight, \
                pair_wise_ious_refine, loss_pred_refine, cos_sim_weight_refine
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            loss_theta = pos_theta_preds.sum() * 0

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_refine=loss_bbox_refine,
            loss_theta=loss_theta)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine', 'theta_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bbox_preds_refine,
                   theta_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            theta_pred_list = [
                theta_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']          # [1024, 1024, 3]
            scale_factor = img_metas[img_id]['scale_factor']    # [1., 1., 1., 1.]
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 theta_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           theta_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, theta_pred, points in zip(
                cls_scores, bbox_preds, theta_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            theta_pred = theta_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = torch.cat([bbox_pred, theta_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_arb_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            bbox_type='obb')
        return det_bboxes, det_labels

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
            in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level.
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets[:, :4] = bbox_targets[:, :4] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 5))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_thetas = torch.split(
            gt_bboxes, [2, 2, 1], dim=2)

        Cos, Sin = torch.cos(gt_thetas), torch.sin(gt_thetas)
        Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=-1).reshape(
            num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(Matrix, offset[..., None])
        offset = offset.squeeze(-1)

        W, H = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = W / 2 + offset_x
        right = W / 2 - offset_x
        top = H / 2 + offset_y
        bottom = H / 2 - offset_y
        # 堆叠，box_targets=(num_points,num_gts,4)
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        # if center_sampling is true, also in center bbox.
        # inside_gt_bbox_mask: (num_points,num_gts)
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(
                inside_center_bbox_mask, inside_gt_bbox_mask)

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        theta_targets = gt_thetas[range(num_points), min_area_inds]
        bbox_targets = torch.cat([bbox_targets, theta_targets], dim=1)
        return labels, bbox_targets

def gating_func(mloss, gating_factor=20.0, shift_factor = 0.5):
    re = 1.0 - 1.0 / (1.0 + (-1.0 * gating_factor * (mloss - shift_factor)).exp())
    return re