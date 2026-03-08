import torch
import math
from torch import nn
from layers import *
from detect import *
import torch.nn.functional as F
from utils.utils_bbox import encode

class IoU_Loss(nn.Module):
    """
    支持: IoU, CIoU, SIoU, MPDIoU
    输入格式: [N, 4] -> (x1, y1, x2, y2)
    """
    def __init__(
        self, 
        loss_type: str = 'ciou', 
        eps: float = 1e-7
        ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.eps = eps

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        p1_x1, p1_y1, p1_x2, p1_y2 = predict.chunk(4, -1)
        t2_x1, t2_y1, t2_x2, t2_y2 = target.chunk(4, -1)
        
        w1, h1 = (p1_x2 - p1_x1).clamp(0), (p1_y2 - p1_y1).clamp(0)
        w2, h2 = (t2_x2 - t2_x1).clamp(0), (t2_y2 - t2_y1).clamp(0)

        # 计算 IoU
        inter = (torch.min(p1_x2, t2_x2) - torch.max(p1_x1, t2_x1)).clamp(0) * \
                (torch.min(p1_y2, t2_y2) - torch.max(p1_y1, t2_y1)).clamp(0)
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        # 根据类型计算惩罚项 
        if self.loss_type == 'iou':
            return 1.0 - iou

        cw = torch.max(p1_x2, t2_x2) - torch.min(p1_x1, t2_x1)
        ch = torch.max(p1_y2, t2_y2) - torch.min(p1_y1, t2_y1)

        if self.loss_type == 'ciou':
            c2 = cw**2 + ch**2 + self.eps  # 对角线平方
            rho2 = ((p1_x1 + p1_x2 - t2_x1 - t2_x2)**2 + (p1_y1 + p1_y2 - t2_y1 - t2_y2)**2) / 4
            v = (4 / math.pi**2) * (torch.atan(w2 / (h2 + self.eps)) - torch.atan(w1 / (h1 + self.eps)))**2
            with torch.no_grad():
                alpha = v / (v - iou + (1.0 + self.eps))
            return 1.0 - iou + (rho2 / c2 + v * alpha)

        elif self.loss_type == 'siou':
            s_cw = (p1_x1 + p1_x2 - t2_x1 - t2_x2) / 2
            s_ch = (p1_y1 + p1_y2 - t2_y1 - t2_y2) / 2
            sigma = torch.pow(s_cw**2 + s_ch**2, 0.5) + self.eps
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            # 角度惩罚
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            # 距离惩罚
            rho_x = (s_cw / (cw + self.eps))**2
            rho_y = (s_ch / (ch + self.eps))**2
            gamma = 2 - angle_cost
            distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
            # 形状惩罚
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            return 1.0 - iou + (shape_cost + distance_cost) * 0.5

        elif self.loss_type == 'mpdiou':
            # 直接计算左上角点对和右下角点对的距离
            d1 = (p1_x1 - t2_x1)**2 + (p1_y1 - t2_y1)**2
            d2 = (p1_x2 - t2_x2)**2 + (p1_y2 - t2_y2)**2
            # 使用最小外接矩形的对角线作为分母对齐
            c2 = cw**2 + ch**2 + self.eps
            return 1.0 - iou + d1/c2 + d2/c2

class TAL(nn.Module):
    def __init__(self, num_cls=80, topk=13, alpha=1.0, beta=6.0):
        super().__init__()
        self.topk = topk
        self.num_cls = num_cls
        self.alpha = alpha
        self.beta = beta

    def get_scores(self, predict_cls, target_cls, num_box, B, device):
        idx = torch.arange(B, device=device).view(-1, 1).repeat(1, num_box)
        return predict_cls[idx, :, target_cls.squeeze(-1).long()]
        # 输出 [B, num_gt, 8400]，含义为8400个预测框相较于num_gt个真实框分类上的得分
    
    def get_iou(self, predict_box, target_box):
        """计算 [B, 8400, 4] 和 [B, num_gt, 4] 之间的成对 IoU"""
        predict_box = predict_box.unsqueeze(1) # [B, 1, 8400, 4]
        target_box = target_box.unsqueeze(2) # [B, num_gt, 1, 4]，以下利用广播机制
        
        inter_x1 = torch.max(predict_box[..., 0], target_box[..., 0])
        inter_y1 = torch.max(predict_box[..., 1], target_box[..., 1])
        inter_x2 = torch.min(predict_box[..., 2], target_box[..., 2])
        inter_y2 = torch.min(predict_box[..., 3], target_box[..., 3])
        
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        a1 = (predict_box[..., 2] - predict_box[..., 0]) * (predict_box[..., 3] - predict_box[..., 1])
        a2 = (target_box[..., 2] - target_box[..., 0]) * (target_box[..., 3] - target_box[..., 1])
        return inter / (a1 + a2 - inter + 1e-7) 
        # 输出 [B, num_gt, 8400]，含义为8400个预测框相较于num_gt个真实框的iou

    def select_in(self, predict_box, target_box):
        pd_centers = (predict_box[:, :, :2] + predict_box[:, :, 2:]) / 2 
        gt_x1y1, gt_x2y2 = target_box.chunk(2, -1)
        
        lt = pd_centers.unsqueeze(1) - gt_x1y1.unsqueeze(2)
        rb = gt_x2y2.unsqueeze(2) - pd_centers.unsqueeze(1)
        
        is_in = torch.cat([lt, rb], dim=-1).min(-1).values > 1e-9
        return is_in.float()

    def select_topk(self, scores_in):
        _, topk_indices = torch.topk(scores_in, self.topk, dim=-1)
        mask = torch.zeros_like(scores_in)
        # 使用 scatter_ 将 topk 位置设为 1
        mask.scatter_(-1, topk_indices, 1.0)
        return mask
        # 输出[B, num_gt, 8400]，将 topk 位置设为 1，其余位置为 0，为筛选掩码
    
    def resolve_conflicts(self, mask, metrics):
        mask_sum = mask.sum(1) 
        if (mask_sum > 1).any():
            max_indices = metrics.argmax(1) 
            num_gt = metrics.shape[1]
            one_hot_mask = F.one_hot(max_indices, num_gt).to(metrics.dtype)
            mask = one_hot_mask.permute(0, 2, 1) * mask
            
        fg_mask = mask.sum(1) > 0 
        target_gt_idx = mask.argmax(1)
        return fg_mask, target_gt_idx

    @torch.no_grad()
    def forward(self, predict_cls, predict_box, target_cls, target_box, target_mask):
        """
        predict_cls: [B, 8400, nc]
        predict_box: [B, 8400, 4]
        target_cls:  [B, num_gt, 1]
        target_box:  [B, num_gt, 4]
        target_mask: [B, num_gt, 1]（真实框掩码，为了batch形状占位的空gt框数值为0）
        """
        B = predict_cls.shape[0]
        n_max = predict_cls.shape[1]
        num_gt = target_box.shape[1]
        device = predict_cls.device

        iou = self.get_iou(predict_box, target_box) 
        box_scores = self.get_scores(predict_cls, target_cls, num_gt, B, device)
        scores = box_scores.pow(self.alpha) * iou.pow(self.beta) # [B, num_gt, 8400]，含义为8400个预测框相较于num_gt个真实框的加权总得分

        is_in = self.select_in(predict_box, target_box)  # 预测框中心点要在真实框内才参与 topk
        mask = self.select_topk(scores * is_in) * target_mask 
        
        # 处理一个预测框被分配给多个 gt 的情况
        # target_gt_idx[B, 8400]标记所有框对应类别 id ，fg_mask[B, 8400]将所有正样本位置标记为 1
        fg_mask, target_gt_idx = self.resolve_conflicts(mask, scores)

        b_idx = torch.arange(B, device=device).unsqueeze(1) 
        a_idx = torch.arange(n_max, device=device).unsqueeze(0)

        # 利用 target_gt_idx 从 target_cls 里抠出正确的类别
        target_labels = target_cls[b_idx, target_gt_idx].squeeze(-1) # [B, 8400]
        target_labels[~fg_mask] = self.num_cls # 将负样本标记为背景 id，形状为 [B, 8400]，意为8400个预测框的类别 id (含背景)

        target_bboxes = target_box[b_idx, target_gt_idx] # [B, 8400, 4]，意为8400个预测框的最终匹配到的 gt 框的四条边

        matched_iou = iou[b_idx, target_gt_idx, a_idx] # [B, 8400]，意为8400个预测框最终对应类别的 iou

        # 将预测框最终分到的类别处设为对应的 IoU，其余类别为 0（软标签）
        target_scores = torch.zeros_like(predict_cls)
        target_scores.scatter_(
            2, 
            target_labels.unsqueeze(-1).long().clamp(0, self.num_cls - 1), 
            matched_iou.unsqueeze(-1).to(target_scores.dtype)
        )
        
        # 负样本分数归 0，避免逻辑残留
        target_scores *= fg_mask.unsqueeze(-1)

        return target_bboxes, target_scores, fg_mask

class BCE_Loss(nn.Module):
    """
    YOLOv12 分类损失模块
    采用带有 Logits 的二元交叉熵，支持 TAL 产生的软标签（Soft Label）
    """
    def __init__(self):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, cls_outs, target_scores):
        """
        - cls_outs: Detect层出来的分类原始值 [B, nc, 8400]
        - target_scores: TAL算出的软标签 [B, 8400, nc]
        """
        cls_outs = cls_outs.permute(0, 2, 1)

        # 8400个点中，分别的 nc 个类别的损失
        loss = self.loss_fcn(cls_outs, target_scores)
        return loss.sum()

        
class DFL_Loss(nn.Module):
    def __init__(
        self, 
        reg_max: int = 16
        ):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target_ltrb):
        """
        pred_dist:   正样本的原始回归分布 [N_pos, 64] (4条边 * 16)
        target_ltrb: 正样本对应的真实距离 [N_pos, 4] (ltrb 格式)
        """
        target_ltrb = target_ltrb.clamp(0, self.reg_max - 1.01) # 防越界
        
        # 找到 gt 坐标左右两侧的整数索引，并计算线性插值的权重
        left_idx = target_ltrb.long()
        right_idx = left_idx + 1
        weight_right = target_ltrb - left_idx
        weight_left = 1.0 - weight_right
        
        pred_dist = pred_dist.view(-1, self.reg_max) # [N*4, 16]，代表每条边的 16 个概率分布
        
        # 交叉熵
        loss_left = F.cross_entropy(pred_dist, left_idx.view(-1), reduction='none')
        loss_right = F.cross_entropy(pred_dist, right_idx.view(-1), reduction='none')
        loss = loss_left.view(target_ltrb.shape) * weight_left + \
               loss_right.view(target_ltrb.shape) * weight_right
        return loss.sum()
    
class Yolo12_Loss(nn.Module):
    def __init__(self, num_cls: int = 80, reg_max: int = 16):
        super().__init__()
        self.nc = num_cls
        self.reg_max = reg_max

        self.iou_loss = IoU_Loss(loss_type="ciou")
        self.tal = TAL(num_cls=num_cls, topk=13)
        self.bce_loss = BCE_Loss()
        self.dfl_loss = DFL_Loss(reg_max=reg_max)

        self.hyp_box = 7.5
        self.hyp_cls = 0.5
        self.hyp_dfl = 1.5
        self.mean_fg_scores = 0.1

    def forward(self, predicts, targets, imgs):
        """
        predicts: (pred_bboxes, cls_logits, reg_dist, anchors)
        targets:  [B, M, 5] (normalized 0-1)
        imgs:     [B, 3, H, W]
        """
        p_box, p_cls, p_reg, anchors, strides = predicts
        device = p_box.device
        batch_size = p_box.shape[0]
        _, _, img_h, img_w = imgs.shape
        img_size = torch.tensor([img_w, img_h, img_w, img_h], device=device)

        pd_box = p_box.permute(0, 2, 1)      # [B, 8400, 4]
        pd_cls = p_cls.permute(0, 2, 1)      # [B, 8400, nc]
        
        gt_labels = targets[:, :, 4:5]            # [B, num_gt, 1]
        gt_bboxes = targets[:, :, 0:4] * img_size # [B, num_gt, 4] 
        mask_gt   = (targets.sum(-1, keepdim=True) > 0)

        t_bboxes, t_scores, fg_mask = self.tal(
            pd_cls.detach().sigmoid(), 
            pd_box.detach(),
            gt_labels, gt_bboxes, mask_gt
        )

        loss_cls = self.bce_loss(p_cls, t_scores) 
        loss_iou = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)

        if fg_mask.any():
            p_box_pos = pd_box[fg_mask] # [N_pos, 4]，正样本预测框
            t_box_pos = t_bboxes[fg_mask] # [N_pos, 4]，正样本所匹配的 gt 框
            
            pos_anchors = anchors.transpose(1, 2).expand(batch_size, -1, -1)[fg_mask] # [N_pos, 2]，为正样本对应的锚点中心
            strides = strides.transpose(1, 2) 
            pos_strides = strides.expand(batch_size, -1, -1)[fg_mask] # 用于对齐不同特征尺度
            
            loss_iou = self.iou_loss(p_box_pos, t_box_pos).sum()

            target_ltrb = encode(t_box_pos, pos_anchors) / pos_strides # 归一化
            p_dist_pos = p_reg.permute(0, 2, 1)[fg_mask] # [N_pos, 64]
            loss_dfl = self.dfl_loss(p_dist_pos, target_ltrb)

        target_scores_sum = t_scores.sum()
        norm_factor = max(target_scores_sum, 1.0)  # 正则化项，工程上可能改为 max(fg_mask.sum(), 1.0)，收敛快但精度下降

        total_loss = (loss_iou * self.hyp_box + 
                      loss_cls * self.hyp_cls + 
                      loss_dfl * self.hyp_dfl)

        return total_loss / norm_factor
         