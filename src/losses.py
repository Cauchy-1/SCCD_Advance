# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class NoiseRobustLoss(nn.Module):
    """
    组合: Label Smoothing + GCE + ELR 正则 + 软标签蒸馏 + 小损筛选
    mode 示例:
      'ce' 仅交叉熵
      'gce' 使用 GCE
      'gce_elr' GCE + ELR
    """
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        mode: str = "gce_elr",
        gce_q: float = 0.7,
        elr_lambda: float = 3.0,
        ema_alpha: float = 0.7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mode = mode
        self.gce_q = gce_q
        self.elr_lambda = elr_lambda
        self.ema_alpha = ema_alpha
        self.register_buffer("eps", torch.tensor(1e-8))
        self.pred_history = {}  # sample_id -> prob tensor

    def update_ema(self, sample_ids: List[str], probs: torch.Tensor):
        for sid, p in zip(sample_ids, probs.detach()):
            if sid not in self.pred_history:
                self.pred_history[sid] = p.clone()
            else:
                self.pred_history[sid] = (
                    self.ema_alpha * self.pred_history[sid] + (1 - self.ema_alpha) * p
                )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_ids: List[str],
        soft_targets: Optional[List[torch.Tensor]] = None,
        small_loss_mask: Optional[torch.Tensor] = None,
    ):
        probs = torch.softmax(logits, dim=-1)
        self.update_ema(sample_ids, probs)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (self.num_classes - 1)
            ls_target = torch.full_like(probs, smooth)
            ls_target.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            ls_target = F.one_hot(targets, self.num_classes).float()

        ce = -(ls_target * torch.log(probs + self.eps)).sum(dim=1)
        loss = ce

        if "gce" in self.mode:
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            if self.gce_q < 1:
                gce = (1 - pt.pow(self.gce_q)) / self.gce_q
            else:
                gce = -torch.log(pt + self.eps)
            loss = 0.5 * ce + 0.5 * gce

        if "elr" in self.mode:
            reg_terms = []
            for i, sid in enumerate(sample_ids):
                hist = self.pred_history[sid].clamp(1e-6, 1 - 1e-6)
                reg_terms.append(torch.sum(probs[i] * hist))
            reg = torch.stack(reg_terms)
            elr_penalty = self.elr_lambda * (-torch.log(1 - reg + self.eps))
            loss = loss + elr_penalty

        if soft_targets is not None:
            if all(st is not None for st in soft_targets):
                st = torch.stack(soft_targets).to(logits.device)
                kl = F.kl_div(
                    torch.log(probs + self.eps), st, reduction="none"
                ).sum(dim=1)
                loss = 0.7 * loss + 0.3 * kl

        if small_loss_mask is not None:
            loss = loss[small_loss_mask]

        return loss.mean()