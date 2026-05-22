import torch
import torch.nn as nn
import torch.nn.functional as F


class OACSFMJointLoss(nn.Module):


    def __init__(self, consistency_weight=1.0, visibility_weight=0.05,
                 reliability_weight=0.05, rho=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.consistency_weight = consistency_weight
        self.visibility_weight = visibility_weight
        self.reliability_weight = reliability_weight
        self.rho = rho

    @staticmethod
    def _symmetric_kl(logits_a, logits_b):
        prob_a = F.softmax(logits_a, dim=1)
        prob_b = F.softmax(logits_b, dim=1)
        kl_ab = F.kl_div(F.log_softmax(logits_a, dim=1), prob_b.detach(), reduction='batchmean')
        kl_ba = F.kl_div(F.log_softmax(logits_b, dim=1), prob_a.detach(), reduction='batchmean')
        return 0.5 * (kl_ab + kl_ba)

    @staticmethod
    def _masked_mean(values, mask):
        if mask is None:
            return values.mean()
        mask_f = mask.float()
        return (values * mask_f).sum() / (mask_f.sum() + 1e-8)

    def _visibility_regularization(self, visibility, mask):
        l1 = self._masked_mean(visibility, mask)
        if visibility.size(1) > 1:
            diff = torch.abs(visibility[:, 1:] - visibility[:, :-1])
            diff_mask = mask[:, 1:] & mask[:, :-1] if mask is not None else None
            tv = self._masked_mean(diff, diff_mask)
        else:
            tv = visibility.new_tensor(0.0)
        return l1 + tv

    def _reliability_sparsity(self, reliability, mask):
        if mask is None:
            rho_hat = reliability.mean(dim=1)
        else:
            mask_f = mask.float()
            rho_hat = (reliability * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
        rho_hat = torch.clamp(rho_hat, min=1e-5, max=1.0 - 1e-5)
        rho = torch.as_tensor(self.rho, device=reliability.device, dtype=reliability.dtype)
        return (rho * torch.log(rho / rho_hat) +
                (1.0 - rho) * torch.log((1.0 - rho) / (1.0 - rho_hat))).mean()

    def forward(self, logits_tuple, targets, aux_tuple, masks_tuple, lambda_regular=1.0):
        logits_high, logits_middle, logits_low = logits_tuple

        loss_cls = (
            self.ce(logits_high, targets) +
            self.ce(logits_middle, targets) +
            self.ce(logits_low, targets)
        )

        loss_con = (
            self._symmetric_kl(logits_high, logits_middle) +
            self._symmetric_kl(logits_high, logits_low) +
            self._symmetric_kl(logits_middle, logits_low)
        ) / 3.0

        loss_vis = logits_high.new_tensor(0.0)
        loss_rel = logits_high.new_tensor(0.0)
        for aux, mask in zip(aux_tuple, masks_tuple):
            visibility = aux['visibility']
            reliability = aux['reliability']
            loss_vis = loss_vis + self._visibility_regularization(visibility, mask)
            loss_rel = loss_rel + self._reliability_sparsity(reliability, mask)
        loss_vis = loss_vis / len(aux_tuple)
        loss_rel = loss_rel / len(aux_tuple)

        total_loss = (
            loss_cls +
            self.consistency_weight * loss_con +
            lambda_regular * self.visibility_weight * loss_vis +
            lambda_regular * self.reliability_weight * loss_rel
        )
        return total_loss, loss_cls, loss_con, loss_vis, loss_rel
