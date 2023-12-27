import torch
import torch.nn as nn


class Total_loss(nn.Module):

    def __init__(self, loss_opt, ignore_index: int = 255):
        super(Total_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.opt = loss_opt

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target.type(torch.long))

        ce_loss *= self.opt.ce_coef
        loss = ce_loss
        return loss, {'loss': ce_loss}
