import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HingeLoss(nn.Module):

    def __init__(self, margin, reduction='mean', ignore_pad=True, pad_value=0):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.ignore_pad = ignore_pad
        self.pad_value = pad_value

    # labels: 0, 1
    def forward(self, inputs, labels, mask=None):

        assert inputs.shape == labels.shape

        targets = torch.sum(inputs * labels, dim=1, keepdim=True) # (N, 1, H, W)
        loss = inputs - targets + self.margin * (1 - labels) # (N, D, H, W)
        loss = F.relu(loss)

        if mask is not None:
            loss *= mask.float()
        if self.ignore_pad:
            pad_mask = targets != self.pad_value # 패딩된 곳이 true disp일때는 loss 계산하지 않음.
            # print(torch.sum(targets == self.pad_value))
            loss *= pad_mask.float()

        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            ones = torch.ones_like(inputs, dtype=torch.int, device=inputs.device)
            if mask is not None:
                ones *= mask.int()
            if self.ignore_pad:
                ones *= (targets != self.pad_value).int()
            return torch.sum(loss) / (torch.sum(ones).float() + 1e-12)
        if self.reduction == 'sum':
            return torch.sum(loss)

def one_hot(indices, depth, on_value=1, off_value=0, dim=1, dtype=None):
    assert indices.shape[dim] == 1
    eye = torch.eye(depth, dtype=dtype, device=indices.device)
    if on_value != 1 or off_value != 0:
        eye = eye * on_value + (1-eye) * off_value
    if dim == -1:
        out = eye[indices.long()]
    else:
        out = torch.squeeze(eye[indices.long()].transpose(dim,-1), -1)
    return out