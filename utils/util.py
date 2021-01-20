import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HingeLoss(nn.Module):

    def __init__(self, margin, reduction='mean'):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    # labels: 0, 1
    def forward(self, inputs, labels, mask=None):

        assert inputs.shape == labels.shape

        targets = torch.sum(inputs * labels, dim=1, keepdim=True)
        loss = inputs - targets + self.margin * (1 - labels)
        loss = F.relu(loss)

        if mask is not None:
            loss *= mask.float()
        
        if self.reduction == 'none':
            return loss
        if self.reduction == 'mean':
            return torch.mean(loss)
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