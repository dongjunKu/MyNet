import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.pac as pac

class HeadPac_mc(nn.Module):
    def __init__(self, num_channels=[1, 112, 112, 112, 112, 112]):
        super(HeadPac_mc, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # conv1
        self.conv1_1 = nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1)
        self.conv1_2 = nn.Conv2d(num_channels[1], num_channels[2], 3, padding=1)
        self.conv1_3 = nn.Conv2d(num_channels[2], num_channels[3], 3, padding=1)
        self.conv1_4 = nn.Conv2d(num_channels[3], num_channels[4], 3, padding=1)
        self.conv1_5 = pac.PacConv2d(num_channels[4], num_channels[5], 3, padding=1)

    def forward(self, x, kernel_coeff=0.0001):
        h = x
        
        h1 = self.conv1_1(h)
        h1 = self.relu(h1)
        h1 = self.conv1_2(h1)
        h1 = self.relu(h1)
        h1 = self.conv1_3(h1)
        h1 = self.relu(h1)
        h1 = self.conv1_4(h1)
        h1 = self.relu(h1)
        h1, k1 = self.conv1_5(h1, h1 * kernel_coeff, return_kernel=True)
        
        return [h1], [k1]

# 옵션: 모든 컨볼루션을 pac로 할 것인가. transpose conv의 커널로 앞부분의 필터를 넘길 때 어떻게 넘길것인가, channel_wise 옵션 끄거나 켜기
class HeadPac(nn.Module):
    def __init__(self, num_channels=[1, 64, 128, 256, 512, 1024]):
        super(HeadPac, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # conv1
        self.conv1_1 = nn.Conv2d(num_channels[0], num_channels[1], 3, padding=1)
        self.conv1_2 = nn.Conv2d(num_channels[1], num_channels[1], 3, padding=1)
        
        self.conv1_3 = nn.Conv2d(num_channels[1], num_channels[1], 3, stride=2, padding=1)  # 1/2

        # conv2
        self.conv2_1 = pac.PacConv2d(num_channels[1], num_channels[2], 3, padding=1)
        self.conv2_2 = pac.PacConv2d(num_channels[2], num_channels[2], 3, padding=1)
        
        self.conv2_3 = pac.PacConv2d(num_channels[2], num_channels[2], 3, stride=2, padding=1)  # 1/4

        # conv3
        self.conv3_1 = pac.PacConv2d(num_channels[2], num_channels[3], 3, padding=1)
        self.conv3_2 = pac.PacConv2d(num_channels[3], num_channels[3], 3, padding=1)
        
        self.conv3_3 = pac.PacConv2d(num_channels[3], num_channels[3], 3, stride=2, padding=1)  # 1/8

        # conv4
        self.conv4_1 = pac.PacConv2d(num_channels[3], num_channels[4], 3, padding=1)
        self.conv4_2 = pac.PacConv2d(num_channels[4], num_channels[4], 3, padding=1)
        
        self.conv4_3 = pac.PacConv2d(num_channels[4], num_channels[4], 3, stride=2, padding=1)  # 1/16

        # conv5
        self.conv5_1 = pac.PacConv2d(num_channels[4], num_channels[5], 3, padding=1)
        self.conv5_2 = pac.PacConv2d(num_channels[5], num_channels[5], 3, padding=1)

    def forward(self, x, kernel_coeff=0.0001):
        h = x
        
        h1 = self.conv1_1(h)
        h1 = self.relu(h1)
        h1 = self.conv1_2(h1)
        h1 = self.relu(h1)
        
        h2 = self.conv1_3(h1)
        h2 = self.relu(h2)

        h2, k2 = self.conv2_1(h2, h2 * kernel_coeff, return_kernel=True) # 커널 shape: (N, C, kernel_h, kernel_w, out_h, out_w)
        h2 = self.relu(h2)
        h2 = self.conv2_2(h2, kernel=k2) # 커널 재사용
        h2 = self.relu(h2)
        
        h3 = self.conv2_3(h2, kernel=k2[:,:,:,:,::2,::2]) # 여기서는 커널을 스트라이드 2에 맞게 변형해야 함.
        h3 = self.relu(h3)
        
        h3, k3 = self.conv3_1(h3, h3 * kernel_coeff, return_kernel=True)
        h3 = self.relu(h3)
        h3 = self.conv3_2(h3, kernel=k3)
        h3 = self.relu(h3)
        
        h4 = self.conv3_3(h3, kernel=k3[:,:,:,:,::2,::2])
        h4 = self.relu(h4)

        h4, k4 = self.conv4_1(h4, h4 * kernel_coeff, return_kernel=True)
        h4 = self.relu(h4)
        h4 = self.conv4_2(h4, kernel=k4)
        h4 = self.relu(h4)

        h5 = self.conv4_3(h4, kernel=k4[:,:,:,:,::2,::2])
        h5 = self.relu(h5)

        h5, k5 = self.conv5_1(h5, h5 * kernel_coeff, return_kernel=True)
        h5 = self.relu(h5)
        h5 = self.conv5_2(h5, kernel=k5)
        h5 = self.relu(h5)
        
        return [h1, h2, h3, h4, h5], [None, k2, k3, k4, k5]

class BodyFst(nn.Module):
    def __init__(self):
        super(BodyFst, self).__init__()

        # self.deconv4 = pac.PacConvTranspose2d(1, 1, 3, stride=2, padding=1)

    def forward(self, left_features, right_features, max_disparity, kernels=None):
        max_disparity = min(max_disparity, left_features[0].shape[-1])

        cost_volumes = []
        for i, (left_feature, right_feature) in enumerate(zip(left_features, right_features)):
            left_feature = F.normalize(left_feature, p=2)
            right_feature = F.normalize(right_feature, p=2)
            dot_volume = left_feature.permute(0,2,3,1).matmul(right_feature.permute(0,2,1,3)) # dot product, (N, H, W, W)
            ww2wd = []
            for j in range(max_disparity // (2**i)):
                ww2wd.append(F.pad(torch.diagonal(dot_volume, offset=-j, dim1=-2, dim2=-1), (j,0), "constant", 0)) # (N, H, W)
            cost_volume = torch.stack(ww2wd, dim=1) # (N, D, H, W)
            cost_volumes.append(cost_volume)
        """
        d5 = torch.argmax(cost_volumes[5-1], dim=1, keepdim=True) * (2**(5-1)) # (N, 1, H, W)
        d4 = self.deconv4(d5, kernel=kernels[4-1] if kernels is not None else kernels)
        """

        return cost_volumes

    def cost_volume_fst():
        return
    def cost_volume_acrt():
        return

        