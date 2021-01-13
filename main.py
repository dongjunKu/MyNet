import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

import params
from dataloader.read_data import SceneFlow
import torch.optim as optim

from utils.python_pfm import *
from models.myNet import *

import matplotlib.pyplot as plt

p = params.Params()

head = HeadPac([3, 64, 128, 256, 512, 1024])
body = BodyFst(p.train_disparity)

criterion = nn.HingeEmbeddingLoss(margin=0.2, reduction='mean')
optimizer = torch.optim.Adam(head.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor()])
# transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# transforms.Grayscale(num_output_channels=1) - grayscale로 변환한다.

dataset = SceneFlow("train", transform=transform, crop_size=p.train_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

for epoch in range(5):
    data_iter = iter(dataloader)

    for step in range(len(dataloader)):
        data = next(data_iter)
        
        featuresL, kernelsL = head(data['imL'], multiscale=True)
        featuresR, kernelsR = head(data['imR'], multiscale=True)
        cost_vols = body(featuresL, featuresR, kernelsL)

        gt = data['dispL']
        gt[gt > p.train_disparity - 1] = p.train_disparity - 1 # 최대 넘는 값 처리.
        disps = []
        for i in range(len(cost_vols)):
            eye = torch.eye(p.train_disparity // 2**i)
            disp = (F.interpolate(gt, scale_factor=1/2**i, mode='bilinear', align_corners=False))
            disp = ((disp + 0.5) // 2**i).long() # (N, 1, H, W)
            disp = torch.squeeze(eye[disp].transpose(1,-1), -1) # 원 핫
            disps.append(disp)

        losses = []
        for cost_vol, disp in zip(cost_vols, disps):
            losses.append(criterion(cost_vol, disp))
        loss = sum(losses)

        print("step: ", step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """
        print(data['imL'].numpy().shape)
        print(np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0)).shape)
        plt.subplot(2,2,1), plt.imshow((255 * np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,2,2), plt.imshow((255 * np.transpose(np.squeeze(data['imR'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,2,3), plt.imshow(np.squeeze(data['dispL']), vmin=0)
        plt.subplot(2,2,4), plt.imshow(np.squeeze(data['dispR']), vmin=0)
        plt.show()
        """
        