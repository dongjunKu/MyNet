import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np

import params
from dataloader.read_data import SceneFlow
import torch.optim as optim

from utils.python_pfm import *
from models.myNet import *

import matplotlib.pyplot as plt

p = params.Params()

head = HeadPac([3, 64, 128, 256, 512, 1024])
body = BodyFst(192)

transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([transforms.RandomResizedCrop(256), transforms.ToTensor()]) # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

dataset = SceneFlow("train", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

for epoch in range(5):
    data_iter = iter(dataloader)

    for step in range(len(dataloader)):
        data = next(data_iter)

        featuresL, kernelsL = head(data['imL'], multiscale=True)
        featuresR, kernelsR = head(data['imR'], multiscale=True)
        cost_vols = body(featuresL, featuresR, kernelsL)

        print("step: ", step)

        """
        print(data['imL'].numpy().shape)
        print(np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0)).shape)
        plt.subplot(2,2,1), plt.imshow((255 * np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,2,2), plt.imshow((255 * np.transpose(np.squeeze(data['imR'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,2,3), plt.imshow(np.squeeze(data['dispL']), vmin=0)
        plt.subplot(2,2,4), plt.imshow(np.squeeze(data['dispR']), vmin=0)
        plt.show()
        """