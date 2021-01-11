import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np

import params
from read_data import SceneFlow
import torch.optim as optim

from python_pfm import *

import matplotlib.pyplot as plt

p = params.Params()

dataset = SceneFlow("train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

for epoch in range(5):
    data_iter = iter(dataloader)

    for step in range(len(dataloader)):
        data = next(data_iter)

        plt.subplot(2,2,1), plt.imshow(np.squeeze(data['imL']))
        plt.subplot(2,2,2), plt.imshow(np.squeeze(data['imR']))
        plt.subplot(2,2,3), plt.imshow(np.squeeze(data['dispL']), vmin=0)
        plt.subplot(2,2,4), plt.imshow(np.squeeze(data['dispR']), vmin=0)
        plt.show()