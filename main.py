import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import params
from dataloader.read_data import SceneFlow
import torch.optim as optim

from utils.python_pfm import *
from utils.util import *
from models.myNet import *

import matplotlib.pyplot as plt

p = params.Params()

head = HeadPac(p.feature_num_list).to(p.device)
body = BodyFst(p.train_disparity).to(p.device)  

criterion = HingeLoss(margin=0.2, reduction='mean').to(p.device)
optimizer = torch.optim.Adam(head.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor()])
# transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# transforms.Grayscale(num_output_channels=1) - grayscale로 변환한다.

dataset = SceneFlow("train", transform=transform, crop_size=p.train_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

pre_epoch = 0
pre_step = 0
try:
    checkpoint = torch.load(p.SAVE_PATH + 'checkpoint.tar')
    head.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    pre_epoch = checkpoint['epoch']
    pre_step = checkpoint['step']
    print("restored successfully.. start from {} epoch {} step".format(pre_epoch, pre_step))
except:
    print("can't find saved model")

if p.mode == 'train':
    head.train()
if p.mode == 'eval':
    head.eval()

for epoch in range(pre_epoch, 5):
    data_iter = iter(dataloader)

    for step in range(pre_step, len(dataloader)):
        data = next(data_iter)
        
        featuresL, kernelsL = head(data['imL'].to(p.device), multiscale=True)
        featuresR, kernelsR = head(data['imR'].to(p.device), multiscale=True)
        cost_vols = body(featuresL, featuresR, kernelsL)

        gt = data['dispL'].to(p.device)
        masks = []
        disps = []
        for i in range(len(cost_vols)):
            disp = (F.interpolate(gt, scale_factor=1/2**i, mode='bilinear', align_corners=False))
            mask = (disp < p.train_disparity - 0.5).float()
            disp *= mask
            masks.append(mask)
            disp = ((disp + 0.5) // 2**i).long() # (N, 1, H, W)
            disp = one_hot(disp, p.train_disparity // 2**i, dim=1)
            disps.append(disp)

        losses = []
        for cost_vol, disp, mask in zip(cost_vols, disps, masks):
            losses.append(criterion(cost_vol, disp, mask))
        loss = sum(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step:", step, ", loss1: {}, loss2: {}, loss3: {}, loss4: {}, loss5: {}".format(*losses))
        # print(cost_vols[-1].shape)
        # print(cost_vols[-1])

        if step % 100 == 0:
            torch.save({'epoch': epoch,
                        'step': step + 1,
                        'model_state_dict': head.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        p.SAVE_PATH + 'checkpoint.tar')

        """
        print(data['imL'].numpy().shape)
        print(np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0)).shape)
        plt.subplot(2,2,1), plt.imshow((255 * np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,2,2), plt.imshow((255 * np.transpose(np.squeeze(data['imR'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,2,3), plt.imshow(np.squeeze(data['dispL']), vmin=0)
        plt.subplot(2,2,4), plt.imshow(np.squeeze(data['dispR']), vmin=0)
        plt.show()
        """

    pre_step = 0
        