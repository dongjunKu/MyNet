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
from dataloader.read_data import *
import torch.optim as optim

from utils.python_pfm import *
from utils.util import *
from models.myNet import *

import matplotlib.pyplot as plt

p = params.Params()

head = HeadPac(p.feature_num_list).to(p.device)
body = BodyFst().to(p.device)

criterion = HingeLoss(margin=0.2, reduction='mean').to(p.device)
optimizer = torch.optim.Adam(head.parameters(), lr=0.001) # TODO: body 파라미터, 가변 lr

transform = transforms.Compose([transforms.ToTensor()])
# transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# transforms.Grayscale(num_output_channels=1) - grayscale로 변환한다.

def train(imgL, imgR, disp_true, kernel_coeff=0):
    head.train()

    optimizer.zero_grad()

    featuresL, kernelsL = head(imgL.to(p.device), kernel_coeff=kernel_coeff, multiscale=True)
    featuresR, kernelsR = head(imgR.to(p.device), kernel_coeff=kernel_coeff, multiscale=True)
    cost_vols = body(featuresL, featuresR, p.train_disparity, kernelsL)

    gt = disp_true.to(p.device)
    masks = []
    disps = []
    for i in range(len(cost_vols)):
        disp = (F.interpolate(gt, scale_factor=1/2**i, mode='bilinear', align_corners=False))
        mask = disp < p.train_disparity - 0.5
        disp[mask == 0] = 0
        masks.append(mask)
        disp = ((disp + 0.5) // 2**i).long() # (N, 1, H, W)
        disp = one_hot(disp, p.train_disparity // 2**i, dim=1)
        disps.append(disp)

    losses = []
    for cost_vol, disp, mask in zip(cost_vols, disps, masks):
        losses.append(criterion(cost_vol, disp, mask))
    loss = sum(losses)
    
    loss.backward()
    optimizer.step()

    return [loss.data for loss in losses]

def test(imgL, imgR, disp_true, kernel_coeff=0.0001):
    head.eval()

    optimizer.zero_grad()

    with torch.no_grad():
        featuresL, kernelsL = head(imgL.to(p.device), kernel_coeff=kernel_coeff, multiscale=True)
        featuresR, kernelsR = head(imgR.to(p.device), kernel_coeff=kernel_coeff, multiscale=True)
        cost_vols = body(featuresL, featuresR, p.test_disparity, kernelsL)

    gt = disp_true.to(p.device)
    masks = []
    disps = []
    for i in range(len(cost_vols)):
        disp = (F.interpolate(gt, scale_factor=1/2**i, mode='bilinear', align_corners=False))
        mask = disp < p.test_disparity - 0.5
        disp[mask == 0] = 0
        masks.append(mask)
        disp = ((disp + 0.5) // 2**i).long() # (N, 1, H, W)
        disp = one_hot(disp, p.test_disparity // 2**i, dim=1)
        disps.append(disp)

    losses = []
    for cost_vol, disp, mask in zip(cost_vols, disps, masks):
        losses.append(criterion(cost_vol, disp, mask))

    return cost_vols, [loss.data for loss in losses]

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
    train_dataset = Sceneflow('train', transform=transform, crop_size=p.train_size)
    test_dataset = Sceneflow('test', transform=transform, crop_size=p.train_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

    for epoch in range(pre_epoch, 5):
        train_data_iter = iter(train_dataloader)
        test_data_iter = iter(test_dataloader)

        for step in range(pre_step, len(train_dataloader)):
            data = next(train_data_iter)

            losses = train(data['imL'], data['imR'], data['dispL'], p.train_kernel_coeff)
            print("step:", step, ",\tloss1: {},\tloss2: {},\tloss3: {},\tloss4: {},\tloss5: {}".format(*losses))

            if step % 500 == 0:
                torch.save({'epoch': epoch,
                            'step': step + 1,
                            'model_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            p.SAVE_PATH + 'checkpoint.tar')

                test_steps = 100
                test_losses = [0., 0., 0., 0., 0.]
                for step in range(test_steps):
                    data = next(test_data_iter)
                    cost_vols, losses = test(data['imL'], data['imR'], data['dispL'], p.test_kernel_coeff)

                    for i in range(len(losses)):
                        test_losses[i] += losses[i] / test_steps
                
                print("\ntest loss\tloss1: {},\tloss2: {},\tloss3: {},\tloss4: {},\tloss5: {}\n".format(*test_losses))

        pre_step = 0

if p.mode == 'test':
    test_dataset = Middlebury('train', transform=transform, resize=p.test_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)
    
    test_data_iter = iter(test_dataloader)

    for step in range(len(test_dataloader)):
        data = next(test_data_iter)

        cost_vols, losses = test(data['imL'], data['imR'], data['dispL'], p.test_kernel_coeff)
        print("step:", step, ",\tloss1: {},\tloss2: {},\tloss3: {},\tloss4: {},\tloss5: {}".format(*losses))

        plt.subplot(2,4,1), plt.imshow((255 * np.transpose(np.squeeze(data['imL'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,4,2), plt.imshow((255 * np.transpose(np.squeeze(data['imR'].numpy()), (1,2,0))).astype(np.uint8))
        plt.subplot(2,4,3), plt.imshow(np.squeeze(data['dispL']), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,4,4), plt.imshow(np.squeeze(np.argmax(cost_vols[0], axis=1)), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,4,5), plt.imshow(np.squeeze(np.argmax(cost_vols[1], axis=1)*2), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,4,6), plt.imshow(np.squeeze(np.argmax(cost_vols[2], axis=1)*4), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,4,7), plt.imshow(np.squeeze(np.argmax(cost_vols[3], axis=1)*8), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,4,8), plt.imshow(np.squeeze(np.argmax(cost_vols[4], axis=1)*16), vmin=0, vmax=p.test_disparity)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.08, hspace=0)
        plt.show()