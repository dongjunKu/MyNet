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
from viewer.view import *
from models.myNet import *

import matplotlib.pyplot as plt

p = params.Params()

head = HeadPac_mc().to(p.device)
body = BodyFst().to(p.device) # pad: -2

criterion = HingeLoss(margin=0.2, reduction='mean', ignore_pad=True, pad_value=0).to(p.device) # pad: -2
# optimizer = torch.optim.Adam(head.parameters(), lr=0.001) # TODO: body 파라미터, 가변 lr
optimizer = torch.optim.SGD(head.parameters(), lr=0.002, momentum=0.9) # for mc-cnn

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#transforms.Grayscale(num_output_channels=1) # grayscale로 변환한다.

def train(imgL, imgR, disp_true):
    head.train()

    optimizer.zero_grad()

    featuresL, kernelsL = head(imgL.to(p.device), kernel_coeff=p.train_kernel_coeff)
    featuresR, kernelsR = head(imgR.to(p.device), kernel_coeff=p.train_kernel_coeff)
    cost_vols = body(featuresL, featuresR, p.train_disparity, kernelsL)

    gt = disp_true.to(p.device)
    masks = []
    disps = []
    for i in range(len(cost_vols)):
        disp = (F.interpolate(gt, scale_factor=1/2**i, mode='bilinear', align_corners=False))
        temp = torch.tensor(disp) # debug
        mask = disp < p.train_disparity - 0.5
        disp[mask == 0] = 0
        if torch.isnan(disp).any(): # debug
            print(temp[torch.isnan(temp)]) # debug
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

    return [loss.cpu().detach().numpy() for loss in losses]

def validate(imgL, imgR, disp_true):
    head.eval()

    optimizer.zero_grad()

    with torch.no_grad():
        featuresL, kernelsL = head(imgL.to(p.device), kernel_coeff=p.train_kernel_coeff)
        featuresR, kernelsR = head(imgR.to(p.device), kernel_coeff=p.train_kernel_coeff)
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

    return [loss.cpu().numpy() for loss in losses]

def test(imgL, imgR, disp_true):
    head.eval()

    optimizer.zero_grad()

    with torch.no_grad():
        featuresL, kernelsL = head(imgL.to(p.device), kernel_coeff=p.test_kernel_coeff)
        featuresR, kernelsR = head(imgR.to(p.device), kernel_coeff=p.test_kernel_coeff)
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

    return [cost_vol.cpu().numpy() for cost_vol in cost_vols], [loss.cpu().numpy() for loss in losses]

pre_epoch = 0
pre_step = 0
min_loss = float('inf')
try:
    checkpoint = torch.load(p.SAVE_PATH + 'checkpoint.tar')
    head.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    pre_epoch = checkpoint['epoch']
    pre_step = checkpoint['step']
    min_loss = checkpoint['min_loss']
    print("restored successfully.. start from {} epoch {} step, min_loss: {}".format(pre_epoch, pre_step, min_loss))
except:
    print("can't find saved model")

if p.mode == 'train':
    train_dataset = Sceneflow('train', transform=transform, normalize=True, crop_size=p.train_size)
    validate_dataset = Sceneflow('test', transform=transform, normalize=True, crop_size=p.train_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

    for epoch in range(pre_epoch, 5):
        train_data_iter = iter(train_dataloader)
        validate_data_iter = iter(validate_dataloader)

        for step in range(pre_step, len(train_dataloader)):
            data = next(train_data_iter)

            losses = train(data['imL'], data['imR'], data['dispL'])
            print_losses("step: " + str(step), losses)

            if step % 500 == 0:
                validate_steps = 100
                validate_losses = [0., 0., 0., 0., 0.]
                for val_step in range(validate_steps):
                    data = next(validate_data_iter)
                    losses = validate(data['imL'], data['imR'], data['dispL'])

                    for i in range(len(losses)):
                        validate_losses[i] += losses[i] / validate_steps
                
                print_losses("\nvalidate loss", losses, "\n")

                if min_loss > sum(losses):
                    min_loss = sum(losses)
                    torch.save({'epoch': epoch,
                                'step': step + 1,
                                'min_loss': min_loss,
                                'model_state_dict': head.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                                p.SAVE_PATH + 'checkpoint' + str(epoch) + 'ep' + str(step) + 'step.tar')

                torch.save({'epoch': epoch,
                            'step': step + 1,
                            'min_loss': min_loss,
                            'model_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            p.SAVE_PATH + 'checkpoint.tar')
                    

        pre_step = 0

if p.mode == 'test':
    test_dataset = Middlebury('train', transform=transform, normalize=True, resize=p.test_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=p.batch_size, shuffle=False, num_workers=1)
    
    test_data_iter = iter(test_dataloader)

    for step in range(len(test_dataloader)):
        data = next(test_data_iter)

        cost_vols, losses = test(data['imL'], data['imR'], data['dispL'])

        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)

        print("---------------", step, "th image ---------------")
        print_losses("test loss", losses)
        plt.subplot(2,3,1), plt.title("imageL"), plt.imshow((np.transpose(np.squeeze(data['imL'].numpy(), axis=0), (1,2,0))), cmap='gray')
        plt.subplot(2,3,2), plt.title("imageR"), plt.imshow((np.transpose(np.squeeze(data['imR'].numpy(), axis=0), (1,2,0))), cmap='gray')
        plt.subplot(2,3,3), plt.title("disp_true"), plt.imshow(np.squeeze(data['dispL']), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,3,4), plt.title("disp_pred"), plt.imshow(np.squeeze(np.argmax(cost_vols[0], axis=1)), vmin=0, vmax=p.test_disparity)
        plt.subplot(2,3,5), plt.title("diff")
        s = plot_error(cost_vols[0], data['dispL_raw'].numpy(), p.test_disparity)
        textbox = plt.subplot(2,3,6)
        plt.title('kernel coeff: 0.1'), plt.axis('off'), plt.text(0.05, 0.3, s, fontsize=8)
        #plt.subplot(2,4,5), plt.imshow(np.squeeze(np.argmax(cost_vols[1], axis=1)*2), vmin=0, vmax=p.test_disparity)
        #plt.subplot(2,4,6), plt.imshow(np.squeeze(np.argmax(cost_vols[2], axis=1)*4), vmin=0, vmax=p.test_disparity)
        #plt.subplot(2,4,7), plt.imshow(np.squeeze(np.argmax(cost_vols[3], axis=1)*8), vmin=0, vmax=p.test_disparity)
        #plt.subplot(2,4,8), plt.imshow(np.squeeze(np.argmax(cost_vols[4], axis=1)*16), vmin=0, vmax=p.test_disparity)
        # plt.tight_layout()
        plt.subplots_adjust(left=0.02, bottom=0, right=0.98, top=1, wspace=0.08, hspace=0)
        plt.savefig('/home/gu/workspace/Gu/EANet/EANet_0.0/result/mc-cnn/step' + str(step) + '_1e-1.png', dpi=300)        
        #plt.show()
        textbox.clear()
        print("\n")