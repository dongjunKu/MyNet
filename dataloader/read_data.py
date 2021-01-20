from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pickle
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

from utils.python_pfm import readPFM
import params

class Sceneflow(Dataset):

    def __init__(self, setType, transform=None, crop_size=(256,512), root_dir=None): # setType: "train" or "test"

        p = params.Params()

        if root_dir is None:
            root_dir = p.DATA_PATH
        else:
            root_dir = root_dir

        self.setType = setType
        self.crop_size = crop_size

        if setType == "train":
            path_paths_img_left = root_dir + p.sceneflow_paths_train_img_left
            path_paths_img_right = root_dir + p.sceneflow_paths_train_img_right
            path_paths_disp_left = root_dir + p.sceneflow_paths_train_disp_left
            path_paths_disp_right = root_dir + p.sceneflow_paths_train_disp_right
        if setType == "test":
            path_paths_img_left = root_dir + p.sceneflow_paths_test_img_left
            path_paths_img_right = root_dir + p.sceneflow_paths_test_img_right
            path_paths_disp_left = root_dir + p.sceneflow_paths_test_disp_left
            path_paths_disp_right = root_dir + p.sceneflow_paths_test_disp_right

        finl = open(path_paths_img_left,'rb')
        finr = open(path_paths_img_right, 'rb')
        self.paths_img_left = pickle.load(finl)
        self.paths_img_right = pickle.load(finr)        
        finl.close()
        finr.close()
        finl = open(path_paths_disp_left, 'rb')
        finr = open(path_paths_disp_right, 'rb')
        self.paths_disp_left = pickle.load(finl)
        self.paths_disp_right = pickle.load(finr)
        finl.close()
        finr.close()

        assert len(self.paths_img_left) == len(self.paths_img_right) == len(self.paths_disp_left) == len(self.paths_disp_right)
        
        self.transform = transform

    def __len__(self):
        return len(self.paths_img_left)

    def __getitem__(self, idx):

        # print(self.paths_img_left[idx])
        # print(self.paths_img_right[idx])
        # print(self.paths_disp_left[idx])
        # print(self.paths_disp_right[idx])
        imageL = Image.open(self.paths_img_left[idx])
        imageR = Image.open(self.paths_img_right[idx])
        dispL = readPFM(self.paths_disp_left[idx])[0].astype(np.float32).reshape(540,960,1).transpose((2, 0, 1))
        dispR = readPFM(self.paths_disp_right[idx])[0].astype(np.float32).reshape(540,960,1).transpose((2, 0, 1))
        
        # randomcrop
        if self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(imageL, output_size=self.crop_size)
            imageL = TF.crop(imageL, i, j, h, w)
            imageR = TF.crop(imageR, i, j, h, w)
            dispL = dispL[:,i:i+h,j:j+w] # pillow image가 아니므로 이렇게 한다.
            dispR = dispR[:,i:i+h,j:j+w]

        sample = {'imL': imageL, 'imR': imageR, 'dispL': dispL, 'dispR': dispR}
        if self.transform is not None:
            sample['imL']=self.transform(sample['imL'])
            sample['imR']=self.transform(sample['imR'])
        return sample

class Middlebury(Dataset):

    def __init__(self, setType, transform=None, resize=None, crop_size=None, root_dir=None): # setType: "train" or "test"

        p = params.Params()

        if root_dir is None:
            root_dir = p.DATA_PATH
        else:
            root_dir = root_dir

        self.setType = setType
        self.resize = resize
        self.crop_size = crop_size

        if setType == "train":
            path_paths_img_left = root_dir + p.middlebury_paths_train_img_left
            path_paths_img_right = root_dir + p.middlebury_paths_train_img_right
            path_paths_disp_left = root_dir + p.middlebury_paths_train_disp_left
        if setType == "test":
            path_paths_img_left = root_dir + p.middlebury_paths_test_img_left
            path_paths_img_right = root_dir + p.middlebury_paths_test_img_right
            path_paths_disp_left = None

        finl = open(path_paths_img_left,'rb')
        finr = open(path_paths_img_right, 'rb')
        self.paths_img_left = pickle.load(finl)
        self.paths_img_right = pickle.load(finr)
        finl.close()
        finr.close()       
        if path_paths_disp_left is not None:
            finl = open(path_paths_disp_left, 'rb')
            self.paths_disp_left = pickle.load(finl)
            finl.close()
        else:
            self.paths_disp_left = None

        assert len(self.paths_img_left) == len(self.paths_img_right)
        
        self.transform = transform

    def __len__(self):
        return len(self.paths_img_left)

    def __getitem__(self, idx):

        # print(self.paths_img_left[idx])
        # print(self.paths_img_right[idx])
        # print(self.paths_disp_left[idx])
        # print(self.paths_disp_right[idx])
        imageL = Image.open(self.paths_img_left[idx])
        imageR = Image.open(self.paths_img_right[idx])
        dispL = None
        if self.paths_disp_left is not None:
            dispL = readPFM(self.paths_disp_left[idx])[0].astype(np.float32)
        
        # resize
        if self.resize is not None:
            imageL = imageL.resize((self.resize[1], self.resize[0]))
            imageR = imageR.resize((self.resize[1], self.resize[0]))
            if self.paths_disp_left is not None:
                dispL = resize(dispL, self.resize) / dispL.shape[1] * self.resize[1]

        # randomcrop
        if self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(imageL, output_size=self.crop_size)
            imageL = TF.crop(imageL, i, j, h, w)
            imageR = TF.crop(imageR, i, j, h, w)
            if self.paths_disp_left is not None:
                dispL = dispL[i:i+h,j:j+w] # pillow image가 아니므로 이렇게 한다.

        if dispL is not None:
            sample = {'imL': imageL, 'imR': imageR, 'dispL': np.expand_dims(dispL, axis=0)}
        else:
            sample = {'imL': imageL, 'imR': imageR}
        if self.transform is not None:
            sample['imL']=self.transform(sample['imL'])
            sample['imR']=self.transform(sample['imR'])
        return sample
