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

    def __init__(self, setType, transform=None, normalize=False, crop_size=(256,512), root_dir=None): # setType: "train" or "test"

        p = params.Params()

        if root_dir is None:
            root_dir = p.DATA_PATH
        else:
            root_dir = root_dir

        self.setType = setType
        self.normalize = normalize
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

        sample = {'imL': imageL, 'imR': imageR, 'dispL': dispL, 'dispR': dispR}
        if self.transform is not None:
            sample['imL']=self.transform(sample['imL'])
            sample['imR']=self.transform(sample['imR'])
        if self.normalize:
            sample['imL'] = (sample['imL'] - torch.mean(sample['imL'].view(sample['imL'].size()[0], -1))) / torch.std(sample['imL'].view(sample['imL'].size()[0], -1))
            sample['imR'] = (sample['imR'] - torch.mean(sample['imR'].view(sample['imR'].size()[0], -1))) / torch.std(sample['imR'].view(sample['imR'].size()[0], -1))
        if self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(imageL, output_size=self.crop_size)
            sample['imL'] = sample['imL'][:,i:i+h,j:j+w]
            sample['imR'] = sample['imR'][:,i:i+h,j:j+w]
            sample['dispL'] = sample['dispL'][:,i:i+h,j:j+w]
            sample['dispR'] = sample['dispR'][:,i:i+h,j:j+w]
        return sample

class Middlebury(Dataset):

    def __init__(self, setType, transform=None, resize=None, normalize=False, crop_size=None, root_dir=None): # setType: "train" or "test"

        p = params.Params()

        if root_dir is None:
            root_dir = p.DATA_PATH
        else:
            root_dir = root_dir

        self.setType = setType
        self.resize = resize
        self.normalize = normalize
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
        imageL_raw = Image.open(self.paths_img_left[idx])
        imageR_raw = Image.open(self.paths_img_right[idx])
        dispL_raw = None
        if self.paths_disp_left is not None:
            dispL_raw = readPFM(self.paths_disp_left[idx])[0].astype(np.float32)
        
        imageL = imageL_raw.copy()
        imageR = imageR_raw.copy()
        dispL = dispL_raw.copy()

        imageL_raw = torch.tensor(np.transpose(np.array(imageL_raw), (2,0,1)))
        imageR_raw = torch.tensor(np.transpose(np.array(imageR_raw), (2,0,1)))
        if self.paths_disp_left is not None:
            dispL_raw = torch.tensor(np.expand_dims(dispL_raw, axis=0))
        
        # resize
        if self.resize is not None:
            imageL = imageL.resize((self.resize[1], self.resize[0]))
            imageR = imageR.resize((self.resize[1], self.resize[0]))
            if self.paths_disp_left is not None:
                dispL = resize(dispL, self.resize) / dispL.shape[1] * self.resize[1]

        if dispL is not None:
            sample = {'imL': imageL, 'imR': imageR, 'dispL': np.expand_dims(dispL, axis=0),
                        'imL_raw': imageL_raw, 'imR_raw': imageR_raw, 'dispL_raw': dispL_raw}
        else:
            sample = {'imL': imageL, 'imR': imageR, 'imL_raw': imageL_raw, 'imR_raw': imageR_raw}
        if self.transform is not None:
            sample['imL'] = self.transform(sample['imL'])
            sample['imR'] = self.transform(sample['imR'])
        if self.normalize:
            sample['imL'] = (sample['imL'] - torch.mean(sample['imL'].view(sample['imL'].size()[0], -1))) / torch.std(sample['imL'].view(sample['imL'].size()[0], -1))
            sample['imR'] = (sample['imR'] - torch.mean(sample['imR'].view(sample['imR'].size()[0], -1))) / torch.std(sample['imR'].view(sample['imR'].size()[0], -1))
        if self.crop_size is not None:
            i, j, h, w = transforms.RandomCrop.get_params(imageL, output_size=self.crop_size)
            sample['imL'] = sample['imL'][:,i:i+h,j:j+w]
            sample['imR'] = sample['imR'][:,i:i+h,j:j+w]
            sample['dispL'] = sample['dispL'][:,i:i+h,j:j+w]
            sample['dispR'] = sample['dispR'][:,i:i+h,j:j+w]
        return sample
