from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
from utils.python_pfm import readPFM
import numpy as np
import torch
import params

class SceneFlow(Dataset):

    def __init__(self, setType, transform=None, root_dir=None): # setType: "train" or "test"

        p = params.Params()

        if root_dir is None:
            root_dir = p.DATA_PATH
        else:
            root_dir = root_dir

        self.setType = setType

        if setType == "train":
            path_paths_img_left = root_dir + p.paths_train_img_left
            path_paths_img_right = root_dir + p.paths_train_img_right
            path_paths_disp_left = root_dir + p.paths_train_disp_left
            path_paths_disp_right = root_dir + p.paths_train_disp_right
        if setType == "test":
            path_paths_img_left = root_dir + p.paths_test_img_left
            path_paths_img_right = root_dir + p.paths_test_img_right
            path_paths_disp_left = root_dir + p.paths_test_disp_left
            path_paths_disp_right = root_dir + p.paths_test_disp_right

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
        imageL = cv2.cvtColor(cv2.imread(self.paths_img_left[idx]), cv2.COLOR_BGR2RGB).reshape(540,960,3)#.transpose((2, 0, 1))
        imageR = cv2.cvtColor(cv2.imread(self.paths_img_right[idx]), cv2.COLOR_BGR2RGB).reshape(540,960,3)#.transpose((2, 0, 1))
        dispL = readPFM(self.paths_disp_left[idx])[0].astype(np.float32).reshape(540,960,1).transpose((2, 0, 1))
        dispR = readPFM(self.paths_disp_right[idx])[0].astype(np.float32).reshape(540,960,1).transpose((2, 0, 1))
        sample = {'imL': imageL, 'imR': imageR, 'dispL': dispL, 'dispR': dispR}
        if self.transform is not None:
            sample['imL']=self.transform(sample['imL'])
            sample['imR']=self.transform(sample['imR'])
        return sample
