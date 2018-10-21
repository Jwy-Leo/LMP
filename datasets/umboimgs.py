from __future__ import print_function
import random
import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import maskUtils
import numpy as np
import json
import os
import os.path
import cv2
import pdb

__all__ = ['Umbot7', 'get_dts_idx_list', 'get_iter_dts_loder']


def get_dts_idx_list(loader_list):
    dts_size_list = []
    for loader in loader_list:
        dts_size_list.append(len(loader))
    n_batch = sum(dts_size_list)
    dts_idx_list = torch.LongTensor(n_batch)
    cnt = 0
    for i in range(0, len(dts_size_list)):
        dts_idx_list[cnt:cnt + dts_size_list[i]] = i
        cnt += dts_size_list[i]
    return dts_idx_list


def get_iter_dts_loder(loader_list):
    iter_dts_loader_list = []
    for loader in loader_list:
        iter_dts_loader_list.append(iter(loader))
    return iter_dts_loader_list


class Umbot7(data.Dataset):
    def __init__(self, img_dir, label_dir, n_class,train=False,resized=False):

        assert os.path.isdir(img_dir), \
            '%s is not a directory' % img_dir

        assert os.path.isdir(label_dir), \
            '%s is not a directory' % label_dir

        self._img_dir = img_dir
        self._label_dir = label_dir
        self.filenames = os.listdir(self._img_dir)
        self._n_class = n_class
        self.train = train
        self.resized = resized
    def __getitem__(self, index):
        
        file_name = self.filenames[index]
        
        img_file = os.path.join(self._img_dir,file_name)
        label_file = os.path.join(self._label_dir,file_name)

        img = transforms.ToTensor()(Image.open(img_file).convert('RGB'))
        label = transforms.ToTensor()(Image.open(label_file))
        label = label[0].long()
        img_w, img_h = img.size(2), img.size(1)
        if(self.train==False): #no the case
            #pass
            if self.resized:
                img = img[:,:640,:] #gym
                label = label[:640,:]
            else:
                img = img[:,:1024,:]
                label = label[:1024,:]
        else:
            crop_size = 256 #512
            x0 = random.randint(0,img_w-crop_size)
            y0 = random.randint(0,img_h-crop_size)
            img = img[:, y0:y0+crop_size, x0:x0+crop_size]
            label = label[y0:y0+crop_size, x0:x0+crop_size]
        return img, label, file_name

    def __len__(self):
        return len(self.filenames)

