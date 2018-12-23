#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 17:47:33 2018

@author: xie
"""



import os
import torch
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset_floder(data.Dataset):
    def __init__(self, data_root, data_list, transform = None, target_transform=None, loader=default_loader):
        with open (data_list) as f:
            lines=f.readlines()
        imgs=[]
        for line in  lines:
            cls = line.split() 
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(data_root, fn)):
#                imgs.append((fn, tuple([float(v) for v in cls])))
                imgs.append((fn, tuple([int(v) for v in cls])))
        self.data_root = data_root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.data_root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.LongTensor(label)

    def __len__(self):
        return len(self.imgs)
    

