"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import os.path as osp
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob


class Fashion(datasets.ImageFolder):
    def __init__(self, root=MyPath.db_root_dir('fashion'), split='train', transform=None):
        super(Fashion, self).__init__(root=os.path.join(root),
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize((224, 224))
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img_name = path.split('/')[-1]
        url = osp.join("https://cdn.theyes.com/images/full", img_name)
        img = Image.open(path)
        #print(img.mode)
        if img.mode == 'RGBA':
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask = img.split()[3])
            img = background

        img = img.convert("RGB")
        img = self.resize(img) 
        im_size = img.size

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'url': url}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        #print(img.mode)
        if img.mode == 'RGBA':
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask = img.split()[3])
            img = background
        img = img.convert("RGB")

        img = self.resize(img) 
        return img


