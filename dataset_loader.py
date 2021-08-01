import os
import numpy as np
import PIL.Image
import random
import torch
from torch.utils import data
from functions import imutils


class DutsclsData(data.Dataset):
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False, resize = 512):
        super(DutsclsData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        self.img_list = []
        self.cls_list = []

        self.cls = os.listdir(self.root)
        self.cls2idx = dict(zip(self.cls, list(range(1, len(self.cls) + 1))))

        for cls_idx in self.cls:
            img_names = os.listdir(os.path.join(self.root, cls_idx))
            for img_name in img_names:
                self.img_list.append(img_name)
                self.cls_list.append([cls_idx])


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        lbl_cls = self.cls_list[index][0]

        img_file, lbl = self.img_list[index], self.cls2idx[lbl_cls]
        img = PIL.Image.open(os.path.join(self.root, str(lbl_cls), img_file)).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.array(img)

        onehot = np.zeros(len(self.cls))
        lbl = np.array(lbl) - 1
        onehot[lbl] = 1


        if self._transform:
            return self.transform(img, onehot)
        else:
            return img, onehot


    def transform(self, img, lbl):
        img = img.astype(np.float64) / 255.0
        lbl = lbl.astype(np.float32)
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify #256*256*3 to 3*256*256
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

class ImageNetClsData(data.Dataset):
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False, resize=256):
        super(ImageNetClsData, self).__init__()
        self.root = root
        self._transform = transform
        self.resize = resize
        txts = os.listdir(os.path.join(self.root, 'data/det_lists'))
        txts = filter(lambda x: x.startswith('train_pos') or x.startswith('train_part'), txts)
        file2lbl = {}
        for txt in txts:
            files = open(os.path.join(self.root, 'data/det_lists', txt)).readlines()
            for f in files:
                f = f.strip('\n')+'.JPEG'
                if f in file2lbl:
                    file2lbl[f] += [int(txt.split('.')[0].split('_')[-1])]
                else:
                    file2lbl[f] = [int(txt.split('.')[0].split('_')[-1])]
        self.file2lbl = file2lbl.items()
        self.file2lbl = list(self.file2lbl)

    def __len__(self):
        return len(self.file2lbl)

    def __getitem__(self, index):
        # load image

        # print(len(self.file2lbl))
        img_file, lbl = self.file2lbl[index]
        img = PIL.Image.open(os.path.join(self.root, 'ILSVRC2014_DET_train', img_file)).convert('RGB')
        img = img.resize((self.resize, self.resize))
        img = np.array(img)
        onehot = np.zeros(200)
        lbl = np.array(lbl)-1
        onehot[lbl] = 1

        if self._transform:
            return self.transform(img, onehot)
        else:
            return img, onehot

    def transform(self, img, lbl):
        img = img.astype(np.float64)/255.0
        lbl = lbl.astype(np.float32)
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify #256*256*3 to 3*256*256
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl