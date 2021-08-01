import PIL.Image
import numpy as np
import os.path
from torch.utils import data
from functions import imutils





class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class VOC12ClassificationDatasetMSF(data.Dataset):

    def __init__(self, root, img_normal=TorchvisionNormalize(), scales=(1.0,)):
        super(VOC12ClassificationDatasetMSF, self).__init__()
        self.img_normal = img_normal
        self.scales = scales
        self.root = root
        self.img_names = []
        self.names = []
        img_root = os.path.join(self.root, 'DUTS-TR-Image')  # DUTS-TR-ImageECSSD-image
        file_names = os.listdir(img_root)

        for i, names in enumerate(file_names):
            if not names.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, names)
            )
            self.names.append(names[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        sal = True
        if sal:

            # img =
            img_file = self.img_names[idx]
            img = PIL.Image.open(img_file)
            img = img.convert('RGB')
            img = np.array(img, dtype=np.uint8)
            name_str = self.names[idx]

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))  # why here -> flip to CAM
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1])}
        return out





