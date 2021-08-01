import torch
import cv2
from torch import multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as f
from torch.backends import cudnn
from functions.pamr import BinaryPamr
import importlib
import os
import dataloader
from functions import torchutils, imutils

cudnn.enabled = True


def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    print("generating the cam maps as well as cam_pamr maps ... ")

    with torch.no_grad():

        model.cuda()
        for _, pack in enumerate(data_loader):
            # model.zero_grad
            rgb = pack['img'][0][0][0]
            img_name = pack['name'][0]
            size = pack['size']
            strided_size = imutils.get_strided_size(size, 4)
            size = (int(size[0].item()), int(size[1].item()))
            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [f.interpolate(torch.unsqueeze(o[1], 0), strided_size, mode='bilinear',
                               align_corners=False)[0] for o in outputs]), 0)

            cls = (outputs[0][0] + outputs[1][0] + outputs[2][0] + outputs[3][0])/4
            cls = (cls[0] + cls[1])/2
            sig = nn.Sigmoid()
            cls = sig(cls).view(200, 1, 1)
            cam = strided_cam * cls
            cam = torch.sum(cam, 0).unsqueeze(0)
            cam /= f.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

            rgb = f.interpolate(rgb.unsqueeze(0), size, mode='bilinear')
            cam = f.interpolate(cam.unsqueeze(0), size, mode='bilinear')
            cam_pamr = BinaryPamr(rgb.cuda(), cam, binary=None)
            cam_pamr = f.interpolate(cam_pamr, size, mode='bilinear')
            cam_pamr = cam_pamr.squeeze().cpu().numpy()*255.0
            cv2.imwrite(os.path.join(args.cam_pamr_out_dir, img_name + '.png'), cam_pamr)

            cam = cam.squeeze().cpu().numpy()*255.0
            cv2.imwrite(os.path.join(args.cam_out_dir, img_name + '.png'), cam)


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.load_state_dict(torch.load(args.cam_weights_dir), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()
    dataset = dataloader.VOC12ClassificationDatasetMSF(scales=args.cam_scales, root=args.sal_dataset_dir)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)

    torch.cuda.empty_cache()
