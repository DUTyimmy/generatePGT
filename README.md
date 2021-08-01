# generatePGT
Generating pseudo labels for MFNet.

This can also be served as a pipeline for all weakly supervised salient object detection (WSOD) methods. This code can generate class activation maps (CAMs) as well as two kinds of pseudo labels for WSOD. We sincerely hope that this will contribute to the community.

## Prerequisites
### environment
  - Windows 10
  - Torch 1.8.1
  - CUDA 10.0
  - Python 3.7.4
  - other environment requirment can be found in requirments.txt 

### training dataset (ImageNet)
you can download **ImageNet** dataset from this [official website](https://image-net.org/).

### inference dataset (DUTS-Train RGB image)
you can download **DUTS-Train** dataset from this [official website](http://saliencydetection.net/duts/). Only RGB images are used in our MFNet.

## Training
### Firstly, 
you should set your training and inference dataset root in ```--cls_dataset_dir``` and ```--sal_dataset_dir``` in ```run_sample.py```, respectively.
### Secondly,
setting ```--train_cam_pass``` to True, and run ```run_sample.py```.

## inference
### Firstly, 
setting ```--make_cam_pass``` to True, and run ```run_sample.py```. Here you can get ①CAMs and ②the pixel-wise pseudo labels in root ```./result/```, the latter is .
### Secondly,
Setting your inference dataset root in ```img_root``` in ```run_slic.py```, and run. Here you can get ③superpixel-wise pseudo labels.

## Checkpoint & Maps
### Checkpoint
link: https://pan.baidu.com/s/1G-YHYsfho-rWwMv6VMFT4g.    code: oipw
### Maps: CAMs & pseudo labels
link: .    code: oipw

## Acknowledge
Thanks to pioneering helpful works:

  - [IRNet](https://github.com/jiwoon-ahn/irn):  Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, CVPR2019, by Jiwoon Ahn et al.
  - [MSW](https://github.com/zengxianyu/mws/tree/new):  Multi-source weak supervision for saliency detection, CVPR2019, by Yu Zeng et al.
  - [SSSS](https://github.com/visinf/1-stage-wseg):  Single-stage Semantic Segmentation from Image Labels, CVPR2020, by Nikita Araslanov et al.
