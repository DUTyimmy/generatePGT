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
you can download ImageNet from this [official website](https://image-net.org/).

### inference dataset (DUTS-Train RGB image)
you can download DUTS-Train from this [official website](http://saliencydetection.net/duts/). Only RGB images are used in our MFNet.

## Training
### Firstly, 
you should set your trining and inference dataset root in ```--cls_dataset_dir``` and ```--sal_dataset_dir``` in ```run_sample.py```, respectively.
### Secondly,
setting ```--train_cam_pass``` to True, and run ```run_sample.py```.

## inference
### Firstly, 
setting ```--make_cam_pass``` to True, and run ```run_sample.py```. here you can get *CAMs* & *CAMs+PAMR* in root "result", latter is the semi-finished pixel-wise pseudo label.
### Secondly,

## Acknowledge
Thanks to pioneering helpful works:

  - [IRNet](https://github.com/jiwoon-ahn/irn):  Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, CVPR2019, by Jiwoon Ahn et al.
  - [MSW](https://github.com/zengxianyu/mws/tree/new):  Multi-source weak supervision for saliency detection, CVPR2019, by Yu Zeng et al.
  - [SSSS](https://github.com/visinf/1-stage-wseg):  Single-stage Semantic Segmentation from Image Labels, CVPR2020, by Nikita Araslanov et al.
