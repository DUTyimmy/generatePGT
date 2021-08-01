# generatePGT
generating pseudo labels for MFNet

## Prerequisites
### environment
  - Windows 10
  - Torch 1.8.1
  - CUDA 10.0
  - Python 3.7.4
  - other environment requirment can be found in requirments.txt 

### training data (ImageNet)
you can download ImageNet from this [official website](https://image-net.org/).

### testing datasets
you can download DUTS-Train from this [official website](http://saliencydetection.net/duts/). Only RGB images are used in our MFNet.

## Acknowledge
Thanks to pioneering helpful works:

  - [IRNet](https://github.com/jiwoon-ahn/irn):  Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations, CVPR2019, by Jiwoon Ahn et al.
  - [MSW](https://github.com/zengxianyu/mws/tree/new):  Multi-source weak supervision for saliency detection, CVPR2019, by Yu Zeng et al.
  - [SSSS](https://github.com/visinf/1-stage-wseg):  Single-stage Semantic Segmentation from Image Labels, CVPR2020, by Nikita Araslanov et al.
