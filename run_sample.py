import argparse
import os
from functions import pyutils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=12, type=int)

    # Class Activation Map
    parser.add_argument("--cam_network", default="network.densenet_cam", type=str)
    parser.add_argument("--cam_crop_size", default=256, type=int)
    parser.add_argument("--cam_batch_size", default=8, type=int)
    parser.add_argument("--cam_num_epoches", default=10, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0), help="Multi-scale inferences")

    # Paths
    parser.add_argument("--cls_dataset_dir", default="", type=str, help='the dir of ImageNet')
    parser.add_argument("--sal_dataset_dir", default="", type=str, help='the dir of DUTS-Train')
    parser.add_argument("--cam_weights_dir", default="ckpt/Cam_18000.pth", type=str, help='the dir of trained ckpt')
    parser.add_argument("--cam_out_dir", default="result/cam", type=str, help='the dir of generated cams')
    parser.add_argument("--cam_pamr_out_dir", default="result/cam_pamr", type=str, help='the dir of generated cam-pamr')

    # Steps
    parser.add_argument("--train_cam_pass", default=True, help='training the classification network')
    parser.add_argument("--make_cam_pass", default=True, help='inferring the class activation maps')

    args = parser.parse_args()

    os.makedirs("ckpt", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.cam_pamr_out_dir, exist_ok=True)

    print(vars(args))

    if args.train_cam_pass is True:
        import train_cam
        timer = pyutils.Timer('step.train_cam:')
        train_cam.run(args)

    if args.make_cam_pass is True:
        import make_cam
        timer = pyutils.Timer('step.make_cam:')
        make_cam.run(args)
