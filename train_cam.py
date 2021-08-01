import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as f
import importlib
import dataset_loader
from functions import pyutils, torchutils
from tensorboardX import SummaryWriter

writer = SummaryWriter('log')
cudnn.enabled = True
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'Net')()

    train_data = DataLoader(dataset_loader.ImageNetClsData(args.cls_dataset_dir, transform=True,
                                                           resize=args.cam_crop_size), batch_size=args.cam_batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)

    max_step = 20000  # (len(train_data_dut) // args.cam_batch_size) * args.cam_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.cam_num_epoches):
        print('Epoch %d/%d' % (ep + 1, args.cam_num_epoches))
        for step, pack in enumerate(train_data):

            img = pack[0].cuda()
            label = pack[1].cuda(non_blocking=True)

            x, cam = model(img)

            loss = f.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 10 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                # print the training parameter on the terminal
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.5f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

                # Visualize the loss & image & cam during the training process on TensorBoardX
                writer.add_scalar('cam loss', loss, step)
                valid_cat = torch.nonzero(label[0])[:, 0]

                single = cam[valid_cat]
                single /= f.adaptive_max_pool2d(single, (1, 1)) + 1e-5
                single = torch.sum(single, 0)
                size = list(img.size())
                single = f.interpolate(torch.unsqueeze(torch.unsqueeze(single, 0), 0), (size[2], size[3]),
                                       mode='bilinear', align_corners=False)

                img = (img[0][0] * std[0] + mean[0]) * 0.299 + (img[0][1] * std[1] + mean[1]) * 0.587 + (
                        img[0][2] * std[2] + mean[2]) * 0.114
                img = img.unsqueeze(0).unsqueeze(0).cuda()
                image = torch.cat((img, single), 0)
                writer.add_images('the results', image, step, dataformats='NCHW')

            if optimizer.global_step % 2000 == (2000 - 1):
                torch.save(model.module.state_dict(), 'ckpt/Cam_%d.pth' % (optimizer.global_step + 1))
                print('save: (snapshot:%3d)' % step)

    # torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    # torch.cuda.empty_cache()
