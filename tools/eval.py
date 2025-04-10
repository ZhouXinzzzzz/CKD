import argparse
import torch
import torch.backends.cudnn as cudnn

import os
import sys
# 把上一级目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg

# device_ids = [0, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            # model.load_state_dict(load_checkpoint(args.ckpt)["model"]) #用来测试代码训练出来的
            model.load_state_dict(load_checkpoint(args.ckpt)) #测试下载的
    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        # model.load_state_dict(load_checkpoint(ckpt)["model"]) #用来测试代码训练出来的
        model.load_state_dict(load_checkpoint(ckpt)) #测试下载的

    model = Vanilla(model)
    model = model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = torch.nn.DataParallel(model)
    test_acc, test_acc_top5, test_loss = validate(val_loader, model)


# python3 tools/eval.py -m resnet56 -c /root/nas-public-tju/zhou/DKD/mdistiller-master/download_ckpts/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth
# CUDA_VISIBLE_DEVICES=3 python3 tools/eval.py -d imagenet -m ResNet34 -c /root/.cache/torch/hub/checkpoints/resnet34\-333f7ec4.pth