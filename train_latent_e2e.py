import argparse
import math
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from PIL import Image
import numpy as np
#import thop
#from ptflops import get_model_complexity_info
#from torchstat import stat
#from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import thop
from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM, ConvTransBlock
from torch.utils.tensorboard import SummaryWriter
import os

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def activation(x):
    return torch.exp(F.softsign(x))

class map_contextual(nn.Module):
    def __init__(self,lc):
        super(map_contextual, self).__init__()
        self.lc = int(lc)
        self.lmd_map = nn.Parameter(torch.ones((1, self.lc, 1, 1)))
        self.latent_attn = ConvTransBlock(self.lc//2, self.lc//2, 32, 4, 0, 'SW')
        self.alpha = nn.Parameter(torch.randn(1))
        nn.init.constant_(self.alpha,0)

    def forward(self, x):
        rd_info = self.latent_attn(x)
        return x*activation(self.lmd_map + self.alpha*rd_info)

class map(nn.Module):
    def __init__(self,lc):
        super(map, self).__init__()
        self.lc = int(lc)
        self.lmd_map = nn.Parameter(torch.ones((1, self.lc, 1, 1)))

    def forward(self, x):
        return x*activation(self.lmd_map)

class latent_e2e(nn.Module):
    def __init__(self,lc,contextual=True):
        super(latent_e2e, self).__init__()
        if contextual:
            self.g_a = map_contextual(lc)
            self.g_s = map_contextual(lc)
        else:
            self.g_a = map(lc)
            self.g_s = map(lc)

    def forward(self, x):
        latent = self.g_a(x)
        x_hat = self.g_s(latent)
        return latent, x_hat

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

def adjust_learning_rate(optimizer, epoch, init_lr):
    if epoch < 50:
        lr = init_lr
    elif epoch < 100:
        lr = init_lr*0.5
    elif epoch < 200:
        lr = init_lr*0.1
    else:
        lr = init_lr * 0.05

    try:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    except:
        1
    return lr

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def train_one_epoch(
    model, high, low, train_dataloader, optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device
    mse_func = nn.MSELoss()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            high_latent = high.g_a(d)
            low_latent = low.g_a(d)

        latent, x_hat = model(high_latent)
        loss_high = mse_func(x_hat, high_latent)
        loss_low = mse_func(latent, low_latent)
        loss = loss_high + loss_low

        loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss_high: {loss_high.item():.3f} |'
                f'\tLoss_low: {loss_low.item():.3f} |'
            )

def test_latent_e2e(epoch, test_dataloader, model, high, low):
    model.eval()
    device = next(model.parameters()).device
    quality_high = AverageMeter()
    quality_low = AverageMeter()
    mse_func = nn.MSELoss()
    print(f"Test epoch {epoch}:")
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            high_latent = high.g_a(d)
            low_latent = low.g_a(d)
            latent, x_hat = model(high_latent)
            loss_high = mse_func(x_hat, high_latent)
            loss_low = mse_func(latent, low_latent)

            quality_high.update(-10.*math.log10(loss_high))
            quality_low.update(-10.*math.log10(loss_low))

        print(
            f"High PSNR: {quality_high.avg:.2f}dB, Low PSNR: {quality_low.avg:.2f}dB"
        )

def test_vbr(epoch, test_dataloader, model, high):
    model.eval()
    device = next(model.parameters()).device
    bpp = AverageMeter()
    psnr = AverageMeter()
    mse_func = nn.MSELoss()
    print(f"Test epoch {epoch}:")
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            b,_,h,w = d.size()
            num_pixels = b*h*w
            out_net = high.forward_latent_e2e(d, model)
            bpp_loss = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
            mse_loss = mse_func(out_net["x_hat"], d)
            bpp.update(bpp_loss)
            psnr.update(-10. * math.log10(mse_loss))


        print(
            f"bpp: {bpp.avg:.3f} psnr: {psnr.avg:.2f}dB"
        )


def save_checkpoint(state, filename):
    torch.save(state, filename + ".pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/backup5/zqge/flicker', help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=300,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=32,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--batch_size", '-bs', type=int, default=12, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--type", type=str, default='mse', help="loss type", choices=['mse', "ms-ssim"])
    parser.add_argument("--save_path", type=str, default='./vbr_models/', help="save_path")
    parser.add_argument(
        "--skip_epoch", type=int, default=0
    )
    parser.add_argument(
        "--N", type=int, default=64,
    )

    parser.add_argument(
        "--continue_train", action="store_true", default=False
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    print(type)

    save_path = args.save_path + 'latent_e2e'

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    #'/backup5/zqge/DIV2K_train_LR_bicubic/X2/'
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder('/backup5/zqge/Kodak/kodak', split="", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    #device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = latent_e2e(320, contextual=False)
    net = net.to(device)

    parameters = {
        n
        for n, p in net.named_parameters()
        if p.requires_grad
    }
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    high = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    checkpoint = torch.load("/backup5/zqge/pretrained_small/6.pth.tar", map_location=device)
    high.load_state_dict(checkpoint["state_dict"])
    high = high.to(device)
    low = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    checkpoint = torch.load("finetune5.pth.tar", map_location=device)
    low.load_state_dict(checkpoint["state_dict"])
    low = low.to(device)
    del checkpoint

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])

        del checkpoint


    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        print(f'Training with {torch.cuda.device_count()} GPUs.')

    test_latent_e2e(0, test_dataloader, net, high, low)

    for epoch in range(last_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.learning_rate)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        t0 = time.time()
        train_one_epoch(
            net,
            high,
            low,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
        )

        print('time of this epoch: '+str(time.time()-t0))

        test_latent_e2e(epoch, test_dataloader, net, high, low)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )



if __name__ == "__main__":
    main(sys.argv[1:])