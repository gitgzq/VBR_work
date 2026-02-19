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

from torch.utils.data import DataLoader
from torchvision import transforms
import thop
from compressai.datasets import ImageFolder
from compressai.zoo import models
from pytorch_msssim import ms_ssim

from models import TCM, TCM_VBR
from torch.utils.tensorboard import SummaryWriter
import os

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

def adjust_learning_rate(optimizer, epoch, init_lr):
    if epoch < 50:
        lr = init_lr
    elif epoch < 100:
        lr = init_lr*0.5
    elif epoch < 150:
        lr = init_lr*0.1
    else:
        lr = init_lr * 0.05

    try:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    except:
        1
    return lr


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, type='mse'):
        super().__init__()
        #self.mse = nn.MSELoss(reduction='none')
        self.type = type

    def forward(self, output, target):
        lmd_set = torch.Tensor([0.0025, 0.0035, 0.0067, 0.0130, 0.0250, 0.05]).cuda().to(torch.float32)
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        lmd_set = lmd_set.repeat(1,N)

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.type == 'mse':
            out["mse_loss"] = torch.mean((output["x_hat"]-target)**2, dim=(1,2,3), keepdim=False)
            out["loss"] = (255**2) * torch.mean(torch.index_select(lmd_set,1,output["lmd_index"].view(N).to(torch.int64))*out["mse_loss"]) + out["bpp_loss"]
            out["mse_loss"] = torch.mean(out["mse_loss"])
        # else: 还没修改
        #     out['ms_ssim_loss'] = compute_msssim(output["x_hat"], target)
        #     out["loss"] = self.lmbda * (1 - out['ms_ssim_loss']) + out["bpp_loss"]

        return out


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

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    print(len(aux_parameters))
    if len(aux_parameters)<2:
        aux_optimizer = None

    return optimizer, aux_optimizer

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse', static=False, pretrained=None
):
    model.train()
    device = next(model.parameters()).device
    mse_func = nn.MSELoss()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)

        L2=0
        norm_list = [model.modnet_ga,model.modnet_ga]
        #sum_params=0
        for k in norm_list:
            for p in k.parameters():
                L2 += torch.norm(p,2)*1e-4
                #sum_params+=1
        #print(sum_params,L2)

        if static:
            with torch.no_grad():
                yq = []
                for q in range(len(pretrained)):
                    yq.append(pretrained[q].g_a(d).unsqueeze(0))
            yq = torch.cat(yq, dim=0).cuda().to(torch.float32)
            _,b,c,h,w = yq.size()

            joint = torch.gather(yq, 0, out_net['lmd_index'].view(1,b,1,1,1).expand(1, b, c, h, w).to(torch.int64))
            latent_loss = mse_func(joint, out_net['latent']) * 0.03
            #print(out_criterion["loss"].item(), L2.item(), latent_loss.item())

            loss = out_criterion["loss"] + L2 + latent_loss
        else:
            loss = out_criterion["loss"]

        loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        if aux_optimizer is not None:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        if i % 100 == 0:
            if type == 'mse':
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    #f"\tAux loss: {aux_loss.item():.2f}"
                )
            else:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    #f"\tAux loss: {aux_loss.item():.2f}"
                )

def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()

    if type == 'mse':
        print(f"Test epoch {epoch}:")
        with torch.no_grad():
            for lmd_index in [0,1,2,3,4,5]:
                bpp = AverageMeter()
                psnr = AverageMeter()
                for d in test_dataloader:
                    d = d.to(device)
                    out_net = model(d, lmd_index)

                    out_criterion = criterion(out_net, d)
                    loss.update(out_criterion["loss"])
                    bpp.update(out_criterion["bpp_loss"])
                    psnr.update(-10.*math.log10(out_criterion["mse_loss"]))

                print(
                    f"Test quality index {lmd_index}: bpp={bpp.avg:.3f} PSNR={psnr.avg:.2f}dB"
                )

    else:
        loss = AverageMeter()

        with torch.no_grad():
            for d in test_dataloader:
                d = d.to(device)
                out_net = model(d)
                out_criterion = criterion(out_net, d)
                loss.update(out_criterion["loss"])

        print(
            f"Test epoch {epoch}: "
            f"\tLoss: {loss.avg:.3f} |"
        )

    return loss.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        torch.save(state, filename + "_best.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='/backup5/zqge/flicker', help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
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
        "--continue_train", action="store_true", default=True
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    type = args.type
    print(type)


    save_path = args.save_path + 'tcm_mse_vbr'


    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )
    #'/backup5/zqge/DIV2K_train_LR_bicubic/X2/'
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

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

    net = TCM_VBR(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    #net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)

    net = net.to(device)

    # x = torch.randn(1, 3, 512, 768).cuda()
    # flops, params = thop.profile(net, inputs=(x,))
    # print((params)/1e6, flops / 1e9)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss(type=type)

    last_epoch = 1
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        if args.continue_train:
            last_epoch = checkpoint["epoch"] + 1
            print(last_epoch)
            optimizer.load_state_dict(checkpoint["optimizer"])
            if aux_optimizer is not None:
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        del checkpoint

    else:
        checkpoint = torch.load('/backup5/zqge/pretrained_small/6.pth.tar', map_location=device)
        baseline = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=args.N,
                       M=320)

        baseline.load_state_dict(checkpoint["state_dict"])
        del checkpoint

        for i, _ in baseline.named_parameters():
            try:
                net.state_dict()[i].copy_(baseline.state_dict()[i])
            except:
                1  # print(i)

        print('initialized with base')
        del baseline

    #stage 1: static learning of latents
    if last_epoch==0:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=14*torch.cuda.device_count(),
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )

        pretrained=[]
        for i in range(6):
            pi = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
            checkpoint = torch.load(f"/backup5/zqge/pretrained_small/{i+1}.pth.tar", map_location=device)
            pi.load_state_dict(checkpoint["state_dict"])
            pretrained.append(pi.to(device))
            del pi, checkpoint

        net.requires_grad_(False)
        net.modnet_ga.requires_grad_(True)
        net.modnet_gs.requires_grad_(True)

        if args.cuda and torch.cuda.device_count() > 1:
            net = CustomDataParallel(net)
            print(f'Training with {torch.cuda.device_count()} GPUs.')

        best_loss = float("inf")
        #test_epoch(0, test_dataloader, net, criterion, type)

        for epoch in range(last_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.learning_rate)
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            t0 = time.time()
            train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                type,
                static=True,
                pretrained=pretrained
            )

            print('time of this epoch: '+str(time.time()-t0))
            loss = test_epoch(epoch, test_dataloader, net, criterion, type)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    },
                    is_best,
                    save_path+'_e2e',
                )

    else:
        net.requires_grad_(True)
        if args.cuda and torch.cuda.device_count() > 1:
            net = CustomDataParallel(net)
            print(f'Training with {torch.cuda.device_count()} GPUs.')

        best_loss = float("inf")


        train_dataloader = DataLoader(
            train_dataset,
            batch_size=9 * torch.cuda.device_count(),
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )

        for epoch in range(last_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch, args.learning_rate)
            print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            t0 = time.time()
            train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                type
            )

            print('time of this epoch: ' + str(time.time() - t0))

            loss = test_epoch(epoch, test_dataloader, net, criterion, type)
            is_best = False #loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    },
                    is_best,
                    save_path,
                )

if __name__ == "__main__":
    main(sys.argv[1:])