import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import TCM, TCM_VBR
import warnings
import torch
import copy
import os
import sys
import math
import argparse
import time
import warnings
from pytorch_msssim import ms_ssim
from PIL import Image
from torch import nn
from torch.autograd import Variable

warnings.filterwarnings("ignore")
import torch.optim as optim
import matplotlib.pyplot as plt

print(torch.cuda.is_available())


def distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2)).item()


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return -10 * math.log10(1 - ms_ssim(a, b, data_range=1.).item())


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--lmd_index",
        '-l',
        default=0,
        type=int,
    )
    parser.add_argument(
        "--total_steps",
        '-t',
        default=20,
        type=int,
    )
    parser.add_argument("--checkpoint", type=str, default='vbr_models/modnet.pth.tar',
                        help="Path to a checkpoint")
    parser.add_argument("--data", '-d', type=str, default='/backup5/zqge/Kodak/kodak', help="Path to dataset")
    parser.add_argument(
        "--real", action="store_true", default=False
    )
    parser.set_defaults(real=False)
    args = parser.parse_args(argv)
    return args


def blocked_update(net, x, lmd):
    args = parse_args(sys.argv[1:])
    func = nn.MSELoss()
    device = 'cuda:0'
    s= time.time()
    num_pixels = x.size(2)*x.size(3)
    with torch.no_grad():
        torch.cuda.synchronize()
        y = net.g_a(x)
        lmd_index = torch.ones((1, 1)).to(device)
        lmd_index *= lmd
        lmd_set = [0.0025, 0.0035, 0.0067, 0.0130, 0.0250, 0.05]
        lmd = lmd_set[lmd]

        y = net.modnet_ga(y, lmd_index)
        z = net.h_a(y)

        y = Variable(y, requires_grad=True)
        z = Variable(z, requires_grad=True)

        lr = 800.
        # lr = 1e-4
        best_loss = 100.
        last_loss = 100.
        best_y = y + 0.
        best_z = z + 0.

    total_steps = args.total_steps
    for stp in range(total_steps):

        lr *= 0.99
        opt = optim.SGD([y, z], lr=lr)
        opt.zero_grad()
        out_net = net.forward_fromlatent(y, z, lmd_index)
        bpp = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_net["likelihoods"].values()
        )
        mse = func(x, out_net["x_hat"])
        loss = bpp + lmd * 255 * 255 * mse
        loss.backward()

        opt.step()
        #print(stp, lr, loss.item(), best_loss)

        if stp == 0:
            loss0 = loss.item() + 0.
            bpp0 = bpp + 0.
        if loss.item() < best_loss:
            best_loss = loss.item() + 0.
            best_y = y + 0.
            best_z = z + 0.

        if loss.item() == last_loss:
            total_steps = stp
            break
        if loss.item() > last_loss:
            lr *= 0.5

        last_loss = loss.item() + 0.

    bdrate = (best_loss - loss0) / bpp0
    #print(f'BD-rate after {total_steps:d} steps: {bdrate * 100:.2f}%')

    out_net = net.test_fromlatent(best_y, best_z, lmd_index)
    out_net['x_hat'].clamp_(0, 1)

    rate = sum(len(s[0]) for s in out_net["strings"]) * 8.0
    return rate, out_net['x_hat'], bdrate

def main(argv):
    args = parse_args(argv)
    p = 128
    func = nn.MSELoss()
    path = args.data
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    if args.cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'
    net = TCM_VBR(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    net = net.to(device)

    data_dir = args.data.split('/')[-1]


    net.eval()
    net.requires_grad_(False)
    count = 0
    PSNR = 0
    Bit_rate = 0
    MS_SSIM = 0
    BD_RATE = 0


    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        dictory = {}
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        net.load_state_dict(dictory)

    if args.real:
        net.update()
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
            x = img.unsqueeze(0)
            x_padded, padding = pad(x, p)
            count += 1
            with torch.no_grad():
                if args.cuda:
                    torch.cuda.synchronize()
                s = time.time()
                out_enc = net.compress(x_padded)
                out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
                if args.cuda:
                    torch.cuda.synchronize()
                e = time.time()
                total_time += (e - s)
                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                print(f'MS-SSIM: {compute_msssim(x, out_dec["x_hat"]):.2f}dB')
                print(f'PSNR: {compute_psnr(x, out_dec["x_hat"]):.2f}dB')
                Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                PSNR += compute_psnr(x, out_dec["x_hat"])
                MS_SSIM += compute_msssim(x, out_dec["x_hat"])

    else:
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)
            H,W = x_padded.size(2), x_padded.size(3)
            count += 1
            num_pixels = x.size(0) * x.size(2) * x.size(3)

            rate, rec, bdrate = blocked_update(net, x_padded, args.lmd_index)
            bpp = rate / num_pixels
            x_rec = crop(rec, padding)
            psnr = compute_psnr(x, x_rec)

            PSNR += psnr
            MS_SSIM += compute_msssim(x, x_rec)
            Bit_rate += bpp
            BD_RATE += bdrate

    PSNR = PSNR / count
    MS_SSIM = MS_SSIM / count
    Bit_rate = Bit_rate / count
    BD_RATE = BD_RATE / count
    print(f'average_PSNR: {PSNR:.4f}dB')
    print(f'average_MS-SSIM: {MS_SSIM:.4f}')
    print(f'average_Bit-rate: {Bit_rate:.4f} bpp')
    print(f'average_BD-RATE: {BD_RATE * 100:.2f}%')
    


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
