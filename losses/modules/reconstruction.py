import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def gaussian(window_size, sigma, device=None, dtype=torch.float32):
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0: x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size, sigma, force_even=False, device=None, dtype=torch.float32):
    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d = gaussian(kernel_size, sigma, device, dtype)
    return window_1d

def get_gaussian_kernel2d(kernel_size, sigma, force_even=False, device=None, dtype=torch.float32):
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, force_even, device, dtype)
    kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, force_even, device, dtype)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise AssertionError(img.shape)
    return img[:, :, :-1, :] - img[:, :, 1:, :]


class ReconstrLoss(nn.Module):

    def __init__(self, alpha=0.5, window_size=3) -> None:
        super(ReconstrLoss, self).__init__()
        
        self.alpha = alpha
        self.window_size = window_size

    def forward(self, img1, img2, reduction='none'):
        device = img1.device

        img1_dx, img1_dy = _gradient_x(img1), _gradient_y(img1)
        img2_dx, img2_dy = _gradient_x(img2), _gradient_y(img2)

        loss_img = F.smooth_l1_loss(img1, img2, reduction='none')
        loss_gradx = F.smooth_l1_loss(img1_dx, img2_dx, reduction='none')
        loss_grady = F.smooth_l1_loss(img1_dy, img2_dy, reduction='none')

        kernel = get_gaussian_kernel2d(
            (self.window_size, self.window_size), (1.5, 1.5)
        ).unsqueeze(0).unsqueeze(0).to(device)
        kernel = kernel.repeat(3, 1, 1, 1)
        gauss_loss_img = F.conv2d(
            loss_img, kernel, groups=3, padding=self.window_size//2).mean(1, keepdim=True)
        gauss_loss_gradx = F.conv2d(
            loss_gradx, kernel, groups=3, padding=self.window_size//2).mean(1, keepdim=True)
        gauss_loss_grady = F.conv2d(
            loss_grady, kernel, groups=3, padding=self.window_size//2).mean(1, keepdim=True)

        gauss_loss_gradx = F.pad(gauss_loss_gradx, [0, 1, 0, 0], mode='replicate')
        gauss_loss_grady = F.pad(gauss_loss_grady, [0, 0, 0, 1], mode='replicate')

        loss = (1 - self.alpha) * gauss_loss_img + self.alpha * (gauss_loss_gradx + gauss_loss_grady)

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "sum":
            return torch.sum(loss)
        elif reduction == "none":
            return loss
