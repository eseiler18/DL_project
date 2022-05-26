"""metric"""
import torch


def calculate_psnr(denoised, ground_truth):
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10 ** -8)
