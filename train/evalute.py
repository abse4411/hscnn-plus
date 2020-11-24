import numpy as np
import torch
import torch.nn.functional as F

def imgtensor_flatten_sampler(rgb, spe, random_sample_num=None):
    assert rgb.dim() == 3 #[3, 512, 512]
    rgb = rgb.view(rgb.size(0), -1).transpose(0, 1)
    spe = spe.view(spe.size(0), -1).transpose(0, 1)
    if random_sample_num is None: return rgb, spe
    index = np.random.permutation(rgb.size(0))[0:random_sample_num]
    return rgb[index, :], spe[index, :]

def compare_rmse_g(pred, gt, eps = 1e-2):
    assert pred.size() == gt.size() # torch.Size([1, 31, 512, 512])
    pred = pred.contiguous().view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    ret = torch.sqrt(eps + torch.mean((pred - gt) ** 2, dim=1))
    # assert ret.numel() == 1
    return ret.mean()

def compare_rmse(pred, gt, eps=1e-2):
    pred = pred.contiguous().view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    ret = torch.mean(torch.sqrt(eps + (pred - gt) ** 2), dim=1)

    return ret.mean()

def compare_rrmse(pred, gt, eps = 1e-2):
    pred = pred.contiguous().view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    ret = torch.mean( torch.sqrt((pred-gt)**2)/(gt+eps) + eps )
    return ret.mean()

def compare_rrmse_g(pred, gt, eps = 1e-2):
    pred = pred.contiguous().view(pred.size(0), -1)
    gt = gt.view(gt.size(0), -1)
    gt_mean = gt.mean()
    ret = torch.sqrt( eps + torch.mean( (( pred-gt )/gt_mean)**2 ) )
    return ret.mean()

if __name__=="__main__":
    a = torch.randn([1, 31, 512, 512])
    b = torch.randn([1, 31, 512, 512])
    compare_rmse_g(a, b)