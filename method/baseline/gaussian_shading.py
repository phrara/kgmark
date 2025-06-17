import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc


class GaussianShading:
    def __init__(self, ch_factor=4, hw_factor=8, fpr=0.01, user_number=1):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        random_matrix = torch.rand(1, 4, 64, 64)
        self.fixed_mask = random_matrix < 0.015
        print((self.fixed_mask == True).sum())
        self.mask = self.fixed_mask.flatten().cpu().numpy()
        
        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message, z_T_inv: torch.Tensor):
        z = z_T_inv.flatten().cpu().numpy()
        self.orginal_z = np.copy(z)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            if self.mask[i] == True:
                z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
            self.orginal_z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        self.orginal_z = torch.from_numpy(self.orginal_z).reshape(1, 4, 64, 64).cuda()   
        z = torch.from_numpy(z).reshape(1, 4, 64, 64)
        return z.cuda()

    def create_watermark_and_return_w(self, z_T_inv: torch.Tensor):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        if self.watermark is None:
            self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m, z_T_inv)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        # self.orginal_z[self.fixed_mask] = reversed_m[self.fixed_mask].clone()
        # reversed_m = self.orginal_z.reshape(1, 4, 64, 64)
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count