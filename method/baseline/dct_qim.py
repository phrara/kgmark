import torch 
import numpy as np
from .qim_hide import QIMHide
from .qim_dehide import QIMDehide

class DCT_QIM():
    
    def __init__(self):
        self.watermark = np.random.randint(0, 2, 64)
    
    
    
    def embed(self, z_T_inv: torch.Tensor, batch_size=4):
        z_T_inv = z_T_inv.squeeze()
        l = []
        for i in range(batch_size):
            o = QIMHide(z_T_inv[i].cpu().numpy(), self.watermark, 15.5)
            l.append(torch.tensor(o))
        return torch.stack(l).unsqueeze(0).cuda()
    
    def extract(self, z_T_inv: torch.Tensor, batch_size=4):
        z_T_inv = z_T_inv.squeeze()
        l = []
        for i in range(batch_size):
            o = QIMDehide(z_T_inv[i].cpu().numpy(), 15.5, len(self.watermark))
            l.append(self.similar(o, self.watermark))
        return np.array(l).mean()
    
    def similar(self, x, y):
        len1 = min(len(x), len(y))
        x = np.double(x[:len1])
        y = np.double(y[:len1])
        z = np.sum(x * y) / (np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2)))
        return z
    