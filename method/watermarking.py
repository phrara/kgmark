import scipy
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import *

from .diffusion import SimpleDDIMScheduler, device

def get_watermarking_signature(batch_size, shape=(4, 64, 64)):
    signature = torch.randn(*shape, requires_grad=False)
    signature_fft  = torch.fft.fft2(signature)

    return torch.stack([signature_fft]*batch_size).to(device)



def get_watermarking_mask(inversed_steps_latents: torch.Tensor, 
                          sample_steps_latents: torch.Tensor,
                          signature_fft: torch.Tensor, 
                          sample_steps=[5, 10, 15], 
                          threshold=0.91,
                          num_learning_epoch=50,
                          lr=0.02):

    
    
    x_list = sample_steps_latents[torch.tensor([i-1 for i in sample_steps])].requires_grad_(False)
    y = inversed_steps_latents[
        torch.tensor(([inversed_steps_latents.shape[0]-1]*len(sample_steps))) 
        - 
        torch.tensor(sample_steps)
    ].requires_grad_(False)
    
    z = signature_fft.clone().requires_grad_(False)
    p = torch.rand(x_list[0].shape, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([p], lr=lr)

    for step in tqdm(range(num_learning_epoch)):
        loss = 0
        
        for i, x in enumerate(x_list):     
            
            X = torch.fft.fft2(x)
            mask = torch.sigmoid(p)
            X_prime = mask * z + (1 - mask) * X
            x_prime = torch.fft.ifft2(X_prime).real

            loss += F.mse_loss(x_prime+(z*mask).real, y[i])

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # print(f"Step {step}, Loss: {loss.item()}")

    final_mask = (torch.sigmoid(p) > threshold)
    return final_mask





def embed_watermark(last_inverse_latent, watermarking_mask, signature_fft):
    last_inverse_latent = torch.fft.fft2(last_inverse_latent)
    last_inverse_latent[watermarking_mask] = signature_fft[watermarking_mask].clone()
    watermarked_latents = torch.fft.ifft2(last_inverse_latent).real
    
    return watermarked_latents


def extract_watermark(last_inversed_w_latent, watermarking_mask, signature_fft):
    last_inversed_w_latent_fft = torch.fft.fft2(last_inversed_w_latent)
    v = (
            torch.abs(last_inversed_w_latent_fft[watermarking_mask] - signature_fft[watermarking_mask])
        ).mean().item() / signature_fft.shape[0]
    return v

def get_p_value(last_inversed_w_latent, watermarking_mask, signature_fft):
    last_inversed_w_latent_fft = torch.fft.fft2(last_inversed_w_latent)[watermarking_mask].flatten()
    signature_fft = signature_fft[watermarking_mask].flatten()
    signature_fft = torch.concatenate([signature_fft.real, signature_fft.imag])
    
    last_inversed_w_latent_fft = torch.concatenate(
        [last_inversed_w_latent_fft.real, last_inversed_w_latent_fft.imag]
    )
    
    sigma_w = last_inversed_w_latent_fft.std()
    lambda_w = (signature_fft ** 2 / sigma_w ** 2).sum().item()
    x_w = (((last_inversed_w_latent_fft - signature_fft) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(signature_fft), nc=lambda_w)
    
    return p_w