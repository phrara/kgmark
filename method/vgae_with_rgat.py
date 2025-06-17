from math import e
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import VGAE, RGATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super(Encoder, self).__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv_mu = RGATConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv_logvar = RGATConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.linear_mu = nn.Linear(hidden_channels, out_channels)
        self.linear_logvar = nn.Linear(hidden_channels, out_channels)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.dp = nn.Dropout()
        self.dp1 = nn.Dropout()

    def forward(self, x, edge_index, edge_type):
        
        hidden = F.relu(self.conv1(x, edge_index, edge_type))
        
        mu = self.conv_mu(hidden, edge_index, edge_type)
        mu = self.act1(mu)
        mu = self.dp(mu)
        mu = self.linear_mu(mu)
        
        logvar = self.conv_logvar(hidden, edge_index, edge_type)
        logvar = self.act2(logvar)
        logvar = self.dp1(logvar)
        logvar = self.linear_logvar(logvar)
        
        logstd = 0.5 * logvar
        return mu, logstd

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, in_channels, num_relations):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(out_channels, hidden_channels)
        self.conv_hidden = RGATConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv_x = RGATConv(hidden_channels, in_channels, num_relations=num_relations)
        self.dp = nn.Dropout()
        

    def forward(self, z, edge_index, edge_type):
        
        z = self.linear(z)
        hidden = F.relu(self.conv_hidden(z, edge_index, edge_type))
        hidden = self.dp(hidden)
        recon_x = self.conv_x(hidden, edge_index, edge_type)
        return recon_x

class VGAEWithRGAT(VGAE):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        encoder = Encoder(in_channels, hidden_channels, out_channels, num_relations)
        decoder = Decoder(out_channels, hidden_channels, in_channels, num_relations)
        super(VGAEWithRGAT, self).__init__(encoder, decoder)

    def encode_graph(self, x, edge_index, edge_type):
        return self.encode(x, edge_index, edge_type)
    
    def decode_graph(self, z, edge_index, edge_type):
        return self.decoder(z, edge_index, edge_type)
    
    def forward(self, x, edge_index, edge_type):
        z = self.encode(x, edge_index, edge_type)
        recon_x = self.decode(z, edge_index, edge_type)
        return z, recon_x

    def recon_loss(self, recon_x, x):
        mse_x = F.mse_loss(recon_x, x)
        return mse_x

    def total_loss(self, recon_x, x, current_epoch, total_epochs):
        recon_loss = self.recon_loss(recon_x, x)
        kl_loss = self.kl_loss()
        kl_weight = min(1.0, current_epoch / total_epochs)
        return recon_loss + 0.1 * kl_weight * kl_loss


def reform_latent(z, num_nodes, ddim_in_channels, diffusion_latent_size=(64, 64)):
    z = z.reshape(-1, ddim_in_channels, diffusion_latent_size[0]*diffusion_latent_size[1])
    z = z.reshape(-1, diffusion_latent_size[0]*diffusion_latent_size[1])
    z = z[:num_nodes]
    return z

def adapt_latent(z, ddim_in_channels, diffusion_latent_size=(64, 64)) -> tuple[torch.Tensor, int]:
    z, batch_size = padding_latent(z, ddim_in_channels)
    z = z.reshape(-1, ddim_in_channels, z.shape[1])
    z = z.reshape(z.shape[0], ddim_in_channels, diffusion_latent_size[0], diffusion_latent_size[1])
    return z, batch_size

def padding_latent(z, ddim_in_channels) -> tuple[torch.Tensor, int]:
    num_padding_row = int(
        (z.shape[0] / ddim_in_channels if z.shape[0] % ddim_in_channels == 0 else z.shape[0] // ddim_in_channels + 1) 
        * ddim_in_channels - z.shape[0])
    for _ in range(num_padding_row):
        z = F.pad(z, (0, 0, 0, 1), mode='constant', value=0)
    return z, int(z.shape[0] / ddim_in_channels)