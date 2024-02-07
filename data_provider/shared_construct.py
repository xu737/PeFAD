import numpy as np
from scipy.stats import entropy
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score


def vae_loss(vae, inputs, outputs, z_mean, z_log_var):
    # reconstruction_loss = nn.functional.mse_loss(outputs, inputs, reduction='sum') 
    reconstruction_loss = nn.functional.mse_loss(outputs, inputs) 
    # kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    return reconstruction_loss + kl_loss #+ kl_divergence
    


def wasserstein_distance(input_data, input_size, generated_sequence, epsilon=1e-10):
    """
    Calculate the Wasserstein distance between input samples and generated samples.

    Parameters:
    - input_data: Input samples
    - generated_sequence: Generated samples
    - critic: The discriminator used to evaluate the tightness of sample distributions
    - epsilon: A small value used for stable computation

    Returns:
    - wasserstein_loss: Loss estimated by the Wasserstein distance
    """

    critic = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    critic = critic.to(input_data.device)

    input_scores = critic(input_data)
    generated_scores = critic(generated_sequence)
    
    wasserstein_loss = torch.abs(torch.mean(generated_scores - input_scores))
    
    return wasserstein_loss



def manual_info_los1s(seq1, seq2, epsilon=1e-8):  ##final munal_infomation

    flat_seq1 = seq1.view(-1, seq1.size(-1))
    flat_seq2 = seq2.view(-1, seq2.size(-1))

    p_x = F.softmax(flat_seq1, dim=-1)+ epsilon
    p_y = F.softmax(flat_seq2, dim=-1)+ epsilon

    p_xy = F.softmax(flat_seq1 + flat_seq2, dim=-1)

    p_xy = F.softmax(flat_seq1.unsqueeze(2) + flat_seq2.unsqueeze(1), dim=-1)+ epsilon
    mi = torch.mean(p_xy * torch.log(p_xy / (p_x.unsqueeze(2) * p_y.unsqueeze(1)))) / flat_seq1.size(0)

    loss = torch.abs(mi)
    # loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    return loss

#-------------------------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear(x)) #-> bs,hidden_size
        mu = self.mu(x) #-> bs,latent_size
        sigma = self.sigma(x)#-> bs,latent_size
        return mu,sigma

class Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        return x
    
class VAE_git(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)
    def forward(self, x): #x: bs,input_size
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        eps = torch.randn_like(sigma)  #eps: bs,latent_size
        z = mu + torch.exp(0.5 * sigma) * eps
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,mu,sigma

class VAE(nn.Module):
    def __init__(self, original_dim, latent_dim,device):
        super(VAE, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  
            nn.ReLU(),
            nn.Linear(256, self.latent_dim * 2)  
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, original_dim),
            nn.Sigmoid() 
        )
    def _add_noise(self, data, size, mean, std): #
        if self.noise_type == 'Gaussian':
            rand = torch.normal(mean=mean, std=std, size=size).to(self.device)
        if self.noise_type == 'Laplace':
            rand = torch.Tensor(np.random.laplace(loc=mean, scale=std, size=size)).to(self.device)
        data += rand
        return data

    def sample(self, mean, std):
        batch_size, seq_len, self.latent_dim = mean.size()
        epsilon = torch.randn_like(std)
        return mean + torch.exp(0.5 * std) * epsilon

    def forward(self, x):
        # print("x.shape: ",x.shape)  #[32, 100, 38]
        # z_mean_log_var = self.encoder(x)   
        # x = x.to(self.device)
        z_mean_log_var = self.encoder(x)

        z_mean = z_mean_log_var[:, :, :self.latent_dim]
        z_log_var = z_mean_log_var[:, :, self.latent_dim:] 

        # z = self.sample(z_mean, z_log_var)
        z = self.sample(z_mean, z_log_var) 

        reconstructed = self.decoder(z)  

        # return reconstructed, z_mean, z_log_var
        return reconstructed.to(self.device), z_mean.to(self.device), z_log_var.to(self.device)
