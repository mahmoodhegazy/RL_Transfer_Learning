import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedRepresentationNetwork(nn.Module):
    """Network that learns a shared representation space across environments."""
    
    def __init__(self, discrete_input_dim, continuous_input_dim, shared_dim):
        super().__init__()
        
        # Encoder for discrete environment
        self.discrete_encoder = nn.Sequential(
            nn.Linear(discrete_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )
        
        # Encoder for continuous environment
        self.continuous_encoder = nn.Sequential(
            nn.Linear(continuous_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )
        
        # Decoders to reconstruct each domain (for training)
        self.discrete_decoder = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, discrete_input_dim)
        )
        
        self.continuous_decoder = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.ReLU(),
            nn.Linear(256, continuous_input_dim)
        )
        
    def encode_discrete(self, x):
        return self.discrete_encoder(x)
    
    def encode_continuous(self, x):
        return self.continuous_encoder(x)
    
    def decode_discrete(self, z):
        return self.discrete_decoder(z)
    
    def decode_continuous(self, z):
        return self.continuous_decoder(z)
    
    def align_representations(self, discrete_batch, continuous_batch):
        """Train to align representations across domains."""
        # Encode both domains
        discrete_z = self.encode_discrete(discrete_batch)
        continuous_z = self.encode_continuous(continuous_batch)
        
        # Reconstruction loss for each domain
        discrete_recon = self.decode_discrete(discrete_z)
        continuous_recon = self.decode_continuous(continuous_z)
        
        recon_loss_discrete = F.mse_loss(discrete_recon, discrete_batch)
        recon_loss_continuous = F.mse_loss(continuous_recon, continuous_batch)
        
        # Domain alignment loss (minimize distance between distributions)
        alignment_loss = self._compute_alignment_loss(discrete_z, continuous_z)
        
        # Total loss
        total_loss = recon_loss_discrete + recon_loss_continuous + alignment_loss
        return total_loss
    
    def _compute_alignment_loss(self, discrete_z, continuous_z):
        """Compute loss to align representation distributions."""
        # This could use MMD, Wasserstein distance, or other distribution matching
        # For simplicity, using mean and variance matching
        discrete_mean = discrete_z.mean(dim=0)
        continuous_mean = continuous_z.mean(dim=0)
        
        discrete_var = discrete_z.var(dim=0)
        continuous_var = continuous_z.var(dim=0)
        
        mean_loss = F.mse_loss(discrete_mean, continuous_mean)
        var_loss = F.mse_loss(discrete_var, continuous_var)
        
        return mean_loss + var_loss