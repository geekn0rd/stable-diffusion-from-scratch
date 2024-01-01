import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super.__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VAE_ResBlock(128, 128),
            VAE_ResBlock(128, 128),

            # (H, W) -> (H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResBlock(128, 256),
            VAE_ResBlock(256, 256),

            # (H/2, W/2) -> (H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResBlock(256, 512),
            VAE_ResBlock(512, 512),

            # (H/4, W/4) -> (H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            
            VAE_AttentionBlock(512),
            VAE_ResBlock(512),
            
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # Bottleneck
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (4, H/8, W/8) - > [(2, H/8, W/8), (2, H/8, W/8)]
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var. -30, 20)
        
        var = log_var.exp()
        std = var.sqrt()

        # N(0, 1) -> N(mu, sigma)
        x = mean + noise * std

        # scale the output
        x *= 0.18215

        return x

