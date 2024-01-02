import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channs: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channs)
        self.attention = SelfAttention(1, channs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        n, c, h, w = x.shape

        # Reshaping to use attention
        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)

        return  x + residue
        

class VAE_ResBlock(nn.Module):
    
    def __init__(self, in_chann: int, out_chann: int):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_chann)
        self.conv_1 = nn.Conv2d(in_chann, out_chann, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_chann)
        self.conv_2 = nn.Conv2d(out_chann, out_chann, kernel_size=3, padding=1)

        if in_chann == out_chann:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_chann, out_chann, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.res_layer(residue)



class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),

            # (512, H/8, W/8) -> (512, H/4, W/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),
            VAE_ResBlock(512, 512),

            # (512, H/4, W/4) -> (512, H/2, W/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            VAE_ResBlock(512, 256),
            VAE_ResBlock(256, 256),
            VAE_ResBlock(256, 256),

            # (512, H/2, W/2) -> (512, H, W)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            VAE_ResBlock(512, 128),
            VAE_ResBlock(128, 128),
            VAE_ResBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLU(),

            #  (512, H, W) -> (3, H, W)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Undo scaling
        x /= 0.18215

        # (4, H/8, W/8)
        for module in self:
            x = module(x)
        
        # (3, H, W)
        return x