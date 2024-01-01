import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    




class VAE_ResBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_chan)
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(32, out_chan)
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)

        if in_chan == out_chan:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding=0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residue = x

            x = self.groupnorm_1(x)
            x = F.silu(x)

            x = self.conv_1(x)

            x = self.groupnorm_2(x)
            x = F.silu(x)

            x = self.conv_2(x)

            return x + self.res_layer(residue)


