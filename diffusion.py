import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.linear_1(x)
        x = F.silu(x)
        
        x = self.linear_2(x)

        # (1, 1280)
        return x


class Diffusion(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNet_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent (4, H / 8, W / 8)
        # context (Seq_Lenm Dim)
        # time:  (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (4, H / 8, W / 8) -> (320 , H / 8, W / 8)
        output = self.unet(latent, context, time)

        # (320, H / 8, W / 8) -> (4 , H / 8, W / 8)
        output = self.final(output)

        return output

