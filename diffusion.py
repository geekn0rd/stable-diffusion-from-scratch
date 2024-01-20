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


class UNet_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_norm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)

    def forward(self, features: torch.Tensor, time: torch.Tensor) -> torch.Tensor:

        residue = features

        features = self.group_norm(features)

        features = F.silu(features)

        features = self.conv(features)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = features + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.group_norm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
    

class UNet_AttentionBlock(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        pass


class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:

        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.encoders = nn.Module([

            # (4, H / 8, W / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)),

            # (320, H / 8, W / 8) - >   (320, H / 16, W / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)),
            SwitchSequential(UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)),

            # (640, H / 16, W / 16) - >   (640, H / 32, W / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)),

            # (1280, H / 32, W / 32) - >   (1280, H / 64, W / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            SwitchSequential(UNet_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),

            UNet_AttentionBlock(8, 160),

            UNet_ResidualBlock(1280, 1280),

        )

        self.decoders = nn.Module([

            # (1280, H / 64, W / 64)
            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)),

            SwitchSequential(UNet_ResidualBlock(1920, 1280), UNet_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)),

            SwitchSequential(UNet_ResidualBlock(960, 640), UNet_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),

            SwitchSequential(UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)),
        ])


class UNet_OutputLayer(nn.Module):

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.group_norm = nn.GroupNorm(32, in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            x = self.group_norm(x)

            x = F.silu(x)

            x = self.conv(x)

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

