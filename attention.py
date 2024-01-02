import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # K, Q, V together
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.d_head = d_embed // n_heads
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:

        input_shape = x.shape
        batch_size, sequence_len, d_embed =  input_shape

        intermid_shape = (batch_size, sequence_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (seq_len, dim) -> (seq_len, num_heads, dim_head) -> (num_heads, seq_len, dim_head)
        q = q.view(intermid_shape).transpose(1, 2)
        k = k.view(intermid_shape).transpose(1, 2)
        v = v.view(intermid_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            
            # Upper triangle matrix
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v

        output = output.transpose(1, 2)
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output