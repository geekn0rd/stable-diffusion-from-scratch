import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, d_embed))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        
        x = self.token_embedding(tokens)

        x += self.positional_embedding(x)

        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, d_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_embed)
        self.attention = SelfAttention(n_heads, d_embed)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (seq_len, dim_embed)
        residue = x

        # Self-Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residue

        # Feedforward layer
        residue = x

        x = self.layernorm_2(x)

        # (seq_len, dim_embed) ->  (seq_len, 4 * dim_embed)
        x = self.linear_1(x) 
        
        # QuickGELU
        x = x * torch.sigmoid(1.702 * x) 

        # (seq_len, 4 * dim_embed) ->  (seq_len, dim_embed)
        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # seq_len -> (seq_len, embed_dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)

        return output
    
