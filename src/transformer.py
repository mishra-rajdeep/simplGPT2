import torch 
import torch.nn as nn
from attention import Attention

# Transformer block
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = Attention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],context_length=cfg["context_length"],dropout=cfg["drop_rate"],num_heads=cfg["n_heads"],qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attention_mask=None):
        shortcut = x 
        x = self.drop(self.att(self.norm1(x), attention_mask=attention_mask))
        x = shortcut + x
        shortcut = x 
        x = self.drop(self.ff(self.norm2(x)))
        x = shortcut + x
        return x
    
class LayerNorm(nn.Module):
    # Subtract mean and normalize to unit variance by dividing by standard deviation
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.bias

class GELU(nn.Module):
    # Gaussian Error Linear Unit
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # return  0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return torch.nn.functional.gelu(x)

class FeedForward(nn.Module):
    # Feedforward layer of transformer
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )
    def forward(self, x):
        return self.layers(x)

