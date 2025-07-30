import torch 
import torch.nn as nn 

# Implements Causal(Masked) Multiheaded Self-Attention 
class Attention(nn.Module):
    def __init__(self, d_in, d_out, context_length,dropout,num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out 
        self.num_heads = num_heads 
        self.head_dim = d_out // num_heads
        self.Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x, attention_mask=None): 
        batch_size, seq_len, d_in = x.shape
        keys = self.K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        queries = self.Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        values = self.V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask[:seq_len, :seq_len], float('-inf')) 
        if attention_mask is not None:
            attn_scores.masked_fill_(attention_mask[:, None, None, :] == 0, float('-inf'))
        attn_weights = (attn_scores/self.head_dim**0.5).softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights @ values
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)
        out = self.out_proj(context)
        return out




