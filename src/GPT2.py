import torch 
import torch.nn as nn 
from transformer import Transformer, LayerNorm

# GPT-2 Architecture
class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([Transformer(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, attention_mask=None):
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_emb+pos_emb)
        for block in self.trf_blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x
    
# Load GPT-2 Weights from Huggingface
def load_gpt2_weights(device="cuda"):
    from transformers import GPT2LMHeadModel
    config = {"vocab_size": 50257,"context_length": 1024, "emb_dim": 768,"n_heads": 12,"n_layers": 12, "drop_rate": 0.1,"qkv_bias": True} 
    model = GPT2(config).to(device)
    model_state = model.state_dict()
    weights = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    model_state["tok_emb.weight"] = weights["transformer.wte.weight"]
    model_state["pos_emb.weight"] = weights["transformer.wpe.weight"]
    model_state["final_norm.scale"] = weights["transformer.ln_f.weight"]
    model_state["final_norm.bias"] = weights["transformer.ln_f.bias"]
    model_state["out_head.weight"] = weights["lm_head.weight"]
    for i in range(12):
        model_state[f"trf_blocks.{i}.norm1.scale"] = weights[f"transformer.h.{i}.ln_1.weight"]
        model_state[f"trf_blocks.{i}.norm1.bias"] = weights[f"transformer.h.{i}.ln_1.bias"]
        model_state[f"trf_blocks.{i}.norm2.scale"] = weights[f"transformer.h.{i}.ln_2.weight"]
        model_state[f"trf_blocks.{i}.norm2.bias"] = weights[f"transformer.h.{i}.ln_2.bias"]
        
        model_state[f"trf_blocks.{i}.ff.layers.0.weight"] = weights[f"transformer.h.{i}.mlp.c_fc.weight"].T
        model_state[f"trf_blocks.{i}.ff.layers.0.bias"] = weights[f"transformer.h.{i}.mlp.c_fc.bias"]
        model_state[f"trf_blocks.{i}.ff.layers.2.weight"] = weights[f"transformer.h.{i}.mlp.c_proj.weight"].T
        model_state[f"trf_blocks.{i}.ff.layers.2.bias"] = weights[f"transformer.h.{i}.mlp.c_proj.bias"]

        model_state[f"trf_blocks.{i}.att.out_proj.weight"] = weights[f"transformer.h.{i}.attn.c_proj.weight"].T
        model_state[f"trf_blocks.{i}.att.out_proj.bias"] = weights[f"transformer.h.{i}.attn.c_proj.bias"]

        W_q, W_k, W_v = weights[f"transformer.h.{i}.attn.c_attn.weight"].chunk(3, dim=1)
        model_state[f"trf_blocks.{i}.att.Q.weight"] = W_q.T
        model_state[f"trf_blocks.{i}.att.K.weight"] = W_k.T
        model_state[f"trf_blocks.{i}.att.V.weight"] = W_v.T
        b_q, b_k, b_v = weights[f"transformer.h.{i}.attn.c_attn.bias"].chunk(3, dim=0)
        model_state[f"trf_blocks.{i}.att.Q.bias"] = b_q
        model_state[f"trf_blocks.{i}.att.K.bias"] = b_k
        model_state[f"trf_blocks.{i}.att.V.bias"] = b_v
    model.load_state_dict(model_state)
    return model