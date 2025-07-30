import torch 
import heapq

def greedy_decode(model,tokenizer, src, max_len):
    # Take the argmax at each step of the prediction
    model.eval()
    device = next(model.parameters()).device
    input_tokens = torch.tensor([tokenizer.encode(src)], dtype=torch.long, device=device)
    for _ in range(max_len - input_tokens.shape[1]):
        logits = model(input_tokens)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        input_tokens = torch.cat([input_tokens, next_token], dim=1)
    return tokenizer.decode(input_tokens[0].tolist())
    
def beam_search_decode(model, tokenizer, src, max_len, beam_width=5):
    # Maintain a beam of size beam_width containing sequences with highest sum of log-probabilities
    model.eval()
    device = next(model.parameters()).device
    input_tokens = torch.tensor([tokenizer.encode(src)], dtype=torch.long, device=device)
    beams = [(0.0, input_tokens)]  
    for _ in range(max_len - input_tokens.shape[1]):
        new_beams = []
        for score, seq in beams:
            logits = model(seq)
            next_token_logits = logits[:, -1, :]
            probs = torch.log_softmax(next_token_logits, dim=-1) 
            topk_probs, topk_indices = torch.topk(probs, beam_width, dim=-1)  
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_probs[0, i].item()
                new_beams.append((new_score, new_seq))
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])
    best_seq = beams[0][1]
    return tokenizer.decode(best_seq[0].tolist())

def top_p_sampling(model, tokenizer, src, max_len, p = 0.9 , temp = 1.0):
    # Sample next token with top-p sampling after scaling logits by temperature 
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(src)], dtype=torch.long, device=next(model.parameters()).device)
    for _ in range(max_len - input_ids.shape[1]):
        logits = model(input_ids)[:, -1, :] / temp
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > p
        if torch.any(mask):
            sorted_probs[torch.where(mask)[0][0] + 1 :] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        input_ids = torch.cat([input_ids, sorted_indices[next_token].unsqueeze(0)], dim=1)
    return tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    from GPT2 import load_gpt2_weights
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_gpt2_weights(device="cuda")
    str = input("Enter a string for generation: ")
    print("Greedy Decode: ")
    print(greedy_decode(model, tokenizer, str, 25))
    print("Beam Search Decode: ")
    print(beam_search_decode(model, tokenizer, str, 25, beam_width=5))
    print("Top P Sampling: ")
    print(top_p_sampling(model, tokenizer, str, 25, p = 0.3 , temp = 1.0))