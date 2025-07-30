import os, sys
from tqdm.auto import tqdm
import torch, tiktoken
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from GPT2 import load_gpt2_weights
from generation import greedy_decode, beam_search_decode, top_p_sampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

class DailyDialogDataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_len = 256, pad_token = 50256):
        self.tokenizer = tokenizer 
        self.data = []
        with open(txt_path, encoding="utf-8") as fp:
            for line in tqdm(fp.readlines(), desc="Loading DailyDialog Dataset"):
                res = line.split("__eou__")
                res = [resp.strip() for resp in res if resp.strip()]
                text = f"A:{res[0]} B:{res[1]}\n"
                tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
                if len(tokens) > max_len: continue
                target = torch.roll(tokens, shifts=-1)
                target[-1] = pad_token
                self.data.append((tokens, target))
                if len(res) >=4: 
                    text = f"A:{res[0]} B:{res[1]}\nA:{res[2]} B:{res[3]}\n"
                    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
                    if len(tokens) > max_len: continue
                    target = torch.roll(tokens, shifts=-1)
                    target[-1] = pad_token
                    self.data.append((tokens, target))
                if len(res) >= 6:
                    text = f"A:{res[0]} B:{res[1]}\nA:{res[2]} B:{res[3]}\nA:{res[4]} B:{res[5]}\n"
                    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
                    if len(tokens) > max_len: continue
                    target = torch.roll(tokens, shifts=-1)
                    target[-1] = pad_token
                    self.data.append((tokens, target))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate(batch, pad_token = 50256):
    input_tokens, target = zip(*batch)
    input_tokens = torch.nn.utils.rnn.pad_sequence(input_tokens, batch_first=True, padding_value=pad_token) # type: ignore
    target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad_token) # type: ignore
    attention_mask = (input_tokens != pad_token).long()
    return input_tokens, target, attention_mask

if __name__ == "__main__":
    print("Loading model...")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_gpt2_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=50256)

    model_dir = "models/dialog/"
    inference_only = False
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(os.path.join(model_dir, "dialog_model.pt")):
        inference_only = True
        model.load_state_dict(torch.load(os.path.join(model_dir, "dialog_model.pt"), map_location=device))
        print("Loaded model from", model_dir)

    if not inference_only:
        num_epochs = 2
        writer = SummaryWriter("logs/dialog/")
        print("Loading dataset...")
        dataloader = DataLoader(DailyDialogDataset("data/dailydialog/raw/ijcnlp_dailydialog/dialogues_text.txt", tokenizer, 128), batch_size=16, collate_fn=collate, shuffle=True, num_workers=0)
        print("Training model...")
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            batches = 0
            prog = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
            for step, (input_tokens, target, attention_mask) in prog:
                input_tokens = input_tokens.to(device)
                target = target.to(device)
                attention_mask = attention_mask.to(device)
                optimizer.zero_grad()
                output = model(input_tokens, attention_mask=attention_mask)
                loss = criterion(output.view(-1, output.shape[-1]), target.view(-1))
                loss.backward()
                optimizer.step()
                writer.add_scalar("loss", loss.item(), step + epoch * len(dataloader))
                total_loss += loss.item()
                batches += 1
                prog.set_postfix({"Loss": total_loss / batches})
            torch.save(model.state_dict(), os.path.join(model_dir, "dialog_model.pt"))

    print("Starting inference...")
    text = ""
    ind = 1
    while True:
        prompt = input(">>>")
        if prompt == "exit": break
        # print(f"Prompt- {text+f'A: {prompt} B: '}")
        output = beam_search_decode(model, tokenizer, text+f"A: {prompt} B: ", len(text.split())+50, beam_width=3)
        # print(f"Output- {output}")
        output = output.split('B:')[ind].split('A:')[0].strip()
        print(f">>>>>>{output}")
        text+=f"A: {prompt} B: {output}\n"
        ind+=1
        
