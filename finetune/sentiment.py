import os , sys
from tqdm import tqdm
import torch, tiktoken
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from GPT2 import load_gpt2_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDBDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = []
        pos = os.path.join(path, "pos")
        neg = os.path.join(path, "neg")
        for f in tqdm(os.listdir(pos), desc="Loading '+' Dataset"):
            with open(os.path.join(pos, f), "r", encoding="utf-8") as fp:
                text = fp.read()
                tokens = tokenizer.encode(text)
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                self.data.append((torch.tensor(tokens, dtype=torch.long), 1))
        for f in tqdm(os.listdir(neg), desc="Loading '-' Dataset"):
            with open(os.path.join(neg, f), "r", encoding="utf-8") as fp:   
                text = fp.read()
                tokens = tokenizer.encode(text)
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                self.data.append((torch.tensor(tokens, dtype=torch.long), 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def collate(batch):
    input_tokens, labels = zip(*batch)
    input_tokens = torch.nn.utils.rnn.pad_sequence(input_tokens, batch_first=True, padding_value=50256) # type: ignore
    labels = torch.tensor(labels)
    attention_mask = (input_tokens != 50256).long()
    return input_tokens, attention_mask, labels

class GPT2SentimentClassifier(torch.nn.Module):
    def __init__(self, dropout = 0.2, num_classes = 2, fine_tune = False):
        super().__init__()
        self.gpt = load_gpt2_weights()
        if not fine_tune:
            for param in self.gpt.parameters():
                param.requires_grad = False
        self.finetune = fine_tune
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.gpt.tok_emb.embedding_dim, num_classes) 
    def forward(self, in_idx, attention_mask = None):
        batch_size, seq_len = in_idx.shape
        if self.finetune:
            tok_emb = self.gpt.tok_emb(in_idx)
            pos_emb = self.gpt.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = self.gpt.drop_emb(tok_emb+pos_emb)
            for block in self.gpt.trf_blocks: x = block(x, attention_mask=attention_mask)
            x = self.gpt.final_norm(x)
        else :
            with torch.no_grad():
                tok_emb = self.gpt.tok_emb(in_idx)
                pos_emb = self.gpt.pos_emb(torch.arange(seq_len, device=in_idx.device))
                x = self.gpt.drop_emb(tok_emb+pos_emb)
                for block in self.gpt.trf_blocks: x = block(x, attention_mask=attention_mask)
                x = self.gpt.final_norm(x)
        if attention_mask is not None: last_token_idx = attention_mask.sum(dim=1) - 1
        else: last_token_idx = seq_len - 1
        last_token = x[torch.arange(batch_size), last_token_idx]
        return self.classifier(self.dropout(last_token))

if __name__ == "__main__":
    print("Loading GPT-2 weights...")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPT2SentimentClassifier(num_classes=2, dropout=0.2, fine_tune=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model_dir = "models/sentiment/"
    test_only = False
    eval_model = False
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(os.path.join(model_dir, "sentiment_model.pt")):
        test_only = True
        model.load_state_dict(torch.load(os.path.join(model_dir, "sentiment_model.pt"), map_location=device, weights_only=True)["model"])
        print("Loaded model from", model_dir)
        
    runs_dir = "logs/sentiment/"
    num_epochs = 3

    writer = SummaryWriter("logs/sentiment")
    if not test_only:
        print("Loading IMDB dataset...")
        train_dataset = IMDBDataset("data/sentiment/aclImdb/train", tokenizer, max_len=256)
        dataloader = DataLoader(train_dataset, collate_fn=collate, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        print("Training model...")
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            batches = 0
            prog = tqdm(dataloader)
            for batch in prog:
                input_tokens, attention_mask, labels = batch
                input_tokens = input_tokens.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_tokens, attention_mask)    
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1
                prog.set_postfix(epoch=epoch, loss=f"{total_loss / batches:.4f}")
            writer.add_scalar("Loss/train", total_loss/batches, global_step=epoch)
            torch.save({"model": model.state_dict()}, os.path.join(model_dir, "sentiment_model.pt"))

        print("Saving model to", model_dir)
        torch.save({"model": model.state_dict()}, os.path.join(model_dir, "sentiment_model.pt"))

    if eval_model:
        print("Evaluating model...")
        model.eval()
        test_dataset = IMDBDataset("data/sentiment/aclImdb/test", tokenizer, max_len=256)
        test_dataloader = DataLoader(test_dataset,collate_fn=collate,  batch_size=32, shuffle=False, num_workers=4)
        correct = 0
        count = 0
        with torch.no_grad():
            prog = tqdm(test_dataloader)
            for batch in prog:
                input_tokens, attention_mask, labels = batch
                input_tokens = input_tokens.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                logits = model(input_tokens, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += torch.sum(preds == labels).item()
                count += len(labels)
                prog.set_postfix(test_acc=f"{correct / count:.4f}")

        writer.add_scalar("Accuracy/test", correct/count)
        writer.close()
        print("Test Accuracy:", correct / len(test_dataset))
    
    print("Starting inference...")
    while True:
        text = input("Enter a sentence: ")
        tokens = tokenizer.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(tokens)
        preds = torch.argmax(logits, dim=1).item()
        print("Predicted class:", "positive" if preds == 1 else "negative")



