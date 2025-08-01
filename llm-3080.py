import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# Routing Module
# ===============================
class RoutingModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(torch.roll(x, shifts=1, dims=1))
        cosine_sim = F.cosine_similarity(q, k, dim=-1)
        p = 0.5 * (1 - cosine_sim)
        p[:, 0] = 1.0
        b = (p >= 0.5).float()
        return p, b

# ===============================
# Chunking Layer
# ===============================
class ChunkingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.router = RoutingModule(dim)

    def forward(self, x):
        p, b = self.router(x)
        selected = []
        for i in range(x.size(0)):
            selected.append(x[i][b[i] == 1])
        padded = torch.nn.utils.rnn.pad_sequence(selected, batch_first=True)
        return padded.to(x.device), p, b

# ===============================
# Dechunking Layer (no inplace)
# ===============================
class DechunkingLayer(nn.Module):
    def forward(self, z, p, b, original_len):
        batch_size = z.size(0)
        upsampled = torch.zeros(batch_size, original_len, z.size(-1), device=z.device)

        for i in range(batch_size):
            idx = 0
            for t in range(original_len):
                if idx < z.size(1):
                    upsampled[i, t] = z[i, idx]
                    if b[i, t] == 1:
                        idx += 1
                else:
                    upsampled[i, t] = z[i, -1]

        coef = p.unsqueeze(-1)
        smoothed = coef * upsampled + (1 - coef) * torch.roll(upsampled, shifts=1, dims=1)
        smoothed[:, 0] = upsampled[:, 0]
        return smoothed

# ===============================
# Ratio Loss
# ===============================
def ratio_loss(b, p, N=6):
    L = b.size(1)
    F = b.sum(dim=1) / L
    G = p.sum(dim=1) / L
    return ((N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))).mean()

# ===============================
# H-Net Stage
# ===============================
class HNetStage(nn.Module):
    def __init__(self, dim, n_layers=2, n_heads=8):
        super().__init__()
        self.encoder = nn.LSTM(dim, dim, batch_first=True)
        self.chunk = ChunkingLayer(dim)
        self.main = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True),
            num_layers=n_layers
        )
        self.dechunk = DechunkingLayer()
        self.decoder = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        chunks, p, b = self.chunk(enc_out)
        main_out = self.main(chunks)
        dechunked = self.dechunk(main_out, p, b, enc_out.size(1))
        dec_out, _ = self.decoder(dechunked)
        return dec_out, p, b

# ===============================
# Multi-Stage H-Net
# ===============================
class MultiStageHNet(nn.Module):
    def __init__(self, vocab_size=256, dim=256, stages=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.stages = nn.ModuleList([HNetStage(dim) for _ in range(stages)])
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        ratio_losses = []
        for stage in self.stages:
            x, p, b = stage(x)
            ratio_losses.append(ratio_loss(b, p))
        logits = self.fc_out(x)
        return logits, sum(ratio_losses)

# ===============================
# Dataset
# ===============================
class ByteDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=512):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len].clone().detach()
        y = self.data[idx+1:idx+self.seq_len+1].clone().detach()
        return x, y

# ===============================
# Entrenamiento con AMP
# ===============================
def train(model, dataloader, epochs=5, lr=1e-4, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, ratio_l = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)) + 0.03 * ratio_l
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.8, device=0)

    print(f"Training on: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    data = torch.randint(0, 256, (200000,), dtype=torch.long)
    dataset = ByteDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = MultiStageHNet()
    train(model, dataloader)
    torch.save(model.state_dict(), "hnet_model_gpu.pth")
