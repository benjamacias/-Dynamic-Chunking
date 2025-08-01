import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import math

# ===============================
# 🔹 Routing Module
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
        p[:, 0] = 1.0  # primer byte siempre boundary
        b = (p >= 0.5).float()
        return p, b

# ===============================
# 🔹 Chunking Layer
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
        max_len = max(len(s) for s in selected)
        selected_padded = torch.zeros(len(selected), max_len, x.size(-1), device=x.device)
        for i, s in enumerate(selected):
            selected_padded[i, :len(s)] = s
        return selected_padded, p, b

# ===============================
# 🔹 Dechunking Layer (con smoothing)
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


        smoothed = upsampled.clone()
        coef = p.unsqueeze(-1)  # [B, L, 1]
        smoothed_out = coef * smoothed + (1 - coef) * torch.roll(smoothed, shifts=1, dims=1)
        smoothed_out[:, 0] = smoothed[:, 0]  # primer paso no suaviza
        return smoothed_out


# ===============================
# 🔹 Ratio Loss
# ===============================
def ratio_loss(b, p, N=6):
    L = b.size(1)
    F = b.sum(dim=1) / L
    G = p.sum(dim=1) / L
    loss = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
    return loss.mean()

# ===============================
# 🔹 Multi-Stage H-Net
# ===============================
class HNetStage(nn.Module):
    def __init__(self, dim, n_layers=4, n_heads=8):
        super().__init__()
        self.encoder = nn.LSTM(dim, dim, batch_first=True)
        self.chunk = ChunkingLayer(dim)
        self.main = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads),
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
# 🔹 Dataset Dummy (ejemplo)
# ===============================
class ByteDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=1024):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# ===============================
# 🔹 Entrenamiento
# ===============================
def train(model, dataloader, epochs=5, lr=1e-4):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    for epoch in range(epochs):
        total_ce = 0.0
        total_ratio = 0.0
        correct = 0
        count = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, ratio_l = model(x)
            ce_loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = ce_loss + 0.03 * ratio_l
            loss.backward()
            optimizer.step()
            total_ce += ce_loss.item()
            total_ratio += ratio_l.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == y).float().sum().item()
            count += y.numel()
            progress.set_postfix({"CE": ce_loss.item(), "Ratio": ratio_l.item()})
        scheduler.step()
        avg_ce = total_ce / len(dataloader)
        avg_ratio = total_ratio / len(dataloader)
        acc = correct / count
        print(f"Epoch {epoch+1}/{epochs} - CE Loss: {avg_ce:.4f} - Ratio Loss: {avg_ratio:.4f} - Acc: {acc:.4f}")

# ===============================
# 🔹 Ejemplo de uso
# ===============================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
    print(torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU')
    torch.autograd.set_detect_anomaly(True)
    data = torch.randint(0, 256, (50000,), dtype=torch.long)  # Datos aleatorios
    dataset = ByteDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = MultiStageHNet()
    train(model, dataloader)
