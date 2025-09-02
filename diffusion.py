# tiny text->RGB diffusion (ε-prediction, DDIM η=0)
# pip: torch (1.13+). No deps beyond PyTorch.

import math, random, torch, torch.nn as nn, torch.nn.functional as F

# ---------------------------
# 0) Toy data (name, RGB in [0,1])
# ---------------------------
NAMED_COLORS = {
  "red":   (1.00, 0.00, 0.00),
  "green": (0.00, 1.00, 0.00),
  "blue":  (0.00, 0.00, 1.00),
  "white": (1.00, 1.00, 1.00),
  "black": (0.00, 0.00, 0.00),
  "gray":  (0.50, 0.50, 0.50),
  "pink":  (1.00, 0.75, 0.80),
  "orange":(1.00, 0.55, 0.00),
  "purple":(0.50, 0.00, 0.50),
  "yellow":(1.00, 1.00, 0.00),
  "beige": (0.96, 0.96, 0.86),   # some approximations
  "tan":   (0.82, 0.71, 0.55),
  "cream": (1.00, 0.99, 0.82),
}

def make_dataset():
    items = []
    for name, rgb in NAMED_COLORS.items():
        x = torch.tensor(rgb, dtype=torch.float32)
        x = x * 2 - 1  # -> [-1,1]
        items.append((name, x))
    return items

# ---------------------------
# 1) Tiny text "encoder"
#    (replace this with BERT/CLIP later)
# ---------------------------
class TinyTextEnc(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(32, 128), nn.SiLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, names):
        # letter-frequency across 26 letters + 6 extras (space, dash, etc.)
        B = len(names)
        X = torch.zeros(B, 32)
        for i, s in enumerate(names):
            s = s.lower()
            for ch in s:
                if 'a' <= ch <= 'z':
                    X[i, ord(ch)-97] += 1.0
                else:
                    X[i, 26 + (hash(ch) % 6)] += 1.0
        X = X / (X.sum(dim=1, keepdim=True) + 1e-6)
        return self.proj(X)

# ---------------------------
# 2) Time embedding (sin/cos)
# ---------------------------
def t_embed(t, dim=32):
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1e-4), math.log(1.0), half, device=t.device))
    a = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)

# ---------------------------
# 3) ε-prediction MLP
#    Input: concat(x_t, t_emb, y_emb) -> ε_hat (3D)
# ---------------------------
class EpsMLP(nn.Module):
    def __init__(self, ydim=64, h=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 + 32 + ydim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 3)
        )
    def forward(self, x_t, t_cont, y):
        te = t_embed(t_cont)              # [B,32]
        h = torch.cat([x_t, te, y], dim=1)
        return self.fc(h)                 # ε_hat

# ---------------------------
# 4) Beta schedule + ᾱ precompute
# ---------------------------
def make_alpha_bar(T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)           # [T]
    abar = torch.cat([torch.tensor([1.0], device=device), abar])  # 0..T
    return betas, alphas, abar

# ---------------------------
# 5) Training step (one formula, one loss)
#     x_t = sqrt(abar_t) * x0 + sqrt(1-abar_t) * ε
#     L = || ε - ε_hat(x_t, t, y) ||^2
# ---------------------------
def train_step(model, txtenc, optim, batch, abar, device="cpu"):
    names, x0 = zip(*batch)
    x0 = torch.stack(x0).to(device)                   # [B,3] in [-1,1]
    y  = txtenc(list(names)).to(device)               # [B,ydim]
    B, T = x0.size(0), len(abar)-1
    t_idx = torch.randint(1, T+1, (B,), device=device)
    abar_t = abar[t_idx].view(B,1); one_minus = (1-abar_t).clamp_min(1e-6)
    eps = torch.randn(B,3, device=device)
    x_t = abar_t.sqrt()*x0 + one_minus.sqrt()*eps
    t_cont = t_idx.float() / T
    eps_hat = model(x_t, t_cont, y)
    loss = F.mse_loss(eps_hat, eps)
    optim.zero_grad(); loss.backward(); optim.step()
    return loss.item()

# ---------------------------
# 6) Deterministic DDIM (η=0) sampling
#     x_{t-1} = sqrt(abar_{t-1}) * x0_hat + sqrt(1-abar_{t-1}) * ε_hat
#     with x0_hat = (x_t - sqrt(1-abar_t)*ε_hat) / sqrt(abar_t)
# ---------------------------
@torch.no_grad()
def sample(model, txtenc, prompt, abar, steps=100, device="cpu"):
    T = len(abar)-1
    idxs = torch.linspace(T, 1, steps, dtype=torch.long, device=device)
    x = torch.randn(1,3, device=device)             # start from N(0,I)
    y = txtenc([prompt]).to(device)
    for i in range(len(idxs)):
        ti = idxs[i].item()
        a_t  = abar[ti]; a_prev = abar[max(ti-1,0)]
        t_cont = torch.full((1,), ti/float(T), device=device)
        eps_hat = model(x, t_cont, y)
        x0_hat = (x - torch.sqrt(1-a_t)*eps_hat) / torch.sqrt(a_t)
        x = torch.sqrt(a_prev)*x0_hat + torch.sqrt(1-a_prev)*eps_hat
    return x.clamp(-1,1)     # RGB in [-1,1]

# ---------------------------
# 7) Tiny demo
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = make_dataset()
    txtenc = TinyTextEnc(64).to(device)
    model  = EpsMLP(ydim=64, h=128).to(device)
    opt    = torch.optim.AdamW(list(model.parameters())+list(txtenc.parameters()), lr=2e-3)
    _,_,abar = make_alpha_bar(T=1000, device=device)

    # quick-n-dirty training
    for step in range(2000):  # ~ a few seconds on GPU / ~tens on CPU
        batch = [random.choice(data) for _ in range(64)]
        loss = train_step(model, txtenc, opt, batch, abar, device)
        if (step+1) % 200 == 0:
            print(f"step {step+1}: loss {loss:.4f}")

    # sample some prompts
    for prompt in ["red", "blue", "beige", "purple", "cream", "warm gray"]:
        x = sample(model, txtenc, prompt, abar, steps=200, device=device)[0]
        rgb = ((x.cpu().numpy()+1)/2).tolist()
        print(f"{prompt:>10s} -> RGB {tuple(round(v,3) for v in rgb)}")
