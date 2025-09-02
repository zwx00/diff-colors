import math, random, torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from dataset import get_css_colors, get_xkcd_colors


def make_dataset():
    items = []
    all_colors = get_css_colors() | get_xkcd_colors()
    for name, rgb in all_colors.items():
        x = torch.tensor(rgb, dtype=torch.float32)
        x = x * 2 - 1  # -> [-1,1]
        items.append((name, x))
    return items

# ---------------------------
# 2) Time embedding (sin/cos)
# This is basically the same as positional embedding in transformers
# It lets the network know the timestep - eg. earlier timesteps mean more noise
# ---------------------------
def t_embed(t, dim=32):
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1), math.log(10000.0), half, device=t.device))
    a = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)

# ---------------------------
# 3) ε-prediction MLP
#    Input: concat(x_t, t_emb, y_emb) -> ε_hat (3D)
#    The y_emb is the embedding of the color name eg. label
#    x_t is the RGB of the color at the current timestep
#    t_cont is the timestep as a continuous value between 0 and 1
#    The output is the predicted noise ε_hat, a 3 value vector in [-1,1]
#  Architecture:
#    - yproj: Project the label embedding to 64 dimensions
#    - hidden layer: 256 dimensions
#    - FiLM: Factorized Layer Normalization
#       - a way to inject the label embedding into the MLP continuously instead of just at the input
#    - SiLU activation - a smooth activation function that is similar to ReLU but has a smooth curve
# ---------------------------
class EpsMLP(nn.Module):
    def __init__(self, y_in=384, y_proj=64, h=256, tdim=32):
        super().__init__()

        self.yproj = nn.Sequential(nn.Linear(y_in, y_proj), nn.LayerNorm(y_proj), nn.SiLU())
        self.in_lin = nn.Linear(tdim + 3, h)

        # h * 2 because we have scale and shift (eg. gamma * x + beta)
        self.in_film = nn.Linear(y_proj, 2 * h)
        # initialize to zero
        nn.init.zeros_(self.in_film.weight)
        nn.init.zeros_(self.in_film.bias)

        self.h_lin = nn.Linear(h, h)
        self.h_film = nn.Linear(y_proj, 2 * h)
        nn.init.zeros_(self.h_film.weight)
        nn.init.zeros_(self.h_film.bias)

        self.out_lin = nn.Linear(h, 3)

    def forward(self, x_t, t_cont, y):
        te = t_embed(t_cont)
        y = self.yproj(y)

        h = torch.cat([x_t, te], dim=1)

        # input layer
        h = self.in_lin(h)
        g_in, b_in = self.in_film(y).chunk(2, dim=1) # split into two tensors
        h = h * (g_in + 1) + b_in # apply film scale and shift

        h = F.silu(h) # activation

        # hidden layer
        h = self.h_lin(h)
        g_h, b_h = self.h_film(y).chunk(2, dim=1)

        h = h * (g_h + 1) + b_h
        h = F.silu(h)

        return self.out_lin(h) # ε_hat

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
    y  = txtenc.encode(list(names), convert_to_tensor=True, device=device, normalize_embeddings=True).clone().detach()
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
    model.eval()
    txtenc.eval()
    T = len(abar)-1
    idxs = torch.linspace(T, 1, steps, device=device).long()
    x = torch.randn(1,3, device=device) 
    y = txtenc.encode([prompt], convert_to_tensor=True, device=device, normalize_embeddings=True).clone().detach()
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
    txtenc = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    model  = EpsMLP().to(device)
    opt    = torch.optim.AdamW(list(model.parameters()), lr=2e-3)
    _,_,abar = make_alpha_bar(T=1000, device=device)

    # quick-n-dirty training
    for step in range(6000):  # ~ a few seconds on GPU / ~tens on CPU
        batch = [random.choice(data) for _ in range(128)]
        loss = train_step(model, txtenc, opt, batch, abar, device)
        if (step+1) % 200 == 0:
            print(f"step {step+1}: loss {loss:.4f}")

    # sample some prompts
    for prompt in ["red", "blue", "beige", "purple", "cream", "warm gray"]:
        x = sample(model, txtenc, prompt, abar, steps=500, device=device)[0]
        rgb_norm = ((x.cpu().numpy()+1)/2).tolist()
        rgb_rgb = tuple(round(v * 255) for v in rgb_norm)
        print(f"{prompt:>10s} -> RGB \033[38;2;{rgb_rgb[0]};{rgb_rgb[1]};{rgb_rgb[2]}m█\033[0m {rgb_rgb}")
