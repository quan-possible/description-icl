"""Meta-trained sequence models for in-context regression with descriptions.

A prompt is an optional description token followed by examples. The model
predicts each y_k from x_k and everything before it, as a mixture of
Gaussians, and is trained on log loss so it can match the Bayes-optimal
predictive (a two-component Gaussian mixture under the robust prior).

Units: sigma_y = 1. The description states only its vector m. Its precision
r = a_desc / a_base and its reliability p are fixed for each trained model
and never stated, so the model learns both from the training prompts.
"""

import math

import torch
from torch import nn


def sample_batch(B, K, d, a_base, rs, p, p_desc, device, generator=None):
    """Draw B prompts of K examples. Returns a dict of tensors.

    rs: tensor of precision ratios to sample from. p: probability a
    description is correct. p_desc: probability the prompt has one.
    """
    g = generator
    rn = lambda *s: torch.randn(*s, device=device, generator=g)
    ru = lambda *s: torch.rand(*s, device=device, generator=g)
    r = rs[torch.randint(len(rs), (B,), device=device, generator=g)]
    has_desc = ru(B) < p_desc
    correct = ru(B) < p
    m = torch.sqrt(a_base * (1 - r))[:, None] * rn(B, d)
    w_right = m + torch.sqrt(a_base * r)[:, None] * rn(B, d)
    w_wrong = math.sqrt(a_base) * rn(B, d)
    # without a description the task is a plain draw from the base prior
    w = torch.where((has_desc & correct)[:, None], w_right, w_wrong)
    X = rn(B, K, d)
    Y = torch.einsum("bkd,bd->bk", X, w) + rn(B, K)
    return dict(X=X, Y=Y, m=m, r=r, has_desc=has_desc, correct=correct, w=w)


def tokens(batch):
    """Prefix layout of Huang & Ge (2025). Each token is (vector, previous
    answer, has-description flag). Token 0 is the description (m, 0, flag),
    all zeros when the prompt has none; token k >= 1 is (x_k, y_{k-1}, 0), so
    a causal model at position k has seen examples 1..k-1 and the query x_k."""
    X, Y = batch["X"], batch["Y"]
    B, K, d = X.shape
    h = batch["has_desc"].float()[:, None]
    z = lambda *s: torch.zeros(*s, device=X.device)
    desc = torch.cat([batch["m"] * h, z(B, 1), h], dim=1)
    prev_y = torch.cat([z(B, 1), Y[:, :-1]], dim=1)
    ex = torch.cat([X, prev_y[..., None], z(B, K, 1)], dim=2)
    return torch.cat([desc[:, None, :], ex], dim=1)  # (B, K+1, d+2)


class Head(nn.Module):
    def __init__(self, width, components):
        super().__init__()
        self.out = nn.Linear(width, 3 * components)

    def forward(self, h):
        logit, mean, log_sd = self.out(h).chunk(3, dim=-1)
        return torch.log_softmax(logit, -1), mean, log_sd.clamp(-7, 7)


class Transformer(nn.Module):
    def __init__(self, d, width=128, layers=6, heads=4, components=2, max_len=64):
        super().__init__()
        self.embed = nn.Linear(d + 2, width)
        self.pos = nn.Parameter(torch.zeros(max_len, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = Head(width, components)

    def forward(self, tok):
        T = tok.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tok.device)
        h = self.body(self.embed(tok) + self.pos[:T], mask=mask, is_causal=True)
        return self.head(self.norm(h)[:, 1:])  # predictions for y_1..y_K


class LSTM(nn.Module):
    def __init__(self, d, width=256, layers=2, components=2):
        super().__init__()
        self.embed = nn.Linear(d + 2, width)
        self.body = nn.LSTM(width, width, layers, batch_first=True)
        self.head = Head(width, components)

    def forward(self, tok):
        return self.head(self.body(self.embed(tok))[0][:, 1:])


def log_density(pred, y):
    """log of the predicted mixture density at y; pred from a model, y (B, K)
    or a grid (B, K, G)."""
    log_w, mean, log_sd = pred
    if y.dim() == 3:
        y, log_w, mean, log_sd = (
            y[..., None], log_w[:, :, None], mean[:, :, None], log_sd[:, :, None]
        )
    else:
        y = y[..., None]
    comp = -0.5 * ((y - mean) / log_sd.exp()) ** 2 - log_sd - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(log_w + comp, dim=-1)
