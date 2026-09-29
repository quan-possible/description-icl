"""Meta-trained sequence models for in-context regression with descriptions.

A prompt is an optional description, n examples, and one question, in the
prefix layout of Huang & Ge (2025). The model predicts the answer to the
question as a mixture of Gaussians and is trained on log loss, so it can
match the Bayes-optimal predictive (a two-component Gaussian mixture under
the robust prior).

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


def tokens(batch, n):
    """Prompts with n examples each (n a tensor of B counts, below K).

    Every row is (is-description, is-example, vector, answer). Row 0 is the
    description (1, 0, m, 0); rows 1..n are examples (0, 1, x_k, y_k); row
    n + 1 is the question (0, 0, x_{n+1}, 0). Returns the rows and a mask of
    the hidden ones: later rows, and row 0 when there is no description.
    """
    X, Y = batch["X"], batch["Y"]
    B, K, d = X.shape
    k = torch.arange(K, device=X.device)[None, :]
    is_ex = (k < n[:, None]).float()[..., None]
    rows = torch.cat([torch.zeros_like(is_ex), is_ex, X, Y[..., None] * is_ex], dim=2)
    zero = torch.zeros(B, 1, device=X.device)
    desc = torch.cat([zero + 1, zero, batch["m"], zero], dim=1)
    tok = torch.cat([desc[:, None, :], rows], dim=1)  # (B, K+1, d+3)
    hidden = torch.cat([~batch["has_desc"][:, None], k > n[:, None]], dim=1)
    return tok, hidden


class Head(nn.Module):
    def __init__(self, width, components):
        super().__init__()
        self.out = nn.Linear(width, 3 * components)

    def forward(self, h):
        logit, mean, log_sd = self.out(h).chunk(3, dim=-1)
        return torch.log_softmax(logit, -1), mean, log_sd.clamp(-7, 7)


class Transformer(nn.Module):
    """No positions and no causal mask: the examples of a prompt have no
    order, and the marker columns tell the three kinds of row apart."""

    def __init__(self, d, width=128, layers=6, heads=4, components=2):
        super().__init__()
        self.embed = nn.Linear(d + 3, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = Head(width, components)

    def forward(self, tok, hidden, n):
        h = self.norm(self.body(self.embed(tok), src_key_padding_mask=hidden))
        return self.head(h[torch.arange(len(n), device=n.device), n + 1])  # the question row


def log_density(pred, y):
    """log of the predicted mixture density at y. pred is a model's output, or
    several stacked to (B, K, components); y matches its leading shape, or is
    a grid (B, K, G)."""
    log_w, mean, log_sd = pred
    if y.dim() == 3:
        y, log_w, mean, log_sd = (
            y[..., None], log_w[:, :, None], mean[:, :, None], log_sd[:, :, None]
        )
    else:
        y = y[..., None]
    comp = -0.5 * ((y - mean) / log_sd.exp()) ** 2 - log_sd - 0.5 * math.log(2 * math.pi)
    return torch.logsumexp(log_w + comp, dim=-1)
