"""Meta-trained sequence models for in-context regression with descriptions.

A prompt is an optional description, n examples, and one question, in the
prefix layout of Huang & Ge (2025). The model predicts a Gaussian for the
answer to the question, a mean and a log variance, and is trained on log
loss as in Genewein et al. (2025), so its target is the Bayes-optimal
predictive distribution.

Units: prompts are drawn with sigma_y = 1 and w of variance a_base, the units
of the exact learner. The model sees them in the units of Garg et al. and
Huang & Ge, w ~ N(0, I): m and every y are divided by sqrt(a_base), which
makes the noise variance 1 / a_base. The description states only its vector m. Its precision
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
    return dict(X=X, Y=Y, m=m, r=r, has_desc=has_desc, correct=correct, w=w, a_base=a_base)


def tokens(batch, n):
    """Prompts with n examples each (n a tensor of B counts, below K).

    Every row is (is-description, is-example, vector, answer), with m and
    the answers in the model's units. Row 0 is the
    description (1, 0, m, 0); rows 1..n are examples (0, 1, x_k, y_k); row
    n + 1 is the question (0, 0, x_{n+1}, 0). Returns the rows and a mask of
    the hidden ones: later rows, and row 0 when there is no description.
    """
    scale = batch["a_base"] ** -0.5
    X, Y, m = batch["X"], batch["Y"] * scale, batch["m"] * scale
    B, K, d = X.shape
    k = torch.arange(K, device=X.device)[None, :]
    is_ex = (k < n[:, None]).float()[..., None]
    rows = torch.cat([torch.zeros_like(is_ex), is_ex, X, Y[..., None] * is_ex], dim=2)
    zero = torch.zeros(B, 1, device=X.device)
    desc = torch.cat([zero + 1, zero, m, zero], dim=1)
    tok = torch.cat([desc[:, None, :], rows], dim=1)  # (B, K+1, d+3)
    hidden = torch.cat([~batch["has_desc"][:, None], k > n[:, None]], dim=1)
    return tok, hidden


class Transformer(nn.Module):
    """The size of Garg et al. (2022): 12 layers, 8 heads, width 256, GELU,
    no dropout. No positions and no causal mask: the examples of a prompt have no
    order, and the marker columns tell the three kinds of row apart."""

    def __init__(self, d, width=256, layers=12, heads=8):
        super().__init__()
        self.embed = nn.Linear(d + 3, width)  # one linear layer reads each row
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)  # mean and log variance

    def forward(self, tok, hidden, n):
        """(mean, log variance) of the answer to the question, each (B,)."""
        h = self.norm(self.body(self.embed(tok), src_key_padding_mask=hidden))
        out = self.head(h[torch.arange(len(n), device=n.device), n + 1])  # the question row
        return out[:, 0], out[:, 1].clamp(-12.0, 12.0)


def log_loss(mean, log_var, y):
    """Negative log density of y under N(mean, exp(log_var)), per prompt."""
    return 0.5 * (math.log(2 * math.pi) + log_var + (y - mean) ** 2 * torch.exp(-log_var))
