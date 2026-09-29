import torch

from descriptor_icl import meta


def test_token_layout():
    """One description slot, then one slot per example; the precision is
    never written into the prompt."""
    d, K = 3, 4
    g = torch.Generator().manual_seed(0)
    b = meta.sample_batch(6, K, d, 10.0, torch.tensor([0.05]), 0.7, 0.5, "cpu", g)
    b["has_desc"] = torch.tensor([True, False] * 3)
    tok = meta.tokens(b)
    assert tok.shape == (6, K + 1, d + 2)
    # description slot: m and the flag, or all zeros without a description
    assert torch.equal(tok[0, 0], torch.cat([b["m"][0], torch.tensor([0.0, 1.0])]))
    assert torch.all(tok[1, 0] == 0)
    # example slot k: x_k and the previous answer
    assert torch.equal(tok[:, 1:, :d], b["X"])
    assert torch.all(tok[:, 1, d] == 0)
    assert torch.equal(tok[:, 2:, d], b["Y"][:, :-1])
    assert torch.all(tok[:, 1:, d + 1] == 0)


def test_models_predict_a_two_component_mixture():
    d, K = 3, 4
    b = meta.sample_batch(2, K, d, 10.0, torch.tensor([0.5]), 1.0, 1.0, "cpu")
    for model in (meta.Transformer(d, width=16, layers=1), meta.LSTM(d, width=16, layers=1)):
        pred = model(meta.tokens(b))
        assert all(t.shape == (2, K, 2) for t in pred)
        assert meta.log_density(pred, b["Y"]).shape == (2, K)
