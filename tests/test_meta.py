import torch

from descriptor_icl import meta

D, K = 3, 4


def batch(has_desc):
    g = torch.Generator().manual_seed(0)
    b = meta.sample_batch(4, K, D, 10.0, torch.tensor([0.05]), 0.7, 1.0, "cpu", g)
    b["has_desc"] = torch.tensor(has_desc)
    return b


def test_prompt_layout():
    """Description row, n example rows with their own answers, one question
    row with a blank answer; everything else hidden."""
    b = batch([True, False, True, True])
    n = torch.tensor([2, 2, 0, 3])
    tok, hidden = meta.tokens(b, n)
    m, Y = b["m"] / 10.0**0.5, b["Y"] / 10.0**0.5  # the model sees w ~ N(0, I)
    assert tok.shape == (4, K + 1, D + 3) and hidden.shape == (4, K + 1)
    # prompt 0: description, two examples, question x_3
    assert torch.allclose(tok[0, 0], torch.cat([torch.tensor([1.0, 0.0]), m[0], torch.zeros(1)]))
    for k in (0, 1):
        want = torch.cat([torch.tensor([0.0, 1.0]), b["X"][0, k], Y[0, k : k + 1]])
        assert torch.allclose(tok[0, k + 1], want)
    assert torch.allclose(tok[0, 3], torch.cat([torch.zeros(2), b["X"][0, 2], torch.zeros(1)]))
    assert hidden[0].tolist() == [False, False, False, False, True]
    # prompt 1 has no description: its first row is hidden
    assert hidden[1].tolist() == [True, False, False, False, True]
    # prompt 2 has no examples: the question follows the description
    assert hidden[2].tolist() == [False, False, True, True, True]
    assert hidden[3].tolist() == [False] * 5


def test_model_reads_only_the_prompt():
    """The prediction ignores hidden rows and the order of the examples."""
    torch.manual_seed(0)
    model = meta.Transformer(D, width=16, layers=2).eval()
    b = batch([True] * 4)
    n = torch.tensor([2, 2, 2, 2])
    tok, hidden = meta.tokens(b, n)
    pred = model(tok, hidden, n)
    assert pred.shape == (4,)

    changed = tok.clone()
    changed[:, 4] = 7.0  # a hidden row
    swapped = tok.clone()
    swapped[:, [1, 2]] = tok[:, [2, 1]]  # the two examples
    for other in (changed, swapped):
        assert torch.allclose(pred, model(other, hidden, n), atol=1e-5)
