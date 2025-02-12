import torch
from challenge1 import MultiHead, Head

import torch.nn as nn

def test_head_forward():
    block_size = 4
    embd_size = 8
    head_size = 2
    head = Head(block_size, embd_size, head_size)
    x = torch.randn(1, block_size, embd_size)
    out = head(x)
    assert out.shape == (1, block_size, head_size), f"Expected shape (1, {block_size}, {head_size}), but got {out.shape}"

def test_multihead_forward():
    num_heads = 4
    block_size = 4
    embd_size = 8
    multihead = MultiHead(num_heads, block_size, embd_size)
    x = torch.randn(1, block_size, embd_size)
    out = multihead(x)
    expected_shape = (1, block_size, embd_size)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, but got {out.shape}"

if __name__ == "__main__":
    test_head_forward()
    test_multihead_forward()
    print("All tests passed.")