import torch
import torch.nn as nn
import pytest
import sys

sys.path.insert(0, ".")

from inference.vmoe import Expert, Router


# Gating Tests
def test_gating_forward_shape():
    """Test Gating output shape"""
    B, S, H = 2, 4, 64
    E, K = 8, 2

    router = Router(d_model=H, num_experts=E, top_k=K)
    x = torch.randn(B * S, H)

    router_output, indices = router(x)
    assert router_output.shape == (B * S, E), (
        f"Expected {(B * S, E)}, got {router_output.shape}"
    )
    assert indices.shape == (B * S, K), f"Expected {(B * S, K)}, got {indices.shape}"


def test_expert():
    expert = Expert(dim=2, d_ff=8)
    x = torch.randn(8, 2)
    output = expert(x)
    print(output)


# Expert 1 Test for MoE Layer
def test_expert_forward():
    """Test Expert 1 SwiGLU forward pass with shape and gradient flow"""
    batch_size, dim, d_ff = 4, 64, 256

    expert = Expert(dim=dim, d_ff=d_ff)
    x = torch.randn(batch_size, dim, requires_grad=True)

    # Forward pass
    output = expert(x)

    # Check output shape
    assert output.shape == (batch_size, dim), (
        f"Expected {(batch_size, dim)}, got {output.shape}"
    )

    # Check gradient flow
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradient should flow back to input"
    assert expert.W1.weight.grad is not None, "Gradient should flow to W1"
    assert expert.W2.weight.grad is not None, "Gradient should flow to W2"
    assert expert.W3.weight.grad is not None, "Gradient should flow to W3"

