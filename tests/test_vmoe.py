import torch
import pytest
import sys

sys.path.insert(0, '.')

from inference.vmoe import Expert, Router, MoE

# Gating Tests
def test_gating_forward_shape():
    """Test Gating output shape"""
    B, S, H = 2, 4, 64
    E, K = 8, 2
    
    router = Router(hidden_size=H, num_experts=E, top_k=K)
    x = torch.randn(B * S, H)
    
    top_indices, top_weights, gate_probs = router(x)
    assert top_indices.shape == (B * S, K), f"Expected {(B*S, K)}, got {top_indices.shape}"
    assert top_weights.shape == (B * S, K), f"Expected {(B*S, K)}, got {top_weights.shape}"
    assert gate_probs.shape == (B * S, E), f"Expected {(B*S, E)}, got {gate_probs.shape}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
