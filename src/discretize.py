import torch
import torch.nn.functional as F

def discretize_zoh(delta, A, B):
    """
    Discretization (Zero-Order Hold)
    delta: (batch, l, d_inner) or (batch, d_inner)
    A: (d_inner, d_state)
    B: (batch, l, d_state) or (batch, d_state)
    
    Returns:
    A_bar: (batch, l, d_inner, d_state)
    B_bar: (batch, l, d_inner, d_state)
    """
    # delta is (B, D) or (B, L, D)
    # A is (D, N)
    # B is (B, N) or (B, L, N)
    
    # We want to compute:
    # A_bar = exp(delta * A)
    # B_bar = (delta * A)^-1 * (exp(delta * A) - I) * (delta * B)
    
    # Note: In Mamba, B is usually already multiplied by delta in the parallel scan, 
    # but the mathematical definition of B_bar in ZOH is as above.
    # However, the Mamba paper simplifies this for the parallel scan.
    # For the RNN recurrence, we use the simple discretization:
    # A_bar = exp(delta * A)
    # B_bar = delta * B
    
    # Let's follow the standard Mamba implementation's discretization which is:
    # A_bar = exp(delta * A)
    # B_bar = delta * B
    
    # If delta has shape (B, D) and A has shape (D, N)
    # delta * A has shape (B, D, N) via broadcasting
    
    dt_a = delta.unsqueeze(-1) * A # (B, D, N)
    A_bar = torch.exp(dt_a)
    
    # B is often (B, N) or (B, L, N)
    # We need B_bar to be (B, D, N)
    # In Mamba, B is usually projected from x, so it's (B, L, N)
    # delta is (B, L, D)
    # B_bar = delta * B -> (B, L, D, N)
    
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2) # (B, L, D, N)
    
    return A_bar, B_bar

def discretize_simple(delta, A, B):
    """
    Simpler version for single token inference.
    delta: (batch, d_inner)
    A: (d_inner, d_state)
    B: (batch, d_state)
    """
    dt_a = delta.unsqueeze(-1) * A # (B, d_inner, d_state)
    A_bar = torch.exp(dt_a)
    B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2) # (B, d_inner, d_state)
    return A_bar, B_bar
