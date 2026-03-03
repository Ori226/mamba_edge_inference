import torch
from .model import MambaRNN
from .constants import D_MODEL

def generate_tokens(model, start_tokens, max_len=20, d_model=D_MODEL):
    """
    Generate tokens one by one using the Mamba recurrence.
    start_tokens: (batch, seq_len, d_model)
    """
    # 1. Warm up the state with the prefix
    model.reset_state()
    for i in range(start_tokens.size(1)):
        x_t = start_tokens[:, i, :]
        _ = model.step(x_t)
    
    # 2. Generate new tokens
    outputs = []
    # Start with the last token's output as the next input if needed, 
    # but here we just show the RNN loop mechanism.
    current_x = start_tokens[:, -1, :]
    
    for _ in range(max_len):
        # In a real model, we would project the output of step() 
        # back to vocab space, sample a token, and use its embedding.
        # Here we just show the recurrence.
        next_x = model.step(current_x)
        outputs.append(next_x)
        current_x = next_x
        
    return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    # Test the loop
    model = MambaRNN(n_layers=12, d_model=D_MODEL)
    prefix = torch.randn(1, 5, D_MODEL)
    generated = generate_tokens(model, prefix)
    print(f"Generated sequence shape: {generated.shape}")
    print("Mamba inference loop successful.")
