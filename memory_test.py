import torch
import time
from src.model import MambaBlock
from src.constants import D_MODEL, D_STATE, D_INNER, D_CONV

def profile_mamba_memory(seq_lengths):
    print("Profiling Mamba State Memory...")
    model = MambaBlock(d_model=D_MODEL, d_state=D_STATE, d_inner=D_INNER, d_conv=D_CONV)
    
    results = []
    for L in seq_lengths:
        model.conv_state.zero_()
        model.ssm_state.zero_()
        
        # Simulate L steps
        for _ in range(L):
            x_t = torch.randn(1, D_MODEL)
            _ = model.step(x_t)
            
        # Measure state size in bytes
        conv_size = model.conv_state.element_size() * model.conv_state.nelement()
        ssm_size = model.ssm_state.element_size() * model.ssm_state.nelement()
        total_state_bytes = conv_size + ssm_size
        
        results.append(total_state_bytes)
        print(f"Seq Len: {L:4d} | Mamba State: {total_state_bytes/1024:.2f} KB")
        
    return results

def profile_transformer_kv_cache(seq_lengths, n_layers=12, d_model=D_MODEL):
    print("\nSimulating Transformer KV-Cache Memory...")
    # Transformer KV cache size: 2 * n_layers * batch * L * d_model * element_size
    # (Assuming d_head * n_heads = d_model)
    element_size = 4 # float32
    
    results = []
    for L in seq_lengths:
        kv_cache_bytes = 2 * n_layers * 1 * L * d_model * element_size
        results.append(kv_cache_bytes)
        print(f"Seq Len: {L:4d} | Transformer KV-Cache: {kv_cache_bytes/1024:.2f} KB")
        
    return results

if __name__ == "__main__":
    seq_lengths = [1, 128, 512, 1024, 2048, 4096]
    
    mamba_mem = profile_mamba_memory(seq_lengths)
    transformer_mem = profile_transformer_kv_cache(seq_lengths)
    
    print("\n--- Summary ---")
    print(f"{'Seq Len':<10} | {'Mamba (KB)':<12} | {'Transformer (KB)':<18}")
    for i, L in enumerate(seq_lengths):
        print(f"{L:<10} | {mamba_mem[i]/1024:<12.2f} | {transformer_mem[i]/1024:<18.2f}")
    
    is_constant = all(m == mamba_mem[0] for m in mamba_mem)
    print(f"\nIs Mamba memory constant? {'YES' if is_constant else 'NO'}")
