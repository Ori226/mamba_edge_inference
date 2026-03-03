import torch
from transformers import AutoTokenizer
from weights_loader import load_official_mamba

def run_pile_inference(prompt="The mathematical proof of Mamba's recurrence is", max_new_tokens=20):
    """
    Demonstrates Mamba inference on a text prompt using official weights
    mapped to our simple RNN implementation.
    """
    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Load our RNN-Mamba with official 130M weights
    try:
        model = load_official_mamba("state-spaces/mamba-130m")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to random weights for structural demonstration...")
        from src.model import MambaLM
        model = MambaLM(n_layers=24, d_model=768, vocab_size=50277)

    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens[0].tolist()}")
    
    # 1. Warm up state with prompt (Standard RNN behavior)
    model.reset_state()
    for i in range(tokens.size(1) - 1):
        _ = model.step(tokens[:, i])
    
    # 2. Generate new tokens one by one
    current_token = tokens[:, -1]
    generated_text = prompt
    
    print("\nGenerating:", end=" ", flush=True)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.step(current_token)
            next_token = torch.argmax(logits, dim=-1)
            
        word = tokenizer.decode(next_token)
        print(word, end="", flush=True)
        
        generated_text += word
        current_token = next_token
        
    print("\n\nFull Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    # Sample from the Pile or a general knowledge prompt
    sample_prompt = "Mamba is a state space model that"
    run_pile_inference(sample_prompt)
