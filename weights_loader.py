import os
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
from src.model import MambaLM
from src.constants import D_MODEL

load_dotenv() # Load HF_TOKEN from .env

def load_official_mamba(model_name="state-spaces/mamba-130m"):
    """
    Loads official Mamba weights and maps them to our RNN implementation.
    """
    token = os.getenv("HF_TOKEN")
    print(f"Downloading/Loading official Mamba from HF: {model_name}")
    
    official_sd = None
    
    try:
        # Try standard transformers loading first
        print("Trying AutoModelForCausalLM...")
        official_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=token if token else None
        )
        official_sd = official_model.state_dict()
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Attempting manual weight download fallback...")
        try:
            # Try pytorch_model.bin or model.safetensors
            try:
                weights_path = hf_hub_download(
                    repo_id=model_name, 
                    filename="pytorch_model.bin",
                    token=token if token else None
                )
                official_sd = torch.load(weights_path, map_location="cpu")
            except:
                print("pytorch_model.bin not found, trying model.safetensors...")
                from safetensors.torch import load_file
                weights_path = hf_hub_download(
                    repo_id=model_name, 
                    filename="model.safetensors",
                    token=token if token else None
                )
                official_sd = load_file(weights_path)
        except Exception as e2:
            raise RuntimeError(f"Manual download failed: {e2}")

    if official_sd is None:
        raise RuntimeError("Failed to load state_dict through any method.")

    # Infer model dimensions from state_dict
    print(f"Mapping weights. Keys found: {len(official_sd)}")
    
    # Embedding weight is usually 'backbone.embedding.weight'
    emb_weight = official_sd.get("backbone.embedding.weight")
    if emb_weight is None:
        # Try alternate names
        emb_weight = official_sd.get("embedding.weight")
    
    if emb_weight is None:
        raise KeyError("Could not find embedding weight in state_dict")

    vocab_size, d_model = emb_weight.shape
    
    # Count layers by looking at keys like 'backbone.layers.X.mixer.A_log'
    layer_indices = set()
    dt_rank = 48 # Default for 130M
    for key, val in official_sd.items():
        if "layers." in key:
            try:
                idx = int(key.split("layers.")[1].split(".")[0])
                layer_indices.add(idx)
                if "dt_proj.weight" in key:
                    dt_rank = val.shape[1]
            except:
                continue
    n_layers = len(layer_indices)
    
    print(f"Inferred Model: {n_layers} layers, d_model={d_model}, vocab={vocab_size}, dt_rank={dt_rank}")
    
    our_model = MambaLM(n_layers=n_layers, d_model=d_model, vocab_size=vocab_size, dt_rank=dt_rank)
    
    # Mapping logic
    new_sd = {}
    
    # Standard Mamba-130M mapping
    mapping = {
        "backbone.embedding.weight": "embedding.weight",
        "backbone.norm_f.weight": "backbone.norm_f.weight", # Final norm
        "lm_head.weight": "lm_head.weight",
    }
    
    for old, new in mapping.items():
        if old in official_sd:
            new_sd[new] = official_sd[old]
        elif old.replace("backbone.", "") in official_sd:
             new_sd[new] = official_sd[old.replace("backbone.", "")]

    for i in range(n_layers):
        # Mamba-1.0 parameters
        mixer_prefix = f"backbone.layers.{i}.mixer"
        our_prefix = f"backbone.layers.{i}"
        
        # Layer norm
        norm_key = f"backbone.layers.{i}.norm.weight"
        if norm_key in official_sd:
            new_sd[f"{our_prefix}.norm.weight"] = official_sd[norm_key]
        elif norm_key.replace("backbone.", "") in official_sd:
            new_sd[f"{our_prefix}.norm.weight"] = official_sd[norm_key.replace("backbone.", "")]

        # If prefix is missing 'backbone.', we handle it
        if f"layers.{i}.mixer.A_log" not in official_sd and f"backbone.layers.{i}.mixer.A_log" not in official_sd:
             # Try without 'mixer'?
             mixer_prefix = f"backbone.layers.{i}"
        
        sub_mapping = {
            "in_proj.weight": "in_proj.weight",
            "out_proj.weight": "out_proj.weight",
            "dt_proj.weight": "dt_proj.weight",
            "dt_proj.bias": "dt_proj.bias",
            "x_proj.weight": "x_proj.weight",
            "A_log": "A_log",
            "D": "D",
            "conv1d.weight": "conv1d.weight",
            "conv1d.bias": "conv1d.bias",
        }
        
        for old_suffix, new_suffix in sub_mapping.items():
            old_key = f"{mixer_prefix}.{old_suffix}"
            if old_key not in official_sd:
                # Try without 'backbone.'
                old_key = old_key.replace("backbone.", "")
            
            if old_key in official_sd:
                new_sd[f"{our_prefix}.{new_suffix}"] = official_sd[old_key]

    our_model.load_state_dict(new_sd, strict=False)
    print("Successfully mapped official weights.")
    return our_model
