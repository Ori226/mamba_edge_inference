# Mamba-Edge-Inference-PoC

> [!IMPORTANT]
> **Work in Progress:** This repository is a proof-of-concept demonstrating Mamba's linear recurrence properties for edge deployment. Architectural refinements and optimized mapping logic are ongoing.

**The Core Mathematical Thesis:**
Mamba inference is mathematically a standard Linear Recurrence (RNN). While it uses **Parallel Associative Scans** for $O(\log L)$ training, the inference mode for a single token $x_t$ is a $O(1)$ state update. This makes it a perfect candidate for NPUs, DSPs, and edge devices that lack custom CUDA/Triton kernel support but excel at fixed-size tensor arithmetic.

**This repository proves that:**
1. **Mamba is "Hardware Agnostic":** Specialized kernels are an optimization, not a requirement.
2. **Constant Memory Footprint:** RAM usage does not increase with context length.
3. **Deployment Readiness:** Coherent text generation is possible using only standard PyTorch ops (ready for ONNX/TFLite export).

## Installation

To run this PoC on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [repository-url]
    cd mamba_edge_inference
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install torch transformers python-dotenv
    ```

4.  **Configure HuggingFace Token:**
    Create a `.env` file in the root directory and add your token:
    ```bash
    echo "HF_TOKEN=your_huggingface_token_here" > .env
    ```

## Key Features

1.  **Hardware Agnostic:** No specialized CUDA/Triton kernels are required for inference. The implementation uses standard PyTorch ops (`matmul`, `add`, `mul`, `exp`).
2.  **Constant Memory Footprint:** RAM usage remains fixed regardless of sequence length. Unlike Transformers, there is no growing KV-Cache.
3.  **Edge-Ready:** The code follows a simple RNN recurrence, making it portable to TFLite, CoreML, or OpenVINO.

## Usage

### 1. Proof of Concept (Structure and Recurrence)
Run the Pile dataset simulation to see Mamba's $O(1)$ recurrence in action:
```bash
python3 pile_test.py
```

### 2. Memory Benchmark
Compare Mamba's constant state vs Transformer's KV-cache:
```bash
python3 memory_test.py
```

## Structure

- `src/discretize.py`: Implements Zero-Order Hold (ZOH) discretization.
- `src/model.py`: Implements the Mamba recurrence ($h_t = \bar{A}h_{t-1} + \bar{B}x_t$).
- `src/inference.py`: Shows the token-by-token generation loop.
- `benchmarks/memory_test.py`: Compares Mamba's fixed state vs. Transformer's linear KV-cache.

## Mathematical Proof

The recurrence step is:
$$h_t = (\exp(\Delta A)) h_{t-1} + (\Delta B) x_t$$
$$y_t = C h_t + D x_t$$

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](file:///home/ori/code/mamba_edge_inference/LICENSE) file for details.

Copyright 2026 Ori
