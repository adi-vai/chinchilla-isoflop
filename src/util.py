from src.model import ModelConfig


def matmul_params(model_config: ModelConfig) -> int:
    # Weights that do a multiply-add per token: the N in 6N
    L, d, V = (
        model_config.model_depth,
        model_config.model_width,
        model_config.vocab_size,
    )
    # 12·L·d² block weight matrices, d·V tied embedding / lm_head
    return 12 * L * d**2 + d * V


def compute_params(model_config: ModelConfig) -> int:
    # Analytic count, no model built; must equal get_compute_params()
    L, d = model_config.model_depth, model_config.model_width
    # Plus (10L + 2)·d biases and LayerNorms (10d per block, 2d final LayerNorm)
    return matmul_params(model_config) + (10 * L + 2) * d


def total_params(model_config: ModelConfig) -> int:
    # Compute params plus the position embedding, T·d
    return (
        compute_params(model_config)
        + model_config.context_length * model_config.model_width
    )


def flops_per_token(model_config: ModelConfig) -> int:
    # Training FLOPs per token, matmuls only: 6N + attention (12·L·H·Q·T).
    # Biases, LayerNorm, softmax etc. are elementwise and left out (NOTES.md)
    L, Q, H, T = (
        model_config.model_depth,
        model_config.model_width // model_config.num_heads,
        model_config.num_heads,
        model_config.context_length,
    )
    return 6 * matmul_params(model_config) + 12 * L * Q * H * T
