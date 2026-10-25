"""Checks on parameter and FLOPs counting: errors here are silent elsewhere.

Run from the repo root: `uv run pytest`.
"""

import pytest

from src.ladder import BRYANT_LIU, make_config, to_sig_figs
from src.model import TransformerLanguageModel
from src.util import compute_params, flops_per_token, matmul_params, total_params

CONFIGS = [(d, L, H) for d, L, H, _, _ in BRYANT_LIU]


@pytest.mark.parametrize(("d", "L", "H"), CONFIGS)
def test_analytic_params_match_model(d: int, L: int, H: int) -> None:
    config = make_config(d, L, H)
    model = TransformerLanguageModel(config)
    assert compute_params(config) == model.get_compute_params()
    assert total_params(config) == model.get_total_params()


@pytest.mark.parametrize(("d", "L", "H", "published", "sig_figs"), BRYANT_LIU)
def test_total_params_match_bryant_liu(
    d: int, L: int, H: int, published: float, sig_figs: int
) -> None:
    assert to_sig_figs(total_params(make_config(d, L, H)), sig_figs) == published


@pytest.mark.parametrize(("d", "L", "H"), CONFIGS)
def test_attention_flops_are_12_L_T_d(d: int, L: int, H: int) -> None:
    # H·Q = d, so the attention term must not depend on how d is split into heads
    config = make_config(d, L, H)
    attention = flops_per_token(config) - 6 * matmul_params(config)
    assert attention == 12 * L * config.context_length * d


def test_attention_flops_independent_of_num_heads() -> None:
    assert len({flops_per_token(make_config(192, 6, H)) for H in (1, 2, 3, 6)}) == 1
