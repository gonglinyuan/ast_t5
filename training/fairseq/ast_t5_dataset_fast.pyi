from typing import List, Tuple

import numpy as np

configured: bool


def configure(
    tok_l: int,
    tok_r: int,
    tok_0: int,
    tok_py_func_def: int,
    tok_py_block: int,
    tok_py_expr_stmt: int,
    tok_py_string: int,
    tok_py_func_body: int,
    identifier_ndtypes: List[int],
    sentinel_idx_begin: int,
    max_positions: int,
    ndtype_begin: int
    ): ...


def generate_masks(
    seq_arr: np.ndarray,
    ndtype_arr: np.ndarray,
    mask_prob: float,
    flat_multiple_len: int,
    threshold_lb: int,
    threshold_ub: int,
    obf_prob: float,
    obf_ratio: float,
    t2c_prob: float,
    t2c_ratio: float,
    seed: int
) -> Tuple[np.ndarray, list, list, list, np.ndarray, np.ndarray
]: ...
