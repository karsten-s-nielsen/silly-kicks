"""The REFERENCE ghost leaf walk: the numpy bodies of `_vectorized_leaf_indices` / `_vectorized_leaf_values` as
they stood at `2474b73`, kept VERBATIM as the test oracle (spec r3 D16).

Every committed ghost-GK and ghost-outfield artifact was trained in an HF env without numba, so it ran exactly
these bodies. The flat active-set walk that replaced them must be bit-identical to them (`np.array_equal`); this
module is what "bit-identical" is measured against. Do not edit it to follow the production code.
"""

from __future__ import annotations

import numpy as np


def reference_leaf_indices(nodes_list: list[np.ndarray], X: np.ndarray) -> np.ndarray:  # noqa: N803 -- verbatim
    """Shape (n_samples, n_trees): the LOCAL leaf index per sample per tree (no depth-cap raise)."""
    n_samples = X.shape[0]
    n_trees = len(nodes_list)
    leaves = np.zeros((n_samples, n_trees), dtype=np.intp)

    for t_idx, nodes in enumerate(nodes_list):
        current = np.zeros(n_samples, dtype=np.intp)

        for _ in range(100):  # depth bound
            node_data = nodes[current]
            is_leaf = node_data["left"] == 0
            if np.all(is_leaf):
                break

            feature_idx = node_data["feature_idx"]
            thresholds = node_data["num_threshold"]
            go_left_missing = node_data["missing_go_to_left"]

            # Gather feature values for each sample's current split feature
            feat_vals = X[np.arange(n_samples), feature_idx]

            # Decision: NaN handling + threshold comparison
            is_nan = np.isnan(feat_vals)
            go_left = np.where(
                is_nan,
                go_left_missing.astype(bool),
                feat_vals <= thresholds,
            )

            # Advance non-leaf samples
            next_node = np.where(go_left, node_data["left"], node_data["right"])
            current = np.where(is_leaf, current, next_node)

        leaves[:, t_idx] = current

    return leaves


def reference_leaf_values(nodes_list: list[np.ndarray], X: np.ndarray) -> np.ndarray:  # noqa: N803 -- verbatim
    """Shape (n_samples,): each sample's reached-leaf values summed tree by tree (raises past the depth cap)."""
    n_samples = X.shape[0]
    total = np.zeros(n_samples, dtype=np.float64)

    for nodes in nodes_list:
        current = np.zeros(n_samples, dtype=np.intp)

        for _ in range(100):  # depth bound
            node_data = nodes[current]
            is_leaf = node_data["left"] == 0
            if np.all(is_leaf):
                break

            feat_vals = X[np.arange(n_samples), node_data["feature_idx"]]
            go_left = np.where(
                np.isnan(feat_vals),
                node_data["missing_go_to_left"].astype(bool),
                feat_vals <= node_data["num_threshold"],
            )
            next_node = np.where(go_left, node_data["left"], node_data["right"])
            current = np.where(is_leaf, current, next_node)

        # Convergence guard (raise, NOT assert — `python -O` strips asserts): a tree
        # deeper than the cap would otherwise read an internal node's `value` (garbage).
        if not np.all(nodes[current]["left"] == 0):
            msg = "leaf traversal did not converge within depth cap"
            raise RuntimeError(msg)
        total += nodes[current]["value"]

    return total
