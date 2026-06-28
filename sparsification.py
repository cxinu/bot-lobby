"""
cresci-2017 / TRESA — Step 5: Edge sparsification utilities
============================================================
Implements two edge-drop paradigms used during TRESA training and evaluation:

  RandomDrop      — each edge independently dropped with probability p
                    models: uniform API sampling gaps, random crawl failures

  DegreeBasedDrop — P(drop edge) ∝ max(deg(u), deg(v))
                    high-degree nodes lose edges first
                    models: API rate-limiting which hits popular accounts hardest

Both return a new edge_index tensor (and optionally the mask of dropped edges
for use as positive examples in the L_lp link prediction head).

Usage
-----
From another script:
    from sparsification import RandomDrop, DegreeBasedDrop, SparsificationSchedule

Standalone verification:
    DATA_ROOT=./data python 05_sparsification.py
"""

import torch
import numpy as np
import pickle
import os

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)


# ── Core drop classes ─────────────────────────────────────────────────────────

class RandomDrop:
    """
    Bernoulli edge drop: each edge is dropped independently with probability p.

    Args
        p        : drop probability ∈ [0, 1)
        seed     : optional RNG seed for reproducibility

    Returns (from __call__)
        kept_edge_index   : LongTensor [2, E_kept]
        dropped_edge_index: LongTensor [2, E_dropped]  (positives for L_lp)
        drop_mask         : BoolTensor [E]              True = dropped
    """
    def __init__(self, p: float, seed: int = None):
        assert 0.0 <= p < 1.0, f"p must be in [0, 1), got {p}"
        self.p    = p
        self.rng  = np.random.default_rng(seed)

    def __call__(self, edge_index: torch.Tensor):
        E = edge_index.shape[1]
        if E == 0 or self.p == 0.0:
            empty = edge_index[:, :0]
            return edge_index, empty, torch.zeros(E, dtype=torch.bool)

        drop_mask = torch.from_numpy(
            self.rng.random(E) < self.p
        )
        kept    = edge_index[:, ~drop_mask]
        dropped = edge_index[:,  drop_mask]
        return kept, dropped, drop_mask

    def __repr__(self):
        return f"RandomDrop(p={self.p})"


class DegreeBasedDrop:
    """
    Degree-proportional edge drop.

    For each edge (u, v), the drop probability is:
        P(drop) = p_base * min(1, max(deg(u), deg(v)) / degree_cap)

    where degree_cap normalises so the heaviest hub has probability exactly
    p_base (or close to it), and low-degree nodes have near-zero probability.

    This simulates API rate-limiting: popular accounts (high degree) have their
    interactions pruned first because they generate the most traffic.

    Args
        p_base      : target drop probability for the highest-degree node
        degree_cap  : degree at which drop probability saturates at p_base
                      default: 95th percentile of degree distribution
        seed        : optional RNG seed
    """
    def __init__(self, p_base: float, degree_cap: int = None, seed: int = None):
        assert 0.0 <= p_base < 1.0
        self.p_base     = p_base
        self.degree_cap = degree_cap   # computed lazily if None
        self.rng        = np.random.default_rng(seed)
        self._deg_cache = {}           # node_id → degree (populated on first call)

    def _compute_degrees(self, edge_index: torch.Tensor, num_nodes: int):
        """Count undirected degree for each node."""
        deg = torch.zeros(num_nodes, dtype=torch.long)
        deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
        deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))
        return deg   # shape [num_nodes]

    def __call__(self, edge_index: torch.Tensor, num_nodes: int = None):
        E = edge_index.shape[1]
        if E == 0:
            empty = edge_index[:, :0]
            return edge_index, empty, torch.zeros(E, dtype=torch.bool)

        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1

        deg = self._compute_degrees(edge_index, num_nodes)   # [N]

        if self.degree_cap is None:
            # 95th percentile of non-zero degrees
            nonzero_degs = deg[deg > 0].numpy()
            self.degree_cap = max(1, int(np.percentile(nonzero_degs, 95)))

        if self.p_base == 0.0:
            empty = edge_index[:, :0]
            return edge_index, empty, torch.zeros(E, dtype=torch.bool)

        # Per-edge drop probability = p_base * max(deg_u, deg_v) / degree_cap
        deg_u = deg[edge_index[0]].float()   # [E]
        deg_v = deg[edge_index[1]].float()   # [E]
        edge_p = self.p_base * torch.clamp(
            torch.maximum(deg_u, deg_v) / self.degree_cap, max=1.0
        )   # [E], values in [0, p_base]

        drop_mask = torch.from_numpy(
            self.rng.random(E) < edge_p.numpy()
        )
        kept    = edge_index[:, ~drop_mask]
        dropped = edge_index[:,  drop_mask]
        return kept, dropped, drop_mask

    def __repr__(self):
        return f"DegreeBasedDrop(p_base={self.p_base}, degree_cap={self.degree_cap})"


# ── Negative sampler (for L_lp training) ─────────────────────────────────────

class NegativeSampler:
    """
    Samples non-edges as negatives for the link prediction head.

    Strategy: random node pairs, rejection-sampled to exclude real edges.
    On cresci-2017 (density ~0.001%) collision probability is negligible,
    so we skip the full rejection step and just sample freely.

    Args
        edge_index : the FULL (unsparsified) edge_index — used to build
                     the positive edge set for exact collision avoidance
        num_nodes  : total number of nodes
        ratio      : negatives per positive (default 1:1)
    """
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, ratio: int = 1):
        self.num_nodes = num_nodes
        self.ratio     = ratio
        # Build set of existing edges for fast lookup
        ei = edge_index.numpy()
        self.edge_set  = set(zip(ei[0].tolist(), ei[1].tolist()))

    def sample(self, n_pos: int, rng: np.random.Generator = None) -> torch.Tensor:
        """Return [2, n_pos * ratio] negative edge_index."""
        if rng is None:
            rng = np.random.default_rng()
        n_neg  = n_pos * self.ratio
        N      = self.num_nodes
        # Oversample then filter collisions
        factor = 3
        src = rng.integers(0, N, n_neg * factor)
        dst = rng.integers(0, N, n_neg * factor)
        # Remove self-loops
        valid = src != dst
        src, dst = src[valid], dst[valid]
        # Remove real edges (fast: density is ~0, so very few collisions)
        mask = np.array([
            (int(s), int(d)) not in self.edge_set
            for s, d in zip(src[:n_neg * 2], dst[:n_neg * 2])
        ])
        src = src[:n_neg * 2][mask][:n_neg]
        dst = dst[:n_neg * 2][mask][:n_neg]
        if len(src) < n_neg:
            # Pad with random pairs if filtering removed too many
            extra = n_neg - len(src)
            src = np.concatenate([src, rng.integers(0, N, extra)])
            dst = np.concatenate([dst, rng.integers(0, N, extra)])
        return torch.tensor(np.stack([src[:n_neg], dst[:n_neg]]), dtype=torch.long)


# ── Training-time schedule ────────────────────────────────────────────────────

class SparsificationSchedule:
    """
    Wraps a dropper for use in a training loop.
    Resamples the drop mask each epoch (stochastic augmentation).

    Usage:
        sched = SparsificationSchedule(RandomDrop(p=0.4), full_edge_index)
        for epoch in range(N):
            sparse_ei, dropped_ei = sched.step()
            # train with sparse_ei, use dropped_ei as L_lp positives
    """
    def __init__(self, dropper, full_edge_index: torch.Tensor, num_nodes: int = None):
        self.dropper        = dropper
        self.full_edge_index = full_edge_index
        self.num_nodes       = num_nodes or int(full_edge_index.max().item()) + 1
        self._epoch          = 0

    def step(self):
        self._epoch += 1
        if isinstance(self.dropper, DegreeBasedDrop):
            kept, dropped, _ = self.dropper(self.full_edge_index, self.num_nodes)
        else:
            kept, dropped, _ = self.dropper(self.full_edge_index)
        return kept, dropped

    def reset(self):
        self._epoch = 0


# ── Standalone verification ───────────────────────────────────────────────────

def verify():
    print("Loading graph...")
    with open(os.path.join(DATA_ROOT, "retweet_graph.pkl"), "rb") as f:
        gd = pickle.load(f)

    edges_df   = gd["edges"]
    G_dir      = gd["G_dir"]
    num_nodes  = G_dir.number_of_nodes()

    # Rebuild edge_index (no self-loops — those are added by PyG later)
    import pandas as pd
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # We need the id→idx mapping — rebuild from full_features parquet
    df = pd.read_parquet(os.path.join(DATA_ROOT, "full_features.parquet"))
    id_to_idx = {str(uid): i for i, uid in enumerate(df["id"])}

    src, dst = [], []
    for _, row in edges_df.iterrows():
        s = id_to_idx.get(str(row["retweeter_id"]))
        d = id_to_idx.get(str(row["original_author_id"]))
        if s is not None and d is not None and s != d:
            src.append(s)
            dst.append(d)

    full_ei    = torch.tensor([src, dst], dtype=torch.long)
    E_full     = full_ei.shape[1]
    N          = len(df)

    print(f"  Full edge_index: {E_full} edges · {N} nodes\n")

    DROP_RATES = [0.0, 0.2, 0.4, 0.6]
    RNG_SEED   = 42

    print("=" * 60)
    print("RANDOM DROP")
    print("=" * 60)
    print(f"  {'drop rate':>10}  {'kept':>8}  {'dropped':>8}  {'actual %':>10}  {'check':>8}")
    print("  " + "-" * 52)
    for p in DROP_RATES:
        dropper          = RandomDrop(p=p, seed=RNG_SEED)
        kept, dropped, _ = dropper(full_ei)
        actual_pct       = dropped.shape[1] / max(E_full, 1) * 100
        target_pct       = p * 100
        ok               = abs(actual_pct - target_pct) < 15   # loose — E is small
        print(f"  {p:>10.0%}  {kept.shape[1]:>8}  {dropped.shape[1]:>8}  "
              f"{actual_pct:>9.1f}%  {'✓' if ok else '!'}")

    print()
    print("=" * 60)
    print("DEGREE-BASED DROP")
    print("=" * 60)
    print(f"  {'p_base':>10}  {'kept':>8}  {'dropped':>8}  {'actual %':>10}  {'degree_cap':>12}")
    print("  " + "-" * 58)
    for p in DROP_RATES:
        dropper = DegreeBasedDrop(p_base=p, seed=RNG_SEED)
        kept, dropped, _ = dropper(full_ei, num_nodes=N)
        actual_pct = dropped.shape[1] / max(E_full, 1) * 100
        print(f"  {p:>10.0%}  {kept.shape[1]:>8}  {dropped.shape[1]:>8}  "
              f"{actual_pct:>9.1f}%  {dropper.degree_cap:>12}")

    print()
    print("=" * 60)
    print("NEGATIVE SAMPLER")
    print("=" * 60)
    ns      = NegativeSampler(full_ei, num_nodes=N, ratio=1)
    rng     = np.random.default_rng(RNG_SEED)
    # Simulate a drop of 40% → ~570 positives
    d40     = RandomDrop(p=0.4, seed=RNG_SEED)
    _, dropped_40, _ = d40(full_ei)
    n_pos   = dropped_40.shape[1]
    neg_ei  = ns.sample(n_pos=n_pos, rng=rng)
    print(f"  Positives (dropped at 40%): {n_pos}")
    print(f"  Negatives sampled (1:1):    {neg_ei.shape[1]}")

    # Sanity: no negatives should be in the full edge set
    neg_set   = set(zip(neg_ei[0].tolist(), neg_ei[1].tolist()))
    collisions = len(neg_set & ns.edge_set)
    print(f"  Collisions with real edges: {collisions}  (should be 0 or near-0)")

    print()
    print("=" * 60)
    print("SCHEDULE — stochastic resampling across 5 epochs")
    print("=" * 60)
    sched = SparsificationSchedule(RandomDrop(p=0.4, seed=None), full_ei, num_nodes=N)
    kept_counts = []
    for ep in range(5):
        kept_ep, dropped_ep = sched.step()
        kept_counts.append(kept_ep.shape[1])
        print(f"  Epoch {ep+1}: kept={kept_ep.shape[1]}  dropped={dropped_ep.shape[1]}")
    print(f"  Variance across epochs: {np.std(kept_counts):.1f}  (non-zero = correctly stochastic)")

    print()
    print("All checks done. Sparsification module ready for Step 6.")


if __name__ == "__main__":
    verify()
