"""Implementation of GetUselessExperiments (Algorithm 3 from the paper).

Combines two pruning criteria:
1. ID check: single-world experiments whose outcome is identifiable from
   P(V) have zero potency
2. AllPathsBlocked: experiments where all paths from R*_theta to outcomes
   pass through interventions have zero potency

Also provides sampling functions for queries and experiments.
"""
import numpy as np
import networkx as nx
from collections import defaultdict

from .causal_graph import CausalGraph
from .id_algorithm import is_identifiable


# ---------------------------------------------------------------------------
#  Sampling helpers
# ---------------------------------------------------------------------------

def _sample_size(mean, sd, rng, min_val=1):
    """Sample a positive integer from N(mean, sd), clamped to >= min_val."""
    if sd == 0:
        return max(min_val, round(mean))
    return max(min_val, round(rng.normal(mean, sd)))


def sample_query(causal_graph, theta_config, rng):
    """Sample a query theta according to theta_config.

    Returns a list of (Y_set, X_set) tuples representing counterfactual
    worlds.  For each world a seed pair (x, y) is chosen at the target
    shortest-path distance; then both sets are expanded to the desired sizes.

    Args:
        causal_graph: CausalGraph object
        theta_config: dict with keys CF_worlds_mean/sd, W_size_mean/sd,
                      Z_size_mean/sd, intervention_outcome_distance_mean/sd
        rng: numpy random Generator

    Returns:
        list of (frozenset(Y), frozenset(X)) tuples, or None if no valid
        query could be constructed
    """
    variables = causal_graph.observed
    n = len(variables)

    # Build undirected observed graph for distance computation
    obs_undirected = nx.Graph()
    obs_undirected.add_nodes_from(variables)
    obs_undirected.add_edges_from(causal_graph.directed_edges)

    # All-pairs shortest paths (only among observed variables)
    path_lengths = dict(nx.all_pairs_shortest_path_length(obs_undirected))

    # Group variable pairs by distance
    pairs_by_dist = defaultdict(list)
    for x in variables:
        for y in variables:
            if x != y:
                d = path_lengths.get(x, {}).get(y, None)
                if d is not None:
                    pairs_by_dist[d].append((x, y))

    if not pairs_by_dist:
        return None  # disconnected graph with no reachable pairs

    available_dists = sorted(pairs_by_dist.keys())

    n_worlds = _sample_size(
        theta_config['CF_worlds_mean'], theta_config['CF_worlds_sd'], rng
    )

    worlds = []
    for _ in range(n_worlds):
        target_dist = _sample_size(
            theta_config['intervention_outcome_distance_mean'],
            theta_config['intervention_outcome_distance_sd'],
            rng, min_val=1,
        )
        w_size = _sample_size(
            theta_config['W_size_mean'], theta_config['W_size_sd'], rng
        )
        z_size = _sample_size(
            theta_config['Z_size_mean'], theta_config['Z_size_sd'], rng
        )

        # Find closest achievable distance
        closest_dist = min(available_dists, key=lambda d: abs(d - target_dist))
        candidates = pairs_by_dist[closest_dist]

        # Pick a seed pair
        idx = rng.integers(len(candidates))
        x_seed, y_seed = candidates[idx]

        # Build Y_set around y_seed
        Y_set = {y_seed}
        pool = [v for v in variables if v != x_seed and v != y_seed]
        if len(pool) > 0 and w_size > 1:
            extra = min(w_size - 1, len(pool))
            chosen = rng.choice(len(pool), size=extra, replace=False)
            for i in chosen:
                Y_set.add(pool[i])

        # Build X_set around x_seed (disjoint from Y_set)
        X_set = {x_seed}
        pool = [v for v in variables if v not in Y_set and v != x_seed]
        if len(pool) > 0 and z_size > 1:
            extra = min(z_size - 1, len(pool))
            chosen = rng.choice(len(pool), size=extra, replace=False)
            for i in chosen:
                X_set.add(pool[i])

        worlds.append((frozenset(Y_set), frozenset(X_set)))

    return worlds


def sample_experiments(variables, experiment_config, n_experiments, rng):
    """Sample candidate experiments according to experiment_config.

    Each experiment is a tuple of (W, Z) frozenset pairs, one per
    counterfactual world.

    Args:
        variables: list of observed variable names
        experiment_config: dict with keys CF_worlds_mean/sd,
                           W_size_mean/sd, Z_size_mean/sd
        n_experiments: number of experiments to sample
        rng: numpy random Generator

    Returns:
        list of experiment tuples; each experiment is a tuple of
        (frozenset(W), frozenset(Z)) pairs
    """
    n = len(variables)
    experiments = []

    for _ in range(n_experiments):
        n_worlds = _sample_size(
            experiment_config['CF_worlds_mean'],
            experiment_config['CF_worlds_sd'],
            rng,
        )
        worlds = []
        for _ in range(n_worlds):
            w_size = _sample_size(
                experiment_config['W_size_mean'],
                experiment_config['W_size_sd'],
                rng,
            )
            z_size = _sample_size(
                experiment_config['Z_size_mean'],
                experiment_config['Z_size_sd'],
                rng,
            )
            total = min(w_size + z_size, n)
            w_size = min(w_size, total)
            z_size = max(1, min(z_size, total - w_size))
            w_size = max(1, total - z_size)

            indices = rng.choice(n, size=total, replace=False)
            selected = [variables[i] for i in indices]
            W = frozenset(selected[:w_size])
            Z = frozenset(selected[w_size:])
            worlds.append((W, Z))

        experiments.append(tuple(worlds))

    return experiments


# ---------------------------------------------------------------------------
#  GetUselessExperiments (Algorithm 3)
# ---------------------------------------------------------------------------

def get_useless_experiments(causal_graph, query_worlds, experiments):
    """Run GetUselessExperiments (Algorithm 3).

    For each experiment checks:
    1. If single-world and P(W|do(Z)) is identifiable -> useless (Thm 4)
    2. If AllPathsBlocked for every world -> useless (Corollary 1)

    Args:
        causal_graph: CausalGraph object
        query_worlds: list of (Y_set, X_set) tuples (the query theta)
        experiments: list of experiments, each a tuple of (W, Z) pairs

    Returns:
        useless: set of experiment indices that have zero potency
        stats: dict with pruning statistics
    """
    R_theta_star = causal_graph.get_R_theta_star(query_worlds)

    useless = set()
    id_pruned = 0
    blocked_pruned = 0

    for i, experiment in enumerate(experiments):
        # 1. ID check: only for single-world interventional experiments
        if len(experiment) == 1:
            W, Z = experiment[0]
            if is_identifiable(causal_graph, list(W), list(Z)):
                useless.add(i)
                id_pruned += 1
                continue

        # 2. AllPathsBlocked: must hold for ALL worlds in the experiment
        all_blocked = True
        for W, Z in experiment:
            if not causal_graph.all_paths_blocked(R_theta_star, W, Z):
                all_blocked = False
                break
        if all_blocked:
            useless.add(i)
            blocked_pruned += 1

    stats = {
        'id_pruned': id_pruned,
        'blocked_pruned': blocked_pruned,
        'total_pruned': id_pruned + blocked_pruned,
    }
    return useless, stats
