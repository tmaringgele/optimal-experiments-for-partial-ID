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
    worlds.  When intervention_outcome_distance_mean/sd are provided, a
    seed pair (x, y) is chosen at the target distance.  Otherwise, Y is
    picked uniformly at random and X is sampled from ancestors(Y).

    Args:
        causal_graph: CausalGraph object
        theta_config: dict with keys CF_worlds_mean/sd, W_size_mean/sd,
                      Z_size_mean/sd, and optionally
                      intervention_outcome_distance_mean/sd
        rng: numpy random Generator

    Returns:
        list of (frozenset(Y), frozenset(X)) tuples, or None if no valid
        query could be constructed
    """
    variables = causal_graph.observed
    n = len(variables)

    use_distance = (
        theta_config.get('intervention_outcome_distance_mean') is not None
    )

    if use_distance:
        obs_undirected = nx.Graph()
        obs_undirected.add_nodes_from(variables)
        obs_undirected.add_edges_from(causal_graph.directed_edges)
        path_lengths = dict(nx.all_pairs_shortest_path_length(obs_undirected))

        pairs_by_dist = defaultdict(list)
        for x in variables:
            for y in variables:
                if x != y:
                    d = path_lengths.get(x, {}).get(y, None)
                    if d is not None:
                        pairs_by_dist[d].append((x, y))

        if not pairs_by_dist:
            return None
        available_dists = sorted(pairs_by_dist.keys())

    n_worlds = _sample_size(
        theta_config['CF_worlds_mean'], theta_config['CF_worlds_sd'], rng
    )

    max_retries = 50
    worlds = []
    for _ in range(n_worlds):
        w_size = _sample_size(
            theta_config['W_size_mean'], theta_config['W_size_sd'], rng
        )
        z_size = _sample_size(
            theta_config['Z_size_mean'], theta_config['Z_size_sd'], rng
        )

        if use_distance:
            target_dist = _sample_size(
                theta_config['intervention_outcome_distance_mean'],
                theta_config['intervention_outcome_distance_sd'],
                rng, min_val=1,
            )
            closest_dist = min(available_dists,
                               key=lambda d: abs(d - target_dist))
            candidates = pairs_by_dist[closest_dist]
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

            # Build X_set around x_seed, restricted to ancestors(Y)
            anc_Y = causal_graph.observed_ancestors(Y_set)
            X_set = set()
            if x_seed in anc_Y:
                X_set.add(x_seed)
            anc_pool = list(anc_Y - X_set)
            needed = z_size - len(X_set)
            if needed > 0 and len(anc_pool) > 0:
                extra = min(needed, len(anc_pool))
                chosen = rng.choice(len(anc_pool), size=extra, replace=False)
                for i in chosen:
                    X_set.add(anc_pool[i])

            if not X_set:
                # seed pair didn't work; fall through to ancestor-based
                anc_Y = causal_graph.observed_ancestors(Y_set)
                if anc_Y:
                    anc_list = list(anc_Y)
                    z_actual = min(z_size, len(anc_list))
                    idx = rng.choice(len(anc_list), size=z_actual, replace=False)
                    X_set = {anc_list[i] for i in idx}

            if X_set:
                worlds.append((frozenset(Y_set), frozenset(X_set)))
        else:
            # No distance control: pick Y randomly, X from ancestors(Y)
            valid = False
            for _retry in range(max_retries):
                w_actual = min(w_size, n)
                w_indices = rng.choice(n, size=w_actual, replace=False)
                Y_set = {variables[i] for i in w_indices}

                anc_Y = causal_graph.observed_ancestors(Y_set)
                if not anc_Y:
                    continue

                anc_list = list(anc_Y)
                z_actual = min(z_size, len(anc_list))
                z_indices = rng.choice(len(anc_list), size=z_actual,
                                       replace=False)
                X_set = {anc_list[i] for i in z_indices}
                valid = True
                break

            if not valid:
                # Fallback: find any variable with ancestors
                for v in variables:
                    anc = causal_graph.observed_ancestors({v})
                    if anc:
                        Y_set = {v}
                        X_set = {rng.choice(list(anc))}
                        valid = True
                        break

            if valid:
                worlds.append((frozenset(Y_set), frozenset(X_set)))

    return worlds if worlds else None


def sample_experiments(causal_graph, experiment_config, n_experiments, rng):
    """Sample candidate experiments according to experiment_config.

    Each experiment is a tuple of (W, Z) frozenset pairs, one per
    counterfactual world.  Z is always constrained to be a subset of
    the observed ancestors of W in the DAG.

    When intervention_outcome_distance_mean/sd are provided, a seed pair
    (z, w) is chosen at the target shortest-path distance and the sets
    are expanded around it.

    Args:
        causal_graph: CausalGraph object
        experiment_config: dict with keys CF_worlds_mean/sd,
                           W_size_mean/sd, Z_size_mean/sd, and optionally
                           intervention_outcome_distance_mean/sd
        n_experiments: number of experiments to sample
        rng: numpy random Generator

    Returns:
        list of experiment tuples; each experiment is a tuple of
        (frozenset(W), frozenset(Z)) pairs
    """
    variables = causal_graph.observed
    n = len(variables)

    use_distance = 'intervention_outcome_distance_mean' in experiment_config

    if use_distance:
        obs_undirected = nx.Graph()
        obs_undirected.add_nodes_from(variables)
        obs_undirected.add_edges_from(causal_graph.directed_edges)
        path_lengths = dict(nx.all_pairs_shortest_path_length(obs_undirected))

        pairs_by_dist = defaultdict(list)
        for x in variables:
            for y in variables:
                if x != y:
                    d = path_lengths.get(x, {}).get(y, None)
                    if d is not None:
                        pairs_by_dist[d].append((x, y))

        if not pairs_by_dist:
            use_distance = False
        else:
            available_dists = sorted(pairs_by_dist.keys())

    experiments = []
    max_retries = 50

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

            valid = False
            for _retry in range(max_retries):
                if use_distance:
                    target_dist = _sample_size(
                        experiment_config['intervention_outcome_distance_mean'],
                        experiment_config['intervention_outcome_distance_sd'],
                        rng, min_val=1,
                    )
                    closest_dist = min(available_dists,
                                       key=lambda d: abs(d - target_dist))
                    candidates = pairs_by_dist[closest_dist]
                    idx = rng.integers(len(candidates))
                    z_seed, w_seed = candidates[idx]

                    # Build W around w_seed
                    W_set = {w_seed}
                    pool = [v for v in variables if v != z_seed and v != w_seed]
                    if len(pool) > 0 and w_size > 1:
                        extra = min(w_size - 1, len(pool))
                        chosen = rng.choice(len(pool), size=extra, replace=False)
                        for i in chosen:
                            W_set.add(pool[i])

                    # Z must be subset of ancestors(W)
                    anc_W = causal_graph.observed_ancestors(W_set)
                    if not anc_W:
                        continue  # retry: W has no observed ancestors

                    Z_set = set()
                    if z_seed in anc_W:
                        Z_set.add(z_seed)
                    anc_pool = list(anc_W - Z_set)
                    needed = z_size - len(Z_set)
                    if needed > 0 and len(anc_pool) > 0:
                        extra = min(needed, len(anc_pool))
                        chosen = rng.choice(len(anc_pool), size=extra, replace=False)
                        for i in chosen:
                            Z_set.add(anc_pool[i])

                    if not Z_set:
                        continue  # retry: couldn't form a valid Z
                else:
                    # Pick W first
                    w_actual = min(w_size, n)
                    w_indices = rng.choice(n, size=w_actual, replace=False)
                    W_set = {variables[i] for i in w_indices}

                    # Z from ancestors(W)
                    anc_W = causal_graph.observed_ancestors(W_set)
                    if not anc_W:
                        continue  # retry

                    anc_list = list(anc_W)
                    z_actual = min(z_size, len(anc_list))
                    z_indices = rng.choice(len(anc_list), size=z_actual, replace=False)
                    Z_set = {anc_list[i] for i in z_indices}

                valid = True
                break

            if not valid:
                # Fallback: pick a single root->leaf pair
                for v in variables:
                    anc = causal_graph.observed_ancestors({v})
                    if anc:
                        W_set = {v}
                        Z_set = {rng.choice(list(anc))}
                        valid = True
                        break

            if valid:
                worlds.append((frozenset(W_set), frozenset(Z_set)))

        if worlds:
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
