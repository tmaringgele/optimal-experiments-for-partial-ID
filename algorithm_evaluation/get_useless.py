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


def _sample_world(causal_graph, variables, w_size, z_size, rng,
                   use_distance=False, available_dists=None,
                   pairs_by_dist=None, distance_mean=None, distance_sd=None):
    """Sample a single (Y_set, X_set) world with X ⊆ ancestors(Y).

    Returns (frozenset(Y), frozenset(X)) or None on failure.
    """
    n = len(variables)
    max_retries = 50

    if use_distance:
        target_dist = _sample_size(distance_mean, distance_sd, rng, min_val=1)
        closest_dist = min(available_dists, key=lambda d: abs(d - target_dist))
        candidates = pairs_by_dist[closest_dist]
        idx = rng.integers(len(candidates))
        x_seed, y_seed = candidates[idx]

        Y_set = {y_seed}
        pool = [v for v in variables if v != x_seed and v != y_seed]
        if len(pool) > 0 and w_size > 1:
            extra = min(w_size - 1, len(pool))
            chosen = rng.choice(len(pool), size=extra, replace=False)
            for i in chosen:
                Y_set.add(pool[i])

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

        if not X_set and anc_Y:
            anc_list = list(anc_Y)
            z_actual = min(z_size, len(anc_list))
            idx = rng.choice(len(anc_list), size=z_actual, replace=False)
            X_set = {anc_list[i] for i in idx}

        if X_set:
            return (frozenset(Y_set), frozenset(X_set))
        return None
    else:
        for _retry in range(max_retries):
            w_actual = min(w_size, n)
            w_indices = rng.choice(n, size=w_actual, replace=False)
            Y_set = {variables[i] for i in w_indices}

            anc_Y = causal_graph.observed_ancestors(Y_set)
            if not anc_Y:
                continue

            anc_list = list(anc_Y)
            z_actual = min(z_size, len(anc_list))
            z_indices = rng.choice(len(anc_list), size=z_actual, replace=False)
            X_set = {anc_list[i] for i in z_indices}
            return (frozenset(Y_set), frozenset(X_set))

        # Fallback
        for v in variables:
            anc = causal_graph.observed_ancestors({v})
            if anc:
                return (frozenset({v}), frozenset({rng.choice(list(anc))}))
        return None


def _prepare_distance_data(causal_graph):
    """Pre-compute distance data for distance-based sampling."""
    variables = causal_graph.observed
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
        return None, None
    return sorted(pairs_by_dist.keys()), pairs_by_dist


def sample_query(causal_graph, theta_config, rng):
    """Sample a query theta according to theta_config.

    Returns a list of (Y_set, X_set) tuples representing counterfactual
    worlds (1 or 2).  X is always a subset of ancestors(Y).

    CF_fraction controls the probability of a 2-world counterfactual query.
    For 2-world queries the second world shares at least one intervention
    variable with the first.

    Args:
        causal_graph: CausalGraph object
        theta_config: dict with keys CF_fraction, W_size_mean/sd,
                      Z_size_mean/sd, and optionally
                      intervention_outcome_distance_mean/sd
        rng: numpy random Generator

    Returns:
        list of (frozenset(Y), frozenset(X)) tuples, or None if no valid
        query could be constructed
    """
    variables = causal_graph.observed

    use_distance = (
        theta_config.get('intervention_outcome_distance_mean') is not None
    )
    available_dists, pairs_by_dist = None, None
    if use_distance:
        available_dists, pairs_by_dist = _prepare_distance_data(causal_graph)
        if available_dists is None:
            return None

    cf_fraction = theta_config.get('CF_fraction', 0.0)
    is_cf = rng.random() < cf_fraction
    n_worlds = 2 if is_cf else 1

    dist_kwargs = dict(
        use_distance=use_distance, available_dists=available_dists,
        pairs_by_dist=pairs_by_dist,
        distance_mean=theta_config.get('intervention_outcome_distance_mean'),
        distance_sd=theta_config.get('intervention_outcome_distance_sd'),
    )

    # --- First world ---
    w_size = _sample_size(
        theta_config['W_size_mean'], theta_config['W_size_sd'], rng)
    z_size = _sample_size(
        theta_config['Z_size_mean'], theta_config['Z_size_sd'], rng)

    world1 = _sample_world(causal_graph, variables, w_size, z_size, rng,
                           **dist_kwargs)
    if world1 is None:
        return None

    worlds = [world1]

    if n_worlds == 2:
        # Second world must share at least one intervention with the first
        _, X1 = world1
        w_size2 = _sample_size(
            theta_config['W_size_mean'], theta_config['W_size_sd'], rng)
        z_size2 = _sample_size(
            theta_config['Z_size_mean'], theta_config['Z_size_sd'], rng)

        world2 = _sample_cf_world(causal_graph, variables, w_size2, z_size2,
                                  X1, rng, **dist_kwargs)
        if world2 is not None:
            worlds.append(world2)

    return worlds


def _sample_cf_world(causal_graph, variables, w_size, z_size, Z_prev, rng,
                     use_distance=False, available_dists=None,
                     pairs_by_dist=None, distance_mean=None, distance_sd=None):
    """Sample a world for a counterfactual experiment.

    Guarantees at least one shared intervention with Z_prev by picking the
    first intervention variable from Z_prev, then sampling the rest normally.
    """
    n = len(variables)
    max_retries = 50
    Z_prev_list = list(Z_prev)

    for _retry in range(max_retries):
        # Pick W
        if use_distance:
            target_dist = _sample_size(distance_mean, distance_sd, rng,
                                       min_val=1)
            closest_dist = min(available_dists,
                               key=lambda d: abs(d - target_dist))
            candidates = pairs_by_dist[closest_dist]
            idx = rng.integers(len(candidates))
            _, w_seed = candidates[idx]

            W_set = {w_seed}
            pool = [v for v in variables if v != w_seed]
            if len(pool) > 0 and w_size > 1:
                extra = min(w_size - 1, len(pool))
                chosen = rng.choice(len(pool), size=extra, replace=False)
                for i in chosen:
                    W_set.add(pool[i])
        else:
            w_actual = min(w_size, n)
            w_indices = rng.choice(n, size=w_actual, replace=False)
            W_set = {variables[i] for i in w_indices}

        anc_W = causal_graph.observed_ancestors(W_set)
        if not anc_W:
            continue

        # Shared intervention: pick one from Z_prev that is an ancestor of W
        shared_candidates = [z for z in Z_prev_list if z in anc_W]
        if not shared_candidates:
            continue

        shared = shared_candidates[rng.integers(len(shared_candidates))]
        Z_set = {shared}

        # Fill remaining Z from ancestors(W)
        anc_pool = list(anc_W - Z_set)
        needed = z_size - 1
        if needed > 0 and len(anc_pool) > 0:
            extra = min(needed, len(anc_pool))
            chosen = rng.choice(len(anc_pool), size=extra, replace=False)
            for i in chosen:
                Z_set.add(anc_pool[i])

        return (frozenset(W_set), frozenset(Z_set))

    return None


def sample_experiments(causal_graph, experiment_config, n_experiments, rng):
    """Sample candidate experiments according to experiment_config.

    Each experiment is a tuple of (W, Z) frozenset pairs (1 or 2 worlds).
    Z is always a subset of observed ancestors of W.

    CF_fraction controls the probability that an experiment is a 2-world
    counterfactual.  For counterfactual experiments, the second world shares
    at least one intervention variable with the first.

    Args:
        causal_graph: CausalGraph object
        experiment_config: dict with keys CF_fraction, W_size_mean/sd,
                           Z_size_mean/sd, and optionally
                           intervention_outcome_distance_mean/sd
        n_experiments: number of experiments to sample
        rng: numpy random Generator

    Returns:
        list of experiment tuples; each experiment is a tuple of
        (frozenset(W), frozenset(Z)) pairs
    """
    variables = causal_graph.observed

    use_distance = (
        experiment_config.get('intervention_outcome_distance_mean') is not None
    )
    available_dists, pairs_by_dist = None, None
    if use_distance:
        available_dists, pairs_by_dist = _prepare_distance_data(causal_graph)
        if available_dists is None:
            use_distance = False

    cf_fraction = experiment_config.get('CF_fraction', 0.0)

    dist_kwargs = dict(
        use_distance=use_distance, available_dists=available_dists,
        pairs_by_dist=pairs_by_dist,
        distance_mean=experiment_config.get('intervention_outcome_distance_mean'),
        distance_sd=experiment_config.get('intervention_outcome_distance_sd'),
    )

    experiments = []

    for _ in range(n_experiments):
        is_cf = rng.random() < cf_fraction
        n_worlds = 2 if is_cf else 1

        w_size = _sample_size(
            experiment_config['W_size_mean'],
            experiment_config['W_size_sd'], rng)
        z_size = _sample_size(
            experiment_config['Z_size_mean'],
            experiment_config['Z_size_sd'], rng)

        world1 = _sample_world(causal_graph, variables, w_size, z_size, rng,
                               **dist_kwargs)
        if world1 is None:
            continue

        worlds = [world1]

        if n_worlds == 2:
            _, Z1 = world1
            w_size2 = _sample_size(
                experiment_config['W_size_mean'],
                experiment_config['W_size_sd'], rng)
            z_size2 = _sample_size(
                experiment_config['Z_size_mean'],
                experiment_config['Z_size_sd'], rng)

            world2 = _sample_cf_world(causal_graph, variables, w_size2,
                                      z_size2, Z1, rng, **dist_kwargs)
            if world2 is not None:
                worlds.append(world2)

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
