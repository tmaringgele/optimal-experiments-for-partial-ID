"""Main evaluation function for the GetUselessExperiments algorithm.

Evaluates how many experiments can be pruned on BIF-derived causal DAGs
with randomly added latent confounders.
"""
import json
import os
import networkx as nx
import numpy as np
import pandas as pd

from .bif_parser import parse_bif
from .causal_graph import CausalGraph
from .get_useless import sample_query, sample_experiments, get_useless_experiments


DEFAULT_THETA_CONFIG = {
    'CF_fraction': 0.0,
    'W_sizes': [1],
    'Z_sizes': [1],
}

DEFAULT_EXPERIMENT_CONFIG = {
    'CF_fraction': 0.0,
    'W_sizes': [1, 2, 3],
    'Z_sizes': [1, 2, 3],
}


def evaluate(
    bif_path,
    confounder_ratio_range=(0.1, 0.5),
    n_simulations=100,
    seed=42,
    experiment_set_size=200,
    theta_config=None,
    experiment_config=None,
    verbose=False,
    save_graphs=None,
):
    """Evaluate GetUselessExperiments on a BIF graph.

    For each simulation:
    1. Parse the BIF file to get the DAG structure
    2. Sample a target |U|/|V| ratio uniformly from confounder_ratio_range
    3. Add that many latent confounders between randomly chosen variable pairs
    4. Sample a query theta according to theta_config
    5. Sample candidate experiments according to experiment_config
    6. Run GetUselessExperiments to identify useless ones
    7. Record statistics

    Args:
        bif_path: path to a .bif file
        confounder_ratio_range: (a, b) interval; per simulation, |U|/|V| is
            drawn uniformly from [a, b] and that many confounders are added.
            Use (a, a) to fix the ratio exactly.
        n_simulations: number of random simulations
        seed: random seed for reproducibility
        experiment_set_size: number of candidate experiments |A| to sample
        theta_config: dict controlling query sampling (see experiment_setup.md)
        experiment_config: dict controlling experiment sampling
        verbose: if True, print progress
        save_graphs: if set to a file path, saves the full graph/query/experiment
            data for every simulation to a JSON file. Each entry is keyed by
            sim_id = "<graph_name>__sim<sim>", which matches the (graph, sim)
            columns in the returned DataFrame. If the file already exists, new
            entries are merged into it.

    Returns:
        pd.DataFrame with one row per simulation
    """
    if theta_config is None:
        theta_config = DEFAULT_THETA_CONFIG
    if experiment_config is None:
        experiment_config = DEFAULT_EXPERIMENT_CONFIG

    variables, edges = parse_bif(bif_path)
    graph_name = os.path.basename(bif_path)
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Graph: {graph_name}, |V|={len(variables)}, "
              f"|A|={experiment_set_size}, ratio_range={confounder_ratio_range}")

    results = []
    graph_records = {}  # sim_id -> reconstruction dict, populated if save_graphs is set

    # Build a bare graph (no confounders) for query sampling
    bare_cg = CausalGraph(variables, edges, [])

    for sim in range(n_simulations):
        # Sample query theta on the bare graph (before adding confounders)
        query_worlds = sample_query(bare_cg, theta_config, rng)
        if query_worlds is None:
            continue

        # Guarantee that every node in X^(1) has a confounder with at least one Y^(1)
        Y1, X1 = query_worlds[0]
        y1_list = sorted(Y1)
        forced_confounders = {
            (x, rng.choice(y1_list))
            for x in X1
        }

        # Sample target |U|/|V| ratio uniformly from the range.
        # Clamp lower bound to the forced-confounder floor so r is always
        # achievable. Use int() (floor) instead of round() to guarantee
        # target_u / |V| never exceeds r.
        # Edge case: if forced confounders alone push the floor above the upper
        # bound (only possible on very small graphs), we accept the minimum
        # forced ratio and skip the random sample entirely.
        r_lo = max(confounder_ratio_range[0],
                   len(forced_confounders) / len(variables))
        r_hi = confounder_ratio_range[1]
        r = rng.uniform(r_lo, r_hi) if r_lo <= r_hi else r_lo
        target_u = max(len(forced_confounders), int(r * len(variables)))

        # All candidate pairs excluding the forced confounders
        candidate_pairs = [
            (v1, v2)
            for i, v1 in enumerate(variables)
            for v2 in variables[i + 1:]
            if (v1, v2) not in forced_confounders and (v2, v1) not in forced_confounders
        ]

        n_additional = min(target_u - len(forced_confounders), len(candidate_pairs))
        if n_additional > 0:
            indices = rng.choice(len(candidate_pairs), size=n_additional, replace=False)
            confounders = list(forced_confounders) + [candidate_pairs[i] for i in indices]
        else:
            confounders = list(forced_confounders)

        cg = CausalGraph(variables, edges, confounders)

        # Sample candidate experiments
        experiments = sample_experiments(
            cg, experiment_config, experiment_set_size, rng
        )

        if save_graphs is not None:
            sim_id = f"{graph_name}__sim{sim}"
            graph_records[sim_id] = {
                "sim_id": sim_id,
                "graph": graph_name,
                "sim": sim,
                "variables": list(variables),
                "edges": [list(e) for e in edges],
                "confounders": [list(c) for c in confounders],
                "query_worlds": [
                    {"Y": sorted(Y), "X": sorted(X)}
                    for Y, X in query_worlds
                ],
                "experiments": [
                    [{"W": sorted(W), "Z": sorted(Z)} for W, Z in experiment]
                    for experiment in experiments
                ],
            }

        # Compute R_theta and R*_theta for reporting
        R_theta = cg.get_R_theta(query_worlds)
        R_theta_star = cg.get_R_theta_star(query_worlds)

        # Run GetUselessExperiments
        useless, stats = get_useless_experiments(
            cg, query_worlds, experiments
        )

        n_experiments = len(experiments)
        n_cf = sum(1 for e in experiments if len(e) == 2)
        query_str = "; ".join(
            f"Y={set(Y)}, X={set(X)}" for Y, X in query_worlds
        )

        # Min intervention-outcome distance (undirected shortest path)
        obs_undirected = nx.Graph()
        obs_undirected.add_nodes_from(variables)
        obs_undirected.add_edges_from(edges)
        min_xy_dist = float('inf')
        for Y, X in query_worlds:
            for x in X:
                for y in Y:
                    if nx.has_path(obs_undirected, x, y):
                        d = nx.shortest_path_length(obs_undirected, x, y)
                        min_xy_dist = min(min_xy_dist, d)
        if min_xy_dist == float('inf'):
            min_xy_dist = None

        result = {
            'graph': graph_name,
            'n_vars': len(variables),
            'u_v_ratio': len(confounders) / len(variables),
            'sim': sim,
            'n_confounders': len(confounders),
            'n_districts': len(cg.get_districts()),
            'query': query_str,
            'query_is_cf': len(query_worlds) == 2,
            'min_intervention_outcome_dist': min_xy_dist,
            'n_R_theta': len(R_theta),
            'n_R_theta_star': len(R_theta_star),
            'n_experiments': n_experiments,
            'n_cf_experiments': n_cf,
            'fraction_cf': n_cf / n_experiments if n_experiments else 0,
            'n_useless': len(useless),
            'n_useful': n_experiments - len(useless),
            'fraction_pruned': len(useless) / n_experiments if n_experiments else 0,
            'id_pruned': stats['id_pruned'],
            'blocked_pruned': stats['blocked_pruned'],
        }
        results.append(result)

        if verbose and (sim + 1) % 10 == 0:
            print(f"  u/v={result['u_v_ratio']:.2f}, sim {sim+1}/{n_simulations}: "
                  f"pruned {result['fraction_pruned']:.1%}")

    if save_graphs is not None and graph_records:
        existing = {}
        if os.path.exists(save_graphs):
            with open(save_graphs, "r") as f:
                existing = json.load(f)
        existing.update(graph_records)
        with open(save_graphs, "w") as f:
            json.dump(existing, f, indent=2)

    return pd.DataFrame(results)
