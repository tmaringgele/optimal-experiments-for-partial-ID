"""Main evaluation function for the GetUselessExperiments algorithm.

Evaluates how many experiments can be pruned on BIF-derived causal DAGs
with randomly added latent confounders.
"""
import os
import numpy as np
import pandas as pd

from .bif_parser import parse_bif
from .causal_graph import CausalGraph
from .get_useless import generate_experiments, get_useless_experiments


def choose_random_query(causal_graph, rng):
    """Choose a random query P(y|do(x)) by picking Y, then a parent of Y as X.

    Args:
        causal_graph: CausalGraph object
        rng: numpy random generator

    Returns:
        (y_var, x_var) or None if no valid query exists
    """
    variables = causal_graph.observed
    # Find variables that have at least one observed parent
    candidates = []
    for v in variables:
        parents = causal_graph.observed_parents(v)
        if parents:
            candidates.append((v, parents))

    if not candidates:
        return None

    idx = rng.integers(len(candidates))
    y_var, parents = candidates[idx]
    x_var = parents[rng.integers(len(parents))]
    return y_var, x_var


def evaluate(
    bif_path,
    confounding_probs=(0.05, 0.15, 0.25, 0.35),
    n_simulations=100,
    seed=42,
    max_experiment_size=None,
    verbose=False,
):
    """Evaluate GetUselessExperiments on a BIF graph.

    For each confounding probability and simulation:
    1. Parse the BIF file to get the DAG structure
    2. Add random latent confounders with the given probability
    3. Choose a random query P(y|do(x))
    4. Enumerate candidate experiments
    5. Run GetUselessExperiments to identify useless ones
    6. Record statistics

    Args:
        bif_path: path to a .bif file
        confounding_probs: list of probabilities for adding confounders
        n_simulations: number of random simulations per probability
        seed: random seed for reproducibility
        max_experiment_size: max |W|+|Z| for experiments (None = no limit)
        verbose: if True, print progress

    Returns:
        pd.DataFrame with one row per simulation
    """
    variables, edges = parse_bif(bif_path)
    graph_name = os.path.basename(bif_path)
    rng = np.random.default_rng(seed)

    # Pre-generate experiments (same variable set for all simulations)
    experiments = generate_experiments(variables, max_experiment_size)
    n_experiments = len(experiments)

    if verbose:
        print(f"Graph: {graph_name}, |V|={len(variables)}, "
              f"|experiments|={n_experiments}")

    results = []

    for p_conf in confounding_probs:
        for sim in range(n_simulations):
            # Add random confounders for every pair of variables
            confounders = []
            for i, v1 in enumerate(variables):
                for v2 in variables[i + 1:]:
                    if rng.random() < p_conf:
                        confounders.append((v1, v2))

            cg = CausalGraph(variables, edges, confounders)

            # Choose random query P(y|do(x))
            query = choose_random_query(cg, rng)
            if query is None:
                continue
            y_var, x_var = query

            # Compute R*_theta for this query
            R_theta = cg.get_R_theta({y_var}, {x_var})
            R_theta_star = cg.get_R_theta_star({y_var}, {x_var})

            # Run GetUselessExperiments
            useless, stats = get_useless_experiments(
                cg, {y_var}, {x_var}, experiments
            )

            result = {
                'graph': graph_name,
                'n_vars': len(variables),
                'p_conf': p_conf,
                'sim': sim,
                'n_confounders': len(confounders),
                'query_y': y_var,
                'query_x': x_var,
                'n_R_theta': len(R_theta),
                'n_R_theta_star': len(R_theta_star),
                'n_experiments': n_experiments,
                'n_useless': len(useless),
                'n_useful': n_experiments - len(useless),
                'fraction_pruned': len(useless) / n_experiments if n_experiments else 0,
                'id_pruned': stats['id_pruned'],
                'blocked_pruned': stats['blocked_pruned'],
            }
            results.append(result)

            if verbose and (sim + 1) % 10 == 0:
                print(f"  p={p_conf}, sim {sim+1}/{n_simulations}: "
                      f"pruned {result['fraction_pruned']:.1%}")

    return pd.DataFrame(results)
