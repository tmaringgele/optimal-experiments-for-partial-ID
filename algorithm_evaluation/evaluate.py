"""Main evaluation function for the GetUselessExperiments algorithm.

Evaluates how many experiments can be pruned on BIF-derived causal DAGs
with randomly added latent confounders.
"""
import os
import numpy as np
import pandas as pd

from .bif_parser import parse_bif
from .causal_graph import CausalGraph
from .get_useless import sample_query, sample_experiments, get_useless_experiments


DEFAULT_THETA_CONFIG = {
    'CF_worlds_mean': 1,
    'CF_worlds_sd': 0,
    'W_size_mean': 1,
    'W_size_sd': 0,
    'Z_size_mean': 1,
    'Z_size_sd': 0,
    'intervention_outcome_distance_mean': 1,
    'intervention_outcome_distance_sd': 0,
}

DEFAULT_EXPERIMENT_CONFIG = {
    'CF_worlds_mean': 1,
    'CF_worlds_sd': 0,
    'W_size_mean': 1,
    'W_size_sd': 1,
    'Z_size_mean': 1,
    'Z_size_sd': 1,
}


def evaluate(
    bif_path,
    confounding_probs=(0.05, 0.15, 0.25, 0.35),
    n_simulations=100,
    seed=42,
    experiment_set_size=200,
    theta_config=None,
    experiment_config=None,
    verbose=False,
):
    """Evaluate GetUselessExperiments on a BIF graph.

    For each confounding probability and simulation:
    1. Parse the BIF file to get the DAG structure
    2. Add random latent confounders with the given probability
    3. Sample a query theta according to theta_config
    4. Sample candidate experiments according to experiment_config
    5. Run GetUselessExperiments to identify useless ones
    6. Record statistics

    Args:
        bif_path: path to a .bif file
        confounding_probs: list of probabilities for adding confounders
        n_simulations: number of random simulations per probability
        seed: random seed for reproducibility
        experiment_set_size: number of candidate experiments |A| to sample
        theta_config: dict controlling query sampling (see experiment_setup.md)
        experiment_config: dict controlling experiment sampling
        verbose: if True, print progress

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
              f"|A|={experiment_set_size}")

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

            # Sample query theta
            query_worlds = sample_query(cg, theta_config, rng)
            if query_worlds is None:
                continue

            # Sample candidate experiments
            experiments = sample_experiments(
                variables, experiment_config, experiment_set_size, rng
            )

            # Compute R_theta and R*_theta for reporting
            R_theta = cg.get_R_theta(query_worlds)
            R_theta_star = cg.get_R_theta_star(query_worlds)

            # Run GetUselessExperiments
            useless, stats = get_useless_experiments(
                cg, query_worlds, experiments
            )

            n_experiments = len(experiments)
            query_str = "; ".join(
                f"Y={set(Y)}, X={set(X)}" for Y, X in query_worlds
            )

            result = {
                'graph': graph_name,
                'n_vars': len(variables),
                'p_conf': p_conf,
                'sim': sim,
                'n_confounders': len(confounders),
                'query': query_str,
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
