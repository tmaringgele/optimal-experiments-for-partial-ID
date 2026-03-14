"""Implementation of GetUselessExperiments (Algorithm 3 from the paper).

Combines two pruning criteria:
1. ID check: experiments whose outcome is identifiable from P(V) have zero potency
2. AllPathsBlocked: experiments where all paths from R*_theta to outcomes
   pass through interventions have zero potency
"""
import itertools
from .causal_graph import CausalGraph
from .id_algorithm import is_identifiable


def generate_experiments(variables, max_size=None):
    """Generate all experiments (W, Z) where W, Z are subsets of variables.

    Each experiment represents P(W | do(Z)) with W ∩ Z = ∅, W ≠ ∅, Z ≠ ∅.

    Args:
        variables: list of variable names
        max_size: max |W| + |Z| (None = no limit, uses all variables)

    Returns:
        list of (frozenset(W), frozenset(Z)) tuples
    """
    n = len(variables)
    if max_size is None:
        max_size = n

    experiments = []
    for z_size in range(1, max_size):
        for z_vars in itertools.combinations(variables, z_size):
            remaining = [v for v in variables if v not in z_vars]
            max_w = min(len(remaining), max_size - z_size)
            for w_size in range(1, max_w + 1):
                for w_vars in itertools.combinations(remaining, w_size):
                    experiments.append((frozenset(w_vars), frozenset(z_vars)))
    return experiments


def get_useless_experiments(causal_graph, query_y, query_x, experiments):
    """Run GetUselessExperiments (Algorithm 3).

    For each experiment in the candidate set, checks:
    1. If P(W|do(Z)) is identifiable → useless (Theorem 4 in the paper)
    2. If AllPathsBlocked(G, R*_theta, W, Z) → useless (Corollary 1)

    Args:
        causal_graph: CausalGraph object
        query_y: set of outcome variable(s) for the query theta
        query_x: set of intervention variable(s) for the query theta
        experiments: list of (W, Z) tuples (frozensets)

    Returns:
        useless: set of (W, Z) tuples that have zero potency
        stats: dict with pruning statistics
    """
    if isinstance(query_y, str):
        query_y = {query_y}
    if isinstance(query_x, str):
        query_x = {query_x}
    query_y = set(query_y)
    query_x = set(query_x)

    R_theta_star = causal_graph.get_R_theta_star(query_y, query_x)

    useless = set()
    id_pruned = 0
    blocked_pruned = 0

    for W, Z in experiments:
        # 1. Check identifiability (single-world interventional queries)
        if is_identifiable(causal_graph, list(W), list(Z)):
            useless.add((W, Z))
            id_pruned += 1
            continue

        # 2. Check AllPathsBlocked
        if causal_graph.all_paths_blocked(R_theta_star, W, Z):
            useless.add((W, Z))
            blocked_pruned += 1

    stats = {
        'id_pruned': id_pruned,
        'blocked_pruned': blocked_pruned,
        'total_pruned': id_pruned + blocked_pruned,
    }
    return useless, stats
