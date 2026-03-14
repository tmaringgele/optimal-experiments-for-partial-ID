# Experiment Setup

## Overview

The `evaluate()` function runs `GetUselessExperiments` on a BIF-derived causal DAG with randomly added latent confounders. For each simulation it:

1. Parses the BIF file to extract the observed DAG
2. Adds latent confounders between each pair of observed variables with probability `p_conf`
3. Samples a query `theta` according to `theta_config`
4. Samples a set of candidate experiments according to `experiment_config` and `experiment_set_size`
5. Runs `GetUselessExperiments` to identify useless experiments
6. Records pruning statistics

---

## Parameters

### `bif_path` (str)
Path to a `.bif` file. Only the graph structure (variable names and edges) is used; probability tables are ignored.

### `confounding_probs` (list of float)
Probabilities for adding a latent confounder between each pair of observed variables. For each `p` in this list, `n_simulations` simulations are run. Each pair `(V_i, V_j)` independently receives a confounder with probability `p`.

Default: `[0.05, 0.15, 0.25, 0.35]`

### `n_simulations` (int)
Number of random simulations per confounding probability.

Default: `100`

### `seed` (int)
Random seed for reproducibility.

Default: `42`

### `experiment_set_size` (int)
The number of candidate experiments `|A|` to sample per simulation. These are drawn according to `experiment_config`.

Default: `200`

### `verbose` (bool)
Print progress updates.

Default: `False`

---

## `theta_config` (dict) — Query Configuration

Controls how the query `theta` is sampled. The query is a list of counterfactual worlds, each specifying outcome variables `Y^(c)` and intervention variables `X^(c)`.

Sizes are sampled from a normal distribution, rounded to the nearest integer, and clamped to a minimum of 1.

| Key | Type | Description |
|-----|------|-------------|
| `CF_worlds_mean` | float | Mean number of counterfactual worlds `|C|` in the query |
| `CF_worlds_sd` | float | Standard deviation for the number of counterfactual worlds |
| `W_size_mean` | float | Mean number of outcome variables `|Y^(c)|` per world |
| `W_size_sd` | float | Standard deviation for `|Y^(c)|` |
| `Z_size_mean` | float | Mean number of intervention variables `|X^(c)|` per world |
| `Z_size_sd` | float | Standard deviation for `|X^(c)|` |
| `intervention_outcome_distance_mean` | float | Mean shortest-path distance between `X^(c)` and `Y^(c)` (in the undirected observed graph). Sampled per world. |
| `intervention_outcome_distance_sd` | float | Standard deviation for the intervention–outcome distance |

**Distance semantics:** For each counterfactual world, a target distance `d` is sampled from `N(mean, sd)`. A seed pair `(x, y)` is chosen uniformly at random from all pairs whose shortest-path distance in the undirected observed graph equals `d` (or the closest available distance if `d` is not achievable). The sets `X^(c)` and `Y^(c)` are then built around this seed pair.

**Default:**
```python
theta_config = {
    'CF_worlds_mean': 1,
    'CF_worlds_sd': 0,
    'W_size_mean': 1,
    'W_size_sd': 0,
    'Z_size_mean': 1,
    'Z_size_sd': 0,
    'intervention_outcome_distance_mean': 1,
    'intervention_outcome_distance_sd': 0,
}
```
This samples `P(y | do(x))` where `x` is an immediate neighbor of `y`.

---

## `experiment_config` (dict) — Candidate Experiment Configuration

Controls how each candidate experiment in `A` is sampled. Each experiment is a list of counterfactual worlds, each specifying observed variables `W^(c)` and intervention variables `Z^(c)`.

| Key | Type | Description |
|-----|------|-------------|
| `CF_worlds_mean` | float | Mean number of counterfactual worlds `|C|` per experiment |
| `CF_worlds_sd` | float | Standard deviation for the number of counterfactual worlds |
| `W_size_mean` | float | Mean number of observed variables `|W^(c)|` per world |
| `W_size_sd` | float | Standard deviation for `|W^(c)|` |
| `Z_size_mean` | float | Mean number of intervention variables `|Z^(c)|` per world |
| `Z_size_sd` | float | Standard deviation for `|Z^(c)|` |

**Default:**
```python
experiment_config = {
    'CF_worlds_mean': 1,
    'CF_worlds_sd': 0,
    'W_size_mean': 1,
    'W_size_sd': 1,
    'Z_size_mean': 1,
    'Z_size_sd': 1,
}
```

---

## Pruning Criteria

`GetUselessExperiments` (Algorithm 3) applies two criteria to identify experiments with guaranteed zero potency:

1. **ID check** (Theorem 4): If the experiment is a single-world interventional quantity `P(W|do(Z))` and it is identifiable from `P(V)` alone, then `pot(a) = 0`. Only applies when `|C| = 1`.

2. **AllPathsBlocked** (Corollary 1): If for every counterfactual world `c`, all directed paths from `R*_theta` to `Y^(c)` pass through `X^(c)`, then `pot(a) = 0`. Applies to any number of worlds.

---

## Output

`evaluate()` returns a `pd.DataFrame` with one row per simulation and the following columns:

| Column | Description |
|--------|-------------|
| `graph` | BIF file name |
| `n_vars` | Number of observed variables |
| `p_conf` | Confounding probability used |
| `sim` | Simulation index |
| `n_confounders` | Number of latent confounders added |
| `query` | String representation of the sampled query `theta` |
| `n_R_theta` | `|R_theta|` — query-relevant response types |
| `n_R_theta_star` | `|R*_theta|` — district-hull response types |
| `n_experiments` | `|A|` — number of candidate experiments |
| `n_useless` | Number of experiments pruned |
| `n_useful` | Number of remaining experiments |
| `fraction_pruned` | Fraction of experiments pruned |
| `id_pruned` | Number pruned by the ID check |
| `blocked_pruned` | Number pruned by AllPathsBlocked |
