# Experiment Setup

## Overview

The `evaluate()` function runs `GetUselessExperiments` on a BIF-derived causal DAG with randomly added latent confounders. For each simulation it:

1. Parses the BIF file to extract the observed DAG
2. Samples a query `theta` on the bare DAG (no confounders yet) according to `theta_config`
3. Adds a **guaranteed confounder** for every node in `X^(1)`: each `x ∈ X^(1)` is connected to a randomly chosen `y ∈ Y^(1)` via a latent confounder
4. Adds additional latent confounders between each remaining pair of observed variables with probability `p_conf`
5. Samples a set of candidate experiments according to `experiment_config` and `experiment_set_size`
6. Runs `GetUselessExperiments` to identify useless experiments
7. Records pruning statistics

**Why query before confounders?** The query is sampled on the bare DAG so that its structure is independent of the confounder placement. A confounder between `X` and `Y` is then hardcoded to ensure the query is non-trivially confounded — without this, many simulations would produce queries that are already identifiable, reducing the evaluation's informativeness.

---

## Parameters

### `bif_path` (str)
Path to a `.bif` file. Only the graph structure (variable names and edges) is used; probability tables are ignored.

### `confounder_ratio_range` (tuple of float)
An interval `[a, b]` controlling the density of latent confounders. For each simulation, a target ratio `r = |U|/|V|` is drawn uniformly from `[a, b]`, and exactly `floor(r * |V|)` confounders are added in total. Use `(a, a)` to fix the ratio exactly.

The actual lower bound for `r` is clamped to `len(forced_confounders) / |V|` so the target is always achievable. On very small graphs where the forced confounders alone exceed `b` (e.g. `|Y^(1)|=5` on an 8-node graph gives `5/8=0.625`), the ratio will be floored at the forced-confounder minimum — this is unavoidable and does not occur on graphs with `|V| >= 9` when `W_sizes` are bounded by 5 and `b >= 0.6`.

Default: `(0.1, 0.5)`

### `n_simulations` (int)
Number of random simulations.

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

### `save_graphs` (str or None)
If set to a file path, saves the full reconstruction data for every simulation
to a JSON file. Each entry is keyed by `sim_id = "<graph_name>__sim<N>"`, which
matches the `graph` and `sim` columns in the returned DataFrame and can be used
to look up the exact graph/query/experiment set for any row of interest.

The JSON structure per entry is:
```json
{
  "sim_id": "asia.bif__sim3",
  "graph": "asia.bif",
  "sim": 3,
  "variables": ["asia", "tub", ...],
  "edges": [["asia", "tub"], ...],
  "confounders": [["tub", "dysp"], ...],
  "query_worlds": [{"Y": ["dysp"], "X": ["tub"]}, ...],
  "experiments": [
    [{"W": ["lung"], "Z": ["asia"]}],
    ...
  ]
}
```

If the file already exists, new entries are merged into it (existing keys are
overwritten). Defaults to `None` (nothing is saved).

Default: `None`

---

## `theta_config` (dict) — Query Configuration

Controls how the query `theta` is sampled. Each query has 1 or 2 counterfactual worlds, each specifying outcome variables `Y^(c)` and intervention variables `X^(c)`.

Set sizes are drawn uniformly at random from the provided lists. Interventions `X` are always restricted to observed ancestors of `Y` in the DAG.

| Key | Type | Description |
|-----|------|-------------|
| `CF_fraction` | float | Probability that the query is a 2-world counterfactual (0.0 = always single-world) |
| `W_sizes` | list of int | Allowed outcome set sizes `|Y^(c)|`; one is picked uniformly at random per world |
| `Z_sizes` | list of int | Allowed intervention set sizes `|X^(c)|`; one is picked uniformly at random per world |
| `intervention_outcome_distances` | list of int or None | *(optional)* Allowed distances between the seed intervention and outcome (undirected shortest path). One is picked uniformly at random per world. If `None` or omitted, interventions are sampled uniformly from `anc(Y) \ Y`. |

**Counterfactual queries:** When a query has 2 worlds, the second world is guaranteed to share at least one intervention variable with the first (the shared variable is picked at random from `X^(1) ∩ ancestors(Y^(2))`). This ensures meaningful counterfactual contrasts (e.g. PNS-style queries). When distance control is enabled, both worlds use it: in the second world, the seed intervention `z₂` is picked from `Z₁ ∩ anc(W₂)` preferring the target distance (same closest-available-distance fallback as world 1).

**Distance semantics:** When `intervention_outcome_distances` is provided, both worlds use the following procedure for the seed intervention:
1. `Y` is picked uniformly at random from `V`
2. `anc(Y) \ Y` is computed and each ancestor's undirected shortest-path distance to `Y` is measured
3. A target distance `d` is picked uniformly from `intervention_outcome_distances`
4. The first intervention `x` is picked uniformly from ancestors at distance `d`. If no ancestor exists at exactly `d`, the ancestor group with the closest available distance is used instead
5. Remaining interventions are filled from `anc(Y) \ Y`

For the second world of a counterfactual query, the same distance-controlled selection applies to the seed intervention, but restricted to `Z₁ ∩ anc(W₂)` (to guarantee shared intervention variables).

**Without distance control (default):** `Y` is picked uniformly at random, and `X` is sampled uniformly from `anc(Y) \ Y`.

**Default:**
```python
theta_config = {
    'CF_fraction': 0.0,
    'W_sizes': [1],
    'Z_sizes': [1],
}
```
This samples `P(y | do(x))` where `x` is a random ancestor of `y`.

---

## `experiment_config` (dict) — Candidate Experiment Configuration

Controls how each candidate experiment in `A` is sampled. Each experiment has 1 or 2 counterfactual worlds, each specifying observed variables `W^(c)` and intervention variables `Z^(c)`. Interventions `Z` are always restricted to observed ancestors of `W`.

| Key | Type | Description |
|-----|------|-------------|
| `CF_fraction` | float | Probability that an experiment is a 2-world counterfactual (0.0 = all single-world) |
| `W_sizes` | list of int | Allowed outcome set sizes `|W^(c)|`; one is picked uniformly at random per world |
| `Z_sizes` | list of int | Allowed intervention set sizes `|Z^(c)|`; one is picked uniformly at random per world |
| `intervention_outcome_distances` | list of int or None | *(optional)* Allowed distances between seed intervention and outcome. Same semantics as `theta_config`. |

**Counterfactual sampling (2-world experiments):** The second world is seeded from the first:
1. Pick the first outcome `w₂` uniformly from `W₁` (shared outcome)
2. Expand `W₂` to the target size with additional variables from `V`
3. Pick the first intervention `z₂` from `Z₁ ∩ (anc(W₂) \ W₂)` (shared intervention). When distance control is enabled, `z₂` is picked preferring the target distance (same semantics as world 1); otherwise picked uniformly.
4. Expand `Z₂` to the target size with additional variables from `anc(W₂) \ W₂`

This guarantees `W₁ ∩ W₂ ≠ ∅` and `Z₁ ∩ Z₂ ≠ ∅`, mirroring how real counterfactual experiments contrast outcomes under shared but differing interventions.

**Default:**
```python
experiment_config = {
    'CF_fraction': 0.0,
    'W_sizes': [1, 2, 3],
    'Z_sizes': [1, 2, 3],
}
```

---

---

## Visualization

Two functions are available in `algorithm_evaluation.visualize` (also exported from the package root) for inspecting saved simulations:

### `plot_simulation(json_path, graph_name, sim, figsize, seed)`

Draws the causal graph for the given simulation:
- **Directed edges**: arrows between observed variables
- **Confounders**: curved dashed orange edges between the two variables they connect
- **Query Y nodes**: red
- **Query X nodes**: blue
- **Y ∩ X nodes**: purple
- **Other nodes**: grey

### `print_experiments(json_path, graph_name, sim)`

Prints the experiment set partitioned into three groups:
1. **Useful** — not pruned by either criterion
2. **Pruned by ID check** — single-world, identifiable from P(V)
3. **Pruned by AllPathsBlocked** — all paths from R\*\_theta blocked

Both functions accept the JSON file produced by `evaluate(..., save_graphs=...)`.
The `sim_id` key in the JSON (`"<graph_name>__sim<N>"`) links each record to
the corresponding row in the DataFrame via `(graph, sim)`.

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
| `u_v_ratio` | Achieved `|U|/|V|` ratio (actual number of confounders divided by `|V|`) |
| `sim` | Simulation index |
| `n_confounders` | Number of latent confounders added |
| `n_districts` | Number of districts (maximal bidirected-connected components) in the graph |
| `query` | String representation of the sampled query `theta` |
| `query_is_cf` | Whether the query is a 2-world counterfactual |
| `n_R_theta` | `|R_theta|` — query-relevant response types |
| `n_R_theta_star` | `|R*_theta|` — district-hull response types |
| `n_experiments` | `|A|` — number of candidate experiments |
| `n_cf_experiments` | Number of 2-world counterfactual experiments in `A` |
| `fraction_cf` | Fraction of experiments that are counterfactual |
| `n_useless` | Number of experiments pruned |
| `n_useful` | Number of remaining experiments |
| `fraction_pruned` | Fraction of experiments pruned |
| `id_pruned` | Number pruned by the ID check |
| `blocked_pruned` | Number pruned by AllPathsBlocked |
