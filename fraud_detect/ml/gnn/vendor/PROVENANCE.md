# Vendored code provenance — Multi-GNN

The model code in this `vendor/` directory is adapted from the IBM Multi-GNN
research repository. Attribution is retained here per the Apache-2.0 terms (see
`LICENSE` in this directory).

## Source

| Field | Value |
|-------|-------|
| Project | Multi-GNN |
| Upstream URL | https://github.com/IBM/Multi-GNN |
| License | Apache-2.0 (SPDX: `Apache-2.0`) |
| Default branch | `main` |
| Pinned commit | `252b0252afca109d1d216c411c59ff70753b25fc` |

The pinned commit is the revision from which the vendored model classes are
copied. When updating the vendored code, bump this commit hash and note what
changed.

## Scope of what is vendored

Only the minimal model classes needed for Node 6a are vendored:

- The GNN model definitions (GIN / GAT / PNA / RGCN) and the Multi-GNN
  adaptations: reverse message passing, ego IDs, and port numberings.

The upstream **data pipeline is intentionally NOT vendored** — graphs are fed to
these models from `fraud_detect/ml/gnn/graph_builder.py`, which builds a
split-safe internal-transfer graph from this project's dataset.

### Files vendored

| Local file | Upstream origin (@ pinned commit) | What was taken |
|------------|-----------------------------------|----------------|
| `models.py` | `models.py` | `GINe`, `GATe`, `PNA`, `RGCN` model classes. |
| `adaptations.py` | `train_util.py` + `data_util.py` | `AddEgoIds`, `add_arange_ids` (ego IDs); `ports`, `to_adj_nodes_with_times`, `to_adj_edges_with_times`, `time_deltas`, `z_norm` (port numbering helpers); `GraphData`, `HeteroGraphData`, `create_hetero_obj` (homogeneous graph + reverse-MP heterogeneous graph). |
| `__init__.py` | — (new) | Package marker. Imports NO torch submodules so importing the `vendor` package never pulls in torch. |

The upstream `LICENSE` (Apache-2.0) is retained in this directory unmodified.

## Modifications

Per Apache-2.0 §4(b), every modified vendored file carries a prominent notice at
the top stating it was changed, its upstream origin (IBM/Multi-GNN at the pinned
commit above), and the nature of the change. Modifications made:

- **`models.py`** — prepended the §4(b) change notice; import paths left as the
  standard `torch` / `torch_geometric` imports (no local-package imports were
  needed). No functional change to the model classes. Upstream data/CLI glue was
  not copied into this file.
- **`adaptations.py`** — prepended the §4(b) change notice. Combined only the
  needed adaptations from upstream `train_util.py` (`AddEgoIds`,
  `add_arange_ids`) and `data_util.py` (ports / reverse-MP helpers and graph
  objects) into one local module importable as
  `fraud_detect.ml.gnn.vendor.adaptations`. Dropped upstream data-pipeline glue:
  data loaders (`get_loaders`), evaluation/training loops (`evaluate_homo`,
  `evaluate_hetero`), checkpoint IO (`save_model`, `load_model`),
  `model_settings.json` parameter extraction, and the `wandb` / `tqdm` /
  `sklearn` imports those relied on. `create_hetero_obj` was changed to take an
  explicit `ports: bool` parameter instead of an upstream argparse `args` object,
  so the vendored code carries no CLI glue.
- **`__init__.py`** — new file; intentionally imports none of the torch-dependent
  submodules so the live path can import the package without torch installed.

Because `torch` / `torch_geometric` are offline-only dependencies, a plain
`import` of `models.py` / `adaptations.py` fails at the torch import line when
those packages are absent — this is expected. The vendored files are verified
with `python -m py_compile` instead of by importing.

## Reference

- Egressy et al., "Provably Powerful Graph Neural Networks for Directed
  Multigraphs" — the work the Multi-GNN repository accompanies. See the upstream
  repository README for the canonical citation.
