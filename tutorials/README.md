# Tutorials

This folder collects the reference notebooks used to reproduce or inspect the main dataset-construction steps in this repository. The notebooks are designed as **reader-facing walkthroughs**: the generated data already lives under `data/`, while the notebooks document how those artifacts were built.

## Notebook Index

| Notebook | Purpose | Main outputs |
| --- | --- | --- |
| `contextualized_PPI_construction.ipynb` | Build cell-line-specific contextualized PPI subgraphs from RNA expression, a reference PPI network, PrimeKG mappings, and protein sequence embeddings. | `data/MultiOmics_feature/cell_line_data/protein_csv/*.csv`, `data/MultiOmics_feature/cell_line_data/protein_nx/*.pkl` |
| `cell_line_specific_scenario_construction.ipynb` | Rebuild the **cell-line-specific** SLBench scenarios, including random splitting, cold start, cross-functional, and tail-node settings. | `data/SLbench/Scenario/Cell_line_specific/<Scenario>/<CellLine>/...` |
| `cross_cell_line_scenario_constrcution.ipynb` | Rebuild the **cross-cell-line** train/test scenarios where one target cell line is held out as test data. | `data/SLbench/Scenario/Cross_cell_line/Multi_5_to_<CellLine>/...` |

> Note: the filename `cross_cell_line_scenario_constrcution.ipynb` keeps the existing repository spelling (`constrcution`) for compatibility.



