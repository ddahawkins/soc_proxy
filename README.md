# CDS-guided time series aggregation for long-duration energy storage

Research code accompanying **“State of Charge Guided Time Series Aggregation to Capture Long Duration Energy Storage Behaviour in Energy System Models”** by Daniel Hawkins, Machteld van den Broek and Kenneth Bruninx.

The method constructs an **ex ante charge–discharge signal (CDS)** from electricity demand, renewable-availability profiles and techno-economic assumptions. The CDS is supplied to time series aggregation (TSA) as an additional clustering feature to help preserve long-duration energy storage (LDES) behaviour in compressed, multi-year capacity-expansion models. The associated cumulative state-of-charge (SoC) proxy is primarily used for interpreting long-horizon storage events. The repository also contains the numerical experiments and ex post LDES capacity-error diagnostic described in the paper.

## Where to find the method

The method implementation is in [`src/soc_proxy/`](src/soc_proxy/):

| File | Purpose |
| --- | --- |
| [`soc_proxy.py`](src/soc_proxy/soc_proxy.py) | Public `generate_soc_proxy()` interface and `SocProxyResult` outputs. |
| [`_core.py`](src/soc_proxy/_core.py) | Numerical charge–discharge allocation, feasibility search and temporal decomposition. |
| [`_margin.py`](src/soc_proxy/_margin.py) | Endogenous margin selection using economic saturation and persistent system-defining events. |
| [`__init__.py`](src/soc_proxy/__init__.py) | Public package exports. |

The returned `SocProxyResult.data` contains the LDES charge–discharge component (`surplus_LDES`, the **CDS**) and its cumulative counterpart (`soc_proxy_LDES`), alongside other outputs. For the manuscript's selected decomposition, specify **Gaussian smoothing with a 24-hour timescale**; the standalone function's default is a 24-hour FFT low-pass decomposition.

## Research workflow

- [`scripts/pipeline.py`](scripts/pipeline.py) integrates proxy construction, TSA and Calliope model execution.
- [`scripts/experiments/run_experiment_batch.py`](scripts/experiments/run_experiment_batch.py) orchestrates experiment batches, with scenarios defined in [`config/`](config/).
- [`scripts/experiments/consolidate_results.py`](scripts/experiments/consolidate_results.py) consolidates the recorded results.
- [`scripts/helpers/`](scripts/helpers/) provides adapters for timeseries, TSAM, Calliope and result extraction.
- [`scripts/paper_figures/`](scripts/paper_figures/) contains scripts for the manuscript figures and supporting analyses.

The repository uses **Pixi** for dependency management (`pixi.toml`, `pixi.lock`), **TSAM** for aggregation and **Calliope** for capacity-expansion modelling. Solving optimisation models requires an appropriately configured **Gurobi** licence. From the repository root, a figure script can, for example, be invoked with:

```bash
pixi run python -m scripts.paper_figures.fig_LDES_cap_results
```

Figure generation generally reads previously recorded results under `results/`; it does not necessarily rerun the underlying optimisation. Experiment settings and input model definitions are kept under `config/`, with raw timeseries and reference-model files under `resources/`.

## Manuscript figure and table mapping

The numbering below follows the accompanying revised manuscript.

| Manuscript item | Generating script or source |
| --- | --- |
| **Figures 1–2** — method workflow and diagnostic schematic | Conceptual diagrams; no generator identified in the supplied Python scripts. |
| **Figure 3** — SoC proxy versus reference-model SoC | `scripts/paper_figures/fig_soc_proxy.py` |
| **Figure 4** — LDES capacity error versus representative periods | `scripts/paper_figures/fig_LDES_cap_results.py` |
| **Figure 5** — CAPEX-weighted MACME | `scripts/paper_figures/fig_weighted_MACME_results.py` |
| **Figure 6** — sensitivity to modelling horizon | `scripts/paper_figures/fig_horizon_results.py` |
| **Figure 7** — runtime comparison | `scripts/paper_figures/fig_runtime_results.py` |
| **Figure 8** — sensitivity to TSA method | `scripts/paper_figures/fig_tsa_methods.py` |
| **Figure 9** — ex post LDES capacity-error prediction | `scripts/paper_figures/fig_diagnostic_acceptability.py` (`fig_continuous_diagnostic_validation`) |
| **Table 4** — margin and proxy-approximation summary | `scripts/paper_figures/tab_proxy_approximation_stats.py` |
| **Figure A.10** — economic margin selection | `scripts/paper_figures/fig_economic_margin_selector.py` |
| **Figure A.11** — persistent-event margin selection | `scripts/paper_figures/fig_binding_event_selector.py` |
| **Figure A.12** — temporal-decomposition sensitivity | `scripts/paper_figures/fig_temporal_decomposition.py` |
| **Figure D.13** — single-predictor diagnostic screening | `scripts/paper_figures/fig_single_predictor.py` |

Other scripts under `scripts/experiments/` and `scripts/paper_figures/` support the calibration, method comparisons and exploratory analyses underlying the study.

## Data and citation

The case studies use electricity-demand data from **ENTSO-E** and renewable-availability data provided by **Renewables.ninja** (MERRA-2), as described in the paper. Please consult the paper for assumptions, sensitivity settings and interpretation of the model outputs. When using this repository, cite the accompanying paper.

For techno-economic asssumption sources, please consult the Calliope configuration file `config/calliope/technologies.yaml`
