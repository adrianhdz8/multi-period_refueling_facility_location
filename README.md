# Multi-period Flow-based Facility Location Problem

Code, data, and detailed results accompanying the paper

> A. Hernandez et al. *Multi-period flow capture model to support rollout of energy infrastructure on freight rail networks.* European Journal of Operational Research (under revision, manuscript EJOR-D-26-01484).

The repository contains

- a **multi-period (dynamic) flow capture model** for rolling out refueling/charging facilities on a freight rail network, including its single-period (static) special case and the "static boundary condition" variant;
- a **Gurobi branch-and-bound** implementation (benchmark) and three **Benders decomposition (BD)** implementations: multi-cut, single Pareto-optimal cut, and multi Pareto-optimal cut (Magnanti-Wong);
- **all input data** (networks, OD flows, scenario and solver configurations) for the 94 instances reported in the paper (51 dynamic, 43 static);
- the **per-instance results** behind every row of the computational tables in the paper.

The sections below are organised around what is needed to reproduce the paper's numbers: where the data come from, where the results are, and how to run the code.

---

## 1. Repository layout

```
.
├── run.py                  # entry points: run_rollout_file, run_multiple_scenarios
├── solver.py               # abstract solver interface
├── ilp_solver.py           # Gurobi branch-and-bound (benchmark)
├── benders_solver.py       # Benders decomposition (multi-cut, Pareto-optimal / Magnanti-Wong cuts)
├── greedy_solver.py        # greedy heuristic
├── facility_location_oop.py# dispatches a scenario to the selected solver
├── matrix_constructor.py   # builds the coverage / constraint matrices of the model
├── routing.py              # OD-flow selection and post-solution flow routing
├── network_processing.py   # loads and simplifies the rail network graphs
├── input_output.py         # reads scenario / solver-parameter files, writes experiment summaries
├── helper.py, util.py      # shared utilities, global paths and constants
├── plotting.py             # maps and figures of facility deployment
├── data/
│   ├── networks/           # network graphs (.pkl) + shortest-path distance dictionary
│   ├── flows/              # processed freight flow data for the 544-node network
│   ├── parameters/         # SPLC station master (station coordinates), US cities table
│   ├── scenario/           # one CSV per instance (94 files): the problem parameters
│   ├── solver parameters/  # one CSV per solution method: the algorithm settings
│   ├── facility/           # optional facility-specific fixed/barred/cost inputs
│   └── legends/            # documentation of every keyword + templates
├── results/
│   ├── dynamic_instance_results.csv   # per-instance results, instances 1-51
│   ├── static_instance_results.csv    # per-instance results, instances 52-94
│   └── experiments/                   # output folder for new runs
└── LICENSE
```

---

## 2. Data

### 2.1 Networks

| Network (`rr` in scenario files) | Size | Description |
|---|---|---|
| `KCS3*` (e.g., `KCS315`, `KCS3100`, `KCS3b`) | 28 nodes, 32 edges (64 directed arcs) | Small network inspired by the former Kansas City Southern railroad. Used in instances 1-34 and 52-85. The different files share the topology and differ in the OD pairs/flows they carry. |
| `USA1` / `usa1` | 544 nodes, 4,553 arcs | Combined service networks of the US Class I railroads. Used in instances 35-51 and 86-94. |

Networks are stored as pickled `networkx` graphs in `data/networks/` and are read by `network_processing.load_simplified_consolidated_graph`. `data/networks/distances/USA1_SP_dict_miles.json` caches all-pairs shortest-path distances (miles) for the 544-node network.

### 2.2 Demand (OD flows)

- **28-node network.** Demand is synthetic. The paper's illustrative case study uses 10 OD pairs with the per-period flows given in its Table 2 flow schedule. For the computational instances, a random subset of OD pairs is sampled and each is assigned a base flow drawn from U{0, 1000} tons; base flows are then multiplied by a per-period growth/decay factor drawn uniformly from [0.8, 1.2]. **The sampled OD pairs and flows are stored in the network/scenario files, so no re-sampling or random seed is needed to reproduce the instances.**
- **544-node network.** Demand is derived from public freight data: the 2019 *Carload Waybill Sample* summaries published by the US Surface Transportation Board and the *Freight Analysis Framework* (FAF5) forecasts (Hwang et al., 2021), processed to station-to-station (SPLC) flows with a forecast to 2050. The processed file is `data/flows/WB2019_summary_class1_SPLC_forecast_2050.csv`; station coordinates are in `data/parameters/SPLC_station_master.csv`. The instances use the top `od` fraction of OD pairs by ton-miles (the `od_flow_perc` field of the scenario file) and `|Q|` ranges from 736 to 3,250 OD pairs.

All data needed to run the instances in the paper are in this repository; nothing is available only on request.

### 2.3 Scenario files (`data/scenario/`)

Each of the 94 files defines one instance. The file name encodes its main parameters, e.g.

```
sc_KCS315_rt400_t12_b1_od1_r0.1_lnghrz_k2.csv
    │      │     │   │  │   │    │       └─ k: number of allowed paths per OD pair
    │      │     │   │  │   │    └───────── dynamic (long-horizon) instance
    │      │     │   │  │   └────────────── discount rate
    │      │     │   │  └────────────────── fraction of OD pairs retained
    │      │     │   └───────────────────── per-period budget (number of facilities)
    │      │     └───────────────────────── 12 periods (`tstatic` = single-period instance)
    │      └─────────────────────────────── vehicle range in km
    └────────────────────────────────────── network (`rr`)
```

`bcstatic` in a name marks a dynamic instance with a pre-specified (static) boundary condition. The 94 scenario files split into 51 dynamic (`lnghrz`) and 43 static (`tstatic`) instances, matching the 51 + 43 instances in the paper. Every keyword (time horizon, per-period range and budget, discount factors, flow-capture minimums, number of paths, etc.) is documented in `data/legends/input_legend_mp.csv`; `data/legends/scenario_template.csv` is a template for new scenarios.

### 2.4 Solver-parameter files (`data/solver parameters/`)

| File | Method in the paper |
|---|---|
| `sp_gurobi_saoff_0.001_3600_large.csv`, `sp_gurobi_saon_0.001_3600_large.csv` | Gurobi branch-and-bound (without / with solution aids) |
| `sp_bd_mcTonODon_scoff_0.001_3600_large.csv` | BD, standard multi-cut |
| `sp_bd_mcToffODoff_scon_0.001_3600_large.csv` | BD, single Pareto-optimal cut |
| `sp_bd_mcTonODon_scon_0.001_3600_large.csv` | BD, multi Pareto-optimal cut |

The name encodes the settings: `mcT`/`OD` = multi-cut over time / over OD pairs, `sc` = strong (Pareto-optimal, Magnanti-Wong) cuts, then the relative optimality gap (`0.001`) and the time limit in seconds (`3600`). All methods use the settings of the paper: relative gap 10⁻³ and a 1-hour limit.

---

## 3. Detailed results

The result of **every** instance is provided as a CSV in `results/`:

- [`results/dynamic_instance_results.csv`](results/dynamic_instance_results.csv) - instances 1-51 (dynamic)
- [`results/static_instance_results.csv`](results/static_instance_results.csv) - instances 52-94 (static)

These are the same numbers as in the tables of the paper (main text and appendix), in machine-readable form. Columns:

| Column | Meaning |
|---|---|
| `instance` | Instance number used in the paper's tables |
| `arcs`, `nodes`, `num_od_pairs`, `k` | Network size, `\|Q\|`, and number of allowed paths |
| `boundary_condition` | `static` for dynamic instances with a pre-specified boundary condition (dynamic file only) |
| `gurobi_gap`, `gurobi_time_s` | Relative optimality gap at termination and total solution time (s) for Gurobi branch-and-bound |
| `bd_multicut_*`, `bd_single_pareto_*`, `bd_multi_pareto_*` | For each BD variant: `cuts` (number of cuts), `iterations`, `gap`, `time_s` |

Conventions, as in the paper: a gap of `-` means the instance was solved to the 10⁻³ tolerance; a time of `Time` means the 1-hour limit was reached (the reported gap is the relative gap at that point); `inf` is the value reported by Gurobi when no finite gap could be computed at termination (e.g., no feasible incumbent within the limit).

Running the code also writes one CSV per (scenario, method) pair into `results/experiments/` (named `<scenario>_<solver-parameter-file>.csv`), and `input_output.write_experiment_summary_file` aggregates them into a single summary table.

---

## 4. Code

### 4.1 Requirements

- Python 3 (the code uses `gurobipy`, `numpy`/`scipy`, `networkx`, `pandas`, `geopandas`, `shapely`, `osmnx`, `multiprocess`, `tqdm`, `matplotlib`, `seaborn`, `plotly`, `geojson`)
- **Gurobi Optimizer 12.0.1** with a valid license. The larger instances exceed the size limit of Gurobi's restricted pip license; free full-size [academic licenses](https://www.gurobi.com/academia/academic-program-and-licenses/) are available.

```bash
pip install gurobipy==12.0.1 numpy scipy networkx pandas geopandas shapely osmnx multiprocess tqdm matplotlib seaborn plotly geojson
```

Benders subproblems are solved analytically with sparse matrix algebra (NumPy/SciPy); the master problem is solved with Gurobi using lazy-constraint callbacks.

### 4.2 Running an instance

Run from the repository root (paths are resolved relative to the code, see `util.py`). A run takes a scenario file name and a solver-parameter file name, both without the `.csv` extension:

```python
from run import run_rollout_file

G, fig = run_rollout_file(
    scenario_code='sc_KCS315_rt400_t12_b1_od1_r0.1_lnghrz_k1',
    solver_param_code='sp_bd_mcTonODon_scon_0.001_3600_large',
)
```

To run several scenarios with several methods (skipping combinations already present in `results/experiments/`):

```python
from run import run_multiple_scenarios

run_multiple_scenarios(
    scenario_codes=['sc_KCS315_rt400_t12_b1_od1_r0.1_lnghrz_k1',
                    'sc_KCS315_rt400_t12_b1_od1_r0.1_lnghrz_k2'],
    solver_param_files=[['sp_gurobi_saoff_0.001_3600_large',
                         'sp_bd_mcTonODon_scoff_0.001_3600_large',
                         'sp_bd_mcToffODoff_scon_0.001_3600_large',
                         'sp_bd_mcTonODon_scon_0.001_3600_large']] * 2,
)
```

To reproduce a row of the paper's tables, pair the scenario file with each of the four methods in section 2.4 and compare with the corresponding row of `results/*_instance_results.csv`.

### 4.3 Reproducibility notes

- **Hardware.** The reported times were obtained on a 2020 MacBook Pro (Apple M1, 8 cores, 16 GB RAM) with Gurobi 12.0.1 through its Python API, using up to 8 threads. Absolute times will differ on other machines; iteration and cut counts of BD and the final solutions/gaps of runs that terminate within the limit are the more portable quantities. Runs that hit the 1-hour limit can differ between machines because Gurobi's search is time- and thread-dependent.
- **Instances are fixed.** Randomly generated OD sets and flows are stored with the instances (section 2.2).
- **Adding your own instance.** Copy `data/legends/scenario_template.csv`, edit the values, save it in `data/scenario/` as `sc_<name>.csv`, and use `<name>` as the `scenario_code`.

---

## 5. Citation and license

If you use this code or data, please cite the paper above (full reference to be updated upon publication).

The code and data in this repository are released under the [MIT License](LICENSE).

## References

See `references.bib` of the paper: Hwang et al. (2021), *freight analysis framework*, and US Surface Transportation Board (2020), *Carload Waybill Sample 2019*.
