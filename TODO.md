# TODO: Iteration Tracker

Legend: `[x]` Completed · `[-]` In Progress · `[ ]` Not Started

Iterations map to phases in [`ROADMAP.md`](ROADMAP.md). When the Current Iteration changes, update [`HANDOFF.md`](HANDOFF.md) to match.

> ## ▶ Current Iteration: **Iteration 1, Ingest Raw Combine & Physics Metric Engineering** (Phase 2)

---

## [x] Iteration 0: Project Setup & Git Config *(Phase 1)*

- [x] Create project root `NFLBigDataBowl27/`
- [x] `git init` on `main`
- [x] `.gitignore`: all of `data/` excluded (competition data terms), Python/R/notebook artifacts, generated figures
- [x] Directory scaffold: `config/`, `data/{raw,interim,processed}`, `notebooks/`, `src/eeg/{ingest,physics,clustering,ngs,survival}`, `tests/`, `reports/figures/`
- [x] Core docs: `README.md`, `ROADMAP.md`, `TODO.md`, `HANDOFF.md`
- [x] Phase 1 scope fixed: theme "Combine to NFL Performance"; 5 groups (WR, TE, DB, OL, DL); positions used to scope the study, never as model features

---

## [-] Iteration 1: Ingest Raw Combine & Physics Metric Engineering *(Phase 2)*  ← CURRENT

### 1.1 Environment
- [x] Create `.venv` with `uv` (Python 3.12); deps declared in `pyproject.toml` (`polars pandas pyarrow numpy scipy scikit-learn lifelines matplotlib seaborn plotly pyyaml`; dev: `jupyter pytest`)
- [x] Freeze via `uv.lock` and commit
- [x] Initial commit of the scaffold

### 1.2 Data acquisition & inventory
- [ ] Download competition data from Kaggle into `data/raw/combine_tracking/` and `data/raw/ngs/`
- [ ] Write `notebooks/00_data_inventory.ipynb`: row counts per Combine class, drill coverage %, which drills have 10 Hz tracking, NGS season overlap
- [ ] Document the schema of every raw file in `config/data_dictionary.md`

### 1.3 Ingestion & unit conversion (`src/eeg/ingest/`)
- [ ] `load_combine.py`: one row per prospect; columns `player_id, name, combine_year, pos, pos_group, ht_in, wt_lb, arm_in, forty, vertical_in, broad_in, bench_reps, cone, shuttle`
- [ ] Map raw positions → 5 target groups (WR, TE, DB, OL, DL); drop everyone else
- [x] `units.py`: `in_to_m`, `lb_to_kg`, `yd_to_m`, plus constants `G = 9.80665`, `BAR_KG = 102.058` (225 lb)
- [ ] Parse height strings (`"6-2"` → 74 in); coerce DNP / blanks to null with a `*_missing` flag
- [ ] Write `data/interim/combine_si.parquet`

### 1.4 Physics: Joule formulas per drill (`src/eeg/physics/`)
- [ ] **Vertical jump** (`vertical.py`)
  - [ ] Potential energy: $E_{vert} = m g h$, with h in metres
  - [ ] Sayers peak power: $P = 60.7\,h_{cm} + 45.3\,m - 2055$ (W). Store as a separate power channel, **not** summed with Joules
- [ ] **Broad jump** (`broad.py`)
  - [ ] Takeoff velocity: $v_0 = \sqrt{R g / \sin 2\theta}$ with θ configurable (default 45°)
  - [ ] Kinetic energy: $E_{broad} = \tfrac12 m v_0^2$
- [ ] **40-yard dash** (`sprint.py`)
  - [ ] Fit $v(t) = v_{max}(1 - e^{-t/\tau})$ to 10/20/40-yd splits (`scipy.optimize.curve_fit`)
  - [ ] If 10 Hz tracking exists, fit to the measured velocity trace instead
  - [ ] Kinetic energy at $v_{max}$: $\tfrac12 m v_{max}^2$
  - [ ] Drag work: $W_d = \int_0^{T} \tfrac12 \rho C_d A\, v(t)^3\, dt$; frontal area $A$ from height/mass (see HANDOFF hurdles)
  - [ ] $E_{40} = \tfrac12 m v_{max}^2 + W_d$
- [ ] **5-10-5 shuttle & 3-cone** (`cod.py`)
  - [ ] Segment model: 5 / 10 / 5 yd (shuttle); estimate segment entry/exit speeds
  - [ ] Change-of-direction work: $W_{cod} = \sum_i (\tfrac12 m v_{in,i}^2 + \tfrac12 m v_{out,i}^2)$, braking (negative) work counted as absolute
  - [ ] Use tracking-measured speeds at each plant when available
- [ ] **Bench press** (`bench.py`)
  - [ ] $W_{bench} = n_{reps} \cdot m_{bar}\, g \cdot \text{ROM}$, with $\text{ROM} = \alpha \cdot \text{arm length}$ (α from config)
  - [ ] Impute missing arm length from height (regression on the cohort) and flag it

### 1.5 Mass normalization & TEE assembly
- [ ] `tee.py`: build `TEE_Absolute` vector (J) per drill: `E_vert, E_broad, E_40, W_cod_shuttle, W_cod_cone, W_bench`
- [ ] Body-mass normalization → `TEE_Relative` (J/kg) for every channel
- [ ] Composite score: mean of per-drill z-scores (within Combine-year cohort), not a raw Joule sum
- [ ] Missingness policy: require ≥ 3 of 6 drills for a composite; record `n_drills`
- [ ] Write `data/processed/tee.parquet`

### 1.6 Validation
- [x] `tests/test_units.py`: conversion round-trips
- [ ] `tests/test_physics.py`: hand-calculated cases (100 kg, 1.0 m vertical → 980.7 J; 45° broad of 3.0 m at 100 kg → 1471 J)
- [ ] Sanity check: plot `E_40` vs mass and `TEE_Relative` composite by pos_group; flag values more than 4 SD out
- [ ] Sensitivity sweep: re-run with α, θ, $C_d A$ at their low/high bounds; check that prospect rankings are stable (Spearman ρ ≥ 0.9)

---

## [ ] Iteration 2: Kinetic Profile Clustering *(Phase 3)*

- [ ] Feature matrix from `tee.parquet` (no position columns)
- [ ] Standardize; correlation audit; optional PCA
- [ ] K-Means baseline (k = 2..10; silhouette, gap statistic)
- [ ] GMM (BIC over k × covariance type); export membership probabilities
- [ ] Bootstrap stability (ARI)
- [ ] Name archetypes from centroids (Heavy Engines, Hyper-Velocity Rockets, Kinetic Anomalies, …)
- [ ] Post-hoc cross-tab: archetype × position group

## [ ] Iteration 3: NGS Alignment & Explosion Efficiency (η) *(Phase 4a–4b)*

- [ ] Player-ID crosswalk: Combine prospect ↔ NGS `nflId` (via gsis_id / pfr_id)
- [ ] Per-frame $E_k$ and propulsive power from NGS speed/acceleration
- [ ] Context filters: burst-from-rest windows, change-of-direction windows
- [ ] High-percentile aggregation per player-season
- [ ] Compute η per matching channel and the overall Explosion Efficiency Gap
- [ ] η leaderboards per position group

## [ ] Iteration 4: Survival Modeling *(Phase 4c)*

- [ ] Build duration/event table (draft → exit; right-censor active players)
- [ ] Join snap counts, wAV, draft round/pick
- [ ] Kaplan–Meier by archetype
- [ ] Cox PH: archetype + η + TEE_Relative + mass, controlling for draft capital
- [ ] Schoenfeld residuals; concordance index; stratify if PH is violated
- [ ] Secondary models: snap-share trajectory, wAV

## [ ] Iteration 5: Submission *(Phase 4d)*

- [ ] Final figures (archetype map, η vs. tenure, survival curves)
- [ ] Kaggle notebook narrative
- [ ] Reproducibility pass: clean run end-to-end from `data/raw/`
