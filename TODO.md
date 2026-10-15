# TODO: Iteration Tracker

Legend: `[x]` Completed · `[-]` In Progress · `[ ]` Not Started

Iterations map to phases in [`ROADMAP.md`](ROADMAP.md). When the Current Iteration changes, update [`HANDOFF.md`](HANDOFF.md) to match.

> ## ▶ Current Iteration: **Iteration 2, Kinetic Profile Clustering** (Phase 3)

---

## [x] Iteration 0: Project Setup & Git Config *(Phase 1)*

- [x] Create project root `NFLBigDataBowl27/`
- [x] `git init` on `main`
- [x] `.gitignore`: all of `data/` excluded (competition data terms), Python/R/notebook artifacts, generated figures
- [x] Directory scaffold: `config/`, `data/{raw,interim,processed}`, `notebooks/`, `src/eeg/{ingest,physics,clustering,ngs,survival}`, `tests/`, `reports/figures/`
- [x] Core docs: `README.md`, `ROADMAP.md`, `TODO.md`, `HANDOFF.md`
- [x] Phase 1 scope fixed: theme "Combine to NFL Performance"; 5 groups (WR, TE, DB, OL, DL); positions used to scope the study, never as model features

---

## [x] Iteration 1: Ingest Raw Combine & Physics Metric Engineering *(Phase 2)*

### 1.1 Environment
- [x] Create `.venv` with `uv` (Python 3.12); deps declared in `pyproject.toml` (`polars pandas pyarrow numpy scipy scikit-learn lifelines matplotlib seaborn plotly pyyaml`; dev: `jupyter pytest`)
- [x] Freeze via `uv.lock` and commit
- [x] Initial commit of the scaffold

### 1.2 Data acquisition & inventory
- [x] Download competition data from Kaggle into `data/raw/kaggle/nfl-big-data-bowl-2027/`
- [x] Write `notebooks/00_data_inventory.ipynb`: row counts per Combine class, drill coverage %, which drills have 10 Hz tracking, NGS season overlap
- [x] Document the schema of every raw file in `config/data_dictionary.md`

### 1.3 Ingestion & unit conversion (`src/eeg/ingest/`)
- [x] `load_combine.py`: one row per prospect (`player_id` = `nfl_id`); source-unit + SI columns, plus hand, wingspan, 10-yd split, NGS scores
- [x] Map raw positions → 5 target groups: already given as `combine_position`; loader validates it
- [x] `units.py`: `in_to_m`, `lb_to_kg`, `yd_to_m`, plus constants `G = 9.80665`, `BAR_KG = 102.058` (225 lb)
- [x] `NA` → null with a `*_missing` flag per drill (heights are already numeric inches)
- [x] Write `data/interim/combine_si.parquet` (510 prospects)
- [x] `load_tracking.py`: timed-drill reps (40, shuttle, 3-cone), PLAYER rows, yards → m → `timed_drills.parquet`
- [x] Shuttle / 3-cone rep segmentation (start/finish detection): in `cod.py`

### 1.4 Physics: Joule formulas per drill (`src/eeg/physics/`)
- [x] **Vertical jump** (`vertical.py`)
  - [x] Potential energy: $E_{vert} = m g h$, with h in metres
  - [x] Sayers peak power: $P = 60.7\,h_{cm} + 45.3\,m - 2055$ (W). Store as a separate power channel, **not** summed with Joules
- [x] **Broad jump** (`broad.py`)
  - [x] Takeoff velocity: $v_0 = \sqrt{R g / \sin 2\theta}$ with θ configurable (default 45°)
  - [x] Kinetic energy: $E_{broad} = \tfrac12 m v_0^2$
- [x] **40-yard dash** (`sprint.py`)
  - [x] Fit $v(t) = v_{max}(1 - e^{-(t-t_0)/\tau})$ to the 10 Hz tracking trace (`scipy.optimize.curve_fit`), $t_0$ free
  - [x] ~~Fallback via official 10-yd split + 40~~: biased against tracking, not used; 6 prospects get no E_40
  - [x] Kinetic energy at $v_{max}$: $\tfrac12 m v_{max}^2$
  - [x] Drag work: $W_d = \int_0^{T} \tfrac12 \rho C_d A\, v(t)^3\, dt$; frontal area $A$ from height/mass (see `reports/phase2_physics_notes.md`)
  - [x] $E_{40} = \tfrac12 m v_{max}^2 + W_d$
- [x] **5-10-5 shuttle & 3-cone** (`cod.py`)
  - [x] Segment each rep from tracking: first-leg direction, far point / cone-3 loop, finish crossing (or window end within 1 yd); reject jogged-out and truncated reps
  - [x] Change-of-direction work: $W_{cod} = \tfrac12 m \sum |\Delta(v^2)|$ over the measured speed trace (≡ Σ ½ m v_in² + ½ m v_out² per leg), braking counted as absolute
  - [x] ~~k-factor fallback from official time~~: reduces to m/T², ranks tracked W_cod at ρ 0.07–0.56 within groups; not used. 156 prospects per drill get W_cod
- [x] **Bench press** (`bench.py`)
  - [x] $W_{bench} = n_{reps} \cdot m_{bar}\, g \cdot \text{ROM}$, with $\text{ROM} = \alpha \cdot \text{arm length}$ (α from config)

### 1.5 Mass normalization & TEE assembly
- [x] `tee.py`: build `TEE_Absolute` vector (J) per drill: `E_vert, E_broad, E_40, W_cod_shuttle, W_cod_cone, W_bench`
- [x] Body-mass normalization → `TEE_Relative` (J/kg) for every channel
- [x] Composite score: mean of per-drill z-scores (within Combine-year cohort), not a raw Joule sum → `TEE_z_abs`, `TEE_z_rel`
- [x] Missingness policy: require ≥ 3 of 6 drills for a composite; record `n_drills` (412 prospects have one)
- [x] Write `data/processed/tee.parquet`

### 1.6 Validation
- [x] `tests/test_units.py`: conversion round-trips
- [x] `tests/test_physics.py`: hand-calculated cases (100 kg, 1.0 m vertical → 980.7 J; 45° broad of 3.0 m at 100 kg → 1471 J); `tests/test_cod.py`: synthetic shuttle / cone reps with exact Σ|Δv²|
- [x] Sanity check (`sanity.py`): plot `E_40` vs mass and `TEE_Relative` composite by pos_group; flag values more than 4 SD out (none)
- [x] Sensitivity sweep: re-run with α, θ, ρ, $C_d$, A at their low/high bounds (+ all-low / all-high); rankings stable (154/154 cases ρ ≥ 0.9, min 0.9986) → `reports/sensitivity.csv`

---

## [-] Iteration 2: Kinetic Profile Clustering *(Phase 3)*  ← CURRENT

- [ ] Decide mass handling (residualise on log mass?) and missing-drill policy (core 369?): see HANDOFF
- [ ] Feature matrix from `tee.parquet` (no position columns)
- [ ] Standardize; correlation audit; optional PCA
- [ ] K-Means baseline (k = 2..10; silhouette, gap statistic)
- [ ] GMM (BIC over k × covariance type); export membership probabilities
- [ ] Bootstrap stability (ARI)
- [ ] Name archetypes from centroids (Heavy Engines, Hyper-Velocity Rockets, Kinetic Anomalies, …)
- [ ] Post-hoc cross-tab: archetype × position group

## [ ] Iteration 3: NGS Alignment & Explosion Efficiency (η) *(Phase 4a–4b)*

- [ ] Player-ID link: Combine `nfl_id` ↔ NGS `nflId` (same ID in every competition file; confirm coverage)
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
