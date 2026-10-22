# TODO: Iteration Tracker

Legend: `[x]` Completed · `[-]` In Progress · `[ ]` Not Started

Iterations map to phases in [`ROADMAP.md`](ROADMAP.md). When the Current Iteration changes, update [`HANDOFF.md`](HANDOFF.md) to match.

> ## ▶ Current Iteration: **Iteration 4, Explosion Potential → Regular-Season Performance** (Phase 4c)

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

## [x] Iteration 2: Kinetic Profiles *(Phase 3)*

- [x] Decide mass handling: residualise each channel on log mass (pooled, within year) + log mass as one feature; missing drills: core vertical / broad / v_max / τ only (369), bench and COD as descriptors
- [x] Feature matrix (`clustering/features.py`) → `data/processed/features.parquet` (no position columns) + `features_context.parquet`
- [x] Standardize; correlation audit; optional PCA (`clustering/audit.py` → `reports/feature_audit.md`: τ valid vs official split, VIF ≤ 2.1, no PCA)
- [x] K-Means baseline (k = 2..10; silhouette vs a Gaussian null, gap statistic from k = 1): **no natural clusters**; the k = 3 excess is curvature in the power-law mass trend
- [x] Decide: mass trend quadratic in log mass; Phase 3 delivers continuous profiles, not archetypes (HANDOFF decisions C, D)
- [x] GMM (`clustering/gmm.py`; BIC over k = 1–8 × covariance type): engine-only k = 1 (96 % of bootstraps); with mass, 3 size classes × one engine Gaussian; size-class probabilities exported
- [x] Profiles (`clustering/profiles.py`): 3 engine PCA axes (4th is τ noise), Kinetic Anomaly score (Mahalanobis, χ²₃), size class → `data/processed/profiles.parquet`
- [x] Stability: bootstrap congruence median 0.99–1.00 (p5 0.94–0.98), score ρ p5 ≥ 0.96, anomaly ρ p5 0.97; leave-one-year-out ≥ 0.98; score reliability 0.95 / 0.79 / 0.73
- [x] Post-hoc read (`clustering/readout.py` → `reports/phase3_readout.md`): position explains ≤ 4 % of any axis; engine neighbours ≈ chance by position; 95 mutual cross-position twin pairs; OL over-represented among anomalies (wider engine spread); bench / COD nearly independent of the axes; middle-mass bump peaks ~115 kg
- [x] Close Iteration 2: decisions and results → `reports/phase3_profiles_notes.md`; HANDOFF rewritten for Iteration 3

## [x] Iteration 3: NGS Alignment & Explosion Efficiency (η) *(Phase 4a–4b)*

- [x] Player-ID link: `nfl_id` is the same in every file; 497 of 510 prospects appear in game tracking, 292 of the 369 profiled in regular-season games
- [x] Ingest game tracking (`ingest/load_games.py`): REG only (PRE and POST dropped), SI units, speed and signed tangential acceleration re-derived from x/y, time since snap, glitch flags (450 of 251 k tracks)
- [x] Validate field kinematics against the Combine (`ngs/validate.py` → `reports/field_kinematics.md`): reported speed confirmed; NGS `a` is total magnitude, so acceleration = smoothed d(speed)/dt on both sides; glitch rule = drop tracks (450); season peak speed is 0.86 × lab (r 0.89, retest 0.90)
- [x] Per-frame $E_k/m$ and propulsive power per kg (inertia + drag) from reported speed and smoothed d(speed)/dt (`ngs/channels.py`); per play → game → player-season, plus the same estimators on the 40
- [x] Context filters: burst from rest (first 1.5 s after the snap, below 1 m/s at the snap; 92% of plays)
- [ ] Change-of-direction windows (deferred: lab COD covers only 126–128 profiled prospects)
- [x] Season statistic and exposure (`ngs/reliability.py` → `reports/field_reliability.md`): mean of the 3 best game maxima; games count with ≥ 5 plays; ≥ 100 plays and ≥ 6 games (476 player-seasons, 287 prospects); η reliability within position 0.65–0.67
- [x] Compute η per matching channel and the overall Explosion Efficiency Gap (`ngs/eta.py` → `reports/eta.md`): 394 player-seasons, 237 prospects; gap = 1 − η_KE (η_KE = η_speed², median 0.69); η raw, read within position, models stratified by position; burst and power η stored but driven by the lab value
- [x] η leaderboards per position group (`ngs/eta_readout.py` → `reports/eta_readout.md`; named table in `data/` only): empirical-Bayes shrinkage toward the group mean (B 0.44 for one season, 0.70 for three); η_KE barely related to the engine (r −0.08 with 40 speed, −0.17 with `ax_explosive`), +0.23 with NGS college production

- [x] Regular season only (POST dropped 2026-10-22 to match the brief; 4.6% of player-games, headline numbers moved ≤ ~0.01)
- [x] Close Iteration 3: decisions and results → `reports/phase4_eta_notes.md`; ROADMAP gains the competition brief and the strategy (physics model of explosion potential → η → regular-season performance); HANDOFF rewritten for Iteration 4

## [-] Iteration 4: Explosion Potential → Regular-Season Performance *(Phase 4c)*  ← CURRENT

- [x] Outcome table (`eeg.outcomes.table` → `outcomes_seasons.parquet`, `outcomes.parquet`, `reports/outcomes.md`): 1,007 league player-seasons (722 with a play), draft slot and round band, career totals, exits. player_play holds every scrimmage snap for OL/DL/DB but only route plays for WR (58%) and TE (40%). Draft slot drives playing time (within-class ρ −0.57 to −0.76); 48 exits (0 in round 1, 44% of rounds 6–7 who could)
- [x] Position production (`eeg.outcomes.production` → `production_seasons.parquet`, `reports/production.md`): 12 measures with split-half reliability, season retest and validity. Decided (open decision 1): primary outcome = playing time for every group; secondary = YPRR (WR 0.85, TE 0.80), DL pressure rate (0.67), OL peak pressure probability allowed (0.77); DB none credible. DL get-off (rel 0.89, retest 0.84) joins η as an in-game explosion channel
- [x] Sharper lab burst (`eeg.physics.burst` → `lab_burst.parquet`, `reports/lab_burst.md`): averaging two tracked reps lifts reliability to 0.62–0.73 but they stay weakly tied to games; the official 10-yard split, as a10 = 2d/t², tracks field burst (r 0.36) and DL get-off (−0.67, beyond top speed). Decided: `lab_burst.measure` = a10; profiles keep v_max + τ (τ is valid given v_max); η_burst keeps its same-estimator denominator
- [x] What predicts transfer (`eeg.ngs.transfer` → `reports/transfer.md`): the Combine predicts game output (top speed CV R² 0.29, burst 0.15 within group) but not η (CV R² 0.03 career, ~0 first season); college production is the one consistent η signal (β 0.22). DL get-off: CV R² 0.43 within DL, 0.16 within NFL position; a10 r −0.44 within NFL position (light, fast starters get off faster; a10 and mass not separable at n 53)
- [x] Exposure-free season value (found in step 5): the game-based `top3` grew with plays (η r +0.46 with season plays), so η credited playing time to the player. Now `top3_of50` (expected top-3 of 50 random plays; exact): no exposure bias, more reliable for speed (η 0.75 vs 0.67, retest 0.77 vs 0.61). Reran channels → transfer. Changed: η falls with the engine (r −0.26 with 40 speed, −0.36 with `ax_explosive`); the college production link is gone; the Combine now predicts η (CV R² 0.14, negatively) and game speed (0.31)
- [x] Potential and η → regular-season output (`eeg.outcomes.output` → `reports/output.md`): draft control = pick + undrafted (CV R² 0.51; open decision 2). The draft has priced the Combine (Combine alone 0.23, draft + Combine 0.46 vs draft 0.47). First-season η adds nothing to next-season playing time (β 0.03) or to production. DL chain beyond the draft, within NFL position: a10 → get-off (−0.44) → pressure rate (get-off β −0.47, a10 0.36, a10 drops to 0.20 with get-off)
- [x] Under- / over-drafted (`eeg.outcomes.mismatch` → `reports/mismatch.md`; named table in `data/` only): snaps and starts per season against a leave-one-out draft-slot expectation (per group, class intercepts). The top 20% by raw joules within group × class beat their slot 65% of the time vs 50% for the rest (permutation p 0.027; starts 57% vs 38%, p 0.003); a tail effect, flat below it. Not mass (mass tail 54% vs 51%), weaker for the size-adjusted axis (59% vs 52%, p 0.27). Strongest in DL (16 of 18). η not used: it needs playing time to exist, and step 5 found nothing
- [ ] Tenure (secondary): league exit with right-censoring; Kaplan–Meier and stratified Cox if events allow
- [ ] Optional: Combine position skill drills (10 Hz, ~60 drills) as position-specific movement

## [ ] Iteration 5: Submission *(Phase 4d)*

- [ ] Final figures (kinetic profile map, η leaderboards, potential → performance)
- [ ] Kaggle notebook narrative
- [ ] Reproducibility pass: clean run end-to-end from `data/raw/`
