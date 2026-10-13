# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 1, Ingest Raw Combine & Physics Metric Engineering
- **Roadmap Phase:** Phase 2, Physics Engine: Drills → Joules
- **Last updated:** 2026-10-12

---

## Current Objective

Turn the raw Combine results into a **mass-normalized mechanical-energy vector** for every prospect in the 5 target groups (WR, TE, DB, OL, DL):

```
data/raw/combine/*  ──►  data/interim/combine_si.parquet  ──►  data/processed/tee.parquet
                         (SI units, one row per prospect)       (TEE_Absolute in J, TEE_Relative in J/kg)
```

Done means every in-scope prospect has per-drill Joules (`E_vert, E_broad, E_40, W_cod_shuttle, W_cod_cone, W_bench`), their J/kg counterparts, a Sayers peak-power channel, a z-score composite, and missingness flags, and `tests/test_physics.py` passes.

---

## Last Session Accomplishments

- Created the project root `NFLBigDataBowl27/` and ran `git init` on `main` (**nothing committed yet**).
- Added a `.gitignore` that excludes everything under `data/` (competition data terms) plus Python/R/notebook artifacts and generated figures.
- Built the directory scaffold: `config/`, `data/{raw/{combine,combine_tracking,ngs,outcomes},interim,processed}`, `notebooks/`, `src/eeg/{ingest,physics,clustering,ngs,survival}` (empty `__init__.py` files), `tests/`, `reports/figures/`.
- Wrote `README.md`, `ROADMAP.md`, `TODO.md` and this `HANDOFF.md`.
- Fixed Phase 1 scope: theme "Combine to NFL Performance"; position groups decide who is in the study and are **never** model features.

No code, environment or data exists yet.

---

## Next Session Steps

Work in this order. Each step unblocks the next.

1. **Environment and first commit**
   - [x] `pyproject.toml` + `uv.lock` (Python 3.12 pinned in `.python-version`); `uv sync` builds `.venv` and installs `eeg` in editable mode
   - [x] Dependencies frozen in `uv.lock`; first commit of the scaffold made
2. **Constants and assumptions file:** `config/physics.yaml`
   - [ ] Put every tunable from *Active Hurdles & Variables* below in one place (g, ρ, C_d, θ, α, shuttle peak-speed factor), each with a low/high bound for the sensitivity sweep
3. **Units module:** `src/eeg/ingest/units.py` + `tests/test_units.py`
   - [ ] `in_to_m = 0.0254`, `lb_to_kg = 0.45359237`, `yd_to_m = 0.9144`, `G = 9.80665`, `BAR_KG = 225 × 0.45359237 = 102.058`
4. **Combine loader:** `src/eeg/ingest/load_combine.py`
   - [ ] Pull Combine data via `nflreadpy`; cache raw to `data/raw/combine/`
   - [ ] Parse height (`"6-2"` → 74 in), map positions → 5 groups, convert to SI, add `*_missing` flags
   - [ ] Write `data/interim/combine_si.parquet`
5. **Data inventory notebook:** `notebooks/00_data_inventory.ipynb`
   - [ ] Drill coverage by year and group; check whether 40-yd **split times** exist (this decides the sprint-fit strategy, see Hurdle 3)
   - [ ] If the Kaggle competition data is out: which drills have 10 Hz tracking, and the ID keys for joining to NGS
6. **Physics modules, simplest first:** `src/eeg/physics/`
   - [ ] `vertical.py` (mgh + Sayers) → `broad.py` → `bench.py` → `sprint.py` → `cod.py`
   - [ ] Write `tests/test_physics.py` alongside them, against these hand checks:
     - 100 kg, 1.0 m vertical → **980.7 J**
     - 100 kg, 3.0 m broad at θ = 45° → **1471.0 J**
     - 25 reps, ROM 0.40 m → 25 × 102.058 × 9.80665 × 0.40 = **10,009 J**
7. **TEE assembly:** `src/eeg/physics/tee.py` → `data/processed/tee.parquet`
8. **Sanity and sensitivity pass:** plots by group, 4-SD outlier flags, and a Spearman rank-stability check across the assumption bounds

---

## Active Hurdles & Variables

Every assumption below goes in `config/physics.yaml`, and each must survive the sensitivity sweep. If prospect rankings flip when one parameter moves within its bounds, that parameter needs better grounding before Phase 3.

### 1. Vertical jump: Sayers equation scope
- $P_{peak}\,[\text{W}] = 60.7 \cdot h\,[\text{cm}] + 45.3 \cdot m\,[\text{kg}] - 2055$ (Sayers et al., 1999).
- This is **peak power, not energy**. Store it as its own channel and never add it to Joule totals.
- It was derived on a general athletic sample. Extrapolating to 140–160 kg linemen is **out of sample**, so flag OL/DL Sayers values as less reliable.
- The Combine vertical is *jump reach − standing reach*, which approximates the rise of the center of mass. $E = mgh$ is therefore the energy at the apex only and leaves out takeoff-phase work.

### 2. Broad jump: takeoff angle θ
- Default **θ = 45°**, which gives $E = \tfrac12 m g R$.
- 45° minimizes the required $v_0$, so this is a **lower bound** on takeoff KE. Real standing long-jump angles are around 30–40°, and landing below takeoff CoM height adds distance.
- Sweep θ ∈ [30°, 45°]. At 30°, $\sin 2\theta = 0.866$, so E is about 15% higher.

### 3. 40-yard dash: sprint profile and drag
- **Model:** $v(t) = v_{max}(1 - e^{-t/\tau})$, so $x(t) = v_{max}\left[t - \tau(1 - e^{-t/\tau})\right]$.
- **Identifiability problem:** two unknowns ($v_{max}$, τ). The final 40 time alone gives **one equation**. Options, in order of preference:
  1. Fit to the **10 Hz Combine tracking** velocity trace (best).
  2. Fit to 10/20/40 splits if the data has them.
  3. Fix τ from a cohort relationship (e.g., regress τ on mass from prospects with tracking), then solve for $v_{max}$.
- **Drag:** $F_d = \tfrac12 \rho C_d A v^2$, so $W_d = \int_0^T \tfrac12 \rho C_d A\, v^3\, dt$.
  - **ρ ≈ 1.18 kg/m³**: Combine is indoors at Lucas Oil Stadium, Indianapolis (~220 m elevation, ~20 °C).
  - **C_d ≈ 0.9** (Samozino et al., 2016 sprint-profiling convention).
  - **Frontal area:** $A = 0.266 \cdot \text{BSA}$, with Du Bois body surface area $\text{BSA} = 0.2025 \cdot h^{0.725} \cdot m^{0.425}$ (h in m, m in kg). Expect A ≈ 0.54 m² (85 kg DB) to 0.72 m² (140 kg OL).
  - **Scale check:** at 9.5 m/s, $F_d \approx \tfrac12 (1.18)(0.9)(0.6)(9.5^2) \approx 29$ N. Over 36.6 m, drag work comes to roughly 500–800 J against about 4,000 J of peak KE. It is small but systematic, and it penalizes large frontal areas (OL/DL).
- **Timing start:** Combine electronic timing starts at first movement, so no reaction-time correction is needed. Confirm against the tracking data.

### 4. Shuttle (5-10-5) and 3-cone: change-of-direction speeds
- Without tracking we have **no speeds at the plant points**, only total time.
- Placeholder: segment peak speed $v_i \approx k \cdot (\text{segment distance} / \text{segment time})$, where **k ∈ [1.5, 2.0]** (2.0 = a perfect triangular accelerate/brake profile).
- Segment times are not recorded separately either. Split total time in proportion to segment length as a first pass.
- Count braking (eccentric) work as **absolute** work. It is real muscular effort even though it is negative mechanical work.
- This is the **weakest reconstruction**. Make it the top priority for replacement by tracking-measured speeds.

### 5. Bench press: arm length and range of motion
- $W = n_{reps} \cdot 102.058\,\text{kg} \cdot 9.80665 \cdot \text{ROM}$. Count concentric work only.
- **ROM = α × arm length.** Combine arm length runs shoulder to fingertip (~32–36 in, 0.81–0.91 m). Bar travel is roughly 0.35–0.50 m.
- Default **α = 0.50**, sweep **α ∈ [0.40, 0.55]**. Calibrate against published bench ROM data if possible.
- **Missing arm length:** impute from height (cohort regression) and set `arm_imputed = True`.
- Long arms mean **more work per rep and fewer reps**. Joules are designed to correct exactly this rep-count bias, so it's a good early test that the physics adds signal.

### 6. Cross-cutting
- **Mass:** use Combine weigh-in mass for all lab Joules. In Phase 4, use **in-season mass** (NGS `players` table) for field energy, or the η ratio will drift with weight change.
- **Composite scaling:** drill Joules differ by orders of magnitude (bench ~10⁴ J, vertical ~10³ J). The composite must be a **mean of z-scores**, never a raw sum.
- **Cohort drift:** compute z-scores within each Combine year so changes in timing or surface don't leak in as signal.
- **NGS units (Phase 4 preview):** tracking speed is in **yd/s** and acceleration in **yd/s²**. Convert before mixing with lab Joules.
