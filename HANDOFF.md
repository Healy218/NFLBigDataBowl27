# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 1, Ingest Raw Combine & Physics Metric Engineering
- **Roadmap Phase:** Phase 2, Physics Engine: Drills → Joules
- **Last updated:** 2026-10-13

---

## Current Objective

Turn the raw Combine results into a **mass-normalized mechanical-energy vector** for every prospect in the 5 target groups (WR, TE, DB, OL, DL):

```
data/raw/kaggle/…/combine_results.csv  ──►  data/interim/combine_si.parquet  ──►  data/processed/tee.parquet
data/raw/kaggle/…/combine_tracking.csv ─────────────────(40, shuttle, 3-cone)──┘
                                            (SI units, one row per prospect)      (TEE_Absolute in J, TEE_Relative in J/kg)
```

Done means every in-scope prospect has per-drill Joules (`E_vert, E_broad, E_40, W_cod_shuttle, W_cod_cone, W_bench`), their J/kg counterparts, a Sayers peak-power channel, a z-score composite, and missingness flags, and `tests/test_physics.py` passes.

**Data rule:** use only the Kaggle competition data. No external sources.

---

## Last Session Accomplishments (2026-10-13)

- Environment: `pyproject.toml` + `uv.lock`, Python 3.12, `uv sync`. Repo pushed to `github.com/Healy218/NFLBigDataBowl27` (private).
- `config/physics.yaml` + `src/eeg/config.py` (assumptions with sweep bounds), `src/eeg/ingest/units.py`.
- Kaggle data downloaded to `data/raw/kaggle/nfl-big-data-bowl-2027/` (9 CSVs, 2.7 GB).
- `notebooks/00_data_inventory.ipynb` + `config/data_dictionary.md`: full inventory; key findings below.
- `src/eeg/ingest/load_combine.py` → `data/interim/combine_si.parquet` (510 prospects, 33 columns).
- 18 tests pass (`uv run pytest`).

**Key data findings** (details in the data dictionary):
- **510 prospects**, 2023–25 drafts; `combine_position` is already our 5 groups; `nfl_id` links every file.
- Height, weight, **arm length**, hand and wingspan are complete for all 510.
- Coverage: 40 / split / vertical / broad ≈ 80–90%; cone, shuttle, bench only 20–53%. 422 prospects have ≥ 3 drills.
- **Combine tracking** (10 Hz, yards, yd/s) exists for the **40 (418 prospects), shuttle (170) and 3-cone (160)**. None for vertical, broad or bench.
- **`dis` is unreliable** in the Combine tracking. Use `x`/`y` or `s`.
- 40 tracking matches official times on absolute field lines (flying 30 bias −0.025 s, r ≈ 0.98). The rep window starts 0.17–0.29 s before the official clock.
- Shuttle/cone rep windows include pre/post movement (≈ 0.4–0.5 s longer than official; r ≈ 0.2). They need segmentation.
- **No in-season body mass** anywhere in the data (`players.csv` has no weight).

---

## Next Session Steps

Work in this order. Each step unblocks the next.

1. ~~Environment and first commit~~ ✅
2. ~~`config/physics.yaml` + loader~~ ✅ (g stays in `units.py`: a uniform scale can't change rankings, so it isn't swept)
3. ~~`units.py`~~ ✅
4. ~~Combine loader~~ ✅ `uv run python -m eeg.ingest.load_combine`
5. ~~Data inventory notebook + data dictionary~~ ✅
6. **Physics modules, simplest first:** `src/eeg/physics/`
   - [ ] `vertical.py` (mgh + Sayers) → `broad.py` → `bench.py` (measured arm length) → `sprint.py` (fit to tracking) → `cod.py` (tracking where available)
   - [ ] A tracking loader for the timed drills (`src/eeg/ingest/load_tracking.py`): PLAYER rows only, rep key (`nfl_id`, `event_id`, `attempt`), yards → m
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
- No tracking for this drill.

### 2. Broad jump: takeoff angle θ
- Default **θ = 45°**, which gives $E = \tfrac12 m g R$.
- 45° minimizes the required $v_0$, so this is a **lower bound** on takeoff KE. Real standing long-jump angles are around 30–40°, and landing below takeoff CoM height adds distance.
- Sweep θ ∈ [30°, 45°]. At 30°, $\sin 2\theta = 0.866$, so E is about 15% higher.
- No tracking for this drill.

### 3. 40-yard dash: sprint profile and drag
- **Model:** $v(t) = v_{max}(1 - e^{-(t-t_0)/\tau})$, so $x(t) = v_{max}\left[(t-t_0) - \tau(1 - e^{-(t-t_0)/\tau})\right]$.
- **Identifiability: solved.** 418 of 424 prospects with an official 40 have a 10 Hz trace (793 reps), so fit $v_{max}$, τ to the measured speed. The 6 without tracking fall back to the official 10-yd split + 40 (two equations, two unknowns), then τ fallback.
- **Start time:** the tracked window starts 0.17–0.29 s *before* the official clock, and the lead varies by year. Fit $t_0$ as a free parameter (or anchor on the x = 10 crossing). Never use the first frame as t = 0. The flying 30 (x 20 → 50) can validate the fit against `forty − ten_yd_split`.
- **Which rep:** the official time is the best attempt. Use the rep with the fastest x = 10 → 50 crossing (decide when writing `sprint.py`).
- **Drag:** $F_d = \tfrac12 \rho C_d A v^2$, so $W_d = \int \tfrac12 \rho C_d A\, v^3\, dt$.
  - **ρ ≈ 1.18 kg/m³**: Combine is indoors at Lucas Oil Stadium, Indianapolis (~220 m elevation, ~20 °C).
  - **C_d ≈ 0.9** (Samozino et al., 2016 sprint-profiling convention).
  - **Frontal area:** $A = 0.266 \cdot \text{BSA}$, with Du Bois body surface area $\text{BSA} = 0.2025 \cdot h^{0.725} \cdot m^{0.425}$ (h in m, m in kg). Expect A ≈ 0.54 m² (85 kg DB) to 0.72 m² (140 kg OL).
  - **Scale check:** at 9.5 m/s, $F_d \approx \tfrac12 (1.18)(0.9)(0.6)(9.5^2) \approx 29$ N. Over 36.6 m, drag work comes to roughly 500–800 J against about 4,000 J of peak KE. It is small but systematic, and it penalizes large frontal areas (OL/DL).
- Tracking `s` is in **yd/s**; convert to m/s before computing energy.

### 4. Shuttle (5-10-5) and 3-cone: change-of-direction speeds
- **Tracking exists for 170 (shuttle) / 160 (cone) prospects**, so plant-point and segment speeds can be measured. But only ~200 prospects have an official time at all.
- **Segmentation needed first:** the rep windows include pre/post movement (≈ 0.4–0.5 s longer than official, r ≈ 0.2). Detect the start (first movement) and finish of each run before using the trace.
- For the ~40 / ~30 prospects with an official time but no tracking, fall back to the k-factor placeholder: segment peak speed $v_i \approx k \cdot (\text{segment distance} / \text{segment time})$, **k ∈ [1.5, 2.0]**, time split in proportion to segment length. Flag them, or drop the channel for them if the two methods don't agree.
- Count braking (eccentric) work as **absolute** work. It is real muscular effort even though it is negative mechanical work.

### 5. Bench press: arm length and range of motion
- $W = n_{reps} \cdot 102.058\,\text{kg} \cdot 9.80665 \cdot \text{ROM}$. Count concentric work only.
- **ROM = α × arm length.** Measured arm length is **complete for all 510** (28.5–36.9 in, 0.72–0.94 m), so no imputation.
- Default **α = 0.50**, sweep **α ∈ [0.40, 0.55]**.
- Long arms mean **more work per rep and fewer reps**. Joules are designed to correct exactly this rep-count bias, so it's a good early test that the physics adds signal.
- Only 192 prospects (38%) benched.

### 6. Cross-cutting
- **Mass:** use Combine weigh-in mass for all lab Joules. The data has **no in-season weight**, so Phase 4 field energy must use Combine mass too (or the η ratio's mass dependence has to be argued away).
- **Composite scaling:** drill Joules differ by orders of magnitude (bench ~10⁴ J, vertical ~10³ J). The composite must be a **mean of z-scores**, never a raw sum.
- **Cohort drift:** compute z-scores within each Combine year so changes in timing or surface don't leak in as signal. With only ~170 prospects per year (12–17 TEs), these cohorts are small.
- **Units:** all tracking (Combine and game) is in **yards, yd/s, yd/s²**. Convert before mixing with lab Joules. Recompute distance from `x`/`y`; don't trust `dis`.
