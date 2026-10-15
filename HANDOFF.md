# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 1, Ingest Raw Combine & Physics Metric Engineering
- **Roadmap Phase:** Phase 2, Physics Engine: Drills → Joules
- **Last updated:** 2026-10-14

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

## Last Session Accomplishments (2026-10-13 → 10-14)

- Environment: `pyproject.toml` + `uv.lock`, Python 3.12, `uv sync`. Repo pushed to `github.com/Healy218/NFLBigDataBowl27` (private).
- `config/physics.yaml` + `src/eeg/config.py` (assumptions with sweep bounds), `src/eeg/ingest/units.py`.
- Kaggle data downloaded to `data/raw/kaggle/nfl-big-data-bowl-2027/` (9 CSVs, 2.7 GB).
- `notebooks/00_data_inventory.ipynb` + `config/data_dictionary.md`: full inventory; key findings below.
- `src/eeg/ingest/load_combine.py` → `data/interim/combine_si.parquet` (510 prospects, 33 columns).
- Physics: `vertical.py`, `broad.py`, `bench.py`, `sprint.py` (417 prospects with E_40), `cod.py` (156 per drill with W_cod).
- 56 tests pass (`uv run pytest`).

**Key data findings** (details in the data dictionary):
- **510 prospects**, 2023–25 drafts; `combine_position` is already our 5 groups; `nfl_id` links every file.
- Height, weight, **arm length**, hand and wingspan are complete for all 510.
- Coverage: 40 / split / vertical / broad ≈ 80–90%; cone, shuttle, bench only 20–53%. 422 prospects have ≥ 3 drills.
- **Combine tracking** (10 Hz, yards, yd/s) exists for the **40 (418 prospects), shuttle (170) and 3-cone (160)**. None for vertical, broad or bench.
- **`dis` is unreliable** in the Combine tracking. Use `x`/`y` or `s`.
- 40 tracking matches official times on absolute field lines (flying 30 bias −0.025 s, r ≈ 0.98). The rep window starts 0.17–0.29 s before the official clock.
- Shuttle/cone rep windows open ~0.2 s before the official clock and end at or just short of the finish. After segmentation, tracked vs official has Spearman 0.82 / 0.88 (Hurdle 4).
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
   - [x] `vertical.py` (mgh + Sayers), `broad.py`, `bench.py` (measured arm length). Pure arithmetic: work on floats, numpy, polars Series and `pl.col` expressions; assumptions (θ, α, Sayers coefficients) passed in explicitly from config
   - [x] `src/eeg/ingest/load_tracking.py` → `data/interim/timed_drills.parquet`: PLAYER frames of 40/shuttle/cone, `rep_id`, `t_s`, SI units (keeps `x_yd` for yard-line geometry; drops `dis`)
   - [x] `sprint.py` (`uv run python -m eeg.physics.sprint` → `data/interim/forty_fits.parquet`): fits all 793 reps in < 1 s; **417 prospects get E_40**
   - [x] `cod.py` (`uv run python -m eeg.physics.cod` → `data/interim/cod_reps.parquet`): segments every shuttle / 3-cone rep, W_cod = ½ m Σ|Δ(v²)|; **156 prospects per drill get W_cod** (see Hurdle 4)
   - [x] `tests/test_physics.py` + `tests/test_cod.py` (56 tests total) pass these hand checks plus synthetic sprint / COD reps with known answers:
     - 100 kg, 1.0 m vertical → **980.7 J**
     - 100 kg, 3.0 m broad at θ = 45° → **1471.0 J**
     - 25 reps, ROM 0.40 m → 25 × 102.058 × 9.80665 × 0.40 = **10,009 J**
7. ~~**TEE assembly**~~ ✅ `uv run python -m eeg.physics.tee` → `data/processed/tee.parquet` (510 rows; schema in the data dictionary). Takes the rep tables plus a config, so the sweep can rebuild it with `config.with_override`.
   - Channel coverage: E_vert 438, E_broad 423, **E_40 417**, W_bench 192, **W_cod_shuttle 156, W_cod_cone 156**; **412 prospects have a composite** (≥ 3 channels), against 422 with ≥ 3 official drills. 12 lose their composite because a channel needs tracking; 2 gain one from a tracked-only W_cod.
   - Tracked W_cod without an official time (5 shuttle, 9 cone) is **kept and flagged** (`W_cod_<drill>_unofficial`) so the sweep or Phase 3 can drop it.
   - ⚠️ **Both composites are mostly body mass:** r(TEE_z_abs, mass) = +0.89, r(TEE_z_rel, mass) = −0.81. Per-kg channels correlate with each other (vert–broad ρ 0.83, vert–40 0.66, vert–cone 0.34), and bench/kg is independent (ρ ≈ 0). Phase 3 clustering must handle this: e.g. residualise on mass, or cluster absolute and relative channels together.
   - Composites average whatever channels are present (3–6), so a 3-channel composite isn't strictly comparable to a 6-channel one. `n_drills` is stored.
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
- **Done:** `sprint.py` fits $(v_{max}, τ, t_0)$ to each 40 trace (frames up to the x = 50 crossing). Median RMSE 0.18 m/s; v_max ≈ 10.5 (WR) → 8.8 m/s (OL); τ ≈ 1.27 s. Only 1 rep hits a bound.
- **Modelled 40 vs official:** `t40_model = t_finish − t0` has r = 0.91–0.97 against the official time but runs **0.12–0.23 s long** (per year). The model's motion onset comes before the official clock starts. This doesn't affect v_max or τ, but don't present t40_model as a 40 time.
- **No split fallback.** The 6 prospects with a 40 but no trace get **no E_40**. Fitting official 10-yd split + 40 is biased against tracking (v_max 0.5–0.8 m/s low, τ r ≈ 0.06–0.26, even with a per-year timing offset). The τ fallback was removed from config.
- **Rep choice:** reps with RMSE > 0.4 m/s (`sprint.max_fit_rmse_mps`; 13 of 793) are excluded, then each player's fastest `t40_model` rep is used. 1 player loses E_40 this way.
- **Start time:** the tracked window starts 0.17–0.29 s *before* the official clock, and the lead varies by year, so $t_0$ is a free fit parameter. The flying 30 (x 20 → 50) validates the tracking against `forty − ten_yd_split` (r ≈ 0.98).
- **Drag:** $F_d = \tfrac12 \rho C_d A v^2$, so $W_d = \int \tfrac12 \rho C_d A\, v^3\, dt$.
  - **ρ ≈ 1.18 kg/m³**: Combine is indoors at Lucas Oil Stadium, Indianapolis (~220 m elevation, ~20 °C).
  - **C_d ≈ 0.9** (Samozino et al., 2016 sprint-profiling convention).
  - **Frontal area:** $A = 0.266 \cdot \text{BSA}$, with Du Bois body surface area $\text{BSA} = 0.2025 \cdot h^{0.725} \cdot m^{0.425}$ (h in m, m in kg). Expect A ≈ 0.54 m² (85 kg DB) to 0.72 m² (140 kg OL).
  - **Scale check:** at 9.5 m/s, $F_d \approx \tfrac12 (1.18)(0.9)(0.6)(9.5^2) \approx 29$ N. **Actual:** W_drag median 832 J (686–1009) against KE median 5,232 J, about 14% of E_40. It is systematic and penalizes large frontal areas (OL/DL). Drag uses the closed-form ∫v³ for the mono-exponential model.
- Tracking `s` is in **yd/s**; convert to m/s before computing energy.

### 4. Shuttle (5-10-5) and 3-cone: change-of-direction work
- **Done:** `cod.py` measures work straight from the 10 Hz speed trace: $W_{cod} = \tfrac12 m \sum |\Delta(v^2)|$, the total absolute change in kinetic energy from motion onset to the finish. Each leg between plants contributes ≈ $m v_{peak}^2$ (accelerate + brake), the last leg ½ m v_end². Braking counts as **absolute** work. Curved 3-cone turns count only the speed actually shed, so no nominal segment lengths are needed. (The sensor turns ~1 yd short of every line: measured paths are 13.4 m shuttle and 20.1 m cone, against 18.3 m and 27.4 m nominal.)
- **Segmentation** (u = displacement along the first leg's direction):
  - **Shuttle:** out (u ≥ +2.5 yd), then the far leg (u ≤ −2.5 yd). The finish is the first − → + crossing of the start line *after the far point*, because the run already passes u = 0 mid-way through the 10-yd leg. First legs go either way (≈ 75% +x).
  - **3-cone:** out (u ≥ 2.5 yd), then the cone-3 loop (|lateral| ≥ 2.5 yd; always −y). The finish is the first + → − crossing after the loop's extreme.
  - **Window end as finish:** many windows stop just short of the line (sensor behind the hand), and 35 shuttle / 7 cone reps use that rule. The window end counts as the finish when it is within 1 yd of the line.
  - **Rejected reps:** shuttle 35 of 223, cone 13 of 203. Most finish below 0.8 × peak speed ("jogged out": 22 shuttle, 7 cone; these run 0.9–1.8 s over the official time). The rest are aborted or truncated windows: no far leg or loop, or fewer than 2 plants.
  - Energy starts at the last frame below 1 m/s, so idle jitter in long windows isn't counted. Best rep = fastest tracked time, as for the 40.
- **The earlier "2023/2025 cone mismatch" was a detector bug.** The first frame (u = 0) counted as a crossing, and finishes weren't anchored to the far point. Tracked time (onset → finish) vs official now has **Spearman 0.82 (shuttle), 0.88 (cone)**; 2023 is 0.87 / 0.97, with bias +0.24 s (shuttle) and +0.10 s (cone). The remaining 2024–25 scatter is symmetric and concentrated in players with a single tracked rep (MAD 0.10–0.14 s vs 0.04–0.08 s for multi-rep players). The tracked rep is probably not always the official one (an untracked or faulted attempt), not a segmentation error.
- **No k-factor fallback.** For prospects with an official time but no usable trace (50 shuttle, 33 cone), $v_i = k \cdot d_i / t_i$ reduces within a drill to W ∝ m / T². So k only rescales and was never a ranking parameter (it was removed from the sweep). Within position groups that proxy ranks tracked W_cod at Spearman **0.07–0.56, worse than mass alone**, so those prospects get no W_cod (per the plan: "drop the channel if the two methods don't agree").
- **Open question:** 5 shuttle / 9 cone prospects have a valid tracked rep but **no official time**. W_cod is kept for them for now; decide in `tee.py` whether a drill without an official result should count.
- **Scale:** W_cod median ≈ 5,260 J (shuttle) / 4,730 J (cone). Per kg, DB/WR are highest (shuttle ≈ 50 J/kg vs OL 42).

### 5. Bench press: arm length and range of motion
- $W = n_{reps} \cdot 102.058\,\text{kg} \cdot 9.80665 \cdot \text{ROM}$. Count concentric work only.
- **ROM = α × arm length.** Measured arm length is **complete for all 510** (28.5–36.9 in, 0.72–0.94 m), so no imputation.
- Default **α = 0.50**, sweep **α ∈ [0.40, 0.55]**.
- Long arms mean **more work per rep and fewer reps**. **Checked:** within each group, r(arm, reps) = −0.06 to −0.30, so the effect is real. But arm length varies only 3–4% (CV) against 15–22% for reps, so bench Joules rank almost identically to raw reps (Spearman 0.97–0.99). The correction is real but small; don't sell bench Joules as a big re-ranking.
- Only 192 prospects (38%) benched.

### 6. Cross-cutting
- **Mass:** use Combine weigh-in mass for all lab Joules. The data has **no in-season weight**, so Phase 4 field energy must use Combine mass too (or the η ratio's mass dependence has to be argued away).
- **Composite scaling:** drill Joules differ by orders of magnitude (bench ~10⁴ J, vertical ~10³ J). The composite must be a **mean of z-scores**, never a raw sum.
- **Cohort drift:** compute z-scores within each Combine year so changes in timing or surface don't leak in as signal. With only ~170 prospects per year (12–17 TEs), these cohorts are small.
- **Units:** all tracking (Combine and game) is in **yards, yd/s, yd/s²**. Convert before mixing with lab Joules. Recompute distance from `x`/`y`; don't trust `dis`.
