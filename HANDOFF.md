# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 2, Kinetic Profiles
- **Roadmap Phase:** Phase 3, Kinetic Profiles: Position-Blind Engine Analysis
- **Last updated:** 2026-10-18 (GMM test; Phase 3 re-planned around continuous profiles)

---

## Current Objective

Describe each prospect's **mechanical signature alone**, as continuous profile scores rather than archetype labels (K-Means and the GMM both found no natural engine clusters; see below):

```
data/processed/tee.parquet ──► data/processed/features.parquet ──► profiles: engine axes + Kinetic Anomaly score + size class
data/interim/forty_fits.parquet (v_max, τ) ──┘
```

Done means a per-prospect profile table (axis scores, anomaly score, size-class probabilities), stable axes under bootstrap, and a post-hoc read by position group ([`ROADMAP.md`](ROADMAP.md) Phase 3).

**Rules:**
- **Position groups are never features.** They scope the study and are used only to read the results.
- **Data:** only the Kaggle competition data. No external sources without asking.

---

## Where Phase 2 left things

Iteration 1 is closed. The physics engine is `src/eeg/physics/` and runs in this order:

```
uv run python -m eeg.ingest.load_combine     # → data/interim/combine_si.parquet (510 prospects)
uv run python -m eeg.ingest.load_tracking    # → data/interim/timed_drills.parquet
uv run python -m eeg.physics.sprint          # → data/interim/forty_fits.parquet  (417 with E_40)
uv run python -m eeg.physics.cod             # → data/interim/cod_reps.parquet    (156 per drill with W_cod)
uv run python -m eeg.physics.tee             # → data/processed/tee.parquet       (412 composites)
uv run python -m eeg.physics.sanity          # → outliers (none > 4 SD), reports/sensitivity.csv (154/154 pass)
```

Phase 3 so far:

```
uv run python -m eeg.clustering.features     # → data/processed/features.parquet (369), features_context.parquet, reports/feature_fits.csv
uv run python -m eeg.clustering.audit        # → reports/feature_audit.md
uv run python -m eeg.clustering.kmeans       # → reports/kmeans_selection.csv, data/processed/kmeans_labels.parquet
uv run python -m eeg.clustering.gmm          # → reports/gmm_report.md, gmm_selection.csv, data/processed/gmm_membership.parquet
```

84 tests (`uv run pytest`). Every drill model, assumption and rejected alternative from Phase 2 is in [`reports/phase2_physics_notes.md`](reports/phase2_physics_notes.md). Schemas are in [`config/data_dictionary.md`](config/data_dictionary.md).

**Data facts that matter for Phase 3:**
- 510 prospects, 2023–25 drafts; `nfl_id` (= `player_id`) links every file.
- Channel coverage: E_vert 438, E_broad 423, E_40 417, W_bench 192, W_cod 156 per drill. **Only 46 prospects have all six.**
- No in-season body mass anywhere in the data.

---

## Decisions made

### A. Body mass: features are "output for your size", plus size (2026-10-17)

On the 369 prospects with vertical, broad and 40, each feature set was clustered with K-Means (k = 3–6) and compared with pos_group (ARI). **Lower means the clusters are less a restatement of position.** (Run with the degree-1 trend.)

| Feature set | \|r(PC1, mass)\| | ARI vs position |
|---|---|---|
| Mass alone (reference) | 1.00 | 0.36–0.51 |
| Original ROADMAP list: J/kg channels + TEE_z_abs + mass + Sayers + v_max, τ | 0.98 | 0.32–0.52 (same as mass alone) |
| Absolute J + J/kg together | 1.00 | 0.33–0.47 |
| J/kg + v_max, τ | 0.87 | 0.20–0.29 |
| **Chosen: residuals on log mass + log mass as one feature** | 0.00 | 0.16–0.26 |
| Residuals on log mass only | 0.00 | 0.02 |

- Each feature is log(value) minus a trend in log mass fit across **all** prospects (never by position), within combine_year, then z-scored within year. Log mass is z-scored within year too.
- E_40 dropped (its residual correlates 1.00 with v_max's); Sayers dropped (r 0.99 with E_vert). Both stay for reporting.

### B. Missing drills: core only (2026-10-17)

- **Features:** log mass, E_vert, E_broad, v_max, τ for 369 prospects (DB 87, DL 82, OL 85, TE 30, WR 85).
- **Descriptors** (describe profiles after fitting, never define them): E_40, bench (146 of the 369), shuttle (126), cone (128), as the same mass-residual z-scores. Not imputed: who skips bench or agility drills depends on position.

### C. Mass trend is quadratic in log mass (2026-10-18)

`clustering.mass_trend_degree: 2`. A power law (degree 1) is curved in the data: **middle-mass prospects (TE, lighter DL) sit 0.4–0.6 SD above it** on vertical, broad and v_max (x² term F = 39–77; τ is flat). Degree-1 residuals are uncorrelated with mass but still depend on it, which breaks "for your size" and made K-Means find an "explosive middle-mass" cluster.
- Fits (`reports/feature_fits.csv`; slope at mean mass / curvature): E_vert 0.62 / −0.89, E_broad 0.72 / −0.64, v_max −0.34 / −0.51, τ −0.05 / +0.15. Descriptors: E_40 0.28 / −1.03, bench 1.15 / −1.56, shuttle 0.65 / −0.33, cone 0.75 / −0.23.
- The middle-mass bump is a **descriptive finding** for the write-up: per unit of mass, middle-sized prospects produce the most.

### D. Phase 3 delivers continuous profiles, not archetypes (2026-10-18)

Both clustering tests below say the engine is one cloud. Discrete archetypes (Heavy Engines, Hyper-Velocity Rockets) are dropped as types; the ideas survive as **regions of the profile axes** (e.g. high general explosiveness at size class 3). Kinetic Anomalies become a **continuous unusualness score**. Phase 4 Cox models get continuous covariates, which they use more efficiently than a cluster label.

---

## Phase 3 results so far

### Feature audit (step 4): keep all five features, no PCA for clustering

From `uv run python -m eeg.clustering.audit` → [`reports/feature_audit.md`](reports/feature_audit.md) (numbers at degree 2).
- **τ is noisy but real.** Rep-to-rep r is 0.51 (0.43–0.56 by year) against 0.97 for v_max. But with v_max held fixed, τ correlates **0.58** with the official 10-yd split (an independent stopwatch measure), and adding it raises the split R² from 0.80 to 0.87 in every year. 2024's wider spread is mostly real: its partial r is the highest (0.65). Averaging τ over both reps doesn't help (0.55), so best-rep τ stays.
- **No redundancy.** VIF ≤ 1.84. Correlations: vert–broad 0.59, broad–v_max 0.35, v_max–τ 0.56, all with mass ≈ 0 by construction.
- **PCA** (eigenvalues 1.83 / 1.42 / 1.00 / 0.44 / 0.30; 37 / 28 / 20 / 9 / 6 %):
  - PC1: general "explosive for your size" (vert 0.51, broad 0.59, v_max 0.56, τ 0.29)
  - PC2: τ + v_max (0.70, 0.43) against the jumps (−0.45, −0.35): a top speed that builds late vs quick, bouncy power
  - PC3: mass alone
  - PC4: vertical + τ against broad + v_max: up vs out

### K-Means baseline (step 5): no natural clusters

From `uv run python -m eeg.clustering.kmeans` → `reports/kmeans_selection.csv`, `reports/figures/kmeans_selection.png`.
- **Silhouette is 0.19–0.21 at every k**, below the 0.25 usually read as "no substantial structure". A structureless Gaussian cloud scores ~0.20, so the module compares each k with a **Gaussian null** (real log mass, the other four features redrawn from a Gaussian with their observed covariance). The data match it at every k (excess −0.6 to +2.1 null SDs).
- **The gap statistic picks k = 1.** ARI vs position ≈ 0 for k ≤ 3.
- At degree 1 there was one excess (k = 3, 6 null SDs): the middle-mass bump (decision C).

### GMM test (step 6): the engine is one Gaussian cloud; only size is lumpy

From `uv run python -m eeg.clustering.gmm` → [`reports/gmm_report.md`](reports/gmm_report.md), `reports/gmm_selection.csv`, `reports/figures/gmm_bic.png`, `data/processed/gmm_membership.parquet`. BIC over k = 1–8 × full / tied / diag / spherical.
- **Engine features alone: BIC picks k = 1** (tied = full, one Gaussian). **96 % of 50 bootstrap resamples** pick k = 1.
- **All five features: BIC picks tied k = 3**, but the components are **size classes**: z(log mass) −0.99 / +0.25 / +1.18, with engine means within ±0.3 SD.
- **Separable model wins:** a 3-class mixture on log mass × one engine Gaussian, independent of it, beats the joint mixture by **ΔBIC = 46** with 20 parameters instead of 32. Engine and size class carry no information about each other.
- **Size classes** (post hoc): 1 = WR 82 + DB 86; 2 = TE 30 + DL 40 (+ 4 skill players); 3 = OL 85 + DL 42. DL splits evenly across the middle and heavy classes. Soft membership is exported; most prospects are certain (median max-probability > 0.99).

---

## Next Session Steps

1. ~~Close Iteration 1~~ ✅ (2026-10-15)
2. ~~Decide A and B~~ ✅
3. ~~Feature matrix~~ ✅
4. ~~Correlation audit + PCA~~ ✅
5. ~~K-Means baseline~~ ✅ no natural clusters; ~~decide C and D~~ ✅
6. ~~GMM test (BIC from k = 1)~~ ✅ engine is one cloud; size classes only
7. **Profiles** (`clustering/profiles.py` → `data/processed/profiles.parquet`):
   - engine axes: PCA of the four engine features (mass excluded), signs fixed, named from loadings (general explosiveness, speed-build vs bounce, up vs out, …)
   - Kinetic Anomaly score: squared Mahalanobis distance of the engine features (χ²₄ under the Gaussian the GMM test supports) and its tail p-value; size class + probabilities from `gmm_membership.parquet`
8. Stability: bootstrap the axis loadings (Tucker congruence, target ≥ 0.95) and the anomaly ranks (Spearman); τ's noise matters most here
9. Read the results (post hoc): axis distributions by position group; descriptors (bench, COD, E_40) against the axes; **cross-position engine twins** (nearest neighbours in engine space with different positions); the anomaly list; the middle-mass bump figure
10. Close Iteration 2 → Iteration 3 (NGS alignment, η)
