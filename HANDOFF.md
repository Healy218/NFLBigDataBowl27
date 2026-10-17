# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 2, Kinetic Profile Clustering
- **Roadmap Phase:** Phase 3, Kinetic Profiles: Position-Blind Clustering
- **Last updated:** 2026-10-17 (K-Means baseline)

---

## Current Objective

Group prospects by **mechanical signature alone** into engine archetypes, with per-prospect membership probabilities:

```
data/processed/tee.parquet ──► data/processed/features.parquet ──► K-Means / GMM ──► archetype + P(archetype) per prospect
data/interim/forty_fits.parquet (v_max, τ) ──┘
```

Done means a stable solution (bootstrap ARI ≥ 0.7), named archetypes, membership probabilities per prospect, and a post-hoc archetype × position cross-tab ([`ROADMAP.md`](ROADMAP.md) Phase 3).

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

Phase 2 had 66 tests (`uv run pytest`; 71 with the Phase 3 feature tests). Every drill model, assumption and rejected alternative is in [`reports/phase2_physics_notes.md`](reports/phase2_physics_notes.md). Schemas are in [`config/data_dictionary.md`](config/data_dictionary.md).

**Data facts that matter for Phase 3:**
- 510 prospects, 2023–25 drafts; `nfl_id` (= `player_id`) links every file.
- Channel coverage: E_vert 438, E_broad 423, E_40 417, W_bench 192, W_cod 156 per drill. **Only 46 prospects have all six.**
- No in-season body mass anywhere in the data.

---

## Decisions made (2026-10-17)

### A. Body mass: features are "output for your size", plus size

On the 369 prospects with vertical, broad and 40, each feature set was clustered with K-Means (k = 3–6) and compared with pos_group (ARI). **Lower means the clusters are less a restatement of position.**

| Feature set | \|r(PC1, mass)\| | ARI vs position |
|---|---|---|
| Mass alone (reference) | 1.00 | 0.36–0.51 |
| Original ROADMAP list: J/kg channels + TEE_z_abs + mass + Sayers + v_max, τ | 0.98 | 0.32–0.52 (same as mass alone) |
| Absolute J + J/kg together | 1.00 | 0.33–0.47 |
| J/kg + v_max, τ | 0.87 | 0.20–0.29 |
| **Chosen: residuals on log mass + log mass as one feature** | 0.00 | 0.16–0.26 |
| Residuals on log mass only | 0.00 | 0.02 |

- Each feature is log(value) minus a log-log mass trend fit across **all** prospects (never by position), within combine_year, then z-scored within year. Log mass is z-scored within year too.
- Slopes (`reports/feature_fits.csv`): E_vert 0.60, E_broad 0.71, v_max −0.34, τ −0.04. Descriptors: E_40 0.26, bench 1.19, shuttle 0.66, cone 0.77.
- E_40 dropped (its residual correlates 1.00 with v_max's); Sayers dropped (r 0.99 with E_vert). Both stay for reporting.

### B. Missing drills: core only

- **Features:** log mass, E_vert, E_broad, v_max, τ for 369 prospects (DB 87, DL 82, OL 85, TE 30, WR 85).
- **Descriptors** (describe archetypes after fitting, never define them): E_40, bench (146 of the 369), shuttle (126), cone (128), as the same mass-residual z-scores. Not imputed: who skips bench or agility drills depends on position.

### Feature audit (step 4): keep all five features, no PCA

From `uv run python -m eeg.clustering.audit` → [`reports/feature_audit.md`](reports/feature_audit.md).
- **τ is noisy but real.** Rep-to-rep r is 0.51 (0.43–0.56 by year) against 0.97 for v_max. But with v_max held fixed, τ correlates **0.58** with the official 10-yd split (an independent stopwatch measure), and adding it raises the split R² from 0.80 to 0.87 in every year. 2024's wider spread is mostly real: its partial r is the highest (0.65). Averaging τ over both reps doesn't help (0.55), so best-rep τ stays.
- **No redundancy.** VIF ≤ 2.14 (broad). Vert–broad r is 0.64, but merging them would erase the jump-up vs jump-out contrast.
- **PCA** (eigenvalues 1.97 / 1.34 / 1.00 / 0.42 / 0.27; 39 / 27 / 20 / 8 / 5 %):
  - PC1: general "explosive for your size" (vert, broad, v_max load 0.55–0.61)
  - PC2: τ + v_max against the jumps: a top speed that builds late vs quick, bouncy power
  - PC3: mass alone
  - PC4: vertical vs broad / v_max (up vs out)
- **Decision: cluster on the five standardized features directly**, so centroids stay readable. GMM with full covariance handles the correlations. K-Means will give the PC1 direction extra weight (three features load on it); keep that in mind when comparing the two.
- Feature correlations: vert–broad 0.64, broad–v_max 0.46, v_max–τ 0.50, all with mass ≈ 0 by construction.

### K-Means baseline (step 5): no natural clusters

From `uv run python -m eeg.clustering.kmeans` → `reports/kmeans_selection.csv`, `reports/figures/kmeans_selection.png`, `data/processed/kmeans_labels.parquet`.
- **Silhouette is 0.19–0.24 at every k**, below the 0.25 usually read as "no substantial structure". A structureless Gaussian cloud scores ~0.20, so the module compares each k with a **Gaussian null**: real log mass, the other four features redrawn from a Gaussian with their observed covariance.
- **The gap statistic picks k = 1** (no clusters).
- **The only excess over the null is k = 3 (6 null SDs), and it's an artifact of the mass trend.** The log-linear (power-law) trend is curved: middle-mass prospects (TE, DL) sit 0.4–0.6 SD above it on vert, broad and v_max (x² term F = 39–77; τ is flat). So degree-1 residuals are uncorrelated with mass but still depend on it non-linearly, and the k = 3 clusters are "explosive middle-mass" / "heavy" / "light" (ARI vs position 0.23).
- **With a quadratic trend (`clustering.mass_trend_degree: 2`), the data match the null at every k** (excess −0.6 to +2.1 SD; gap still picks k = 1; ARI vs position ≈ 0 for k ≤ 3), and position-group means on every non-mass feature are within ±0.22 SD.
- **Conclusion: for its size, the Combine engine is a continuous, roughly Gaussian 4-D cloud.** The only lumpiness is body mass itself (bimodal: skill-sized vs lineman-sized). The ROADMAP's discrete archetypes aren't supported as natural types.

### Decision needed before step 6

1. **Mass trend:** degree 2 (recommended: degree-1 residuals still carry mass, which breaks the "for your size" definition) or degree 1 (keeps the middle-mass bump as cluster structure). The middle-mass bump can be reported as a descriptive finding either way.
2. **What Phase 3 delivers, given no natural clusters:**
   - (a) GMM anyway, framing archetypes as regions of a continuum (soft membership). Expect bootstrap ARI to struggle against the ≥ 0.7 bar.
   - (b) **Continuous kinetic profiles (recommended):** per-prospect scores on the interpretable axes (general explosiveness, speed-build vs bounce, up vs out, mass) plus a "Kinetic Anomaly" score (Mahalanobis distance / GMM log-likelihood). Phase 4 Cox models take continuous covariates more efficiently than an archetype label.
   - Either way, step 6 should still fit the GMM with BIC **including k = 1**, as the formal test.

---

## Next Session Steps

1. ~~Close Iteration 1~~ ✅ (2026-10-15)
2. ~~Decide A and B~~ ✅ (above)
3. ~~Feature matrix~~ ✅ `uv run python -m eeg.clustering.features` → `data/processed/features.parquet` (clustering input), `features_context.parquet` (pos_group, raw values, descriptors), `reports/feature_fits.csv`. 71 tests pass.
4. ~~Correlation audit + PCA~~ ✅ keep all five features, no PCA (above). 75 tests pass.
5. ~~K-Means baseline~~ ✅ no natural clusters (above). 80 tests pass.
   - **Decide** mass-trend degree and the Phase 3 deliverable (above)
6. GMM: BIC over k (from 1) × covariance type; export membership probabilities
7. Bootstrap stability (ARI ≥ 0.7)
8. Name archetypes from centroids; cross-tab archetype × position group; describe archetypes with bench / COD where present
