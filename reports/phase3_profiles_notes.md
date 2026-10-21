# Phase 3 kinetic profile notes: decisions and results

Moved from `HANDOFF.md` when Iteration 2 closed (2026-10-20). This is the reference for how the Combine engine was turned into position-blind profiles, what was tested and rejected (discrete archetypes), and what the profiles show by position. Use it for the methods and results sections of the write-up.

## Pipeline

```
uv run python -m eeg.clustering.features     # → data/processed/features.parquet (369), features_context.parquet, reports/feature_fits.csv
uv run python -m eeg.clustering.audit        # → reports/feature_audit.md
uv run python -m eeg.clustering.kmeans       # → reports/kmeans_selection.csv, data/processed/kmeans_labels.parquet
uv run python -m eeg.clustering.gmm          # → reports/gmm_report.md, gmm_selection.csv, data/processed/gmm_membership.parquet
uv run python -m eeg.clustering.profiles     # → data/processed/profiles.parquet, reports/profiles_report.md
uv run python -m eeg.clustering.readout      # → reports/phase3_readout.md, data/processed/engine_twins.parquet, anomalies.parquet
```

Settings: `clustering` in `config/physics.yaml` (features, mass_trend_degree 2, profile_axes 3, descriptors). Schemas: `config/data_dictionary.md`.

**What Phase 4 uses:** `data/processed/profiles.parquet`: `ax_explosive`, `ax_speed_vs_bounce`, `ax_up_vs_out` (SD units), `anomaly_d2` / `anomaly_p`, `size_class` and `p_size_*`, for 369 prospects. Bench and agility are not in the axes (|r| ≤ 0.20 with every axis) and would enter separately on their subsets. Axis 2 and 3 effects will be attenuated by τ noise (score reliability 0.79, 0.73).

## Decisions

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

## Results

### Feature audit: keep all five features, no PCA for clustering

From `uv run python -m eeg.clustering.audit` → [`reports/feature_audit.md`](feature_audit.md) (numbers at degree 2).
- **τ is noisy but real.** Rep-to-rep r is 0.51 (0.43–0.56 by year) against 0.97 for v_max. But with v_max held fixed, τ correlates **0.58** with the official 10-yd split (an independent stopwatch measure), and adding it raises the split R² from 0.80 to 0.87 in every year. 2024's wider spread is mostly real: its partial r is the highest (0.65). Averaging τ over both reps doesn't help (0.55), so best-rep τ stays.
- **No redundancy.** VIF ≤ 1.84. Correlations: vert–broad 0.59, broad–v_max 0.35, v_max–τ 0.56, all with mass ≈ 0 by construction.
- **PCA** (eigenvalues 1.83 / 1.42 / 1.00 / 0.44 / 0.30; 37 / 28 / 20 / 9 / 6 %):
  - PC1: general "explosive for your size" (vert 0.51, broad 0.59, v_max 0.56, τ 0.29)
  - PC2: τ + v_max (0.70, 0.43) against the jumps (−0.45, −0.35): a top speed that builds late vs quick, bouncy power
  - PC3: mass alone
  - PC4: vertical + τ against broad + v_max: up vs out

### K-Means baseline: no natural clusters

From `uv run python -m eeg.clustering.kmeans` → `reports/kmeans_selection.csv`, `reports/figures/kmeans_selection.png`.
- **Silhouette is 0.19–0.21 at every k**, below the 0.25 usually read as "no substantial structure". A structureless Gaussian cloud scores ~0.20, so the module compares each k with a **Gaussian null** (real log mass, the other four features redrawn from a Gaussian with their observed covariance). The data match it at every k (excess −0.6 to +2.1 null SDs).
- **The gap statistic picks k = 1.** ARI vs position ≈ 0 for k ≤ 3.
- At degree 1 there was one excess (k = 3, 6 null SDs): the middle-mass bump (decision C).

### GMM test: the engine is one Gaussian cloud; only size is lumpy

From `uv run python -m eeg.clustering.gmm` → [`reports/gmm_report.md`](gmm_report.md), `reports/gmm_selection.csv`, `reports/figures/gmm_bic.png`, `data/processed/gmm_membership.parquet`. BIC over k = 1–8 × full / tied / diag / spherical.
- **Engine features alone: BIC picks k = 1** (tied = full, one Gaussian). **96 % of 50 bootstrap resamples** pick k = 1.
- **All five features: BIC picks tied k = 3**, but the components are **size classes**: z(log mass) −0.99 / +0.25 / +1.18, with engine means within ±0.3 SD.
- **Separable model wins:** a 3-class mixture on log mass × one engine Gaussian, independent of it, beats the joint mixture by **ΔBIC = 46** with 20 parameters instead of 32. Engine and size class carry no information about each other.
- **Size classes** (post hoc): 1 = WR 82 + DB 86; 2 = TE 30 + DL 40 (+ 4 skill players); 3 = OL 85 + DL 42. DL splits evenly across the middle and heavy classes. Soft membership is exported; most prospects are certain (median max-probability > 0.99).

### Profiles: three stable axes, an anomaly score, size class

From `uv run python -m eeg.clustering.profiles` → [`reports/profiles_report.md`](profiles_report.md), `data/processed/profiles.parquet`, `reports/figures/profiles_map.png`.

| Axis | Loadings (vert, broad, v_max, τ) | Variance | Score reliability |
|---|---|---|---|
| 1 `explosive` | +0.51, +0.59, +0.56, +0.29 | 46 % | 0.95 |
| 2 `speed_vs_bounce` | −0.45, −0.35, +0.43, +0.70 | 35 % | 0.79 |
| 3 `up_vs_out` | +0.69, −0.47, −0.36, +0.43 | 11 % | 0.73 |
| 4 (dropped) | +0.25, −0.56, +0.62, −0.50 | 8 % | 0.35 |

- **Score reliability** comes from 40 rep-to-rep noise: the mass residual of v_max is 0.85 reliable and τ's is 0.44 (322 players with two reps). The jumps have one recorded value each, so they count as 1 and the figures are upper bounds. **Axis 4 is mostly τ noise**, so it is dropped (`clustering.profile_axes: 3`) and kept out of the anomaly score.
- **Bootstrap over prospects (200):** Tucker congruence median 1.00 / 1.00 / 0.99 (5th percentile 0.98 / 0.98 / 0.94). Axes 3 and 4 have close eigenvalues (0.44 vs 0.30), so they occasionally rotate into each other. Every prospect's axis scores keep ρ ≥ 0.96, and the anomaly ranks ρ 0.97, in 95 % of resamples.
- **Leave one combine year out:** congruence ≥ 0.98 on every axis. 2024 (the noisy-τ year) moves axis 3 the most.
- **Kinetic Anomaly score:** χ²₃ on the kept axes. It is calibrated: 23 prospects at p < 0.05 (18 expected), 5 at p < 0.01 (3.7), none at p < 0.001.
- **Phase 4 caveat:** axes 2–3 carry τ noise, so their effects on tenure will be attenuated (about 0.8 and 0.7 of the true slope).

### Readout by position group 

From `uv run python -m eeg.clustering.readout` → [`reports/phase3_readout.md`](phase3_readout.md), `reports/figures/readout_axes_by_position.png`, `readout_mass_bump.png`. Positions are used here for the first time, only to read results. Named per-prospect tables (`engine_twins.parquet`, `anomalies.parquet`) stay in `data/processed/`, which is not committed; committed reports hold aggregates only.

- **Position barely shows in the engine.** Position group explains 0.8 % of `explosive` and 0.8 % of `speed_vs_bounce` (ANOVA p 0.55), and 3.8 % of `up_vs_out` (p 0.007). Relative to the others, DL jump out more than up (−0.28), TE and OL more up (+0.26, +0.20).
- **Your engine neighbour's position is close to chance.** The nearest prospect has the same position 26 % of the time in profile space, against 21.5 % under permuted labels (z 1.7). With log mass in the space it's 54 % (z 12): size, not engine, is what tracks position.
- **Engine twins:** 95 mutual cross-position nearest-neighbour pairs, spanning every position pair (DB–DL 16, DB–WR 15, DL–OL 13, …, DL–TE 6). Median distance to the nearest other-position prospect is 0.33 SD. The interesting twins sit far from the centre, e.g. a CB and a DE who are both about +2.1 SD `explosive` for their size.
- **Kinetic Anomalies are concentrated in OL.** At p < 0.05: OL 11 (4.2 expected), WR 1 (4.2); by size class, heavy 14 (6.4 expected). OL engines are simply more spread out (axis SDs 1.14–1.22 against 0.81–1.05 elsewhere), in every direction: the anomalies' largest axes split 8 / 8 / 7. Scoring each size class against its own covariance barely changes the ranking (Spearman 0.97; OL still 8), so the pooled score stays. **Write-up point:** linemen vary more in engine for their size than skill players do.
- **Bench and agility are separate qualities.** E_40 tracks `explosive` (r 0.76) and `speed_vs_bounce` (0.47), as expected since it is built from v_max. Bench, shuttle and 3-cone correlate |r| ≤ 0.20 with every axis, pooled and within position. The core axes don't capture change of direction or upper-body strength; they would have to enter Phase 4 separately, on their sparse subsets.
- **Middle-mass bump:** against a power law, output for size peaks in the ~115 kg decile (vert +5 %, broad +3.7 %, v_max +3 %), and the lightest and heaviest deciles sit 2–4 % below.
