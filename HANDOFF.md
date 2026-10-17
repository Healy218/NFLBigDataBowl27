# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 2, Kinetic Profile Clustering
- **Roadmap Phase:** Phase 3, Kinetic Profiles: Position-Blind Clustering
- **Last updated:** 2026-10-17

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

### Feature caveats for the next steps

- **τ is noisy in 2024:** its log residual SD is ~0.17 against ~0.09 in 2023 / 2025, with values from 0.33 s to 2.2 s (sprinters are ~1.0–1.3 s). It isn't explained by window start speed or a block of player IDs. The within-year z-score scales it down but doesn't remove it. The correlation audit should check whether τ earns its place.
- Feature correlations: vert–broad 0.64, broad–v_max 0.46, v_max–τ 0.50 (a higher top speed takes longer to reach), all with mass ≈ 0 by construction.

---

## Next Session Steps

1. ~~Close Iteration 1~~ ✅ (2026-10-15)
2. ~~Decide A and B~~ ✅ (above)
3. ~~Feature matrix~~ ✅ `uv run python -m eeg.clustering.features` → `data/processed/features.parquet` (clustering input), `features_context.parquet` (pos_group, raw values, descriptors), `reports/feature_fits.csv`. 71 tests pass.
4. **Correlation audit + optional PCA:** does τ add signal or 2024 noise? Do vert and broad collapse to one jump axis?
5. K-Means baseline: k = 2..10; silhouette, gap statistic
6. GMM: BIC over k × covariance type; export membership probabilities
7. Bootstrap stability (ARI ≥ 0.7)
8. Name archetypes from centroids; cross-tab archetype × position group; describe archetypes with bench / COD where present
