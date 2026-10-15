# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 2, Kinetic Profile Clustering
- **Roadmap Phase:** Phase 3, Kinetic Profiles: Position-Blind Clustering
- **Last updated:** 2026-10-15

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

66 tests pass (`uv run pytest`). Every drill model, assumption and rejected alternative is in [`reports/phase2_physics_notes.md`](reports/phase2_physics_notes.md). Schemas are in [`config/data_dictionary.md`](config/data_dictionary.md).

**Data facts that matter for Phase 3:**
- 510 prospects, 2023–25 drafts; `nfl_id` (= `player_id`) links every file.
- Channel coverage: E_vert 438, E_broad 423, E_40 417, W_bench 192, W_cod 156 per drill. **Only 46 prospects have all six.**
- No in-season body mass anywhere in the data.

---

## Decisions needed before building features

### A. Body mass dominates every candidate feature set

On the 369 prospects with vertical, broad and 40 (scratch analysis, 2026-10-15). ARI is K-Means (k = 3–6) against pos_group. **Lower ARI means the clusters are less a restatement of position.**

| Feature set | \|r(PC1, mass)\| | ARI vs position |
|---|---|---|
| Mass alone (reference) | 1.00 | 0.36–0.51 |
| ROADMAP list: J/kg channels + TEE_z_abs + mass + Sayers + v_max, τ | **0.98** | **0.32–0.52** (same as mass alone) |
| Absolute J + J/kg together | 1.00 | 0.33–0.47 |
| J/kg + v_max, τ | 0.87 | 0.20–0.29 |
| Residuals on log mass + log mass as one feature | 0.00 | 0.16–0.26 |
| Residuals on log mass only | 0.00 | 0.02 |

- The ROADMAP feature list as written **clusters on mass**: it reproduces size classes, so mostly position groups.
- **Residualising** means regressing each log channel on log mass across *all* prospects (no positions) and keeping the residual: "how explosive for your size." Pooled log-log slopes: E_vert 0.60, E_broad 0.71, E_40 0.26, v_max −0.34, τ −0.05.
- **Recommendation:** residuals + log mass as one explicit feature. Size stays one honest axis (Heavy Engines can still appear) without driving every dimension, and Kinetic Anomalies (elite output for their mass) become directly visible. Residuals only would discard size entirely.
- **Redundancies to drop either way:** the E_40 and v_max residuals correlate at 1.00 (keep v_max, plus τ). Sayers power correlates with E_vert at 0.99, since it's linear in jump height and mass. It adds nothing to clustering, so keep it for reporting.

### B. Missing drills

GMM and K-Means need complete rows. Options:
1. **Core only (recommended):** vertical, broad, v_max, τ (+ mass): 369 prospects (DB 87, DL 82, OL 85, TE 30, WR 85). Bench and COD are used afterwards to describe archetypes, not to define them.
2. Core + bench: 146 prospects. Core + both COD drills: 89. All six: 46. These are too small for a 5+ cluster GMM with bootstrap stability.
3. Impute bench / COD: the missingness isn't random (who skips bench or agility drills is position- and draft-stock-dependent), so imputation would leak position back in.

---

## Next Session Steps

1. ~~Close Iteration 1~~ ✅ (2026-10-15)
2. **Decide A and B above** (user call)
3. `src/eeg/clustering/features.py` → `data/processed/features.parquet`: chosen features, no position columns (pos_group kept in a separate key table for the post-hoc cross-tab), standardized; tests
4. Correlation audit + optional PCA
5. K-Means baseline: k = 2..10; silhouette, gap statistic
6. GMM: BIC over k × covariance type; export membership probabilities
7. Bootstrap stability (ARI ≥ 0.7)
8. Name archetypes from centroids; cross-tab archetype × position group; describe archetypes with bench / COD where present
