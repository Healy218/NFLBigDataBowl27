# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 5, Submission
- **Roadmap Phase:** Phase 4d
- **Last updated:** 2026-10-22 (Iteration 4 closed; Iteration 5 not started)

---

## Current Objective

Turn the analysis into the **Kaggle notebook** for the NFL Big Data Bowl 2027 ("Combine to NFL Performance"; brief in [`ROADMAP.md`](ROADMAP.md)): a narrative a coach, scout or front office can use, the method, the figures, and specific scouting and development recommendations. First decide the frame (open decision 1), then harden the headline results before writing around them.

Done means a notebook that runs end-to-end from `data/raw/`, tells one focused story with the findings below, and a README that matches it.

**Rules:**
- **Position groups are never features.** They scope the study and are used only to read results; models stratify by group or centre within it.
- **Data:** only the Kaggle competition data. No external sources without asking.
- **No look-ahead:** predictors of later performance use Combine values or first-season η.
- **Committed reports hold aggregates only.** Named tables stay in `data/` (competition terms). `reports/figures/` is gitignored: figures are rebuilt by the pipeline.

---

## Where Iterations 1–4 left things

All four are closed. Reference notes for the write-up:
- Physics engine (drills → joules): [`reports/phase2_physics_notes.md`](reports/phase2_physics_notes.md)
- Kinetic profiles: [`reports/phase3_profiles_notes.md`](reports/phase3_profiles_notes.md)
- Game tracking, field channels and η: [`reports/phase4_eta_notes.md`](reports/phase4_eta_notes.md) (`top3` numbers; correction at the top)
- Outcomes, potential → performance, mismatches, tenure: [`reports/phase4c_outcomes_notes.md`](reports/phase4c_outcomes_notes.md)

```
uv run python -m eeg.ingest.load_combine     # → data/interim/combine_si.parquet (510 prospects)
uv run python -m eeg.ingest.load_tracking    # → data/interim/timed_drills.parquet
uv run python -m eeg.physics.sprint          # → data/interim/forty_fits.parquet  (417 with E_40)
uv run python -m eeg.physics.cod             # → data/interim/cod_reps.parquet
uv run python -m eeg.physics.tee             # → data/processed/tee.parquet
uv run python -m eeg.physics.sanity          # → outlier flags, reports/sensitivity.csv
uv run python -m eeg.clustering.features     # → data/processed/features.parquet (369)
uv run python -m eeg.clustering.audit        # → reports/feature_audit.md
uv run python -m eeg.clustering.kmeans       # → reports/kmeans_selection.csv
uv run python -m eeg.clustering.gmm          # → reports/gmm_report.md, data/processed/gmm_membership.parquet
uv run python -m eeg.clustering.profiles     # → data/processed/profiles.parquet, reports/profiles_report.md
uv run python -m eeg.clustering.readout      # → reports/phase3_readout.md
uv run python -m eeg.ingest.load_games       # → data/interim/game_tracking_{season}.parquet (REG only, ~5 s)
uv run python -m eeg.ngs.validate            # → reports/field_kinematics.md, data/processed/field_validation.parquet
uv run python -m eeg.ngs.channels            # → data/processed/field_{plays,games,seasons}.parquet, lab_channels.parquet
uv run python -m eeg.ngs.reliability         # → reports/field_reliability.md, reports/field_reliability.csv
uv run python -m eeg.ngs.eta                 # → data/processed/eta.parquet, eta_career.parquet, reports/eta.md
uv run python -m eeg.ngs.eta_readout         # → reports/eta_readout.md, data/processed/eta_leaderboard.parquet (named)
uv run python -m eeg.outcomes.table          # → data/processed/outcomes_seasons.parquet, outcomes.parquet, reports/outcomes.md
uv run python -m eeg.outcomes.production     # → data/processed/production_seasons.parquet, reports/production.md
uv run python -m eeg.physics.burst           # → data/processed/lab_burst.parquet, reports/lab_burst.md (needs production_seasons)
uv run python -m eeg.ngs.transfer            # → reports/transfer.md, reports/figures/transfer.png
uv run python -m eeg.outcomes.output         # → reports/output.md, reports/figures/output.png
uv run python -m eeg.outcomes.mismatch       # → reports/mismatch.md, data/processed/mismatch.parquet (named)
uv run python -m eeg.survival.tenure         # → reports/tenure.md, reports/figures/tenure.png
```

155 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml).

**The findings to write around** (details and numbers in the notes):
1. **Explosion potential is real on the field.** Combine values predict game top speed (CV R² 0.31) and burst (0.24) within group.
2. **The draft has priced the Combine on average.** Draft slot alone CV R² 0.47 for playing time; adding the Combine 0.46.
3. **The misses are in the explosive tail.** The top 20% by raw explosive energy (TEE_z_abs) within group × class beat their draft-slot playing time 65% of the time vs 50% (p 0.027); starts 57% vs 38% (p 0.003). A tail effect, not mass. They also exit less than their slot predicts (O/E 0.62; size-adjusted axis 0.36, p 0.06; not confirmed with 48 exits).
4. **DL, position-specific:** Combine start (a10 from the 10-yard split) → in-game get-off (r −0.44 within NFL position) → pressure rate (get-off β −0.47, a10 β 0.36) beyond the draft.
5. **η (exposure-free, reliability 0.75)** is mostly the inverse of engine size (r −0.26 with 40 speed, −0.36 with the explosive axis) and adds nothing to output beyond the draft. Median η_KE 0.57 (WR 0.70, DB 0.62, TE 0.62, DL 0.47, OL 0.34).

**Artifacts for figures and tables:** `tee.parquet`, `profiles.parquet` (engine), `lab_burst.parquet` (a10), `eta.parquet` / `eta_career.parquet` / `eta_leaderboard.parquet` (η; leaderboard named), `outcomes.parquet` / `outcomes_seasons.parquet`, `production_seasons.parquet`, `mismatch.parquet` (named: expected vs actual snaps, tail quantiles per measure). Report figures: `transfer.png`, `output.png`, `mismatch.png`, `tenure.png`, `eta.png`, plus the Phase 3 profile figures.

---

## Open decisions (decide when the data is in front of us)

1. **Narrative frame.** The project is named for η, but Iteration 4 moved the signal elsewhere. (a) Keep η central and report that it mostly measures engine size and doesn't predict output; (b) lead with explosion potential: physics of the Combine → the explosive tail the draft under-rates → the DL start chain, with η as a description of how much top speed reaches games by group. The evidence favours (b); it is the user's call, and it sets the title, README and figures.
2. **η beyond expectation** (deferred): game output relative to what the engine predicts, which removes the engine-size dependence. Only worth building if η stays central.
3. **Brief emphasis on 10 Hz tracking.** The raw-joules composite mixes the tracked 40 (E_40) with results (jumps, bench), and a10 is the official split. Measure how much of the tail signal comes from the tracking before choosing what to headline.

---

## Next Session Steps

1. **Decide the frame** (open decision 1), with the user.
2. **Harden the headline results:** tail cut-offs (10 / 20 / 30%), an alternative expectation model, each class separately; which TEE channels carry the tail (open decision 3); the DL chain by class. Write the checks to a report.
3. **Final figures** for the chosen frame.
4. **Notebook narrative**, then the README executive summary to match.
5. **Reproducibility pass:** clean run end-to-end from `data/raw/`.
6. Optional: Combine position skill drills.
