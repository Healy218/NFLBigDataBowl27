# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 4, Explosion Potential → Regular-Season Performance
- **Roadmap Phase:** Phase 4c
- **Last updated:** 2026-10-22 (Iteration 4 step 1 done: outcome table)

---

## Current Objective

The strategy ([`ROADMAP.md`](ROADMAP.md), "Our strategy"): a **physics model of each prospect's explosion potential** (Combine → joules and watts), **η** for how much of it goes off in games, and now the last link: **does potential, and its realization, predict regular-season performance beyond draft slot?**

```
engine (tee, profiles, lab channels) ─┐
η per season (eta.parquet) ───────────┼──► within-group models, draft slot as control ──► regular-season output
college production (context) ────────┘                                                    (playing time, position production; tenure secondary)
```

Done means an outcome table per player-season (regular season, competition data only), sharper lab burst, and within-group answers to the ROADMAP 4c questions: what predicts transfer (η), whether potential and η predict output beyond draft slot, and which potential × η combinations are under- or over-drafted.

**Rules:**
- **Position groups are never features.** They scope the study and are used only to read results; models stratify by group.
- **Data:** only the Kaggle competition data. No external sources without asking.
- **No look-ahead:** predictors of later performance use first-season or per-season η, never the career value.
- **Committed reports hold aggregates only.** Named tables stay in `data/` (competition terms).

---

## Where Iterations 1–3 left things

All three are closed. Reference notes for the write-up:
- Physics engine (drills → joules): [`reports/phase2_physics_notes.md`](reports/phase2_physics_notes.md)
- Kinetic profiles: [`reports/phase3_profiles_notes.md`](reports/phase3_profiles_notes.md)
- Game tracking, field channels and η: [`reports/phase4_eta_notes.md`](reports/phase4_eta_notes.md)

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
```

131 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml) (`games` and `eta` for Iteration 3, `outcomes`).

**What carries into Iteration 4:**
- **Engine (potential):** `tee.parquet` (J and J/kg per drill, TEE composites; 510), `profiles.parquet` (`ax_explosive`, `ax_speed_vs_bounce`, `ax_up_vs_out`, `anomaly_d2`, `size_class`; 369), `forty_fits.parquet` (v_max rep-to-rep r 0.97; τ r 0.51), `lab_channels.parquet` (40 peaks with the field estimators; 417).
- **Realization:** `eta.parquet` (394 player-seasons, 237 prospects; `eta_ke`, `gap` = 1 − η_KE, plus speed, burst, power), `eta_career.parquet` (`eta_*_first`, `eta_*_career`), `eta_leaderboard.parquet` (shrunk η_KE, engine × efficiency terciles), `field_seasons.parquet` (field output per channel; 476 qualified player-seasons).
- **Key facts:** η_KE median 0.69 (WR 0.77, DB 0.78, TE 0.69, DL 0.61, OL 0.54). Within group, η is nearly independent of the engine (r −0.08 with 40 speed, −0.17 with `ax_explosive`) and goes with NGS college production (+0.23). Speed η changes between seasons (retest 0.38 against within-season reliability 0.67), so one season is a weak read (shrinkage B 0.44).
- **Coverage gaps:** 50 prospects with qualified seasons have no well-fit 40 (no η); 77 of the 369 profiled never play a regular-season snap in tracking, which is itself an outcome (the earliest exit).

---

## Outcome data (step 1, [`reports/outcomes.md`](reports/outcomes.md))

`eeg.outcomes.table` → `outcomes_seasons.parquet` (every league season of every prospect, zeros kept: 1,007, 722 with a play) and `outcomes.parquet` (per prospect: draft, career totals, exits). Schemas in [`config/data_dictionary.md`](config/data_dictionary.md).
- **What a play is:** `player_play` holds every regular-season scrimmage snap for OL, DL and DB (matches career snaps exactly), but **only route plays for WR (58% of snaps) and TE (40%)**; run-blocking snaps are missing from player_play and game tracking alike. Playing time is compared within group only. Career snaps are regular season only.
- **Draft drives playing time:** within group and draft class, ρ of snaps per season with overall pick −0.57 (WR) to −0.76 (OL). Median snaps per season fall from 530–940 (round 1) to 0–14 (undrafted). 61% of undrafted never played a snap, against 0–3% of rounds 1–5 and 21% of rounds 6–7. Any test of potential or η needs the draft control, and the undrafted sit at the floor.
- **Exits** (played, then no play through 2025; right-censored): 48 (26 of the 2023 class, 22 of 2024). By round, among those who could exit: round 1 0 / 28, 2–3 7 / 78, 4–5 14 / 70, 6–7 24 / 55, UDFA 3 / 29. Only 6 prospects have a gap season.
- **Active share by year in the league:** ~0.72–0.76 as rookies, 0.66 in year 3 (2023 class).
- Only prospects are tracked, so snap **share** can't be computed; playing time is plays and games (17 regular-season games per team).

---

## Open decisions (decide when the data is in front of us)

1. **Primary outcome per group:** playing time (plays per season), position production from `player_play`, or games started. Pick on coverage and reliability (split-half, as for the field channels), and keep one per group that a coach would recognize.
2. **Draft control:** overall pick with an undrafted indicator, or round bands. Step 1 shows it is strong and monotone within group (ρ −0.57 to −0.76), with the undrafted at the floor; decide the form in step 5.
3. **Tenure:** 48 exits, concentrated in rounds 4–7 (none in round 1). Enough for a Kaplan–Meier view and a small stratified model, not much more; keep it secondary.

---

## Next Session Steps

1. ~~**Outcome table**~~ Done: `eeg.outcomes.table`, see "Outcome data" above.
2. **Position production** from `player_play` per player-season: candidates per group (WR/TE separation and YAC over expected, DL get-off and time to pressure, OL pressure allowed rate, DB coverage and tackles); coverage and split-half reliability. Decide open decision 1.
3. **Sharper lab burst:** τ, a0 and the 10-yard split from all well-fit 40 reps; reliability against the best-rep values.
4. **What predicts transfer:** η (first-season, per-season) against the explosion profile, the 40 shape and agility traces, within group.
5. **Potential and η → regular-season output**, within group, with the draft control (open decision 2); NGS college production as context.
6. **Under- / over-drafted:** potential × η against draft slot.
7. **Tenure** (secondary, open decision 3), then close Iteration 4 → Iteration 5 (submission).
