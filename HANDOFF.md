# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 4, Explosion Potential → Regular-Season Performance
- **Roadmap Phase:** Phase 4c
- **Last updated:** 2026-10-22 (Iteration 4 steps 1–6 done: outcome table, position production, lab burst, what predicts transfer, potential and η → output, under- / over-drafted; η's season value made exposure-free)

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
uv run python -m eeg.outcomes.production     # → data/processed/production_seasons.parquet, reports/production.md
uv run python -m eeg.physics.burst           # → data/processed/lab_burst.parquet, reports/lab_burst.md (needs production_seasons)
uv run python -m eeg.ngs.transfer            # → reports/transfer.md, reports/figures/transfer.png
uv run python -m eeg.outcomes.output         # → reports/output.md, reports/figures/output.png
uv run python -m eeg.outcomes.mismatch       # → reports/mismatch.md, data/processed/mismatch.parquet (named)
```

152 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml) (`games` and `eta` for Iteration 3, `outcomes` and `lab_burst` for Iteration 4).

**What carries into Iteration 4:**
- **Engine (potential):** `tee.parquet` (J and J/kg per drill, TEE composites; 510), `profiles.parquet` (`ax_explosive`, `ax_speed_vs_bounce`, `ax_up_vs_out`, `anomaly_d2`, `size_class`; 369), `forty_fits.parquet` (v_max rep-to-rep r 0.97; τ r 0.51), `lab_channels.parquet` (40 peaks with the field estimators; 417).
- **Realization:** `eta.parquet` (394 player-seasons, 237 prospects; `eta_ke`, `gap` = 1 − η_KE, plus speed, burst, power), `eta_career.parquet` (`eta_*_first`, `eta_*_career`), `eta_leaderboard.parquet` (shrunk η_KE, engine × efficiency terciles), `field_seasons.parquet` (field output per channel; 476 qualified player-seasons).
- **Key facts (exposure-free season value, see "η season value" below):** η_KE median 0.57 (WR 0.70, DB 0.62, TE 0.62, DL 0.47, OL 0.34). Within group, η falls with the engine (r −0.26 with 40 speed, −0.36 with `ax_explosive`, −0.35 with NGS athleticism) and is unrelated to NGS college production (−0.05). Speed η reliability 0.75, season retest 0.68; shrinkage B 0.64 for one season. The Iteration 3 notes (`phase4_eta_notes.md`) carry the old `top3` numbers with a correction at the top.
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

## Position production (step 2, [`reports/production.md`](reports/production.md))

`eeg.outcomes.production` → `production_seasons.parquet`: 12 measures per player-season (ratios of season sums), checked on 489 qualified player-seasons (293 prospects) for within-group split-half reliability, season-to-season retest and validity (r with plays per game, ρ with draft pick).
- **Reliable and valid:** YPRR (WR rel 0.85, TE 0.80; r with plays per game 0.31 / 0.41), WR target rate (0.87), DL pressure rate (0.67; r 0.46 with plays per game, earlier picks higher), OL peak pressure probability allowed (0.77, against 0.35 for the binary pressure-allowed rate).
- **Noise per season:** sacks, sacks allowed, tackles for loss (0.2–0.4), WR YAC over expected (0.20).
- **Role, not quality:** WR separation (rel 0.75 but more separation goes with *fewer* plays and later picks: slot and short routes); DB tackle rate (0.66, unrelated to playing time). player_play has no coverage outcome for the defender, so DB has no credible production measure.
- **DL get-off** (s from snap): rel 0.89, retest 0.84, faster with earlier picks (ρ 0.19) and with more plays (r −0.21). It is in-game explosion, so it is used alongside η.
- **Plays per game** when active: rel 0.91–0.97 within a season but retest only 0.39–0.56 (roles change between seasons).

---

## Lab burst (step 3, [`reports/lab_burst.md`](reports/lab_burst.md))

`eeg.physics.burst` → `lab_burst.parquet` (425 prospects). Decided (`lab_burst.measure`): **the Combine burst input is a10 = 2 d10 / t10²** from the official 10-yard split (mean acceleration over the first 10 yd from rest, m/s²).
- **Tracked starts stay noisy.** Averaging the two well-fit reps lifts reliability from 0.45–0.58 to 0.62–0.73 (τ, a0, sustained a_s peak, tracked 10-yard time), and the averaged a_s peak tracks field burst better (r 0.23 vs 0.16), but none beats the official split.
- **The official split shows up in games** (within group): r 0.36 with field burst, **−0.67 with DL get-off** (n 59). For get-off it works beyond top speed (β 0.63, v_max −0.05): the Combine start from a stance predicts the in-game start from a stance. For general field burst the split and the rest of the 40 count about equally (β −0.22 / −0.24).
- **Profiles stay as built:** τ alone is ~0 against games, but with v_max fixed it carries the start (β 0.31 for get-off, −0.20 for field burst), and the clustering uses both.
- **η_burst keeps its same-estimator lab peak;** a10 is a predictor, not an η denominator. Idealized power over 10 yd in watts (`p10_w`) is weaker (r 0.13 / −0.33): mass works against get-off.

---

## η season value: exposure-free (found in step 5, [`reports/field_reliability.md`](reports/field_reliability.md))

The season value of every field channel was `top3`, the mean of a player's 3 best **game maxima**. More games and snaps give more chances at a fast one, so it grew with exposure: η had within-group r **+0.46** with season plays, and "η predicts playing time" (r +0.22 with next season's plays) was that artifact. It is now **`top3_of50`** (`games.season_statistic`, `games.season_sample_plays`): the expected mean of the 3 best plays in 50 plays drawn at random from the season, computed exactly over all 50-play subsets (`eeg.ngs.channels.expected_top`), the same functional for every player whatever their exposure.
- **No exposure bias** (speed r −0.06 with plays) and **more reliable** for speed: split-half 0.84 vs 0.80, η 0.75 vs 0.67, season retest 0.77 vs 0.61. For burst, η split-half a little lower (0.61 vs 0.67), retest higher.
- Reran channels, reliability, η, readout, lab burst and transfer. Conclusions that changed: η is not independent of the engine (bigger engines reach a smaller fraction of themselves in games); **the college production link was the artifact** (as was college production's link to game speed, now r 0.06).

---

## What predicts transfer (step 4, [`reports/transfer.md`](reports/transfer.md); numbers with the exposure-free season value)

`eeg.ngs.transfer`: Combine predictors against η_KE, game output and DL get-off, per prospect and within group (univariate r with CIs; compact model with cross-validated R² beyond the group mean). Predictors are all Combine values, so career η is not look-ahead here.
- **The Combine predicts game output.** Compact model (profile axes, a10, mass, college production): game top speed CV R² 0.31, game burst 0.24 (a10 r 0.43 / 0.45, v_max 0.44 / 0.43, mass −0.58 / −0.51).
- **It predicts η only negatively:** CV R² 0.14 (career), 0.10 (first season), through the explosive axis (r −0.38), NGS athleticism (−0.36) and v_max (−0.30). Game speeds compress: the bigger the Combine engine, the smaller the fraction reached in games. Nothing in the Combine or college data predicts *higher* η.
- **DL get-off is predictable** (unchanged; it isn't a season-value channel): CV R² 0.43 within DL, 0.16 within NFL position; a10 r −0.44 (CI −0.63 to −0.20) within position, mass +0.44, college production −0.35. a10 and mass can't be separated at n 53.
- **Lab burst** (step 3) with the new season value: a10 r 0.45 with field burst (was 0.36); the decision stands.

---

## Potential and η → regular-season output (step 5, [`reports/output.md`](reports/output.md))

`eeg.outcomes.output`: within group (centred on group × draft class or group × season), draft slot as control. Figure `reports/figures/output.png`.
- **Draft control (open decision 2 settled, `outcomes.draft_control`):** overall pick (undrafted = 258) + undrafted indicator. CV R² for playing time 0.51, against round bands 0.50 and log pick 0.47. Within-class r of pick with playing time −0.62 (DL) to −0.81 (OL).
- **The draft has priced the Combine.** On 368 prospects: Combine alone CV R² 0.23 (college production β 0.38, explosive axis 0.27), draft alone 0.47, draft + Combine 0.46. Beyond the draft no Combine beta's CI excludes 0.
- **η adds nothing to next season's playing time.** First qualified η season s → plays in s + 1 (151 prospects): draft + plays in s CV R² 0.21; adding η 0.20 (β 0.03, CI −0.12 to 0.18); engine and η together, both ~0. η alone r −0.10.
- **η and production (same seasons, per prospect, beyond draft and plays):** nothing for WR YPRR (r 0.00), OL (−0.09), DL pressure (−0.01); TE n 14.
- **The DL chain is the result.** Per DL prospect, pressure rate beyond the draft and plays, centred on NFL position (role check): **get-off β −0.47** (CI −0.67 to −0.27, n 75), **a10 β 0.36** (0.13 to 0.60, n 59), and a10 drops to 0.20 once get-off is in the model (get-off −0.38). The Combine start → the game start → pressure, beyond draft capital. Centred on DL only, all larger (a10 0.55, get-off −0.63).

---

## Under- / over-drafted (step 6, [`reports/mismatch.md`](reports/mismatch.md))

`eeg.outcomes.mismatch` → `mismatch.parquet` (named, `data/` only). Step 5's linear "nothing beyond the draft" averages over every prospect; mismatches live in the tail. So: snaps (and starts) per available season against a **draft-slot expectation** (per group, pick + undrafted with a class intercept, leave-one-out, clipped to the group's range; the group absorbs "linemen play every snap" and "skill players debut later"), and the **share of a measure's top 20% (within group × class) that beat it**. Permutation p shuffles the measure within group × class. Figure `reports/figures/mismatch.png`.
- **Raw joules is the mismatch signal.** Its top 20% beat their slot **65%** of the time (CI 54–74%, n 79) against 50% for the rest (p 0.027); for games started 57% vs 38% (p 0.003). A **tail effect**: quintiles 2–5 sit at 48–54%.
- **Not mass:** the heaviest 20% beat their slot 54% vs 51% (p 0.65). Post hoc, the raw-joule tail who are *not* the heaviest beat it 75% (n 44): energy, not size.
- **Weaker elsewhere:** size-adjusted axis 59% vs 52% (p 0.27; starts 53% vs 41%, p 0.07), joules per kg 48%, a10 56%.
- **By group (small tails):** DL raw joules 16 of 18 (p 0.013), OL axis 11 of 15 (p 0.024); WR, TE and DB nothing credible. Ten tests at the top level and fifteen by group, so treat single p near 0.05 as weak; the raw-joules result holds on both outcomes.
- **Mean residuals are noisy** (log snaps, 0 vs 1,000): `p_mean` 0.22 for raw joules on snaps; the share is the robust statistic.
- η is not used: it exists only for players who already play, and step 5 found it adds nothing.

---

## Open decisions (decide when the data is in front of us)

1. ~~**Primary outcome per group**~~ Decided 2026-10-22 (`outcomes` in `config/physics.yaml`): primary = **playing time** for every group (all prospects, zeros kept, within group). Secondary production: WR and TE YPRR, DL pressure rate, OL peak pressure probability allowed; DB none. DL get-off is a realization channel next to η, not an outcome.
2. ~~**Draft control**~~ Decided 2026-10-22 (`outcomes.draft_control`): overall pick + undrafted indicator (CV R² 0.51 for playing time, best of four forms).
3. **Tenure:** 48 exits, concentrated in rounds 4–7 (none in round 1). Enough for a Kaplan–Meier view and a small stratified model, not much more; keep it secondary.

---

## Next Session Steps

1. ~~**Outcome table**~~ Done: `eeg.outcomes.table`, see "Outcome data" above.
2. ~~**Position production**~~ Done: `eeg.outcomes.production`, see "Position production" above.
3. ~~**Sharper lab burst**~~ Done: `eeg.physics.burst`, see "Lab burst" above.
4. ~~**What predicts transfer**~~ Done: `eeg.ngs.transfer`, see "What predicts transfer" above.
5. ~~**Potential and η → regular-season output**~~ Done: `eeg.outcomes.output`, see step 5 above (and the η season value fix).
6. ~~**Under- / over-drafted**~~ Done: `eeg.outcomes.mismatch`, see step 6 above.
7. **Tenure** (secondary, open decision 3), then close Iteration 4 → Iteration 5 (submission).
