# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 3, NGS Alignment & Explosion Efficiency (η)
- **Roadmap Phase:** Phase 4a–4b, On-field motor and Explosion Efficiency
- **Last updated:** 2026-10-22 (steps 1–6 done: ingest, validation, field channels, reliability and exposure, η, readout)

---

## Current Objective

Measure how much of each prospect's **lab engine shows up on the field**:

```
data/raw/.../game_tracking_{2023,2024,2025}.csv ──► field kinematics (REG + POST) ──► per player-season field channels
data/interim/forty_fits.parquet (v_max, τ) ───────────────────────────────────────► lab channels ──┴──► η = field / lab
```

Done means a per player-season table of field channels and η on matching channels, with the exposure needed for a stable estimate, and η leaderboards per position group ([`ROADMAP.md`](ROADMAP.md) Phase 4a–4b).

**Rules:**
- **Position groups are never features.** They scope the study and are used only to read the results.
- **Data:** only the Kaggle competition data. No external sources without asking.

---

## Where Phases 2 and 3 left things

Both iterations are closed. Reference notes for the write-up:
- Physics engine (drills → Joules, every assumption and rejected alternative): [`reports/phase2_physics_notes.md`](reports/phase2_physics_notes.md)
- Kinetic profiles (decisions A–D, clustering tests, profiles, readout): [`reports/phase3_profiles_notes.md`](reports/phase3_profiles_notes.md)

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
uv run python -m eeg.ingest.load_games       # → data/interim/game_tracking_{season}.parquet (REG + POST, ~5 s)
uv run python -m eeg.ngs.validate            # → reports/field_kinematics.md, data/processed/field_validation.parquet
uv run python -m eeg.ngs.channels            # → data/processed/field_{plays,games,seasons}.parquet, lab_channels.parquet
uv run python -m eeg.ngs.reliability         # → reports/field_reliability.md, reports/field_reliability.csv
uv run python -m eeg.ngs.eta                 # → data/processed/eta.parquet, eta_career.parquet, reports/eta.md
uv run python -m eeg.ngs.eta_readout         # → reports/eta_readout.md, data/processed/eta_leaderboard.parquet (named)
```

128 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml).

**What carries into Phase 4:**
- **Lab channels** (`forty_fits.parquet`, best rep; `combine_si.parquet` mass): v_max (rep-to-rep r 0.97), τ (r 0.51, noisy), E_40. Derived from the 40 model $v(t) = v_{max}(1 - e^{-t/τ})$: initial acceleration $a_0 = v_{max}/τ$ and peak propulsive power per kg $v_{max}^2 / 4τ$ (before drag).
- **Kinetic profiles** (`profiles.parquet`, 369): `ax_explosive`, `ax_speed_vs_bounce`, `ax_up_vs_out`, `anomaly_d2` / `anomaly_p`, `size_class`. These are survival covariates in Iteration 4, and η can be read against them.
- **Mass:** Combine mass only. There is no in-season weight, so in a ratio of energies or of power per kg **mass cancels**: η on kinetic energy is (v_field / v_lab)².

---

## Game tracking facts (quick scan, 2026-10-20)

From the raw files and [`config/data_dictionary.md`](config/data_dictionary.md):
- 10 Hz, **prospects only** (not all 22 players). Units are yards: `s` in yd/s, `a` in yd/s². `dis` is unverified; re-derive distance and speed from x/y, as for the Combine.
- Seasons by draft class: the 2023 class has 2023–25, the 2024 class 2024–25, the 2025 class 2025 only.

| Season | REG prospects | REG frames | Median REG plays per prospect | Frames with s > 12 yd/s (REG) |
|---|---|---|---|---|
| 2023 | 122 | 2.8 M | 207 | 2 |
| 2024 | 247 | 6.2 M | 239 | 3 |
| 2025 | 353 | 9.1 M | 266 | 87 |

- PRE adds 0.7–1.6 M frames per season but only ~45–65 plays per prospect. POST adds 31–115 prospects.
- Frame-level p99.9 of `s` is ~8.3–8.5 yd/s (7.6–7.8 m/s) in REG, so the glitches (up to 16.9 yd/s) are rare but large, and concentrated in 2025.
- **Ingest (step 1, `src/eeg/ingest/load_games.py`):** a track is one prospect on one play, ~1 s before the snap to the end of the play (median 69 frames); 99.8% have a `ball_snap`, so `t_snap_s` is available for snap-relative windows. Reported `s` agrees with central x/y differences (0.98–1.01 above 9 yd/s). Acceleration is `a_s_mps2` (see step 2 findings).
- **Glitches:** 457 of 264 k tracks (0.17%) have an impossible frame: mostly x/y teleports at the snap, plus a few broken tracks where reported speed jumps (0 → 7.5 m/s in a frame, or to 13 yd/s) while `a` stays 0. **Rule: drop whole `glitch_track` tracks.** `flag_mismatch` (reported vs x/y speed off by > 1 m/s) is a quality flag only: it fires mostly at low-speed contact.
- **Coverage of the 369 profiled prospects:** 292 appear in a REG game, and **219** have a season with ≥ 100 REG plays. The 77 with no REG snaps get no η, but they matter for Iteration 4: never playing is the earliest form of league exit.


## Validation findings (step 2, [`reports/field_kinematics.md`](reports/field_kinematics.md))

- **Speed: use reported `s_mps`, on both sides.** x/y speed agrees to 0.3–0.4% above 6 m/s, with no lag. The lab v_max was fit to the same reported speed.
- **Acceleration: NGS `a` is total magnitude** (turning included, never negative): r 0.88 with |a| from x/y, 0.61 with |dv/dt|. So both sides use `a_s` = Savitzky–Golay d(reported speed)/dt, window 5 (`games.accel_window_frames`), applied to the Combine 40 too (`eeg.ngs.kinematics`). Compare field with the **lab peak of the same estimator**, not the model's a0 = v_max/τ: smoothed traces never reach a0 (lab peak = 0.65 × a0, r 0.92).
- **Top speed (426 player-seasons with ≥ 100 plays, 252 prospects):** field season peak (held 0.3 s) is a median **0.86 × the lab 40 peak** (p10–p90 0.74–0.93); only 0.2% beat the lab. r 0.90 pooled, 0.34–0.79 within position; season-to-season r **0.90**; no exposure trend above 100 plays. η_speed is the clean channel, and η_KE = 0.86² ≈ 0.73.
- **Role gap is real** (open decision 3): median speed ratio OL 0.78, DL 0.81, TE 0.86, WR 0.89, DB 0.90.
- **Peak acceleration:** field season peak is *above* the lab 40 peak for 75% (median 1.11): game bursts beat the 40 start. Only r 0.45 with the lab pooled and ~0–0.4 within position; season-to-season r 0.59, and it was 0.30 before 5 broken tracks were removed, so season maxima are fragile. 55% of peaks fall in the first 1.5 s after the snap, 29% are re-accelerations out of a cut. Step 4 needs a robust statistic here; window 9 would lower field peaks more than lab ones (ratio 1.03) and nudge field–lab ρ to 0.57.


## Field channels (step 3, [`reports/field_channels.md`](reports/field_channels.md))

- `src/eeg/ngs/channels.py`: per play (clean tracks) sustained peaks of speed, $E_k/m = ½v^2$, acceleration, **burst** (a_s in the first 1.5 s after the snap, plays starting below 1 m/s: 92% of plays) and **propulsive power per kg** $(a_s + k v^2) v$, with $k = ½ρC_dA/m$ from Combine height and mass, as in E_40. Then max per game, then per player-season three candidates per channel: `_max` (season peak), `_game_med`, `_top3`.
- Lab side (`lab_channels.parquet`, 417): the same estimators on the best 40 rep. The 40 starts from rest, so its acceleration peak is the burst analogue.
- 510 player-seasons with ≥ 100 plays (302 prospects); median 14–16 games. Season-peak medians: speed 8.3 m/s, burst 5.5 m/s², power 20 W/kg (WR 24, OL 13).
- First look against the lab (season peaks): speed 0.86 (r 0.90), $E_k$ 0.73, power **0.83 (r 0.65, ρ 0.75)**, burst 1.04 (r 0.39). Power, mostly a speed channel at the top, is the second usable one; burst is the weak one.


## Reliability and exposure (step 4, [`reports/field_reliability.md`](reports/field_reliability.md))

- **Season value = `top3`** (mean of the 3 best game maxima; `games.season_statistic`), for every channel: best or tied within-position η reliability. The single season peak is the worst (speed η 0.53 vs 0.69).
- **Exposure:** games count with ≥ 5 plays (drops 11%, mostly special-teams cameos, whose maxima are jogs); a player-season needs ≥ 100 plays and ≥ 6 counting games → `qualified` in `field_seasons.parquet`: **481 player-seasons, 288 prospects**. Over 6 games, reliability within position is 0.65 for speed and 0.73–0.84 for burst and power.
- Within position (pooled figures are inflated by role):

| Channel | Field rel. | Lab rel. (1 rep) | Field–lab r | η rel. | SD of log η |
|---|---|---|---|---|---|
| speed | 0.81 | 0.95 | 0.59 | **0.69** | 0.052 |
| burst | 0.91 | 0.45 | 0.12 | 0.67 | 0.150 |
| power | 0.85 | 0.69 | 0.32 | 0.65 | 0.150 |

- **Reading:** speed η is a genuine ratio: the lab is precise and predicts the field. Burst is the most reliable *field* channel, but the lab 40 acceleration is noisy and barely related to it, so burst η is mostly game burst over a noisy constant. Power is in between. Single-game speed is weak (0.23): top speed needs an opportunity.

---

## η and the Explosion Efficiency Gap (step 5, [`reports/eta.md`](reports/eta.md))

- **Coverage:** 398 of 481 qualified player-seasons have a well-fit 40 (50 prospects have none); **238 prospects**, 212 of them profiled. Qualified seasons per prospect: 1 (120), 2 (76), 3 (42).
- **Headline:** gap = 1 − η_KE, η_KE = E_field / E_lab = η_speed² (`eta.headline`). Median η_KE 0.70 (p10–p90 0.51–0.82), i.e. η_speed 0.83. By group: WR 0.77, DB 0.78, TE 0.69, DL 0.61, OL 0.54.
- **Why speed:** within position, log η_speed is unrelated to the lab value (r −0.03) and tracks the field value (r 0.79), so it is game output against a fixed engine. Burst and power η are mostly minus the lab value (r −0.61, −0.65; noisy 40 acceleration), so they're stored and reported but are not the gap. Within position, η_burst and η_power correlate 0.78 and only 0.19–0.29 with η_speed.
- **Role:** position explains 68% of the variance of log η_speed (10% for burst, 32% for power). η stays raw (`eta.role`); leaderboards are read within position and **Iteration 4 stratifies Cox models by position group**.
- **Season to season** (157 consecutive pairs, within position): η_speed r 0.42 against a within-season reliability of 0.69, so speed η changes between seasons (role, development, injury). Burst 0.62 and power 0.74 look steadier only because the shared lab value repeats.
- **Per prospect:** `eta_*_first` (first qualified season) and `eta_*_career` (plays-weighted geometric mean). The career value uses later seasons, so for survival it is look-ahead; prefer the first-season value or a time-varying season value.

---

## Reading η (step 6, [`reports/eta_readout.md`](reports/eta_readout.md))

- **Shrinkage** (empirical Bayes on log η_KE, toward the group mean): season-to-season SD 0.080 against a true within-group spread of 0.068, so one season is a weak read. B = 0.42 for one season, 0.59 for two, 0.68 for three. Shrunk η_KE spans about ±0.03–0.05 around each group mean (p10–p90). Credibly off the group mean (90%): WR 0 / 0, DB 3 / 1, TE 1 / 0, DL 8 / 4, OL 6 / 6 (above / below).
- **Sub-roles** inside a group barely differ in median η_KE (DL: OLB 0.62, NT 0.63, DE 0.60, DT 0.59; DB: CB 0.77, FS and SS 0.76), so the position group is enough role control. OLBs do fill the top of the DL board.
- **η is nearly independent of the engine** (within group): r −0.05 with 40 speed, −0.15 with `ax_explosive`, −0.03 with the Kinetic Anomaly, −0.07 with TEE_z_rel. A slight lean: bigger engines transfer a little less (−0.22 for `ax_explosive` within listed position). In within-group terciles (24 per cell if independent), low engine → high η holds 30 and low engine → low η 15; high engine → high η 18. So η adds information the Combine doesn't hold.
- **η goes with college production** (NGS college production score r +0.21, CI 0.09–0.33; +0.20 within listed position) and slightly against the NGS athleticism score (−0.14). The mass link (−0.16) is sub-role: −0.05 within listed position.
- **Named leaderboards** print to the terminal and live in `data/processed/eta_leaderboard.parquet` (never committed), with "Combine warrior" (high engine, low η) and opposite-corner tags.

---

## Open decisions (decide when the data is in front of us)

1. ~~**Season types**~~ **Decided 2026-10-21:** REG + POST, PRE dropped (preseason snaps are few, and effort and rosters aren't comparable). `games.season_types` in `config/physics.yaml`.
2. ~~**Field statistic**~~ **Decided 2026-10-21:** `top3`, the mean of the 3 best game maxima, with the exposure rules above (`games.*` in `config/physics.yaml`).
3. ~~**Role dependence of field output**~~ **Decided 2026-10-22:** η stays raw; leaderboards within position group; Iteration 4 stratifies Cox models by position group, so η acts only within groups and position is never a covariate (`eta.role`). The headline gap is 1 − η_KE (`eta.headline`).

---

## Next Session Steps

1. ~~**Ingest game tracking**~~ Done (see facts above).
2. ~~**Validate field kinematics**~~ Done (see findings above).
3. ~~**Field channels**~~ Done (see above). Change of direction deferred (lab COD covers only 126–128 profiled prospects).
4. ~~**Exposure and reliability**~~ Done (see above).
5. ~~**η per matching channel**~~ Done (see above).
6. ~~**Read the results**~~ Done (see above).
7. Close Iteration 3 → Iteration 4 (survival).
