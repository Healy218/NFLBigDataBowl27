# Phase 4a–4b notes: game tracking, field channels and Explosion Efficiency η

Moved from `HANDOFF.md` when Iteration 3 closed (2026-10-22). This is the reference for how game tracking was turned into field channels, how η was defined and why, and what it shows. Use it for the methods and results sections of the write-up. All figures are **regular season only** (see decision 1).

## Pipeline

```
uv run python -m eeg.ingest.load_games       # → data/interim/game_tracking_{season}.parquet (REG, SI, flags; ~5 s)
uv run python -m eeg.ngs.validate            # → reports/field_kinematics.md, data/processed/field_validation.parquet
uv run python -m eeg.ngs.channels            # → data/processed/field_{plays,games,seasons}.parquet, lab_channels.parquet
uv run python -m eeg.ngs.reliability         # → reports/field_reliability.md, reports/field_reliability.csv
uv run python -m eeg.ngs.eta                 # → data/processed/eta.parquet, eta_career.parquet, reports/eta.md
uv run python -m eeg.ngs.eta_readout         # → reports/eta_readout.md, data/processed/eta_leaderboard.parquet (named)
```

Settings: `games` and `eta` in `config/physics.yaml`. Schemas: `config/data_dictionary.md`.

**What Iteration 4 uses:** `eta.parquet` (per player-season: field, lab and η per channel, `gap`), `eta_career.parquet` (per prospect: `eta_*_first`, `eta_*_career`), `eta_leaderboard.parquet` (shrunk η_KE, engine × efficiency terciles), and `field_seasons.parquet` (field output per channel, for game-output outcomes).

## Decisions

1. **Season types: regular season only** (2026-10-22). Preseason was dropped first (~45–65 plays per prospect per season; effort and rosters not comparable). Playoffs were dropped to match the brief ("performance on the field during the NFL regular season"); they were 4.6% of player-games and moved no headline number by more than ~0.01.
2. **Speed: reported `s`, on both sides.** x/y central-difference speed agrees to 0.3–0.4% above 6 m/s with no lag, and the lab v_max was fit to the same reported speed.
3. **Acceleration: `a_s`, Savitzky–Golay d(reported speed)/dt** (window 5, both sides). NGS `a` is total acceleration magnitude (r 0.87 with |a| from x/y, 0.61 with |dv/dt|) and never negative, so it can't separate propulsion from braking or cutting. Field acceleration is compared with the lab peak of the same estimator, not the model's a0 = v_max/τ (smoothed traces reach only 0.65 × a0).
4. **Glitches: drop whole tracks** with an impossible frame (time gap, jump > 11 m/s, speed > 11 m/s, |a| or |a_s| > 12 m/s²): 450 of 251 k tracks (0.18%). A speed mismatch with x/y is a quality flag only (it fires at low-speed contact).
5. **Peaks are sustained** for 3 frames (0.3 s, rolling minimum). **Burst** = a_s in the first 1.5 s after the snap, on plays starting below 1 m/s (92% of plays): the 40's start-from-rest analogue. **Power per kg** = (a_s + k v²) v, k = ½ρC_dA/m from Combine height and mass, as for E_40.
6. **Season value = `top3`**, the mean of a player's 3 best game maxima, for every channel: best or tied within-position η reliability; the single season peak is the worst (speed η 0.50 against 0.67).
7. **Exposure:** a game counts with ≥ 5 plays (removes 11% of games, mostly special-teams cameos whose maxima are jogs); a player-season qualifies with ≥ 100 plays and ≥ 6 counting games (476 player-seasons, 287 prospects).
8. **Headline: Explosion Efficiency Gap = 1 − η_KE** (2026-10-22), η_KE = E_field / E_lab on kinetic energy per kg = η_speed² (mass cancels). Burst and power η are stored and reported, not part of the gap.
9. **Role: η stays raw** (2026-10-22). Leaderboards are read within position group, and models that use η stratify by position group, so position never enters as a covariate.
10. **Per prospect:** first qualified season η and a career η (plays-weighted geometric mean). The career value uses later seasons, which is look-ahead for anything that predicts the future; prefer the first-season or per-season value there.

## Results

### Validation (step 2, `reports/field_kinematics.md`)

- **Top speed** (419 player-seasons with ≥ 100 plays and a usable 40, 248 prospects): field season peak is a median **0.86 × the lab 40 peak** (p10–p90 0.74–0.93); 0.5% beat the lab. r 0.89 pooled, 0.31–0.72 within position; season-to-season r 0.89–0.90; no exposure trend above 100 plays.
- **Role gap:** median speed ratio OL 0.78, DL 0.81, TE 0.86, WR 0.89, DB 0.90.
- **Peak acceleration** is *above* the lab 40 peak for 75% of player-seasons (median 1.11): game bursts beat the 40 start. r only 0.44 pooled and ~0–0.4 within position; season-to-season 0.59. 55% of peaks fall in the first 1.5 s after the snap, 28% are re-accelerations out of a cut.

### Field channels (step 3, `reports/field_channels.md`)

- 250,657 plays, 7,899 player-games, 476 qualified player-seasons (287 prospects).
- Season values (`top3`, medians): speed 8.0 m/s, kinetic energy 32 J/kg, burst 5.3 m/s², power 19.5 W/kg (WR 23, OL 12).
- Against the lab 40 (same estimators): speed 0.83, KE 0.69, power 0.80 (r 0.72), burst 1.00 (r 0.43).

### Reliability (step 4, `reports/field_reliability.md`)

Within position group (pooled figures are inflated by role):

| Channel | Field rel. | Lab rel. (1 rep) | Field–lab r | η rel. | SD of log η |
|---|---|---|---|---|---|
| speed | 0.80 | 0.95 | 0.59 | **0.67** | 0.051 |
| burst | 0.91 | 0.45 | 0.11 | 0.67 | 0.151 |
| power | 0.85 | 0.69 | 0.31 | 0.65 | 0.151 |

- Speed η is a genuine ratio: the lab is precise and predicts the field. Burst is the most reliable field channel, but the lab 40 acceleration is noisy and barely related to it. Power is in between.
- Over 6 games, field reliability within position is 0.60 for speed and 0.74–0.79 for burst and power. Single-game speed is weak (0.28): top speed needs an opportunity.

### η and the gap (step 5, `reports/eta.md`)

- **Coverage:** 394 of 476 qualified player-seasons have a well-fit 40; **237 prospects**, 211 with a kinetic profile. Qualified seasons per prospect: 1 (121), 2 (75), 3 (41).
- **η_KE median 0.69** (p10–p90 0.51–0.82), i.e. η_speed 0.83; gap median 0.31. By group: WR 0.77, DB 0.78, TE 0.69, DL 0.61, OL 0.54.
- **Why speed is the gap:** within position, log η_speed is unrelated to the lab value (r −0.04) and follows the field value (r 0.79). Burst and power η mostly follow minus the lab value (r −0.61, −0.66). Within position, η_burst and η_power correlate 0.78 with each other and only 0.18–0.28 with η_speed.
- **Role:** position explains 69% of the variance of log η_speed (10% for burst, 32% for power).
- **Season to season** (154 consecutive pairs, within position): η_speed r 0.38, below its within-season reliability of 0.67: speed η changes between seasons (role, development, health). Burst (0.62) and power (0.74) look steadier only because the shared lab value repeats.

### Reading η (step 6, `reports/eta_readout.md`)

- **Shrinkage** (empirical Bayes on log η_KE toward the group mean): season-to-season SD 0.078 against a true within-group spread of 0.069, so one season is a weak read. A prospect keeps B = 0.44 of their distance from the group mean with one season, 0.61 with two, 0.70 with three. Credibly off the group mean (90%, above / below): WR 0 / 0, DB 3 / 1, TE 1 / 0, DL 8 / 5, OL 5 / 7.
- **Sub-roles** barely differ in median η_KE (DL: OLB 0.62, NT 0.62, DE 0.60, DT 0.59; DB: CB 0.77, FS 0.75, SS 0.76), so the position group is enough role control. OLBs do fill the top of the DL board.
- **η is nearly independent of the engine** (within group): r −0.08 with 40 speed, −0.17 with `ax_explosive` (−0.24 within listed position), −0.02 with the Kinetic Anomaly, −0.09 with TEE_z_rel. In within-group terciles (~23 per cell if independent), low engine × high η holds 29 and low engine × low η 15; high engine × high η 19. **Potential and realization are separate dimensions.**
- **η goes with college production:** NGS college production score r +0.23 (95% CI 0.10–0.35; +0.21 within listed position), and slightly against the NGS athleticism score (−0.16). The body-mass link (−0.14) is sub-role (−0.04 within listed position).
- Named leaderboards live in `data/processed/eta_leaderboard.parquet` only (never committed).

## What was deferred

- **Change-of-direction windows** in games: the lab COD tracking covers only ~170 prospects (126–128 profiled).
- **Better lab burst:** τ (rep-to-rep r 0.51) and the 40 acceleration peak (0.45) are the weakest lab measures. Averaging over all well-fit reps is the first fix (Iteration 4).
- **Combine skill drills** (position drill sets, ~60 drill names, 10 Hz) are untouched.
