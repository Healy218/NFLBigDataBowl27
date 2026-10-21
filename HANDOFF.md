# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 3, NGS Alignment & Explosion Efficiency (η)
- **Roadmap Phase:** Phase 4a–4b, On-field motor and Explosion Efficiency
- **Last updated:** 2026-10-21 (steps 1–2 done: game tracking ingested and validated)

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
```

110 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml).

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

---

## Open decisions (decide when the data is in front of us)

1. ~~**Season types**~~ **Decided 2026-10-21:** REG + POST, PRE dropped (preseason snaps are few, and effort and rosters aren't comparable). `games.season_types` in `config/physics.yaml`.
2. **Field statistic:** a high percentile of frame values, or per-game maxima summarised over the season. Pick on split-half (odd vs even games) reliability at step 4. Speed is already stable as a season peak (retest 0.90); acceleration is not (0.59).
3. **Role dependence of field output:** an OL rarely needs top speed in a game, so raw η for speed will track role. Options: read η within position group (a scope use, already how the leaderboards work), restrict channels to contexts every role shares (e.g. the first second after the snap), or both. Decide at step 5. The gap is measured: OL 0.78 → DB 0.90 of lab speed.

---

## Next Session Steps

1. ~~**Ingest game tracking**~~ Done (see facts above).
2. ~~**Validate field kinematics**~~ Done (see findings above).
3. **Field channels** (`src/eeg/ngs/channels.py`): per frame $E_k/m = \tfrac12 v^2$ and propulsive power per kg $a_s v$ (+ drag, as in the lab); per play → per game → per player-season: top speed, burst acceleration (from near rest, first 1.5 s after the snap), peak power per kg. Lab side: the same estimators on the best 40 rep (`validate.lab_peaks`). Change of direction can follow if time allows (lab COD covers only 126–128 profiled prospects).
4. **Exposure and reliability:** split-half by game; minimum plays per player-season.
5. **η per matching channel** (speed, burst acceleration, peak power) and the overall Explosion Efficiency Gap; decide on role dependence (open decision 3). η on acceleration and power inherits τ's noise; η on speed is the clean one.
6. **Read the results** (post hoc): η leaderboards per position group; η against the profile axes.
7. Close Iteration 3 → Iteration 4 (survival).
