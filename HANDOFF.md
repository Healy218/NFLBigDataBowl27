# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 3, NGS Alignment & Explosion Efficiency (η)
- **Roadmap Phase:** Phase 4a–4b, On-field motor and Explosion Efficiency
- **Last updated:** 2026-10-20 (Iteration 2 closed; Iteration 3 opened)

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
```

97 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml).

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
- **Coverage of the 369 profiled prospects:** 292 appear in a REG game, and **219** have a season with ≥ 100 REG plays. The 77 with no REG snaps get no η, but they matter for Iteration 4: never playing is the earliest form of league exit.

---

## Open decisions (decide when the data is in front of us)

1. **Season types:** REG + POST, PRE dropped (recommended: preseason snaps are few and effort and rosters aren't comparable). Decide at step 1.
2. **Field statistic:** a high percentile of frame values, or per-game maxima summarised over the season. Pick on split-half (odd vs even games) reliability at step 4.
3. **Role dependence of field output:** an OL rarely needs top speed in a game, so raw η for speed will track role. Options: read η within position group (a scope use, already how the leaderboards work), restrict channels to contexts every role shares (e.g. the first second after the snap), or both. Decide at step 5, once the role gap is measured.

---

## Next Session Steps

1. **Ingest game tracking** (`src/eeg/ingest/load_games.py` → `data/interim/game_tracking_{season}.parquet`): join `games.csv` for season type; keep REG + POST; convert to SI; re-derive speed and acceleration from x/y; flag (don't drop) glitch frames. Tests on synthetic frames.
2. **Validate field kinematics:** derived vs reported speed; glitch rule; field top speed against lab v_max per prospect (how many exceed the lab value, and by how much).
3. **Field channels** (`src/eeg/ngs/`): per frame $E_k/m = \tfrac12 v^2$ and propulsive power per kg $a v$ (+ drag); per play → per game → per player-season: top speed, burst acceleration from near rest, peak power per kg. Change of direction can follow if time allows (lab COD covers only 126–128 profiled prospects).
4. **Exposure and reliability:** split-half by game; minimum plays per player-season.
5. **η per matching channel** (speed, burst acceleration, peak power) and the overall Explosion Efficiency Gap; decide on role dependence (open decision 3). η on acceleration and power inherits τ's noise; η on speed is the clean one.
6. **Read the results** (post hoc): η leaderboards per position group; η against the profile axes.
7. Close Iteration 3 → Iteration 4 (survival).
