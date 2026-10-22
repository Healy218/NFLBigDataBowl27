# HANDOFF: Read This First

Session-start state for the Explosion Efficiency Gap project. Everything here comes from the **Current Iteration** in [`TODO.md`](TODO.md). When the iteration changes, rewrite this file.

- **Current Iteration:** Iteration 5, Submission
- **Roadmap Phase:** Phase 4d
- **Last updated:** 2026-10-22 (Iteration 5: frame decided, "True ball is an explosion"; headline results hardened; final figures built; notebook drafted)

---

## Current Objective

Turn the analysis into the **Kaggle notebook** for the NFL Big Data Bowl 2027 ("Combine to NFL Performance"; brief in [`ROADMAP.md`](ROADMAP.md)): a narrative a coach, scout or front office can use, the method, the figures, and specific scouting and development recommendations. The frame is set (open decision 1): **"True ball is an explosion"**, explosion potential as the differentiator. Harden the headline results before writing around them.

Done means a notebook that runs end-to-end from `data/raw/`, tells one focused story with the findings below, and a README that matches it.

**Rules:**
- **Position groups are never features.** They scope the study and are used only to read results; models stratify by group or centre within it.
- **Data:** only the Kaggle competition data. No external sources without asking.
- **No look-ahead:** predictors of later performance use Combine values or first-season η.
- **Committed reports hold aggregates only.** Named tables stay in `data/` (competition terms: don't redistribute the datasets). The notebook names players, as Big Data Bowl entries do (the user's decision, 2026-10-22). `reports/figures/` is gitignored: figures are rebuilt by the pipeline.

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
uv run python -m eeg.outcomes.robustness     # → reports/robustness.md, reports/figures/robustness.png
uv run python -m eeg.viz.figures             # → reports/figures/final_{1..5}_*.png (the write-up figures)
```

158 tests (`uv run pytest`). Schemas: [`config/data_dictionary.md`](config/data_dictionary.md). Settings: [`config/physics.yaml`](config/physics.yaml).

**The findings to write around** (details and numbers in the notes):
1. **Explosion potential is real on the field.** Combine values predict game top speed (CV R² 0.31) and burst (0.24) within group.
2. **The draft has priced the Combine on average.** Draft slot alone CV R² 0.47 for playing time; adding the Combine 0.46.
3. **The misses are in the explosive tail.** The top 20% by raw explosive energy (TEE_z_abs) within group × class beat their draft-slot playing time 65% of the time vs 50% (p 0.027); starts 57% vs 38% (p 0.003). A tail effect, not mass. They also exit less than their slot predicts (O/E 0.62; size-adjusted axis 0.36, p 0.06; not confirmed with 48 exits).
4. **DL, position-specific:** Combine start (a10 from the 10-yard split) → in-game get-off (r −0.44 within NFL position) → pressure rate (get-off β −0.47, a10 β 0.36) beyond the draft.
5. **η (exposure-free, reliability 0.75)** is mostly the inverse of engine size (r −0.26 with 40 speed, −0.36 with the explosive axis) and adds nothing to output beyond the draft. Median η_KE 0.57 (WR 0.70, DB 0.62, TE 0.62, DL 0.47, OL 0.34).

**Artifacts for figures and tables:** `tee.parquet`, `profiles.parquet` (engine), `lab_burst.parquet` (a10), `eta.parquet` / `eta_career.parquet` / `eta_leaderboard.parquet` (η; leaderboard named), `outcomes.parquet` / `outcomes_seasons.parquet`, `production_seasons.parquet`, `mismatch.parquet` (named: expected vs actual snaps, tail quantiles per measure). Report figures: `transfer.png`, `output.png`, `mismatch.png`, `tenure.png`, `eta.png`, plus the Phase 3 profile figures.

---

## Robustness (step 2, [`reports/robustness.md`](reports/robustness.md))

`eeg.outcomes.robustness`: the raw-joules tail (share beating its draft slot, permutation p within group × class) under every check, plus the DL chain by class. Figure `reports/figures/robustness.png` (a forest plot fit for the notebook).
- **The direction never flips.** In all 25 tail checks on raw joules the tail beats its slot more often than the rest. The 20% cut was the only one tried before this step.
- **Cut-off:** 10% 59% vs 52% (snaps, p 0.47; starts 57% vs 41%, p 0.08), 20% 65% vs 50% (p 0.026), 30% 59% vs 50% (p 0.10; starts p 0.008). Strongest at 20%; the top 10% alone is too small (37) to tell.
- **Expectation model:** pick 65% (p 0.026), log pick 67% (0.011), round bands 70% (0.002), nearest-picks 65% vs 52% (0.07); starts p ≤ 0.016 under all four.
- **Each class:** snaps 65% / 57% / 72% (2023 / 2024 / 2025) against ~50%; none significant alone (25–28 per tail). Starts 62% / 61% / 48% (rookie starts in 2025).
- **Groups left out:** holds without WR, TE, DB or OL (64–67%, p 0.01–0.06); **without DL it drops to 57% vs 48% (p 0.23)**, so DL carries much of it.
- **Channels:** the 10 Hz tracked 40 energy alone 63% vs 51% (snaps p 0.08; starts p 0.03); vertical 60% (starts p 0.008), broad 59%, bench nothing; jumps + bench together 65% (p 0.023). r(E_40, jumps + bench) 0.63: the **composite** carries it, each channel alone is weaker. The tracked 40 is one of its pillars, not the whole signal.
- **DL chain by class:** a10 → get-off negative in all three (−0.35, −0.50, −0.44); get-off → pressure −0.45 and −0.51 in 2023 and 2024, −0.15 in 2025 (rookies only); a10 → pressure 0.32, 0.34, 0.18. Same direction everywhere; single classes are too small for tight intervals.

---

## Final figures (step 3, `eeg.viz.figures` → `reports/figures/final_*.png`, not committed)

Built from the analysis functions (no typed-in numbers), one shared style; the example prospect is described, not named (`figures.example_prospect` in `config/physics.yaml`).
1. **`final_1_physics`: from drill to joules.** One edge rusher's tracked 40 (10 Hz points, fitted v(t), ½mv² at top speed) and his energy per drill against his Combine class (+1.25 SD total); pick 77, expected ~279 snaps a season from that slot, played 889.
2. **`final_2_priced`: real on the field, priced by the draft.** Combine → game top speed CV R² 0.31, burst 0.24; playing time: Combine 0.23, draft 0.47, both 0.46.
3. **`final_3_tail`: the explosive tail beats its slot.** Share beating the slot by quintile (raw energy vs mass) and 11 robustness checks as a forest plot.
4. **`final_4_dl_chain`: the DL start.** Partial plots within NFL position: a10 → get-off (β −0.44), get-off → pressure beyond the draft and snaps (β −0.47).
5. **`final_5_eta`: how much of the 40 reaches the game.** Game top speed as % of Combine top speed by group (WR 83%, DB 78%, TE 79%, DL 69%, OL 58%), and within group, bigger engines reach a smaller share (r −0.27).

---

## Notebook draft (step 4, [`notebooks/true_ball_is_an_explosion.ipynb`](notebooks/true_ball_is_an_explosion.ipynb))

Source: `notebooks/build_true_ball.py` (edit there, then build and execute: `uv run python notebooks/build_true_ball.py` and `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/true_ball_is_an_explosion.ipynb`). ~1,460 words of narrative, 5 figures and 1 table, executed with outputs.
- Sections: intro (three questions, short answer) → 1. the explosion in joules (Byron Young) → 2. the Combine shows up on Sundays and the draft knows it → 3. where the draft misses: the explosive tail (+ table of the 10 tail players with the most extra snaps over their slot; misses named too) → 4. up front, explosion is the job (DL chain) → 5. how much of the 40 reaches the game → recommendations → limits → method.
- **Still to do:** check the 2027 rules for length and figure limits (earlier Big Data Bowls capped the write-up at about 2,000 words and a handful of figures/tables; not verified for 2027); make it run on Kaggle (it imports `eeg` and reads `data/processed`, so the package and pipeline need to ship as a Kaggle utility script or dataset, or the notebook must rebuild from `/kaggle/input`); a read-through with the user.

---

## Side projects (`fun/`)

Outside the competition work: week 1 and eight-week team explosion rankings, team explosion vs season results (no signal: only young starters are tracked), and the 2026 rookie class in joules. Public nflverse data is allowed there only (downloaded to `data/external/`, never committed); see [`fun/README.md`](fun/README.md).

---

## Open decisions (decide when the data is in front of us)

1. ~~**Narrative frame**~~ Decided 2026-10-22 (the user's original intent): **(b) explosion potential is the differentiator.** Physics of the Combine → the explosive tail the draft under-rates → the DL start chain; η stays as a description of how much top speed reaches games, by group. **Presentation title: "True ball is an explosion".**
2. ~~**η beyond expectation**~~ Not needed: η is descriptive under frame (b).
3. **Brief emphasis on 10 Hz tracking** (measured in step 2): the tracked 40 energy alone carries most of the tail signal (63% vs 51%), the composite with jumps and bench is stronger (65% vs 50%). Frame it as total explosive energy, with the 10 Hz 40 trace as its tracked core; a10 (DL chain) is the official split.

---

## Next Session Steps

1. ~~**Decide the frame**~~ Done: (b), titled "True ball is an explosion".
2. ~~**Harden the headline results**~~ Done: `eeg.outcomes.robustness`, see "Robustness" above.
3. ~~**Final figures**~~ Done: `eeg.viz.figures`, see "Final figures" above.
4. **Notebook narrative:** drafted (see "Notebook draft" above); next the user's read-through, the 2027 rules check and the README executive summary to match.
5. **Reproducibility pass:** clean run end-to-end from `data/raw/`.
6. Optional: Combine position skill drills.
