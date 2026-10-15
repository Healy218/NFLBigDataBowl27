# Data dictionary: Kaggle NFL Big Data Bowl 2027

Raw files live in `data/raw/kaggle/nfl-big-data-bowl-2027/` (gitignored, competition terms). Missing values are the literal string `NA`, so read with `null_values="NA"`. Checks and numbers behind the notes below are in [`notebooks/00_data_inventory.ipynb`](../notebooks/00_data_inventory.ipynb).

**Key:** `nfl_id` identifies a prospect and links every file. The population is **510 prospects** from the 2023–2025 drafts. Game tracking and `player_play` contain **only these prospects**, not all 22 players on the field.

| File | Rows | Grain |
|---|---|---|
| `combine_results.csv` | 510 | prospect |
| `players.csv` | 510 | prospect |
| `player_career_successes.csv` | 510 | prospect |
| `combine_tracking.csv` | 463,189 | frame (10 Hz) × entity, Combine drills |
| `games.csv` | 1,002 | game (2023–25, PRE/REG/POST) |
| `player_play.csv` | 314,197 | prospect × play |
| `game_tracking_{2023,2024,2025}.csv` | 3.7 M / 7.8 M / 11.2 M | frame (10 Hz) × prospect, games |

---

## `combine_results.csv`

| Column | Type | Units / values | Notes |
|---|---|---|---|
| `draft_year` | int | 2023–2025 | Combine class; cohort for z-scores |
| `nfl_id` | int | | Unique |
| `combine_position` | str | `WR, TE, DB, OL, DL` | Exactly the 5 study groups. Scope only, **never a model feature**. DL includes 30 NFL OLBs (edge rushers) |
| `combine_height` | float | in | Fractional, e.g. 74.625. Complete |
| `combine_weight` | int | lb | Complete |
| `hand_size` | float | in | Complete |
| `arm_length` | float | in | Complete. Bench ROM = α × arm length |
| `wing_span` | float | in | Complete |
| `ten_yd_split` | float | s | Present whenever `forty` is |
| `forty` | float | s | Official 40-yd time (best attempt) |
| `vertical` | float | in | Jump reach − standing reach |
| `broad_jump` | int | in | |
| `three_cone` | float | s | ~35% coverage |
| `short_shuttle` | float | s | 5-10-5; ~39% coverage |
| `bench_reps` | int | reps @ 225 lb | ~38% coverage |
| `ngs_athleticism_score` | int | 0–99 | NGS's own composite. Useful benchmark, not an input |
| `ngs_college_production_score` | int | 0–99 | 1 null |
| `ngs_final_score` | int | 0–99 | 1 null |

## `players.csv`

| Column | Type | Notes |
|---|---|---|
| `nfl_id` | int | |
| `display_name` | str | |
| `draft_year` | int | Same as `combine_results.draft_year` |
| `nfl_position` | str | Finer NFL label (CB, SS, FS, DT, DE, OLB, NT, G, T, C, …). A few differ from the Combine group (e.g. 2 DB → ILB) |
| `birth_date` | str | `YYYY-MM-DD` |
| `college_name`, `college_conference` | str | |
| `draft_round`, `draft_pick_within_round`, `draft_overall_pick` | int | Null = **undrafted** (127 prospects) |

## `player_career_successes.csv`

All integer counts, complete for 510 prospects. Careers are right-censored at the end of the 2025 season.

| Column | Notes |
|---|---|
| `career_offensive_snaps`, `career_defensive_snaps`, `career_special_teams_snaps` | |
| `career_games_active`, `career_games_started` | Max 51 |
| `ap_all_pro_1st_team`, `ap_all_pro_2nd_team`, `pro_bowl_original_ballot` | Rare (1.2% / 0.4% / 2.2% of prospects non-zero) |

## `combine_tracking.csv`

| Column | Type | Units / values | Notes |
|---|---|---|---|
| `draft_year` | int | | |
| `event_id` | str | | Session ID. Format differs for 2023 (hex) vs 2024–25 (timestamp); treat as opaque |
| `nfl_id` | int | | |
| `entity_type` | str | `PLAYER`, `BALL` | BALL rows only in WR/TE skill drills; **filter to PLAYER** |
| `time` | datetime | UTC, 0.1 s steps | Exactly 10 Hz |
| `drill_type` | str | `FORTY_YARD_DASH`, `SHORT_SHUTTLE`, `THREE_CONE_DRILL`, `SKILL_DRILLS_{WR,TE,DB,OL,DL,LB}` | No tracking for vertical, broad or bench |
| `drill_name` | str | | Equals `drill_type` for the timed drills. ~60 skill-drill names, some duplicated with variant spellings (`PASS_PRO-MIRROR_DRILL` / `PASS_PRO_MIRROR_DRILL`) |
| `attempt` | int | 1–4 | A rep = (`nfl_id`, `event_id`, `attempt`) |
| `x`, `y` | float | yd | Field coordinates. 40-yd start line at x = 10, finish at x = 50 |
| `s` | float | yd/s | Consistent with x/y displacement (±1%). Max 11.85; no glitches > 13 |
| `a` | float | yd/s² | |
| `dis` | float | | **Do not use.** 0.25–0.90 × the x/y step depending on drill. Derive distance from `x`/`y` |
| `dir` | float | deg | |

Coverage by timed drill: 40 = 418 prospects / 793 reps, shuttle = 170 / 223, 3-cone = 160 / 203.

**Alignment with official results:**
- **40:** on absolute lines, tracked flying 30 (x 20 → 50) vs official `forty − ten_yd_split` has bias −0.025 s, r = 0.97–0.99. The rep window starts 0.17–0.29 s *before* the official clock, and the lead varies by year, so don't use the first frame as t = 0.
- **Shuttle / 3-cone:** raw windows start ~0.2 s before the official clock and end at (or just short of) the finish line. After segmentation (`src/eeg/physics/cod.py`: onset at 1 m/s → finish crossing anchored to the far point), tracked time vs official has Spearman 0.82 (shuttle) / 0.88 (cone), with bias +0.24 s / +0.10 s. The sensor turns ~1 yd short of every line. The shuttle's first leg goes either way; the 3-cone always runs +x first with the cone-3 loop on the −y side.

## `games.csv`

`game_id` (int, `YYYYMMDDnn`), `game_key` (int), `season` (int), `season_type` (`PRE`/`REG`/`POST`), `week` (int), `home_team_abbr`, `visitor_team_abbr`. 334 games per season (49 PRE, 272 REG, 13 POST).

## `game_tracking_{season}.csv`

| Column | Type | Units | Notes |
|---|---|---|---|
| `game_id`, `play_id` | int | | Join to `player_play`, `games` |
| `nfl_id` | int | | Prospects only (156 / 318 / 448 distinct per season; 497 of 510 appear at least once) |
| `time` | str | UTC ISO timestamp | 10 Hz |
| `x`, `y` | float | yd | |
| `s` | float | yd/s | p99.9 ≈ 9.4, but max up to 16.9: **needs glitch filtering** |
| `a` | float | yd/s² | |
| `dis` | float | yd | Unverified; given the Combine finding, recompute from x/y |
| `o`, `dir` | float | deg | Orientation, direction of motion |
| `event` | str | | Sparse (~7% non-null): `ball_snap`, `pass_forward`, `tackle`, `first_contact`, `pass_arrived`, `play_action`, … |

## `player_play.csv`

One row per prospect per play (64 columns). Grouped:

- **Keys:** `game_id`, `play_id`, `nfl_id`, `team_abbr`, `lined_up_position`
- **Situation:** `play_description`, `quarter`, `down`, `yards_to_go`, `possession_team`, `yardline_side`, `yardline_number`, `game_clock` (`MM:SS`), `pre_snap_home_score`, `pre_snap_visitor_score`, `play_direction` (`left`/`right`)
- **Win prob / EP:** `pre_snap_home_team_win_probability`, `pre_snap_visitor_team_win_probability`, `home_team_win_probability_added`, `visitor_team_win_probility_added` (sic: typo in source), `expected_points`, `expected_points_added`
- **Play type:** `offense_formation`, `receiver_alignment`, `pass_result`, `pass_length`, `play_nullified_by_penalty` (`Y`/`N`), `penalty_yards`, `pre_penalty_yards_gained`, `yards_gained`, `team_coverage_man_zone`, `team_coverage_type`
- **Offense (player):** `in_motion_at_ball_snap`, `target`, `rec_yards`, `rush_yards`, `yards_after_catch`, `expected_yards_after_catch`, `fumble`, `fumble_lost`, `route_ran`, `separation_at_pass_forward`, `cushion`
- **Pass protection (player):** `dropback_duration`, `extended_sack_allowed`, `pass_rushers_encountered`, `peak_pressure_probability_allowed`, `pressure_allowed`, `sack_allowed`, `time_to_pressure_allowed`
- **Pass rush / defense (player):** `quick_pressure`, `unblocked_pressure`, `blitzing`, `player_get_off`, `sack`, `tackle`, `assist`, `caused_forced_fumble`, `recovered_fumble`, `tackle_for_loss`, `time_to_pressure`, `time_to_qb_hurry`, `coverage_assignment`, `coverage_assignment_at_snap`

Most player-level columns are null unless relevant to that player's role on the play.

---

# Derived files

## `data/processed/tee.parquet`

Built by `uv run python -m eeg.physics.tee` from `combine_si.parquet`, `forty_fits.parquet` and `cod_reps.parquet`. One row per prospect (510).

| Column | Units | Notes |
|---|---|---|
| `player_id`, `combine_year`, `pos_group`, `mass_kg` | | `pos_group` scopes the study; never a model feature |
| `E_vert` | J | m g h |
| `E_broad` | J | ½ m R g / sin 2θ (θ = `broad.theta_deg`) |
| `E_40` | J | ½ m v_max² + drag work; tracking only (417) |
| `W_cod_shuttle`, `W_cod_cone` | J | ½ m Σ\|Δ(v²)\|; tracking only (156 each) |
| `W_bench` | J | reps · 102.058 kg · g · α · arm length |
| `<channel>_per_kg` | J/kg | TEE_Relative |
| `P_sayers` | W | Sayers peak power; separate channel, not in composites |
| `sayers_extrapolated` | bool | OL/DL: out of the Sayers sample |
| `W_cod_<drill>_unofficial` | bool | tracked W_cod but no official time (5 shuttle, 9 cone) |
| `<channel>_missing` | bool | channel is null |
| `n_drills` | int | non-null channels (0–6) |
| `z_<channel>`, `z_<channel>_per_kg` | SD | z-score within `combine_year` |
| `TEE_z_abs`, `TEE_z_rel` | SD | mean of available absolute / per-kg z-scores; null if `n_drills` < 3 (412 non-null) |
