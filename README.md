# The Explosion Efficiency Gap: Bridging Lab Power and On-Field Motor for NFL Talent Evaluation

> NFL Big Data Bowl 2027 submission. Theme: **Combine to NFL Performance**.

---

## Executive Summary

Scouts read the Combine through position labels. A 4.45 forty is "elite" for a defensive tackle and "average" for a corner, so every drill gets graded against a named-position curve. That framing hides the question that matters: **how much mechanical energy can this body produce, and how much of it shows up on Sundays?**

This project drops named positions as model inputs and works in **physics**:

1. **Lab Power.** Each Combine drill (vertical, broad, 40-yard dash, 5-10-5 shuttle, 3-cone, bench) becomes the **mechanical work it required, in Joules**, using biomechanical equations (Sayers peak-power regression, projectile kinetic energy, sprint impulse with aerodynamic drag). Where the 10 Hz Combine sensor tracking is available, we measure the velocity profiles directly instead of reconstructing them from split times. The result is a **Total Explosive Energy** vector per prospect, reported as absolute Joules (`TEE_Absolute`) and as mass-normalized J/kg (`TEE_Relative`).
2. **Kinetic Profiles.** Each prospect's drill output relative to body size is described on continuous engine axes (general explosiveness, late-building speed vs quick bounce, jumping up vs out) plus a *Kinetic Anomaly* score for unusual engines, with no positional label in the feature set. Clustering (K-Means, Gaussian Mixture Models) was tested first: for its size, the engine has no natural archetypes; only body size itself comes in classes.
3. **On-Field Motor.** Next Gen Stats (NGS) player tracking from regular-season games gives realized on-field kinetic output: peak speed, acceleration bursts, and kinetic energy at the moments that matter.
4. **Explosion Efficiency (η).** The ratio of realized on-field output to lab capacity:

$$
\eta = \frac{E_{\text{field}}}{E_{\text{lab}}}
\qquad
\text{Explosion Efficiency Gap} = 1 - \eta
$$

A prospect with a large lab engine and a low η is a "Combine warrior" whose power doesn't transfer to the field. A modest engine with high η is a player who gets the most out of the engine they have.

5. **Regular-Season Performance.** Models within each position group test whether explosion potential and η predict regular-season output (playing time, position-specific production from `player_play`, games started) beyond what draft slot already tells us. Tenure (league-exit risk, Cox Proportional Hazards) is a secondary outcome: the draft classes have at most three seasons.

The deliverable is a scouting signal that doesn't depend on positions. It measures how much engine a player has, separately from how efficiently they use it.

---

## Project Architecture & Tech Stack

```
 Combine drills ──► Physics engine ──► TEE vector (J, J/kg) ──► Clustering ──► Kinetic Profiles
 10 Hz Combine tracking ─┘                                                         │
                                                                                   ▼
 NGS game tracking ──► On-field kinetic output ──────────────────────────► η = E_field / E_lab
                                                                                   │
 Snaps, production, draft slot (competition data) ──────────────────► potential → performance models
```

| Layer | Primary (Python) | Alternative (R) |
|---|---|---|
| Data wrangling | **Polars** (large tracking frames), Pandas (interop) | dplyr, data.table |
| Physics / numerics | NumPy, SciPy (`curve_fit` for sprint profiles) | base R, `minpack.lm` |
| Clustering | **scikit-learn** (`KMeans`, `GaussianMixture`, `StandardScaler`, silhouette / BIC) | **tidymodels**, `tidyclust`, `mclust` |
| Survival analysis | **lifelines** (`CoxPHFitter`, `KaplanMeierFitter`) | `survival`, `survminer` |
| Visualization | Matplotlib, Seaborn, Plotly | ggplot2, gganimate |
| Notebooks / submission | Jupyter, Kaggle Notebooks | Quarto / R Markdown |
| Testing | pytest | testthat |

Python is the primary implementation language. R is an optional mirror for the survival and visualization layers.

---

## File Structure

```
NFLBigDataBowl27/
├── README.md              # This file: concept, stack, layout
├── ROADMAP.md             # Macro timeline: Phase 1 (competition theme) → Phase 4 (η + regular-season performance)
├── TODO.md                # Micro task tracker by iteration; marks the Current Iteration
├── HANDOFF.md             # Session-start state: read this first when resuming work
├── .gitignore             # Excludes all data (competition terms) and build artifacts
├── config/                # Physical constants & modeling assumptions (YAML)
├── data/                  # Git-ignored
│   ├── raw/kaggle/nfl-big-data-bowl-2027/   # Competition files: Combine results + 10 Hz tracking,
│   │                                         # game tracking, player_play, players, career successes
│   ├── interim/           # Cleaned / unit-converted intermediates
│   └── processed/         # Model-ready tables (TEE vectors, η, survival frames)
├── notebooks/             # Exploration & the final Kaggle submission notebook
├── src/eeg/               # "Explosion Efficiency Gap" package
│   ├── ingest/            # Loaders + unit conversion (imperial → SI)
│   ├── physics/           # Joule formulas per drill, mass normalization, TEE assembly
│   ├── clustering/        # Feature scaling, K-Means / GMM, profile labeling
│   ├── ngs/               # Player-ID crosswalk, on-field kinetic output, η computation
│   └── survival/          # Duration/event construction, Cox PH, diagnostics
├── tests/                 # pytest: physics sanity checks, unit conversions, joins
└── reports/
    └── figures/           # Generated plots (git-ignored)
```

---

## Getting Started

```bash
cd NFLBigDataBowl27
uv sync                                   # creates .venv (Python 3.12) from uv.lock, incl. dev tools
source .venv/bin/activate                 # fish: source .venv/bin/activate.fish
uv run pytest                             # or run anything without activating via `uv run`
```

Dependencies live in `pyproject.toml` and are pinned in `uv.lock`. Add new ones with `uv add <pkg>` (or `uv add --dev <pkg>`).

Competition data goes in `data/raw/` once it's downloaded from Kaggle. See [`HANDOFF.md`](HANDOFF.md) for the current state and next steps.

## Status

See [`ROADMAP.md`](ROADMAP.md) for the macro plan and [`TODO.md`](TODO.md) for the active iteration.
