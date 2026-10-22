# ROADMAP: The Explosion Efficiency Gap

This is the macro timeline, from the official competition brief to the final deliverable. Each phase maps to one or more iterations in [`TODO.md`](TODO.md).

---

## The competition: NFL Big Data Bowl 2027, "Combine to NFL Performance"

**The challenge (official):** build an analytical framework that uncovers **non-obvious, actionable linkages between 10 Hz Combine sensor tracking data and future regular-season NFL game performance**. Look beyond conventional scouting wisdom and identify novel signals that help coaches and evaluators:

1. **identify hidden talent,**
2. **avoid costly draft busts,** and
3. **design position-specific player development protocols.**

**Overview (official):**

> Every spring, college football's top prospects gather in Indianapolis for the NFL Scouting Combine. For decades, player evaluation at the Combine has relied on standardized stopwatch timings and physical testing numbers, such as the 40-yard dash, short shuttle, 3-cone drill, and vertical jump. While these numbers provide a baseline measure of straight-line speed and explosiveness, player tracking sensors worn during workouts now capture the full details of how athletes move throughout each drill.
>
> The 2027 Big Data Bowl provides player tracking data from the NFL Scouting Combine alongside regular-season NFL game tracking data. The goal of this year's competition is to better understand how player movement during Combine drills translates to performance on the field during the NFL regular season.
>
> In this competition, participants are tasked with analyzing prospect movement in a way that is actionable and accessible to coaches, scouts, and front offices. This could include creating a new metric or comparing movement traits across position groups. As is often the case in Big Data Bowl analytics competitions, participants are encouraged to focus on one small aspect of movement, a specific position group, or an individual drill, rather than attempting to evaluate every prospect across all drills.

**What the brief asks of a submission:**
- **Source:** movement *during Combine drills* from the 10 Hz sensor tracking, not only the stopwatch and tape-measure results.
- **Target:** performance *on the field during the NFL regular season*, from game tracking.
- **Non-obvious:** beyond conventional scouting wisdom (more than "fast 40, high pick").
- **Actionable and accessible:** a coach, scout or front office can use it. A new metric is explicitly welcome.
- **Focused:** depth on one aspect of movement, one position group or one drill beats breadth across every drill.

**How this project answers it:**

| Brief | This project |
|---|---|
| Combine movement from 10 Hz tracking | The 40 trace fit to $v(t) = v_{max}(1 - e^{-t/τ})$ → v_max, τ, sprint energy and power (Phase 2); engine profiles (Phase 3) |
| Regular-season game performance | The same speed, energy and power estimators on game tracking (Phase 4a) |
| A new metric | **Explosion Efficiency η = E_field / E_lab**, and the **Explosion Efficiency Gap** 1 − η (Phase 4b) |
| Hidden talent | Modest engine, high η: gets more of the engine onto the field than the Combine suggests |
| Draft busts | "Combine warriors": big engine, low η |
| Position-specific development | η read within each of the 5 groups: where a group's gap sits and who is off the group norm |
| Focus | One aspect of movement: how much of the 40's straight-line engine shows up in games |
| Future performance | η and the profiles against NFL outcomes, controlling for draft slot (Phase 4c) |

---

## Phases

| Phase | Name | Iterations | Output |
|---|---|---|---|
| 1 | Competition Grounding: *Combine to NFL Performance* | 0 | Scoped question, data inventory, repo |
| 2 | Physics Engine: Drills → Joules | 1 | `TEE_Absolute` (J) and `TEE_Relative` (J/kg) per prospect |
| 3 | Kinetic Profiles: Position-Blind Engine Analysis | 2 | Engine axis scores, Kinetic Anomaly score and size class per prospect |
| 4 | Explosion Efficiency (η) + Survival | 3, 4, 5 | η per player, Cox PH tenure model, final submission |

---

## Phase 1: The Start. Competition Grounding

The scope comes from the competition brief above:

- **Data in play:**
  - The **10 Hz Combine sensor tracking data**: player-level position/speed/acceleration traces captured during Combine drills.
  - **Regular-season NFL game tracking data** (Next Gen Stats): player positions at 10 Hz across real snaps.
- **Target:** uncover **non-obvious, actionable linkages** between Combine movement and future regular-season performance. That means more than a correlation between the 40 time and draft slot.
- **Population:** the **5 target position groups** defined by the competition:

| Group | Code |
|---|---|
| Wide Receivers | WR |
| Tight Ends | TE |
| Defensive Backs | DB |
| Offensive Line | OL |
| Defensive Line | DL |

**How positions are used.** Position groups decide *who is in the study* and are used afterward to *read the results*. They are **never model features**. The thesis is that the engine is position-agnostic, and position labels would sneak traditional scouting priors back into the model.

**Phase 1 exit criteria**
- Repository, documentation and task tracking in place.
- Data inventory: which drills have tracking coverage, which seasons of NGS overlap which Combine classes, and the player-ID crosswalk between the two.

---

## Phase 2: Physics Engine. Drills → Joules

Convert distances, rep counts and times into **mechanical work / energy in Joules**, using physics-grounded biomechanical equations. All inputs go to SI units first (in → m, lb → kg, yd → m).

| Drill | Model | Core equation |
|---|---|---|
| Vertical jump | Potential energy at apex; Sayers peak-power regression | $E = m g h$ ; $P_{peak} = 60.7\,h_{cm} + 45.3\,m - 2055$ (W) |
| Broad jump | Projectile kinetic energy at takeoff | $v_0 = \sqrt{\dfrac{R g}{\sin 2\theta}}$ ; $E = \tfrac12 m v_0^2$ (at θ = 45°: $E = \tfrac12 m g R$) |
| 40-yard dash | Mono-exponential sprint profile + aerodynamic drag | $v(t) = v_{max}(1 - e^{-t/\tau})$ ; $W = \tfrac12 m v_{max}^2 + \int F_d\, v\, dt$, $F_d = \tfrac12 \rho C_d A v^2$ |
| 5-10-5 shuttle / 3-cone | Brake-and-reaccelerate work at each change of direction | $W \approx \sum_i \left(\tfrac12 m v_{in,i}^2 + \tfrac12 m v_{out,i}^2\right)$ |
| Bench press (225 lb) | Concentric work per rep × reps | $W = n_{reps} \cdot m_{bar}\, g \cdot \text{ROM}$, ROM ≈ α · arm length |

Notes:
- The Sayers equation gives **peak power (W)**, not energy. It stays in the feature set as a power channel next to the Joule channels and is not added into them.
- Where 10 Hz Combine tracking exists for a drill, **measured velocity replaces reconstructed velocity** (fitted sprint curves, assumed change-of-direction speeds). The reconstructed version is the fallback for prospects without tracking.

**Outputs**
- `TEE_Absolute`: per-drill Joule vector plus a composite (sum of drill-wise z-scores, so no single high-magnitude drill dominates).
- `TEE_Relative`: the same vector divided by body mass (**J/kg**), which separates "big engine" from "efficient engine per kilogram."

**Phase 2 exit criteria:** every prospect in scope has a TEE vector with explicit missingness flags. Physics unit tests pass (e.g., a 1.0 m rise of 100 kg = 981 J).

---

## Phase 3: Kinetic Profiles. Position-Blind Engine Analysis

Group prospects by **mechanical signature alone**.

- **Features: output for your size, plus size.** Absolute Joules and J/kg both mostly measure body mass: clustering on them (or on the original list of J/kg channels, absolute composite, mass, Sayers power, $v_{max}$, τ) recovers size classes and therefore position groups. So each core channel (vertical energy, broad energy, $v_{max}$, τ) enters as its **residual on log mass**: log value minus a trend in log mass (quadratic: a power law leaves middle-mass prospects 0.4–0.6 SD above it) fit across all prospects, never by position. Log mass is one explicit feature. **No position labels.**
  - Core channels only, so every clustered prospect has every feature: 369 of 510. Bench and shuttle / 3-cone are too sparse to define clusters and too position-dependent to impute; they get the same residual treatment and **describe** profiles after fitting.
  - Dropped as redundant: 40-yd energy (its mass residual correlates 1.00 with $v_{max}$'s) and Sayers power (r 0.99 with vertical energy).
- **Preprocessing:** residuals and log mass are z-scored within Combine year (cohort drift); check collinearity (vertical and broad residuals r ≈ 0.64) and consider PCA before clustering. Built by `src/eeg/clustering/features.py`.
- **Clustering test (done): there are no engine archetypes.**
  - **K-Means**: silhouette ≈ 0.20 at every *k*, the same as a structureless Gaussian cloud with the observed covariance; the gap statistic picks *k* = 1.
  - **Gaussian Mixture Model**: BIC over *k* = 1–8 × covariance type picks *k* = 1 on the engine features (96 % of bootstrap resamples). With log mass added, the best model is three **size classes** (skill-sized, TE / lighter DL, OL / heavier DL) times **one** engine Gaussian, independent of size (ΔBIC 46 over the joint mixture).
  - So for its size, the Combine engine is a continuous cloud. The planned archetypes (*Heavy Engines*, *Hyper-Velocity Rockets*) aren't natural types; they become regions of the profile axes below.
- **Deliverable: continuous kinetic profiles.**

| Output | Definition |
|---|---|
| **Engine axes** | PCA of the four engine features (mass excluded), named from loadings: general explosiveness for your size; late-building top speed vs quick, bouncy power; jumping up vs out. The fourth axis is dropped: it is mostly τ rep-to-rep noise (score reliability 0.35) |
| **Kinetic Anomaly score** | Squared Mahalanobis distance on the three kept axes (χ²₃ under the single Gaussian the GMM test supports) and its tail probability |
| **Size class** | The three-class mass mixture, with membership probabilities |

**Validation (post hoc, by position group):** axis distributions per group, and **cross-position engine twins**: nearest neighbours in engine space with different positions, such as a DL and a TE who share an engine. Descriptors (bench, shuttle / 3-cone, 40 energy) are read against the axes.

**Phase 3 exit criteria:** per-prospect profile table; axes stable under bootstrap (Tucker congruence ≥ 0.95); anomaly ranks stable (bootstrap Spearman reported). Met (`reports/profiles_report.md`): median congruence 0.99–1.00, axis-score and anomaly-rank ρ ≥ 0.96 in 95 % of resamples. Post-hoc read: `reports/phase3_readout.md`.

---

## Phase 4: The End Destination. Explosion Efficiency (η) and NFL Tenure

### 4a. On-field motor from NGS

From regular-season game tracking, compute each player's **realized kinetic output**:

- Instantaneous kinetic energy $E_k(t) = \tfrac12 m v(t)^2$ and propulsive power $P(t) = m\,a(t)\,v(t) + F_d\,v(t)$.
- Use **high-percentile** values (e.g., p95 per game, aggregated over a season) rather than means. Snaps where a player is not asked to be explosive shouldn't dilute capacity.
- Match the context to each drill where possible: burst from a standstill (40 analogue), change of direction (shuttle analogue).

### 4b. Explosion Efficiency

$$
\eta = \frac{E_{\text{field}}}{E_{\text{lab}}}
\qquad\text{(computed on matching channels, e.g. sprint KE vs. 40-yd KE; J/kg vs. J/kg)}
$$

- η near 1: the player turns lab capacity into game output.
- η well below 1: an **Explosion Efficiency Gap**. The engine is there but doesn't transfer (technique, processing speed, role, or injury).

### 4c. Survival analysis (Cox Proportional Hazards)

- **Duration:** seasons (or games) from draft to last meaningful participation.
- **Event:** league exit (no snaps for a full season, or last season on record). Active players are **right-censored**.
- **Covariates:** Kinetic Profile axis scores and Kinetic Anomaly score (continuous), size class, η, `TEE_Relative` composite, body mass. **Controls:** draft round/pick, so the model measures signal *beyond* draft capital.
- **Secondary outcomes:** snap share trajectory (from `player_play` / game tracking per season) and career totals in `player_career_successes.csv` (snaps, games active and started, All-Pro / Pro Bowl), modeled as tenure-quality measures. Competition data only: no external value metrics such as wAV.
- **Diagnostics:** Schoenfeld residuals for the proportional-hazards assumption; Kaplan–Meier curves by profile-axis tertile; concordance index.

**Questions to answer**
1. Do certain engine profiles have **lower league-exit hazard** regardless of position?
2. Does η predict tenure **after controlling for draft slot**? (This is where the actionable signal is.)
3. Which profile × η combinations are systematically **under- or over-drafted**?

### 4d. Submission

A Kaggle notebook with the narrative, method, kinetic profile visuals, η leaderboards for the 5 position groups, survival curves, and the specific scouting recommendations that follow.
