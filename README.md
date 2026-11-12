# RescueBot: Intelligent Autonomous Navigation and Survivor Detection Using POMDPs

RDMU Session III, Lab Activity Exercise 2. A rescue robot must find a trapped survivor in a foggy 4×4 grid, using a noisy heat sensor and unreliable movement, while avoiding hazards.

The project has two deliverables built on **one shared model**:

| Deliverable | File | Purpose |
|---|---|---|
| Analysis notebook | `RescueBot_POMDP_Analysis.ipynb` | 17 substantive analytical sections, followed by Section 18: References. Covers formulation, derivations, validation, simulation, POMCP, policy comparison, sensitivity analysis and conclusions |
| Command-centre dashboard | `app.py` (+ `assets/dashboard.css`) | Interactive Streamlit "mission control": live simulation, belief and sensor intelligence, decision analytics, mission report |
| Shared POMDP implementation | `rescuebot_core.py` | Transition, observation and reward models, simulator, Bayes filter, POMCP, baselines, evaluation statistics |

The notebook and the dashboard both `import rescuebot_core`. Neither contains a second copy of the model.

---

## 1. Objectives
1. Implement the full stochastic transition model.
2. Represent beliefs as probability distributions over the robot's and the survivor's positions.
3. Simulate navigation with exact Bayesian belief updates and online action selection.
4. Optimise the policy with Monte Carlo planning (POMCP) and compare it with simpler policies.
5. Visualise the environment, the belief evolution, the robot's path, the sensor readings, the decisions and the performance.

## 2. POMDP formulation: (S, A, T, O, Z, R, γ)

| Component | Definition |
|---|---|
| **S** | s = (r, q): robot cell r (16 cells) × survivor cell q (14 non-hazard cells) = 224 states. r = q is terminal and absorbing. |
| **A** | Up, Down, Left, Right, Scan |
| **T** | A move succeeds with p = 0.80, otherwise the robot stays put (debris). An off-grid move always stays (probability 1, outcomes aggregated). Scan never moves the robot. The survivor is static. |
| **O** | `None` after a move; `Cold` / `Warm` / `Hot` after a Scan |
| **Z** | Imperfect, distance-banded Cold/Warm/Hot sensor (three Manhattan-distance bands d): d ≤ 1 → (0.05, 0.20, 0.75); 2 ≤ d ≤ 3 → (0.20, 0.65, 0.15); d ≥ 4 → (0.75, 0.20, 0.05) for (Cold, Warm, Hot) |
| **R** | +100 for reaching the survivor; −10 per attempted move (also when it fails or is blocked); an extra −50 for entering a hazard. Scan costs 0 in the assignment model (the slide specifies no scan penalty); the optional extension in notebook Section 13b uses a scan cost of 2 |
| **γ** | 0.95. Discounting expresses a preference for earlier reward (and in the infinite-horizon theory keeps returns finite). The implemented mission is finite anyway: it is limited to 30 decisions. At decision t the planner receives H_rem = 30 − t, its tree depth is min(10, H_rem) (configured tree depth 10), and its rollouts stop at the deadline. |

**Belief update** (`Belief.update`): b′(s′) = η · Z(o | s′, a) · Σₛ T(s′ | s, a) · b(s). The update predicts first, then conditions on the observation and on the "not yet rescued" signal, then normalises. An observation with zero likelihood raises `ImpossibleObservationError`; the belief is never silently reset to uniform.

## 3. File structure
```
RescueBot_POMDP_Project/
├── RescueBot_POMDP_Analysis.ipynb   # executed notebook (all outputs included)
├── rescuebot_core.py                # the single POMDP implementation
├── app.py                           # Streamlit dashboard
├── requirements.txt
├── README.md
├── POMCP planning for RescueBot.md  # literature review behind the planner and evaluation design
├── .streamlit/config.toml           # dark theme base for the dashboard
├── assets/dashboard.css             # command-centre styling
├── results/                         # evaluation outputs written by the notebook, read by the dashboard
│   ├── evaluation_summary.csv / evaluation_episodes.json        (100 episodes x 5 policies)
│   ├── evaluation_fixed_summary.csv / evaluation_fixed_episodes.json (survivor at (4,4))
│   ├── evaluation_costed_scan_summary.csv / evaluation_costed_scan_episodes.json (scan cost = 2 extension)
│   ├── episode_results.csv          (flat per-episode table: main + fixed-survivor runs)
│   ├── paired_comparisons.csv       (seed-matched POMCP − heuristic and POMCP − QMDP differences)
│   ├── sensitivity_results.csv, decision_consistency.csv, demo_history.json
└── tests/
    ├── conftest.py
    ├── test_transition.py  test_observation.py  test_belief.py
    ├── test_reward.py      test_environment.py   # environment, runner and leakage tests
    └── test_planner.py     # rollout-leakage and remaining-horizon tests for POMCP
```

## 4. Installation
```bash
pip install -r requirements.txt
```

## 5. Run the tests
```bash
python -m pytest -q tests
```

## 6. Notebook execution
Open it interactively:
```bash
jupyter notebook RescueBot_POMDP_Analysis.ipynb
```
or re-execute it from top to bottom in place, which also regenerates `results/`:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 RescueBot_POMDP_Analysis.ipynb
```
A full run takes about 20 minutes on a 2-core machine (measured: 18.5 minutes); the sensitivity sweeps take most of that time. The notebook writes every file in `results/` from scratch.

## 7. Dashboard execution
```bash
pip install -r requirements.txt
streamlit run app.py
```
Sections, chosen in the sidebar:
1. **Mission Overview**: KPI cards, the tactical 4×4 map (survivor-belief intensity, robot estimate, destination estimate, last heat reading, path) and the robot-localisation map.
2. **Live Robot Simulation**: Start / Pause / Next step / Run to end / Reset, with step telemetry, updated beliefs and the action history.
3. **Belief & Sensor Intelligence**: survivor and robot heatmaps, a belief-evolution scrubber, entropy over time, the observation history and likelihoods, and expected information gain as a function of sensor noise.
4. **Policy & Decision Analytics**: POMCP Q-value estimates and visit counts, Q-value trajectories, reward progression, and a policy comparison (notebook benchmark, or re-run with the current settings).
5. **Performance & Mission Report**: the mission report card, reward components, benchmark charts, sensitivity analysis and an automated interpretation.

The sidebar holds the environment settings (start, hazards, hidden survivor, p, sensor noise, step limit, scan cost), the policy and POMCP parameters, the seeds, and the **instructor ground-truth toggle**. Changing any setting resets the mission automatically.

## 8. Technical assumptions
* **Coordinates.** Zero-based (row, col) internally, with row 0 at the top and "Up" decreasing the row. Every displayed position is **one-based**: `(row+1, col+1)`, from (1,1) to (4,4).
* **Default scenario** (the slide prescribes no positions): start (1,1), survivor (4,4), hazards (2,2) and (3,3), shown one-based.
* **Known hazard map.** Hazards are a static parameter the robot knows. They can be traversed but entering one costs −50. Staying in a hazard after a failed move is not charged again. The notebook (Section 16) explains how to extend the model to hidden hazards with a separate hazard sensor.
* **No odometry.** Movement outcomes are not observed, so the robot's position belief can spread. Contact with the survivor ends the episode, so "mission still running" also tells the robot that it is not co-located with the survivor.
* **Information hiding.** Policies receive only a `Belief` built from `RescueConfig.public_model()`, which has no survivor field. `MissionRunner` locks the simulator while a policy decides, and any read of the hidden state raises `HiddenStateAccessError`. Inside POMCP, sampled hidden states drive only the generative model. Tree and rollout actions depend only on simulated histories and their exact posteriors (tested in `tests/test_planner.py`). Ground-truth overlays in the dashboard are display-only. The *clairvoyant* policy is a labelled, non-deployable reference that is explicitly given the true state.

## 9. Planner and experiment configuration
| Setting | Default |
|---|---|
| POMCP simulations per decision | 800 |
| Search (tree) depth | 10 |
| Horizons | Remaining mission horizon H_rem = 30 − t is passed to the planner at every decision. The tree depth is min(10, H_rem), and rollouts stop at total depth H_rem, so no simulated reward is collected after the real deadline. |
| UCB exploration constant c | **150**. The pre-declared rule was the highest mean *discounted* return (the planner's objective) on development seeds (base seed 77, 40 episodes) over c ∈ {5, 20, 50, 100, 150, 300}. c = 150 had the highest mean discounted return there, while c = 100 had a higher success-rate point estimate: it is a choice of objective, not a value that maximises every metric. It is empirically tuned; its scale is broadly consistent with Silver & Veness's reward-range guidance (c ≈ R_hi − R_lo), but it was **not** computed with that procedure. c = 20 was an earlier candidate, tested and not selected. The full sweep is in notebook Section 14. |
| Rollout policy | `belief` (default, deployable): each rollout step chooses its move from the exact posterior only (most likely survivor cell, then the lowest expected hazard-aware cost-to-go over the robot belief, with 10% random moves), then Bayes-updates that posterior. It never reads the sampled robot or survivor cell. `oracle`: fully observable rollout on the sampled state, a **non-deployable comparison only**. `random`: uniform. |
| Root belief | Exact Bayes posterior, re-planned at every step (receding-horizon POMCP / PO-UCT) |
| Evaluation | 100 episodes per policy on evaluation seeds (base seed 2026, separate from the development seeds); survivor drawn from the prior; common random numbers (same survivor, movement and sensor random numbers for every policy); plus a 50-episode run with the survivor fixed at (4,4), and a 50-episode costed-scan extension (scan cost 2) that includes QMDP |
| Confidence intervals | Wilson 95% for success rates; percentile bootstrap 95% (5,000 resamples) for mean cumulative reward, mean discounted return and mean steps; paired percentile bootstrap for seed-matched POMCP − heuristic and POMCP − QMDP differences (cumulative reward and discounted return). Other metrics are point estimates |
| Sensitivity | One factor at a time, 40 episodes per point: p, sensor noise, simulations, depth, scan cost, c, rescue reward |
| Seeds | `numpy.random.default_rng` and `SeedSequence` throughout; benchmark base seed 2026; demo mission env seed 2026, planner seed 7 |

Compared policies: **POMCP**, a **risk-aware belief heuristic** (scans while uncertain, otherwise moves toward the most likely survivor cell by expected hazard-aware cost), **QMDP**, **Random**, and a **clairvoyant** value-iteration reference (idealised, non-deployable).

The risk-aware heuristic is **not** a Most-Likely-State (MLS) policy. MLS acts on the single most probable joint state. The heuristic scans while uncertain, averages its move choice over the robot-position belief, and charges hazards at full penalty. No standalone MLS baseline was implemented. The initial survivor belief is uniform over the 13 eligible safe cells, so its entropy is log₂ 13 ≈ 3.70 bits (not 4 bits).

## 10. Reading the results
The headline comparison is in notebook Sections 12–13 (Table A, paired seed-matched differences) and Section 13b (costed-scan extension).
The executed notebook reports the confidence intervals listed in Section 11 below; the other metrics are point estimates. A key modelling observation: under the slide's reward values (+100 / −10 / −50, γ = 0.95, free scans, no penalty for failing the mission), a long search for an uncertain survivor is close to break-even in expected reward, and "scan forever" scores exactly 0. Judge POMCP on the objective it optimises, the expected **discounted** cumulative reward (discounted return), alongside the undiscounted cumulative reward, the rescue rate and hazard avoidance. *Immediate reward* is the reward of a single step. *Cumulative reward* is the undiscounted episode total. *Discounted return* is Σ γᵗ rₜ. Differences whose paired confidence interval includes zero are not resolved at the sample size used. See Section 13b for the costed-scan extension and Section 14 for how movement reliability, sensor quality, scan cost and rescue reward shift the results.

## 11. Statistical methods
**Implemented**
* Wilson score 95% intervals for rescue success rates.
* Percentile-bootstrap 95% intervals (5,000 resamples, fixed seed, NumPy) for mean cumulative reward, mean discounted return and mean steps. No head-to-head comparison against BCa was conducted. Constant-valued samples return a point interval.
* Seed-matched paired percentile-bootstrap 95% intervals for POMCP − heuristic and POMCP − QMDP differences in cumulative reward and discounted return, plus discordant rescue-outcome counts (`results/paired_comparisons.csv`).
* Point estimates only (no intervals) for mean scans, mean hazard entries, hazard-entry rate, survivor MAP accuracy, final belief entropy and robot-location uncertainty.
* One-factor-at-a-time sensitivity sweeps with 40 episodes per point: local, descriptive checks.

**Not implemented** (disclosed extensions): McNemar p-values, Bonferroni/Holm multiplicity corrections, correlation diagnostics across paired seeds, median/IQM reporting, BCa intervals, and two-factor sensitivity analysis. The costed-scan extension (50 episodes per policy) reports marginal point estimates and intervals only; no paired test was performed for it.
