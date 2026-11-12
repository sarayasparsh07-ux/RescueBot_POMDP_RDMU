"""
rescuebot_core.py
=================

Shared implementation of the RescueBot POMDP used by BOTH the analysis notebook
(RescueBot_POMDP_Analysis.ipynb) and the Streamlit dashboard (app.py).

POMDP = (S, A, T, O, Z, R, gamma)

* S : s = (r, q) where r is the robot's true cell and q the survivor's true cell.
      q is restricted to safe (non-hazard) cells. States with r == q are terminal
      (the survivor has been reached) and absorbing.
* A : Up, Down, Left, Right, Scan.
* T : a movement succeeds with probability p (default 0.80) and otherwise fails
      (robot stays put). A move off the grid always leaves the robot in place.
      Scan never moves the robot. The survivor never moves.
* O : None (after movement), Cold, Warm, Hot (after Scan).
* Z : heat reading depends on the Manhattan distance between robot and survivor.
* R : +100 for reaching the survivor, -10 per attempted move (even if it fails),
      an additional -50 for entering a hazard cell, 0 for Scan (optional cost).
* gamma = 0.95.

Coordinate convention
---------------------
Internally every position is ZERO-BASED (row, col) with row 0 at the TOP of the
grid and col 0 at the LEFT.  "Up" decreases the row index.  For display, every
position is converted to ONE-BASED coordinates with ``to_display((r, c)) ==
(r + 1, c + 1)`` so the grid runs from (1,1) to (4,4).

Information-hiding contract
---------------------------
The true survivor position lives only inside :class:`RescueBotEnv`.  Policies
receive a :class:`Belief` built from a :class:`POMDPModel`, and the model is
constructed from :meth:`RescueConfig.public_model`, which deliberately drops the
survivor coordinates.  While a policy is choosing an action the
:class:`MissionRunner` locks the environment, so any attempt to read the hidden
state raises :class:`HiddenStateAccessError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ACTIONS", "ACTION_NAMES", "ACTION_SYMBOLS", "MOVE_ACTIONS", "SCAN",
    "OBS_NONE", "OBS_COLD", "OBS_WARM", "OBS_HOT", "OBS_NAMES", "HEAT_OBS",
    "RescueConfig", "POMDPModel", "Belief", "RescueBotEnv", "StepResult",
    "HiddenStateAccessError", "ImpossibleObservationError",
    "POMCPPlanner", "RandomPolicy", "RiskAwareHeuristicPolicy", "QMDPPolicy",
    "ClairvoyantPolicy", "PolicySpec", "make_policy", "MissionRunner",
    "run_episode", "evaluate_policies", "summarise_evaluation",
    "wilson_interval", "bootstrap_ci", "entropy_bits", "sensor_with_noise",
    "value_iteration", "to_display", "fmt_cell", "decision_consistency",
    "sensitivity_sweep", "DEFAULT_POMCP", "DEFAULT_SPECS", "Decision", "Policy",
    "UP", "DOWN", "LEFT", "RIGHT",
]

# --------------------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT, SCAN = 0, 1, 2, 3, 4
ACTIONS: Tuple[int, ...] = (UP, DOWN, LEFT, RIGHT, SCAN)
MOVE_ACTIONS: Tuple[int, ...] = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = {UP: "Up", DOWN: "Down", LEFT: "Left", RIGHT: "Right", SCAN: "Scan"}
ACTION_SYMBOLS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→", SCAN: "◎"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), SCAN: (0, 0)}

OBS_NONE, OBS_COLD, OBS_WARM, OBS_HOT = 0, 1, 2, 3
OBS_NAMES = {OBS_NONE: "None", OBS_COLD: "Cold", OBS_WARM: "Warm", OBS_HOT: "Hot"}
HEAT_OBS: Tuple[int, ...] = (OBS_COLD, OBS_WARM, OBS_HOT)

PROB_TOL = 1e-9


class HiddenStateAccessError(RuntimeError):
    """Raised when hidden simulator state is read while a policy is deciding."""


class ImpossibleObservationError(ValueError):
    """Raised when an observation has zero likelihood under the current belief.

    The belief updater never silently replaces the posterior by a uniform
    distribution: an impossible observation signals a modelling inconsistency.
    """


def to_display(cell: Tuple[int, int]) -> Tuple[int, int]:
    """Zero-based (row, col) -> one-based display coordinates."""
    return (int(cell[0]) + 1, int(cell[1]) + 1)


def fmt_cell(cell: Optional[Tuple[int, int]]) -> str:
    if cell is None:
        return "—"
    r, c = to_display(cell)
    return f"({r},{c})"


def entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits of a (possibly multi-dimensional) distribution."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
DEFAULT_SENSOR = (
    (0.05, 0.20, 0.75),  # Manhattan distance 0-1   (Cold, Warm, Hot)
    (0.20, 0.65, 0.15),  # Manhattan distance 2-3
    (0.75, 0.20, 0.05),  # Manhattan distance >= 4
)


@dataclass(frozen=True)
class RescueConfig:
    """Complete mission configuration (including the hidden survivor location).

    Only the simulator should ever see ``survivor``.  Policies get
    :meth:`public_model`.
    """

    rows: int = 4
    cols: int = 4
    start: Tuple[int, int] = (0, 0)
    survivor: Tuple[int, int] = (3, 3)
    hazards: Tuple[Tuple[int, int], ...] = ((1, 1), (2, 2))
    p_success: float = 0.80
    sensor_band_edges: Tuple[int, int] = (1, 3)
    sensor_table: Tuple[Tuple[float, float, float], ...] = DEFAULT_SENSOR
    rescue_reward: float = 100.0
    move_cost: float = 10.0
    hazard_penalty: float = 50.0
    scan_cost: float = 0.0
    gamma: float = 0.95
    max_steps: int = 30

    def __post_init__(self):
        object.__setattr__(self, "start", tuple(int(v) for v in self.start))
        object.__setattr__(self, "survivor", tuple(int(v) for v in self.survivor))
        object.__setattr__(self, "hazards", tuple(sorted({(int(a), int(b)) for a, b in self.hazards})))
        object.__setattr__(self, "sensor_table", tuple(tuple(float(x) for x in row) for row in self.sensor_table))
        object.__setattr__(self, "sensor_band_edges", tuple(int(v) for v in self.sensor_band_edges))
        self.validate()

    # ---------------------------------------------------------------- validation
    def _in_grid(self, cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def validate(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("Grid must be at least 2x2.")
        if not self._in_grid(self.start):
            raise ValueError(f"Start {self.start} lies outside the grid.")
        if not self._in_grid(self.survivor):
            raise ValueError(f"Survivor {self.survivor} lies outside the grid.")
        for h in self.hazards:
            if not self._in_grid(h):
                raise ValueError(f"Hazard {h} lies outside the grid.")
        if self.survivor in self.hazards:
            raise ValueError("The survivor cannot be placed in a hazard cell.")
        if self.start in self.hazards:
            raise ValueError("The robot cannot start inside a hazard cell.")
        if self.survivor == self.start:
            raise ValueError("The survivor cannot start in the robot's initial cell.")
        if not (0.0 <= self.p_success <= 1.0):
            raise ValueError("p_success must lie in [0, 1].")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must lie in (0, 1].")
        if len(self.sensor_table) != 3 or any(len(r) != 3 for r in self.sensor_table):
            raise ValueError("sensor_table must be 3 bands x 3 observations.")
        for row in self.sensor_table:
            if min(row) < 0 or abs(sum(row) - 1.0) > 1e-6:
                raise ValueError(f"Sensor row {row} is not a probability distribution.")
        e0, e1 = self.sensor_band_edges
        if not (0 <= e0 < e1):
            raise ValueError("sensor_band_edges must satisfy 0 <= e0 < e1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if self.scan_cost < 0 or self.move_cost < 0 or self.hazard_penalty < 0:
            raise ValueError("Costs and penalties are given as non-negative magnitudes.")
        n_safe = self.rows * self.cols - len(self.hazards) - 1
        if n_safe < 1:
            raise ValueError("No admissible survivor locations remain.")

    def with_(self, **kw) -> "RescueConfig":
        return replace(self, **kw)

    def public_model(self) -> "POMDPModel":
        """Model parameters the agent is allowed to know (no survivor location)."""
        return POMDPModel(
            rows=self.rows, cols=self.cols, start=self.start, hazards=self.hazards,
            p_success=self.p_success, sensor_band_edges=self.sensor_band_edges,
            sensor_table=self.sensor_table, rescue_reward=self.rescue_reward,
            move_cost=self.move_cost, hazard_penalty=self.hazard_penalty,
            scan_cost=self.scan_cost, gamma=self.gamma, max_steps=self.max_steps,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def sensor_with_noise(level: float, base: Sequence[Sequence[float]] = DEFAULT_SENSOR):
    """Return a sensor table on a single noise axis.

    level = 0 -> noiseless (each band always gives its most likely reading)
    level = 1 -> the baseline table
    level = 2 -> completely uninformative (uniform 1/3 for every band)
    Values in between are linear interpolations, so every row stays a
    probability distribution.
    """
    level = float(level)
    if not 0.0 <= level <= 2.0:
        raise ValueError("noise level must lie in [0, 2]")
    base = np.asarray(base, dtype=float)
    perfect = np.zeros_like(base)
    perfect[np.arange(3), base.argmax(axis=1)] = 1.0
    uniform = np.full_like(base, 1.0 / 3.0)
    if level <= 1.0:
        tab = (1 - level) * perfect + level * base
    else:
        tab = (2 - level) * base + (level - 1) * uniform
    tab = tab / tab.sum(axis=1, keepdims=True)
    return tuple(tuple(float(x) for x in row) for row in tab)


# --------------------------------------------------------------------------------------
# Public POMDP model (everything the agent may know)
# --------------------------------------------------------------------------------------
class POMDPModel:
    """Transition, observation and reward model WITHOUT the survivor location.

    The joint state (r, q) is stored with r and q as flat cell indices
    ``idx = row * cols + col``.  Belief arrays have shape ``(n_cells, n_cells)``
    indexed ``[r, q]``; columns belonging to hazard cells are identically zero.
    """

    def __init__(self, rows, cols, start, hazards, p_success, sensor_band_edges,
                 sensor_table, rescue_reward, move_cost, hazard_penalty, scan_cost,
                 gamma, max_steps):
        self.rows, self.cols = int(rows), int(cols)
        self.n = self.rows * self.cols
        self.start = tuple(start)
        self.hazards = tuple(hazards)
        self.p_success = float(p_success)
        self.sensor_band_edges = tuple(sensor_band_edges)
        self.sensor_table = np.asarray(sensor_table, dtype=float)
        self.rescue_reward = float(rescue_reward)
        self.move_cost = float(move_cost)
        self.hazard_penalty = float(hazard_penalty)
        self.scan_cost = float(scan_cost)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self._precompute()

    # ---------------------------------------------------------------- geometry
    def idx(self, cell: Tuple[int, int]) -> int:
        return int(cell[0]) * self.cols + int(cell[1])

    def cell(self, i: int) -> Tuple[int, int]:
        return (int(i) // self.cols, int(i) % self.cols)

    def manhattan(self, i: int, j: int) -> int:
        a, b = self.cell(i), self.cell(j)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def band(self, d: int) -> int:
        e0, e1 = self.sensor_band_edges
        return 0 if d <= e0 else (1 if d <= e1 else 2)

    def intended_cell(self, i: int, a: int) -> int:
        """Cell reached if action ``a`` succeeds (stays put at the boundary)."""
        r, c = self.cell(i)
        dr, dc = ACTION_DELTAS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return nr * self.cols + nc
        return i

    def _precompute(self) -> None:
        n = self.n
        self.hazard_mask = np.zeros(n, dtype=bool)
        for h in self.hazards:
            self.hazard_mask[self.idx(h)] = True
        self.safe_cells = [i for i in range(n) if not self.hazard_mask[i]]
        # valid survivor columns: any non-hazard cell
        self.valid_q = ~self.hazard_mask
        self.start_idx = self.idx(self.start)

        # next-cell table and robot-only movement matrices M_a[r, r']
        self.next_cell = [[self.intended_cell(i, a) for i in range(n)] for a in ACTIONS]
        self.move_matrix = np.zeros((len(ACTIONS), n, n))
        for a in ACTIONS:
            for i in range(n):
                j = self.next_cell[a][i]
                if a == SCAN:
                    self.move_matrix[a, i, i] += 1.0
                elif j == i:  # boundary: both success and failure keep the robot in place
                    self.move_matrix[a, i, i] += self.p_success
                    self.move_matrix[a, i, i] += 1.0 - self.p_success
                else:
                    self.move_matrix[a, i, j] += self.p_success
                    self.move_matrix[a, i, i] += 1.0 - self.p_success

        # distances and observation likelihood tables Z[o][r, q]
        self.dist = np.array([[self.manhattan(i, j) for j in range(n)] for i in range(n)])
        bands = np.vectorize(self.band)(self.dist)
        self.obs_lik = np.zeros((4, n, n))
        self.obs_lik[OBS_NONE] = 1.0
        for k, o in enumerate(HEAT_OBS):
            self.obs_lik[o] = self.sensor_table[bands, k]
        # cumulative thresholds for fast sampling: cum[r][q] = (P(cold), P(cold)+P(warm))
        self._obs_cum = [[(float(self.obs_lik[OBS_COLD, i, j]),
                           float(self.obs_lik[OBS_COLD, i, j] + self.obs_lik[OBS_WARM, i, j]))
                          for j in range(n)] for i in range(n)]
        self._hazard_list = [bool(x) for x in self.hazard_mask]
        self._cost_to_go = {}

    # ---------------------------------------------------------------- reward
    def reward(self, r: int, a: int, r2: int, q: int) -> Tuple[float, Dict[str, float]]:
        """R(s, a, s') with its components.  Rewards are only defined for
        non-terminal s; terminal states are absorbing with zero reward."""
        comp = {"move_cost": 0.0, "hazard": 0.0, "rescue": 0.0, "scan_cost": 0.0}
        if r == q:
            return 0.0, comp
        if a == SCAN:
            comp["scan_cost"] = -self.scan_cost
        else:
            comp["move_cost"] = -self.move_cost
            if r2 != r and self._hazard_list[r2]:
                comp["hazard"] = -self.hazard_penalty
            if r2 == q:
                comp["rescue"] = self.rescue_reward
        return float(sum(comp.values())), comp

    # ---------------------------------------------------------------- generative model (POMCP)
    def generative_step(self, r: int, q: int, a: int, u_move: float, u_obs: float):
        """G(s, a) -> (r', o, reward, terminal) using supplied uniforms."""
        if a == SCAN:
            c0, c1 = self._obs_cum[r][q]
            o = OBS_COLD if u_obs < c0 else (OBS_WARM if u_obs < c1 else OBS_HOT)
            return r, o, -self.scan_cost, False
        target = self.next_cell[a][r]
        r2 = target if u_move < self.p_success else r
        rew = -self.move_cost
        if r2 != r and self._hazard_list[r2]:
            rew -= self.hazard_penalty
        if r2 == q:
            return r2, OBS_NONE, rew + self.rescue_reward, True
        return r2, OBS_NONE, rew, False

    # ---------------------------------------------------------------- full transition model
    def state_space(self, include_terminal: bool = True) -> List[Tuple[int, int]]:
        """All valid joint states (r, q) as flat indices; q excludes hazards."""
        return [(r, q) for q in self.safe_cells for r in range(self.n)
                if include_terminal or r != q]

    def transition_distribution(self, state: Tuple[int, int], a: int) -> Dict[Tuple[int, int], float]:
        """T(. | s, a) as a dict with duplicate outcomes aggregated."""
        r, q = state
        if r == q:  # terminal states are absorbing
            return {(r, q): 1.0}
        out: Dict[Tuple[int, int], float] = {}
        for r2 in range(self.n):
            p = self.move_matrix[a, r, r2]
            if p > 0:
                out[(r2, q)] = out.get((r2, q), 0.0) + float(p)
        return out

    def full_transition_matrix(self, a: int):
        """Return (states, T) where T[i, j] = T(states[j] | states[i], a)."""
        states = self.state_space(include_terminal=True)
        pos = {s: k for k, s in enumerate(states)}
        T = np.zeros((len(states), len(states)))
        for i, s in enumerate(states):
            for s2, p in self.transition_distribution(s, a).items():
                T[i, pos[s2]] += p
        return states, T

    # ---------------------------------------------------------------- planning helpers
    def cost_to_go(self, target: int) -> np.ndarray:
        """Hazard-aware shortest-path cost (in reward points, positive) from every
        cell to ``target`` assuming moves always succeed.  Entering a hazard costs
        move_cost + hazard_penalty."""
        if target in self._cost_to_go:
            return self._cost_to_go[target]
        n = self.n
        dist = np.full(n, np.inf)
        dist[target] = 0.0
        done = np.zeros(n, dtype=bool)
        for _ in range(n):
            cand = np.where(~done, dist, np.inf)
            u = int(np.argmin(cand))
            if not np.isfinite(cand[u]):
                break
            done[u] = True
            # predecessors: cells v with an action leading to u
            for v in range(n):
                for a in MOVE_ACTIONS:
                    if self.next_cell[a][v] == u and v != u:
                        step = self.move_cost + (self.hazard_penalty if (self._hazard_list[u] and u != target) else 0.0)
                        if dist[u] + step < dist[v]:
                            dist[v] = dist[u] + step
        self._cost_to_go[target] = dist
        return dist

    def greedy_move(self, r: int, target: int) -> int:
        """Move that most reduces hazard-aware cost-to-go (deterministic tie-break)."""
        ctg = self.cost_to_go(target)
        best, best_val = UP, np.inf
        for a in MOVE_ACTIONS:
            j = self.next_cell[a][r]
            if j == r:
                continue
            val = ctg[j] + (self.hazard_penalty if (self._hazard_list[j] and j != target) else 0.0)
            if val < best_val - 1e-12:
                best, best_val = a, val
        return best

    def initial_belief(self) -> "Belief":
        return Belief.initial(self)


# --------------------------------------------------------------------------------------
# Belief state
# --------------------------------------------------------------------------------------
class Belief:
    """Exact joint belief b(r, q) over robot cell r and survivor cell q."""

    def __init__(self, model: POMDPModel, joint: np.ndarray):
        self.model = model
        self.joint = np.asarray(joint, dtype=float)
        self.check()

    @classmethod
    def initial(cls, model: POMDPModel) -> "Belief":
        """Robot position known; survivor uniform over safe cells except the start."""
        b = np.zeros((model.n, model.n))
        qs = [q for q in model.safe_cells if q != model.start_idx]
        b[model.start_idx, qs] = 1.0 / len(qs)
        return cls(model, b)

    def copy(self) -> "Belief":
        return Belief(self.model, self.joint.copy())

    # ---------------------------------------------------------------- invariants
    def check(self, tol: float = 1e-9) -> None:
        b = self.joint
        assert b.shape == (self.model.n, self.model.n), "belief has wrong shape"
        assert np.all(b >= -tol), "belief has negative probabilities"
        assert abs(b.sum() - 1.0) < 1e-8, f"belief sums to {b.sum()}"
        assert np.all(b[:, self.model.hazard_mask] == 0.0), "survivor mass on a hazard cell"

    # ---------------------------------------------------------------- Bayes filter
    def predict(self, a: int) -> np.ndarray:
        """Predictive belief  b_pred(s') = sum_s T(s'|s,a) b(s).

        Robot moves with M_a while the survivor column q is unchanged, so
        b_pred[r', q] = sum_r M_a[r, r'] b[r, q].  Terminal states (r == q) are
        absorbing; they carry no mass during an ongoing mission.
        """
        M = self.model.move_matrix[a]
        diag = np.diag(self.joint).copy()
        b = self.joint.copy()
        np.fill_diagonal(b, 0.0)
        pred = M.T @ b
        pred[np.diag_indices_from(pred)] += diag  # absorbing terminal states
        return pred

    def update(self, a: int, o: int, rescued: bool = False, return_details: bool = False):
        """Posterior b'(s') = eta * Z(o|s',a) * sum_s T(s'|s,a) b(s).

        ``rescued`` is the environment's termination signal (the robot recognises
        the survivor on contact).  If the mission continues the update conditions
        on r' != q'; if it ended, on r' == q'.
        """
        if a == SCAN and o not in HEAT_OBS:
            raise ValueError("Scan must produce Cold/Warm/Hot.")
        if a != SCAN and o != OBS_NONE:
            raise ValueError("Movement actions produce the None observation.")
        pred = self.predict(a)
        lik = self.model.obs_lik[o].copy()
        eye = np.eye(self.model.n, dtype=bool)
        lik = np.where(eye, lik, 0.0) if rescued else np.where(eye, 0.0, lik)
        unnorm = lik * pred
        total = unnorm.sum()
        if not np.isfinite(total) or total <= 1e-300:
            raise ImpossibleObservationError(
                f"Observation {OBS_NAMES[o]} after {ACTION_NAMES[a]} has zero likelihood "
                "under the current belief.")
        post = unnorm / total
        post[post < 0] = 0.0
        new = Belief(self.model, post)
        if return_details:
            return new, {"predicted": pred, "likelihood": lik, "unnormalised": unnorm,
                         "evidence": float(total), "eta": float(1.0 / total)}
        return new

    # ---------------------------------------------------------------- summaries
    def robot_marginal(self) -> np.ndarray:
        """P(robot at r) = sum_q b(r, q), as a (rows, cols) grid."""
        return self.joint.sum(axis=1).reshape(self.model.rows, self.model.cols)

    def survivor_marginal(self) -> np.ndarray:
        """P(survivor at q) = sum_r b(r, q), as a (rows, cols) grid."""
        return self.joint.sum(axis=0).reshape(self.model.rows, self.model.cols)

    def joint_entropy(self) -> float:
        return entropy_bits(self.joint)

    def survivor_entropy(self) -> float:
        return entropy_bits(self.survivor_marginal())

    def robot_entropy(self) -> float:
        return entropy_bits(self.robot_marginal())

    def map_survivor(self) -> Tuple[Tuple[int, int], float]:
        m = self.survivor_marginal().ravel()
        i = int(np.argmax(m))
        return self.model.cell(i), float(m[i])

    def map_robot(self) -> Tuple[Tuple[int, int], float]:
        m = self.robot_marginal().ravel()
        i = int(np.argmax(m))
        return self.model.cell(i), float(m[i])

    def predicted_obs_distribution(self) -> Dict[str, float]:
        """P(o | b, Scan) for the three heat readings."""
        return {OBS_NAMES[o]: float((self.model.obs_lik[o] * self.joint).sum()) for o in HEAT_OBS}


# --------------------------------------------------------------------------------------
# Environment simulator
# --------------------------------------------------------------------------------------
@dataclass
class StepResult:
    step: int
    action: str
    intended_cell: Optional[Tuple[int, int]]
    outcome: str
    observation: str
    reward: float
    reward_components: Dict[str, float]
    cumulative_reward: float
    done: bool
    rescued: bool
    truncated: bool


class RescueBotEnv:
    """Ground-truth simulator.  The hidden state is readable only via
    :meth:`get_state_for_evaluation`, and never while the environment is locked
    for policy decision-making."""

    def __init__(self, config: RescueConfig, seed: Optional[int] = 0):
        self.config = config
        self._model = config.public_model()
        self._locked = False
        self.reset(seed)

    # -------------------------------------------- hidden-state guard
    @property
    def _hidden(self):
        if self._locked:
            raise HiddenStateAccessError("Hidden state requested while a policy is deciding.")
        return self.__hidden

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False

    # -------------------------------------------- interface
    def reset(self, seed: Optional[int] = None, survivor: Optional[Tuple[int, int]] = None):
        """Reset the mission.  Randomness for step t (movement success and sensor
        noise) is pre-drawn so that two policies run with the same seed face
        common random numbers."""
        self.rng = np.random.default_rng(seed)
        horizon = self.config.max_steps
        self._u_move = self.rng.random(horizon)
        self._u_obs = self.rng.random(horizon)
        m = self._model
        surv = self.config.survivor if survivor is None else tuple(survivor)
        if surv in self.config.hazards or surv == self.config.start:
            raise ValueError("Invalid survivor location.")
        self.__hidden = {"robot": m.start_idx, "survivor": m.idx(surv)}
        self.t = 0
        self.cumulative_reward = 0.0
        self._done = False
        self._rescued = False
        self._last_obs = OBS_NONE
        self.history: List[StepResult] = []
        return self.get_observation()

    def get_observation(self) -> str:
        return OBS_NAMES[self._last_obs]

    def is_terminal(self) -> bool:
        return self._done

    def get_state_for_evaluation(self) -> Dict[str, Tuple[int, int]]:
        """Ground truth for evaluation / instructor visualisation ONLY."""
        h = self._hidden
        return {"robot": self._model.cell(h["robot"]), "survivor": self._model.cell(h["survivor"])}

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode already finished; call reset().")
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action}")
        m = self._model
        h = self._hidden
        r, q = h["robot"], h["survivor"]
        u_move, u_obs = float(self._u_move[self.t]), float(self._u_obs[self.t])
        r2, o, _, terminal = m.generative_step(r, q, action, u_move, u_obs)
        reward, comp = m.reward(r, action, r2, q)
        if action == SCAN:
            outcome, intended = "Scan (stationary)", None
        else:
            target = m.next_cell[action][r]
            intended = m.cell(target)
            if target == r:
                outcome = "Blocked by boundary"
            elif r2 == target:
                outcome = "Moved"
            else:
                outcome = "Failed (debris)"
        h["robot"] = r2
        self.t += 1
        self.cumulative_reward += reward
        self._last_obs = o
        self._rescued = bool(terminal)
        truncated = (not terminal) and self.t >= self.config.max_steps
        self._done = bool(terminal or truncated)
        res = StepResult(step=self.t, action=ACTION_NAMES[action], intended_cell=intended,
                         outcome=outcome, observation=OBS_NAMES[o], reward=float(reward),
                         reward_components=comp, cumulative_reward=float(self.cumulative_reward),
                         done=self._done, rescued=self._rescued, truncated=truncated)
        self.history.append(res)
        return res


# --------------------------------------------------------------------------------------
# Random streams
# --------------------------------------------------------------------------------------
class _UniformStream:
    """Block-buffered U(0,1) draws from a NumPy Generator (fast scalar access)."""

    __slots__ = ("rng", "buf", "i", "block")

    def __init__(self, rng: np.random.Generator, block: int = 16384):
        self.rng, self.block = rng, block
        self.buf = rng.random(block).tolist()
        self.i = 0

    def __call__(self) -> float:
        if self.i >= self.block:
            self.buf = self.rng.random(self.block).tolist()
            self.i = 0
        v = self.buf[self.i]
        self.i += 1
        return v


# --------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------
@dataclass
class Decision:
    action: int
    info: Dict = field(default_factory=dict)


class Policy:
    """Base class.  ``select_action`` receives the current belief and, from the
    mission runner, the number of real decisions remaining before the mission
    deadline (``steps_remaining``).  Only non-deployable reference policies
    receive ``ground_truth``."""

    name = "policy"
    deployable = True
    requires_ground_truth = False

    def reset(self) -> None:
        pass

    def select_action(self, belief: Belief, ground_truth=None, steps_remaining: Optional[int] = None
                      ) -> Decision:  # pragma: no cover
        raise NotImplementedError

    def notify(self, action: int, observation: int) -> None:
        pass


class _HNode:
    __slots__ = ("n", "actions", "belief")

    def __init__(self, belief=None):
        self.n = 0
        self.actions = None   # list of _ANode once expanded
        self.belief = belief  # exact joint posterior b(r, q) at this simulated history


class _ANode:
    __slots__ = ("n", "q", "children")

    def __init__(self):
        self.n = 0
        self.q = 0.0
        self.children = {}


ROLLOUT_POLICIES = ("belief", "oracle", "random")
ROLLOUT_LABELS = {"belief": "belief (default)",
                  "oracle": "oracle (not deployable)",
                  "random": "random"}


class POMCPPlanner(Policy):
    """Receding-horizon POMCP (Silver & Veness, 2010) with an exact root belief.

    At every real decision the search tree is rebuilt from the EXACT current
    belief (the PO-UCT form of the algorithm, for which the paper states its
    convergence result).  Each simulation

    1. samples a hidden state s = (r, q) ~ b (a hypothesis, never the truth),
    2. descends the tree, choosing actions by UCB1
       ``argmax_a  Q(h,a) + c * sqrt(ln N(h) / N(h,a))``  (untried actions first,
       random tie-breaks),
    3. generates s', o, r with the generative model G(s, a),
    4. expands one new history node, evaluates it with the rollout policy,
    5. backs up the discounted return  R = r + gamma * R'  into Q(h, a).

    Movement actions branch on the observation ``None`` (or end the simulation
    on rescue); Scan branches on Cold / Warm / Hot.  The chosen real action is
    the root action with the highest estimated value.  With a finite budget
    this is an approximation - no global optimality is claimed.

    Information discipline
    ----------------------
    The sampled hidden state (r, q) is used ONLY by the generative model G to
    produce transitions, observations, rewards and terminal events.  Every
    action chosen inside the search - UCB actions in the tree and rollout
    actions beyond it - is a function of the simulated action/observation
    history alone: each tree node stores its exact posterior belief, and the
    default rollout policy updates that belief at every rollout step.

    Rollout policies
    ----------------
    ``"belief"`` (default, deployable) - belief-tracked rollout.  Starting
        from the leaf node's exact posterior, each rollout step (i) targets the
        currently most likely survivor cell, (ii) chooses the move with the
        lowest expected hazard-aware cost-to-go averaged over the robot-position
        belief (10% uniformly random moves), and (iii) updates the belief with
        the Bayes filter: predict with the movement model, then condition on
        "survivor not reached" (moves yield the None observation).  Because
        cells the robot has probably visited lose probability, the target moves
        on to the next candidate - a belief-driven search sweep.  The rollout
        never scans.  Its action choice reads neither the sampled robot cell r
        nor the sampled survivor cell q.
    ``"oracle"`` - fully observable rollout that walks to the SAMPLED survivor
        from the SAMPLED robot cell.  It uses hidden information, so it is a
        non-deployable comparison only (and it is optimistic: with a shallow
        tree the planner then over-values Scan, QMDP-like).
    ``"random"`` - uniform random actions (history-independent).

    Horizons
    --------
    The mission runner passes ``steps_remaining = H_mission - t``.  The tree
    depth is ``H_search = min(horizon, steps_remaining)`` and rollouts stop at
    total depth ``steps_remaining`` (depth already used inside the tree is
    counted), so no simulated reward is collected after the real deadline.

    Exploration constant: default c = 150, empirically tuned by the
    pre-declared rule "highest mean discounted return on development seeds
    (base seed 77, 40 episodes per value)" over c in {5, 20, 50, 100, 150, 300}.
    It does not maximise every metric (c = 100 had a higher success-rate point
    estimate).  Its scale is broadly consistent with Silver & Veness's
    reward-range guidance (c ~ R_hi - R_lo), but it was NOT computed with that
    procedure.  c = 20 was an earlier candidate, tested and not selected: with
    it the first lucky rollout tends to dominate the search.
    """

    name = "POMCP"

    def __init__(self, model: POMDPModel, n_simulations: int = 800, horizon: int = 10,
                 exploration: float = 150.0, rollout: str = "belief",
                 rollout_epsilon: float = 0.1, seed: Optional[int] = 0):
        if rollout not in ROLLOUT_POLICIES:
            raise ValueError(f"rollout must be one of {ROLLOUT_POLICIES}")
        if n_simulations < 1 or horizon < 1:
            raise ValueError("n_simulations and horizon must be positive")
        self.model = model
        self.n_simulations = int(n_simulations)
        self.horizon = int(horizon)
        self.c = float(exploration)
        self.rollout_kind = rollout
        self.rollout_epsilon = float(rollout_epsilon)
        self.gamma = model.gamma
        ss = np.random.SeedSequence(seed)
        g_tree, g_sim, g_pol = (np.random.default_rng(x) for x in ss.spawn(3))
        self.rng = g_tree                      # root sampling
        self.u = _UniformStream(g_tree)        # UCB / final-action tie-breaks
        self.u_sim = _UniformStream(g_sim)     # generative model (transitions, sensor noise)
        self.u_pol = _UniformStream(g_pol)     # rollout-policy randomisation (epsilon moves)
        n = model.n
        self._offdiag = 1.0 - np.eye(n)
        self._MT = [np.ascontiguousarray(model.move_matrix[a].T) for a in ACTIONS]
        self._oracle = [[model.greedy_move(r, q) for q in range(n)] for r in range(n)]
        # expected-cost tables for the belief rollout: C[target][a, r] =
        #   p*(ctg(next(r,a)) + hazard) + (1-p)*ctg(r)
        p = model.p_success
        self._cost_tables = []
        for t in range(n):
            ctg = model.cost_to_go(t)
            tab = np.zeros((len(MOVE_ACTIONS), n))
            for k, a in enumerate(MOVE_ACTIONS):
                for r in range(n):
                    j = model.next_cell[a][r]
                    pen = model.hazard_penalty if (j != r and model.hazard_mask[j] and j != t) else 0.0
                    tab[k, r] = p * (ctg[j] + pen) + (1 - p) * ctg[r]
            self._cost_tables.append(tab)
        self._limit = model.max_steps
        self._h_search = min(self.horizon, self._limit)
        self.max_depth_reached = -1  # deepest simulated step index of the last search (diagnostics)

    # -------------------------------------------- exact belief bookkeeping
    def _child_belief(self, joint: np.ndarray, a: int, o: int) -> Optional[np.ndarray]:
        """Bayes update for a non-terminal transition: predict, weight by Z(o|s'),
        condition on 'not rescued', normalise.  Identical to Belief.update."""
        post = (self._MT[a] @ joint) * (self.model.obs_lik[o] * self._offdiag)
        tot = post.sum()
        return post / tot if tot > 1e-300 else None

    def _advance_belief(self, joint: np.ndarray, a: int) -> Optional[np.ndarray]:
        """Belief update after a rollout MOVE (observation None, mission continues)."""
        post = (self._MT[a] @ joint) * self._offdiag
        tot = post.sum()
        return post / tot if tot > 1e-300 else None

    def rollout_action(self, joint: np.ndarray) -> int:
        """Belief-rollout policy: a function of the belief and public model only."""
        if self.u_pol() < self.rollout_epsilon:
            return MOVE_ACTIONS[int(self.u_pol() * 4)]
        target = int(np.argmax(joint.sum(axis=0)))
        scores = self._cost_tables[target] @ joint.sum(axis=1)
        return MOVE_ACTIONS[int(np.argmin(scores))]

    def _new_child(self, parent: _HNode, a: int, o: int) -> _HNode:
        if self.rollout_kind != "belief":
            return _HNode()
        joint = self._child_belief(parent.belief, a, o)
        return _HNode(joint if joint is not None else parent.belief)

    # -------------------------------------------- search
    def select_action(self, belief: Belief, ground_truth=None, steps_remaining: Optional[int] = None) -> Decision:
        m = self.model
        rem = m.max_steps if steps_remaining is None else int(steps_remaining)
        if rem < 1:
            raise ValueError("No decisions remain before the mission deadline.")
        self._limit = rem
        self._h_search = min(self.horizon, rem)
        self.max_depth_reached = -1
        flat = belief.joint.ravel()
        cdf = np.cumsum(flat)
        cdf[-1] = 1.0
        roots = np.searchsorted(cdf, self.rng.random(self.n_simulations), side="right")
        root = _HNode(belief.joint.copy() if self.rollout_kind == "belief" else None)
        for k in roots.tolist():
            r, q = divmod(int(k), m.n)
            if r == q:
                continue
            self._simulate(r, q, root, 0)
        q_values = {ACTION_NAMES[a]: float(root.actions[a].q) for a in ACTIONS}
        visits = {ACTION_NAMES[a]: int(root.actions[a].n) for a in ACTIONS}
        # final action: highest mean value among visited actions, random tie-break
        visited = [a for a in ACTIONS if root.actions[a].n > 0]
        top = max(root.actions[a].q for a in visited)
        ties = [a for a in visited if root.actions[a].q >= top - 1e-9]
        best = ties[0] if len(ties) == 1 else ties[int(self.u() * len(ties))]
        return Decision(best, {"q_values": q_values, "visits": visits,
                               "estimated_return": float(root.actions[best].q),
                               "simulations": self.n_simulations, "horizon": self._h_search,
                               "configured_horizon": self.horizon, "steps_remaining": rem,
                               "rollout": self.rollout_kind,
                               "tree_nodes": self._count_nodes(root)})

    def _ucb_action(self, node: _HNode) -> int:
        unvisited = [a for a in ACTIONS if node.actions[a].n == 0]
        if unvisited:
            return unvisited[int(self.u() * len(unvisited))]
        log_n = math.log(node.n)
        best, best_val = [], -math.inf
        for a in ACTIONS:
            an = node.actions[a]
            val = an.q + self.c * math.sqrt(log_n / an.n)
            if val > best_val + 1e-12:
                best, best_val = [a], val
            elif val >= best_val - 1e-12:
                best.append(a)
        return best[0] if len(best) == 1 else best[int(self.u() * len(best))]

    def _simulate(self, r: int, q: int, node: _HNode, depth: int) -> float:
        if depth >= self._h_search:  # tree depth exhausted: evaluate by rollout (to the deadline)
            return self._rollout(r, q, depth, node)
        if node.actions is None:  # expansion: add the node, evaluate by rollout
            node.actions = [_ANode() for _ in ACTIONS]
            node.n += 1
            return self._rollout(r, q, depth, node)
        a = self._ucb_action(node)  # history-based: uses node statistics only
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth
        r2, o, rew, term = self.model.generative_step(r, q, a, self.u_sim(), self.u_sim())
        an = node.actions[a]
        if term:  # rescue: terminal, no future value
            ret = rew
        else:
            child = an.children.get(o)
            if child is None:
                child = self._new_child(node, a, o)
                an.children[o] = child
            ret = rew + self.gamma * self._simulate(r2, q, child, depth + 1)
        node.n += 1
        an.n += 1
        an.q += (ret - an.q) / an.n
        return ret

    def _rollout(self, r: int, q: int, depth: int, node: _HNode, trace: Optional[list] = None) -> float:
        """Rollout from a leaf at total depth ``depth`` until the mission deadline.

        (r, q) feed ONLY the generative model.  Action choice:
        belief -> self.rollout_action(belief); oracle -> path to sampled q;
        random -> uniform.  ``trace`` (tests) collects the chosen actions.
        """
        total, disc = 0.0, 1.0
        m, kind, limit = self.model, self.rollout_kind, self._limit
        joint = node.belief if kind == "belief" else None
        while depth < limit:
            if kind == "belief":
                a = self.rollout_action(joint)
            elif kind == "oracle":  # NON-DEPLOYABLE comparison: reads the sampled hidden state
                a = MOVE_ACTIONS[int(self.u_pol() * 4)] if self.u_pol() < self.rollout_epsilon else self._oracle[r][q]
            else:
                a = ACTIONS[int(self.u_pol() * 5)]
            if trace is not None:
                trace.append(a)
            if depth > self.max_depth_reached:
                self.max_depth_reached = depth
            r, _, rew, term = m.generative_step(r, q, a, self.u_sim(), self.u_sim())
            total += disc * rew
            if term:
                break
            if kind == "belief":
                if a == SCAN:  # never chosen by the belief rollout; kept for completeness
                    break
                joint = self._advance_belief(joint, a)
                if joint is None:
                    break
            disc *= self.gamma
            depth += 1
        return total

    @staticmethod
    def _count_nodes(root: _HNode) -> int:
        stack, count = [root], 0
        while stack:
            h = stack.pop()
            count += 1
            if h.actions:
                for an in h.actions:
                    stack.extend(an.children.values())
        return count


class RandomPolicy(Policy):
    """Uniformly random action (reproducible seed)."""

    name = "Random"

    def __init__(self, model: POMDPModel, seed: Optional[int] = 0):
        self.model = model
        self.rng = np.random.default_rng(seed)

    def select_action(self, belief, ground_truth=None, steps_remaining=None):
        return Decision(int(self.rng.integers(len(ACTIONS))), {})


class RiskAwareHeuristicPolicy(Policy):
    """Belief-driven heuristic (no planning, no ground truth).

    1. If the most likely survivor cell has probability below ``scan_threshold``
       and fewer than ``max_consecutive_scans`` scans were made since the last
       move, Scan.
    2. Otherwise target the most likely survivor cell and choose the move that
       minimises the expected hazard-aware cost-to-go, averaging over the
       robot-position belief and the 80/20 movement model.  Entering a hazard
       is charged at its full penalty, so safe detours are preferred.
    """

    name = "Risk-aware heuristic"

    def __init__(self, model: POMDPModel, scan_threshold: float = 0.5,
                 max_consecutive_scans: int = 2, seed: Optional[int] = 0):
        self.model = model
        self.scan_threshold = float(scan_threshold)
        self.max_consecutive_scans = int(max_consecutive_scans)
        self.reset()

    def reset(self):
        self._consecutive_scans = 0

    def select_action(self, belief, ground_truth=None, steps_remaining=None):
        m = self.model
        surv = belief.survivor_marginal().ravel()
        target = int(np.argmax(surv))
        conf = float(surv[target])
        rob = belief.robot_marginal().ravel()
        info = {"target": m.cell(target), "confidence": conf}
        if conf < self.scan_threshold and self._consecutive_scans < self.max_consecutive_scans:
            self._consecutive_scans += 1
            return Decision(SCAN, {**info, "reason": "survivor confidence below threshold"})
        ctg = m.cost_to_go(target)
        scores = {}
        for a in MOVE_ACTIONS:
            exp = 0.0
            for r in np.nonzero(rob > 1e-12)[0]:
                j = m.next_cell[a][r]
                pen = m.hazard_penalty if (j != r and m.hazard_mask[j] and j != target) else 0.0
                exp += rob[r] * (m.p_success * (ctg[j] + pen) + (1 - m.p_success) * ctg[r])
            scores[a] = exp
        a = min(MOVE_ACTIONS, key=lambda x: (round(scores[x], 9), x))
        self._consecutive_scans = 0
        return Decision(a, {**info, "reason": "move toward most likely survivor cell",
                            "expected_cost_to_go": {ACTION_NAMES[k]: float(v) for k, v in scores.items()}})


def value_iteration(model: POMDPModel, tol: float = 1e-10, max_iter: int = 10000):
    """Solve the fully observable MDP over (r, q).  Returns Q[a, r, q] and V[r, q]
    (V = 0 at terminal states r == q).  Used by the clairvoyant reference policy
    and by QMDP."""
    n, g, p = model.n, model.gamma, model.p_success
    V = np.zeros((n, n))
    Q = np.zeros((len(ACTIONS), n, n))
    nxt = np.array(model.next_cell)  # [a, r]
    haz = model.hazard_mask
    eye = np.eye(n, dtype=bool)
    for _ in range(max_iter):
        for a in ACTIONS:
            if a == SCAN:
                Q[a] = -model.scan_cost + g * V
                continue
            j = nxt[a]  # [r]
            moved = j != np.arange(n)
            pen = np.where(moved & haz[j], -model.hazard_penalty, 0.0)[:, None]
            reach = (j[:, None] == np.arange(n)[None, :])  # r' == q
            succ_val = pen + np.where(reach, model.rescue_reward, g * V[j, :])
            fail_val = g * V
            Q[a] = -model.move_cost + p * succ_val + (1 - p) * fail_val
            # moves blocked by the boundary: deterministic stay
            Q[a][~moved, :] = -model.move_cost + g * V[~moved, :]
        newV = Q.max(axis=0)
        newV[eye] = 0.0
        newV[:, haz] = 0.0
        if np.max(np.abs(newV - V)) < tol:
            V = newV
            break
        V = newV
    Q[:, eye] = 0.0
    return Q, V


class QMDPPolicy(Policy):
    """QMDP (Littman et al., 1995): argmax_a sum_s b(s) Q_MDP(s, a).
    Assumes all uncertainty vanishes after one step, so it never values
    information-gathering (Scan)."""

    name = "QMDP"

    def __init__(self, model: POMDPModel, seed: Optional[int] = 0):
        self.model = model
        self.Q, _ = value_iteration(model)

    def select_action(self, belief, ground_truth=None, steps_remaining=None):
        vals = np.tensordot(self.Q, belief.joint, axes=([1, 2], [0, 1]))
        a = int(np.argmax(vals))
        return Decision(a, {"q_values": {ACTION_NAMES[k]: float(vals[k]) for k in ACTIONS}})


class ClairvoyantPolicy(Policy):
    """IDEALISED REFERENCE - NOT DEPLOYABLE.  Knows the true robot and survivor
    positions and follows the optimal fully observable MDP policy obtained by
    value iteration.  It shows the value of perfect information."""

    name = "Clairvoyant (reference)"
    deployable = False
    requires_ground_truth = True

    def __init__(self, model: POMDPModel, seed: Optional[int] = 0):
        self.model = model
        self.Q, self.V = value_iteration(model)

    def select_action(self, belief, ground_truth=None, steps_remaining=None):
        if ground_truth is None:
            raise ValueError("Clairvoyant reference needs ground truth.")
        r = self.model.idx(ground_truth["robot"])
        q = self.model.idx(ground_truth["survivor"])
        a = int(np.argmax(self.Q[:, r, q]))
        return Decision(a, {"q_values": {ACTION_NAMES[k]: float(self.Q[k, r, q]) for k in ACTIONS}})


@dataclass(frozen=True)
class PolicySpec:
    """Picklable policy description so evaluation can run in worker processes."""

    kind: str
    params: Tuple[Tuple[str, object], ...] = ()

    @property
    def label(self) -> str:
        return {"pomcp": "POMCP", "random": "Random", "heuristic": "Risk-aware heuristic",
                "qmdp": "QMDP", "clairvoyant": "Clairvoyant (reference)"}[self.kind]

    @staticmethod
    def of(kind: str, **params) -> "PolicySpec":
        return PolicySpec(kind, tuple(sorted(params.items())))


def make_policy(spec: PolicySpec, model: POMDPModel, seed: int) -> Policy:
    kw = dict(spec.params)
    cls = {"pomcp": POMCPPlanner, "random": RandomPolicy, "heuristic": RiskAwareHeuristicPolicy,
           "qmdp": QMDPPolicy, "clairvoyant": ClairvoyantPolicy}[spec.kind]
    return cls(model, seed=seed, **kw)


# --------------------------------------------------------------------------------------
# Mission runner (decision loop)
# --------------------------------------------------------------------------------------
class MissionRunner:
    """Executes the decision loop:
    belief -> policy -> action -> simulator -> observation -> Bayes update.

    The environment is LOCKED while the policy decides, so the hidden survivor
    coordinates cannot be read (HiddenStateAccessError).  Only the clairvoyant
    reference policy explicitly receives ground truth, and it is labelled as
    non-deployable.
    """

    def __init__(self, config: RescueConfig, policy: Policy, env_seed: int = 0,
                 survivor: Optional[Tuple[int, int]] = None):
        self.config = config
        self.model = config.public_model()
        self.policy = policy
        self.env = RescueBotEnv(config, seed=env_seed)
        self.env.reset(env_seed, survivor=survivor)
        self.policy.reset()
        self.belief = self.model.initial_belief()
        self.records: List[Dict] = []
        self._record_initial()

    def _truth(self):
        return self.env.get_state_for_evaluation()

    def _record_initial(self):
        truth = self._truth()
        self.records.append(self._snapshot(step=0, action=None, decision=None, result=None, truth=truth))

    def _snapshot(self, step, action, decision, result, truth):
        b = self.belief
        rob = b.robot_marginal()
        surv = b.survivor_marginal()
        true_r = self.model.idx(truth["robot"])
        true_q = self.model.idx(truth["survivor"])
        map_r, map_r_p = b.map_robot()
        map_q, map_q_p = b.map_survivor()
        rob_flat = rob.ravel()
        exp_loc_err = float(sum(rob_flat[i] * self.model.dist[i, true_r] for i in range(self.model.n)))
        rec = {
            "step": step,
            "action": ACTION_NAMES[action] if action is not None else None,
            "outcome": result.outcome if result else "Mission start",
            "observation": result.observation if result else "None",
            "reward": result.reward if result else 0.0,
            "move_cost": result.reward_components["move_cost"] if result else 0.0,
            "hazard": result.reward_components["hazard"] if result else 0.0,
            "rescue": result.reward_components["rescue"] if result else 0.0,
            "scan_cost": result.reward_components["scan_cost"] if result else 0.0,
            "cumulative_reward": result.cumulative_reward if result else 0.0,
            "done": result.done if result else False,
            "rescued": result.rescued if result else False,
            "robot_belief": rob.copy(),
            "survivor_belief": surv.copy(),
            "joint_entropy": b.joint_entropy(),
            "survivor_entropy": b.survivor_entropy(),
            "robot_entropy": b.robot_entropy(),
            "est_robot": map_r, "est_robot_prob": map_r_p,
            "est_survivor": map_q, "est_survivor_prob": map_q_p,
            "decision_info": dict(decision.info) if decision else {},
            # evaluation-only ground truth (never given to deployable policies)
            "true_robot": truth["robot"], "true_survivor": truth["survivor"],
            "p_true_survivor": float(surv.ravel()[true_q]),
            "p_true_robot": float(rob_flat[true_r]),
            "expected_localisation_error": exp_loc_err,
            "hazard_entry": bool(result and result.reward_components["hazard"] < 0),
        }
        return rec

    @property
    def done(self) -> bool:
        return self.env.is_terminal()

    def step(self) -> Dict:
        if self.done:
            raise RuntimeError("Mission already finished.")
        gt = None
        if self.policy.requires_ground_truth:
            gt = self._truth()
        self.env.lock()
        try:
            decision = self.policy.select_action(self.belief.copy(), ground_truth=gt,
                                                 steps_remaining=self.config.max_steps - self.env.t)
        finally:
            self.env.unlock()
        a = decision.action
        result = self.env.step(a)
        o = {v: k for k, v in OBS_NAMES.items()}[result.observation]
        self.belief = self.belief.update(a, o, rescued=result.rescued)
        self.policy.notify(a, o)
        rec = self._snapshot(step=result.step, action=a, decision=decision, result=result, truth=self._truth())
        self.records.append(rec)
        return rec

    def run(self) -> List[Dict]:
        while not self.done:
            self.step()
        return self.records

    def history_frame(self) -> pd.DataFrame:
        cols = ["step", "action", "outcome", "observation", "reward", "move_cost", "hazard", "rescue",
                "scan_cost", "cumulative_reward", "joint_entropy", "survivor_entropy", "robot_entropy",
                "est_robot", "est_robot_prob", "est_survivor", "est_survivor_prob", "true_robot",
                "p_true_survivor", "expected_localisation_error", "hazard_entry", "rescued", "done"]
        df = pd.DataFrame([{k: r[k] for k in cols} for r in self.records])
        for c in ("est_robot", "est_survivor", "true_robot"):
            df[c] = df[c].map(fmt_cell)
        return df


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def _episode_seeds(base_seed: int, i: int) -> Tuple[int, int, int]:
    ss = np.random.SeedSequence([int(base_seed), int(i)])
    a, b, c = ss.spawn(3)
    return (int(a.generate_state(1)[0]), int(b.generate_state(1)[0]), int(c.generate_state(1)[0]))


def _survivor_for_episode(config: RescueConfig, seed: int, mode: str) -> Tuple[int, int]:
    if mode == "fixed":
        return config.survivor
    model = config.public_model()
    cands = [model.cell(q) for q in model.safe_cells if q != model.start_idx]
    return cands[int(np.random.default_rng(seed).integers(len(cands)))]


def run_episode(config: RescueConfig, spec: PolicySpec, episode: int, base_seed: int = 2026,
                survivor_mode: str = "random", keep_trace: bool = True) -> Dict:
    env_seed, pol_seed, surv_seed = _episode_seeds(base_seed, episode)
    survivor = _survivor_for_episode(config, surv_seed, survivor_mode)
    policy = make_policy(spec, config.public_model(), pol_seed)
    runner = MissionRunner(config, policy, env_seed=env_seed, survivor=survivor)
    runner.run()
    recs = runner.records
    acts = [r["action"] for r in recs[1:]]
    final = recs[-1]
    pre = recs[-2] if final["rescued"] and len(recs) >= 2 else final
    out = {
        "policy": spec.label, "episode": episode, "survivor": fmt_cell(survivor),
        "success": bool(final["rescued"]), "steps": len(acts),
        "total_reward": float(final["cumulative_reward"]),
        "discounted_return": float(sum(config.gamma ** t * r["reward"] for t, r in enumerate(recs[1:]))),
        "scans": acts.count("Scan"), "moves": len(acts) - acts.count("Scan"),
        "hazard_entries": int(sum(r["hazard_entry"] for r in recs)),
        "steps_to_rescue": len(acts) if final["rescued"] else np.nan,
        # survivor estimate held immediately BEFORE the rescue (or at the end of a failed mission)
        "map_correct": bool(pre["est_survivor"] == pre["true_survivor"]),
        "p_true_survivor_final": float(pre["p_true_survivor"]),
        "final_survivor_entropy": float(pre["survivor_entropy"]),
        "mean_robot_entropy": float(np.mean([r["robot_entropy"] for r in recs])),
        "mean_localisation_error": float(np.mean([r["expected_localisation_error"] for r in recs])),
    }
    if keep_trace:
        out["trace_survivor_entropy"] = [r["survivor_entropy"] for r in recs]
        out["trace_robot_entropy"] = [r["robot_entropy"] for r in recs]
        out["trace_p_true"] = [r["p_true_survivor"] for r in recs]
        out["actions"] = acts
    return out


def _run_one(args):
    return run_episode(*args)


def evaluate_policies(config: RescueConfig, specs: Sequence[PolicySpec], n_episodes: int = 50,
                      base_seed: int = 2026, survivor_mode: str = "random", n_jobs: int = 1,
                      progress=None) -> pd.DataFrame:
    """Run every policy on the SAME episode seeds (common random numbers)."""
    jobs = [(config, s, i, base_seed, survivor_mode, True) for s in specs for i in range(n_episodes)]
    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_run_one, jobs, chunksize=4))
    else:
        rows = []
        for k, j in enumerate(jobs):
            rows.append(_run_one(j))
            if progress:
                progress((k + 1) / len(jobs))
    return pd.DataFrame(rows)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # guard against floating-point round-off at p = 0 or p = 1
    return (min(lo, p), max(hi, p))


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean (returns NaNs for empty input)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    if np.all(x == x[0]):
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarise_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol, g in df.groupby("policy", sort=False):
        n = len(g)
        k = int(g["success"].sum())
        lo, hi = wilson_interval(k, n)
        rlo, rhi = bootstrap_ci(g["total_reward"])
        dlo, dhi = bootstrap_ci(g["discounted_return"])
        slo, shi = bootstrap_ci(g["steps"])
        succ_steps = g.loc[g["success"], "steps_to_rescue"]
        rows.append({
            "Policy": pol, "Episodes": n,
            "Success rate": k / n, "Success 95% CI": f"[{lo:.2f}, {hi:.2f}]",
            "success_lo": lo, "success_hi": hi,
            "Mean reward": g["total_reward"].mean(), "Reward 95% CI": f"[{rlo:.1f}, {rhi:.1f}]",
            "reward_lo": rlo, "reward_hi": rhi,
            "Mean discounted return": g["discounted_return"].mean(), "disc_lo": dlo, "disc_hi": dhi,
            "Mean steps": g["steps"].mean(), "Steps 95% CI": f"[{slo:.1f}, {shi:.1f}]",
            "steps_lo": slo, "steps_hi": shi,
            "Steps to rescue (successes)": succ_steps.mean() if len(succ_steps) else float("nan"),
            "Mean scans": g["scans"].mean(), "Mean hazard entries": g["hazard_entries"].mean(),
            "Hazard-entry rate": (g["hazard_entries"] > 0).mean(),
            "Survivor MAP accuracy": g["map_correct"].mean(),
            "Mean P(true survivor) at end": g["p_true_survivor_final"].mean(),
            "Mean robot entropy (bits)": g["mean_robot_entropy"].mean(),
            "Final survivor entropy (bits)": g["final_survivor_entropy"].mean(),
            "Mean localisation error (cells)": g["mean_localisation_error"].mean(),
        })
    return pd.DataFrame(rows)


def decision_consistency(model: POMDPModel, beliefs: Sequence[Belief], n_repeats: int = 5,
                         seed: int = 0, **pomcp_kwargs) -> pd.DataFrame:
    """Re-plan from the same belief with different planner seeds and report how
    often the modal action is chosen (1.0 = perfectly consistent).  ``beliefs[t]``
    is the belief before real decision t, so steps_remaining = max_steps - t."""
    rows = []
    for t, b in enumerate(beliefs):
        acts = [POMCPPlanner(model, seed=seed + 1000 * t + k, **pomcp_kwargs)
                .select_action(b, steps_remaining=model.max_steps - t).action for k in range(n_repeats)]
        vals, counts = np.unique(acts, return_counts=True)
        rows.append({"step": t, "modal_action": ACTION_NAMES[int(vals[np.argmax(counts)])],
                     "agreement": counts.max() / n_repeats,
                     "actions": ", ".join(ACTION_NAMES[a] for a in acts)})
    return pd.DataFrame(rows)


def sensitivity_sweep(base: RescueConfig, param: str, values: Sequence, specs: Sequence[PolicySpec],
                      n_episodes: int = 30, base_seed: int = 7, survivor_mode: str = "random",
                      n_jobs: int = 1) -> pd.DataFrame:
    """One-factor-at-a-time sweep.  ``param`` is a RescueConfig field, the
    special key ``sensor_noise`` (see :func:`sensor_with_noise`), or a POMCP
    parameter prefixed with ``pomcp.`` (e.g. ``pomcp.n_simulations``)."""
    frames = []
    for v in values:
        cfg = base
        sp = list(specs)
        if param == "sensor_noise":
            cfg = base.with_(sensor_table=sensor_with_noise(v))
        elif param.startswith("pomcp."):
            key = param.split(".", 1)[1]
            sp = [PolicySpec.of(s.kind, **{**dict(s.params), key: v}) if s.kind == "pomcp" else s for s in specs]
        else:
            cfg = base.with_(**{param: v})
        df = evaluate_policies(cfg, sp, n_episodes, base_seed, survivor_mode, n_jobs=n_jobs)
        s = summarise_evaluation(df)
        s.insert(0, "value", v)
        s.insert(0, "parameter", param)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


# c = 150: empirically tuned - highest mean DISCOUNTED return (the planner's objective) on development
# seeds 77 among c in {5, 20, 50, 100, 150, 300}; not the maximiser of every metric and not computed
# from the R_hi - R_lo formula.  Evaluation uses separate seeds (2026).  See notebook Secs. 11 and 14.
DEFAULT_POMCP = PolicySpec.of("pomcp", n_simulations=800, horizon=10, exploration=150.0, rollout="belief")
DEFAULT_SPECS = (DEFAULT_POMCP, PolicySpec.of("heuristic"), PolicySpec.of("qmdp"), PolicySpec.of("random"))
