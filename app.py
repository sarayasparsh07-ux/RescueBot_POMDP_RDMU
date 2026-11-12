"""
RescueBot // Autonomous Mission Control
=======================================
Streamlit command-centre dashboard for the RescueBot POMDP.

Run with:   streamlit run app.py

Every model computation (transitions, observations, rewards, Bayesian belief
updates, POMCP and the baseline policies) is imported from rescuebot_core.py,
the same module used by RescueBot_POMDP_Analysis.ipynb.  This file contains
only presentation and interaction logic.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import rescuebot_core as rb
from rescuebot_core import (ACTION_NAMES, OBS_NAMES, Belief, MissionRunner, PolicySpec, RescueConfig,
                            evaluate_policies, fmt_cell, make_policy, sensor_with_noise,
                            summarise_evaluation)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

# ------------------------------------------------------------------------------------------
# Palette & figure styling
# ------------------------------------------------------------------------------------------
C = {"navy": "#080F1C", "slate": "#101B2D", "slate2": "#15233A", "cyan": "#00D5FF", "blue": "#3478F6",
     "green": "#48E58B", "amber": "#FFB547", "coral": "#FF5364", "ice": "#EEF5FF", "muted": "#9BAEC5",
     "grid": "rgba(155,174,197,0.14)"}
POLICY_COLORS = {"POMCP": C["cyan"], "Risk-aware heuristic": C["blue"], "QMDP": C["amber"],
                 "Random": C["muted"], "Clairvoyant (reference)": C["green"]}
OBS_COLORS = {"Cold": C["blue"], "Warm": C["amber"], "Hot": C["coral"]}
PROB_SCALE = [[0.0, "#0C1729"], [0.2, "#0F2E57"], [0.45, "#1E5FC4"], [0.7, "#00B4E0"], [1.0, "#C9F6FF"]]
ROBOT_SCALE = [[0.0, "#0C1729"], [0.3, "#2B2350"], [0.6, "#7A4FD1"], [1.0, "#E3D2FF"]]
POLICY_KINDS = {"POMCP": "pomcp", "Risk-aware heuristic": "heuristic", "QMDP": "qmdp",
                "Random": "random", "Clairvoyant (reference)": "clairvoyant"}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

ICONS = {
    "status": '<path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"/><path d="M12 6v6l4 2"/>',
    "step": '<path d="M4 4h16v16H4z"/><path d="M9 4v16M15 4v16"/>',
    "reward": '<path d="M12 2l3 7h7l-5.5 4.5 2 7.5L12 17l-6.5 4 2-7.5L2 9h7z"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/>',
    "entropy": '<path d="M3 12c3-6 6 6 9 0s6 6 9 0"/>',
    "actions": '<path d="M5 12h14M13 6l6 6-6 6"/>',
}


def icon(name: str, size: int = 15) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def style_fig(fig: go.Figure, title: str = "", x_title: str = "", y_title: str = "", height: int = 380,
              legend_bottom: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=15, color=C["ice"]), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,28,0.6)",
        font=dict(family="Inter, sans-serif", size=12.5, color=C["ice"]), height=height,
        margin=dict(l=64, r=24, t=56 if title else 24, b=105 if legend_bottom else 72),
        legend=(dict(orientation="h", yanchor="top", y=-0.26, x=0, font=dict(size=11.5), bgcolor="rgba(0,0,0,0)")
                if legend_bottom else dict(font=dict(size=11.5), bgcolor="rgba(0,0,0,0)")),
        hoverlabel=dict(bgcolor=C["slate2"], bordercolor=C["cyan"], font=dict(color=C["ice"], family="Inter")),
    )
    fig.update_xaxes(title=dict(text=x_title, font=dict(size=12.5, color=C["muted"])), gridcolor=C["grid"],
                     zerolinecolor=C["grid"], tickfont=dict(size=11.5, color=C["muted"]), automargin=True)
    fig.update_yaxes(title=dict(text=y_title, font=dict(size=12.5, color=C["muted"])), gridcolor=C["grid"],
                     zerolinecolor="rgba(155,174,197,0.35)", tickfont=dict(size=11.5, color=C["muted"]), automargin=True)
    return fig


def show(fig: go.Figure, key: str | None = None):
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def callout(text: str, kind: str = ""):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------
# Page & CSS
# ------------------------------------------------------------------------------------------
st.set_page_config(page_title="RescueBot // Mission Control", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")
css_path = BASE / "assets" / "dashboard.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

CELLS = [(r, c) for r in range(4) for c in range(4)]
CELL_LABEL = {cell: fmt_cell(cell) for cell in CELLS}
LABEL_CELL = {v: k for k, v in CELL_LABEL.items()}

# ------------------------------------------------------------------------------------------
# Sidebar: branding, status, configuration
# ------------------------------------------------------------------------------------------
ROBOT_SVG = """<svg width="38" height="38" viewBox="0 0 48 48" fill="none">
<rect x="9" y="14" width="30" height="22" rx="6" stroke="#00D5FF" stroke-width="2.4"/>
<circle cx="18" cy="25" r="3.2" fill="#00D5FF"/><circle cx="30" cy="25" r="3.2" fill="#00D5FF"/>
<path d="M24 14V8" stroke="#00D5FF" stroke-width="2.4" stroke-linecap="round"/><circle cx="24" cy="6.5" r="2.2" fill="#FF5364"/>
<path d="M14 40h20" stroke="#3478F6" stroke-width="2.4" stroke-linecap="round"/>
<path d="M4 22v6M44 22v6" stroke="#3478F6" stroke-width="2.4" stroke-linecap="round"/></svg>"""

with st.sidebar:
    st.markdown(f'<div class="brand">{ROBOT_SVG}<div><div class="brand-name">RESCUEBOT</div>'
                f'<div class="brand-sub">Autonomous SAR · POMDP</div></div></div>', unsafe_allow_html=True)
    status_slot = st.empty()

    st.markdown('<div class="side-label">Command sections</div>', unsafe_allow_html=True)
    page = st.radio("Section", ["Mission Overview", "Live Robot Simulation", "Belief & Sensor Intelligence",
                                "Policy & Decision Analytics", "Performance & Mission Report"],
                    label_visibility="collapsed", key="page")

    st.markdown('<div class="side-label">Environment settings</div>', unsafe_allow_html=True)
    with st.expander("Grid, hazards & dynamics", expanded=False):
        start_lbl = st.selectbox("Robot start cell", list(LABEL_CELL), index=0, key="start")
        hazard_lbls = st.multiselect("Hazard cells (known map)", list(LABEL_CELL),
                                     default=["(2,2)", "(3,3)"], key="hazards")
        surv_options = [l for l in LABEL_CELL if l not in hazard_lbls and l != start_lbl]
        default_surv = surv_options.index("(4,4)") if "(4,4)" in surv_options else len(surv_options) - 1
        survivor_lbl = st.selectbox("Hidden survivor cell (simulator only)", surv_options, index=default_surv,
                                    key="survivor", help="Stored only inside the simulator. The planner never receives it.")
        p_success = st.slider("Movement success probability", 0.5, 1.0, 0.80, 0.05, key="p_success")
        noise = st.slider("Sensor noise level", 0.0, 2.0, 1.0, 0.25, key="noise",
                          help="0 = noiseless, 1 = baseline table, 2 = uninformative")
        max_steps = st.slider("Maximum mission steps", 10, 60, 30, 5, key="max_steps")
        scan_cost = st.slider("Scan energy cost (optional extension)", 0.0, 5.0, 0.0, 0.5, key="scan_cost")

    st.markdown('<div class="side-label">Planning configuration</div>', unsafe_allow_html=True)
    policy_name = st.selectbox("Decision policy", list(POLICY_KINDS), index=0, key="policy")
    with st.expander("POMCP parameters", expanded=policy_name == "POMCP"):
        n_sims = st.select_slider("Simulations per decision", [100, 200, 400, 800, 1500, 3000], value=800, key="sims")
        depth = st.slider("Search depth (tree horizon)", 2, 20, 10, 1, key="depth")
        c_ucb = st.select_slider("UCB exploration constant c", [5, 20, 50, 100, 150, 300], value=150, key="c_ucb",
                                 help="Default 150: empirically tuned, the highest mean discounted return in a development-seed sweep "
                                      "(c = 5, 20, 50, 100, 150, 300). It does not maximise every metric. Its scale is broadly "
                                      "consistent with the paper's reward-range guidance, but it was not computed from that formula.")
        rollout = st.selectbox("Rollout policy", list(rb.ROLLOUT_POLICIES), index=0, key="rollout",
                               format_func=lambda k: rb.ROLLOUT_LABELS[k],
                               help="belief = belief-tracked, deployable (default). oracle = reads the sampled hidden state: "
                                    "comparison only, NOT deployable. random = uniform.")
    st.markdown('<div class="side-label">Simulation seeds</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    env_seed = c1.number_input("Environment", 0, 99999, 2026, 1, key="env_seed")
    pol_seed = c2.number_input("Planner", 0, 99999, 7, 1, key="pol_seed")
    st.markdown('<div class="side-label">Instructor</div>', unsafe_allow_html=True)
    reveal = st.toggle("Reveal ground truth", value=False, key="reveal",
                       help="Overlays the simulator's true robot and survivor positions for post-run analysis. "
                            "It is display-only and never reaches the policy.")
    speed = st.slider("Autoplay delay (s)", 0.0, 1.5, 0.5, 0.1, key="speed")

# ------------------------------------------------------------------------------------------
# Configuration → mission state
# ------------------------------------------------------------------------------------------
try:
    cfg = RescueConfig(start=LABEL_CELL[start_lbl], survivor=LABEL_CELL[survivor_lbl],
                       hazards=tuple(LABEL_CELL[h] for h in hazard_lbls), p_success=float(p_success),
                       sensor_table=sensor_with_noise(float(noise)), max_steps=int(max_steps),
                       scan_cost=float(scan_cost))
    cfg_error = None
except ValueError as exc:  # invalid combination chosen in the sidebar
    cfg, cfg_error = RescueConfig(), str(exc)

policy_params = {"n_simulations": int(n_sims), "horizon": int(depth), "exploration": float(c_ucb),
                 "rollout": rollout} if policy_name == "POMCP" else {}
spec = PolicySpec.of(POLICY_KINDS[policy_name], **policy_params)
mission_key = json.dumps({"cfg": cfg.to_dict(), "spec": [spec.kind, list(map(list, spec.params))],
                          "env_seed": int(env_seed), "pol_seed": int(pol_seed)}, default=str)


def new_runner() -> MissionRunner:
    pol = make_policy(spec, cfg.public_model(), int(pol_seed))
    return MissionRunner(cfg, pol, env_seed=int(env_seed))


ss = st.session_state
if "runner" not in ss or ss.get("mission_key") != mission_key:
    changed = "runner" in ss
    ss.runner = new_runner()
    ss.mission_key = mission_key
    ss.playing = False
    ss.plan_time = []
    if changed:
        st.toast("Configuration changed: mission reset with the new settings.", icon="🔄")
runner: MissionRunner = ss.runner
model = runner.model


def do_step():
    if not runner.done:
        t0 = time.time()
        runner.step()
        ss.plan_time.append(time.time() - t0)


# ------------------------------------------------------------------------------------------
# Derived mission state
# ------------------------------------------------------------------------------------------
recs = runner.records
last = recs[-1]
done = runner.done
rescued = bool(last["rescued"])
step_no = last["step"]
disclose = reveal or done
belief = runner.belief
surv_grid = last["survivor_belief"]
rob_grid = last["robot_belief"]
map_q, map_q_p = last["est_survivor"], last["est_survivor_prob"]
map_r, map_r_p = last["est_robot"], last["est_robot_prob"]
n_scans = sum(r["action"] == "Scan" for r in recs[1:])
n_moves = step_no - n_scans

if done and rescued:
    status, chip = "RESCUE COMPLETE", "chip-success"
elif done:
    status, chip = "MISSION FAILED", "chip-failed"
elif ss.playing:
    status, chip = "AUTONOMOUS RUN", "chip-active"
elif step_no > 0:
    status, chip = "PAUSED", "chip-warn"
else:
    status, chip = "STANDBY", "chip-standby"
status_slot.markdown(f'<span class="chip {chip}"><span class="dot"></span>{status}</span>'
                     f'<div style="color:#9BAEC5;font-size:.78rem;margin-top:.45rem">Policy: <b style="color:#EEF5FF">{policy_name}</b>'
                     f' · step {step_no}/{cfg.max_steps}</div>', unsafe_allow_html=True)


def confidence_text() -> str:
    if done and rescued:
        return (f"Survivor reached at {fmt_cell(runner.env.get_state_for_evaluation()['survivor'])} after {step_no} decisions. "
                "Contact tells the robot it is in the same cell as the survivor, but not which cell: its movements were never "
                f"observed, so the post-contact belief still spreads over {int((np.array(surv_grid) > 0.01).sum())} cells.")
    if done:
        return (f"Mission ended at the {cfg.max_steps}-step limit without contact. Final best estimate "
                f"{fmt_cell(map_q)} held only {map_q_p:.0%} of the probability mass.")
    if map_q_p >= 0.6:
        lvl = "HIGH confidence"
    elif map_q_p >= 0.3:
        lvl = "MODERATE confidence"
    else:
        lvl = "LOW confidence"
    prior = 1 / max(1, len(model.safe_cells) - 1)
    return (f"{lvl} in the survivor's location. The most likely cell is <b>{fmt_cell(map_q)}</b> at <b>{map_q_p:.0%}</b> "
            f"(uniform prior {prior:.0%}); survivor-belief entropy is <b>{last['survivor_entropy']:.2f} bits</b>. "
            f"RescueBot believes it is at <b>{fmt_cell(map_r)}</b> with {map_r_p:.0%} certainty.")


# ------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------
def cell_xy(cell):
    return cell[1] + 1, cell[0] + 1


def grid_map(height: int = 560, overlays=("belief", "path", "est", "target", "heat")) -> go.Figure:
    """Tactical 4×4 map: survivor-belief intensity + robot estimate + overlays."""
    R, Cn = cfg.rows, cfg.cols
    z = np.array(surv_grid, dtype=float).copy()
    zmask = z.copy()
    for h in cfg.hazards:
        zmask[h] = np.nan
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=zmask, x=list(range(1, Cn + 1)), y=list(range(1, R + 1)), zmin=0, zmax=max(0.25, float(np.nanmax(zmask))),
        colorscale=PROB_SCALE, xgap=4, ygap=4, showscale=True, hoverongaps=False,
        customdata=np.vectorize(lambda v: f"{v:.1%}")(z),
        hovertemplate="Cell (%{y},%{x})<br>P(survivor) = %{customdata}<extra></extra>",
        colorbar=dict(title=dict(text="P(survivor<br>in cell)", font=dict(size=11, color=C["muted"])), tickformat=".0%",
                      thickness=12, len=0.75, tickfont=dict(size=10.5, color=C["muted"]), outlinewidth=0),
        name="Survivor belief"))
    vmax = max(0.25, float(np.nanmax(zmask)))
    for (r, c) in CELLS:
        x, y = c + 1, r + 1
        if (r, c) in cfg.hazards:
            fig.add_shape(type="rect", x0=x - .46, x1=x + .46, y0=y - .46, y1=y + .46, line=dict(color=C["coral"], width=2),
                          fillcolor="rgba(255,83,100,0.16)", layer="below")
            fig.add_annotation(x=x, y=y, text="⚠<br><b>−50</b>", showarrow=False,
                               font=dict(color=C["coral"], size=12))
        else:
            v = z[r, c]
            fig.add_annotation(x=x, y=y + 0.33, text=f"<b>{v:.0%}</b>" if v >= 0.005 else "0%", showarrow=False,
                               font=dict(family="JetBrains Mono, monospace", size=15,
                                         color=C["navy"] if v > 0.62 * vmax else C["ice"]))
        fig.add_annotation(x=x - 0.36, y=y - 0.36, text=f"{r+1},{c+1}", showarrow=False,
                           font=dict(size=9.5, color="rgba(155,174,197,0.75)"), xanchor="left")
    if cfg.hazards:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Hazard cell (−50 on entry)",
                                 marker=dict(symbol="square-open", size=13, color=C["coral"], line=dict(width=2))))
    est_path = [r["est_robot"] for r in recs]
    if "path" in overlays and len(recs) > 1:
        xs, ys = zip(*[cell_xy(p) for p in est_path])
        fig.add_trace(go.Scatter(x=np.array(xs) - 0.12, y=np.array(ys) - 0.12, mode="lines+markers", name="Estimated path (belief)",
                                 line=dict(color=C["cyan"], width=2, dash="dot"), marker=dict(size=5), hoverinfo="skip"))
    if disclose and len(recs) > 1:
        tp = [r["true_robot"] for r in recs]
        xs, ys = zip(*[cell_xy(p) for p in tp])
        fig.add_trace(go.Scatter(x=np.array(xs) + 0.12, y=np.array(ys) + 0.12, mode="lines+markers",
                                 name="Actual path [ground truth]", line=dict(color=C["ice"], width=2.4),
                                 marker=dict(size=5, symbol="diamond"), hoverinfo="skip"))
    if "target" in overlays and not (done and rescued):
        x, y = cell_xy(map_q)
        fig.add_trace(go.Scatter(x=[x], y=[y - 0.05], mode="markers", name=f"Destination estimate {fmt_cell(map_q)}",
                                 marker=dict(symbol="circle-cross-open", size=40, color=C["amber"], line=dict(width=2.5)),
                                 hovertemplate=f"Most likely survivor cell {fmt_cell(map_q)} ({map_q_p:.0%})<extra></extra>"))
    # robot estimate (+ last heat reading ring)
    x, y = cell_xy(map_r)
    y = y - 0.08
    if "heat" in overlays and last["action"] == "Scan":
        col = OBS_COLORS[last["observation"]]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=f"Last heat reading: {last['observation']}",
                                 marker=dict(symbol="circle-open", size=54, color=col, line=dict(width=3.5)),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", name=f"Robot estimate {fmt_cell(map_r)} ({map_r_p:.0%})",
                             marker=dict(symbol="square", size=26, color=C["cyan"], line=dict(color=C["navy"], width=2)),
                             text=["🤖"], textfont=dict(size=15), textposition="middle center",
                             hovertemplate=f"Most likely robot cell {fmt_cell(map_r)} ({map_r_p:.0%})<extra></extra>"))
    if disclose:
        truth = runner.env.get_state_for_evaluation()
        tx, ty = cell_xy(truth["robot"])
        fig.add_trace(go.Scatter(x=[tx + 0.3], y=[ty + 0.3], mode="markers", name="Actual robot [ground truth]",
                                 marker=dict(symbol="diamond", size=15, color=C["ice"], line=dict(color=C["navy"], width=1.5))))
        sx, sy = cell_xy(truth["survivor"])
        fig.add_trace(go.Scatter(x=[sx + 0.3], y=[sy - 0.3], mode="markers", name="Survivor [ground truth]",
                                 marker=dict(symbol="star", size=22, color=C["green"], line=dict(color=C["navy"], width=1.5))))
    fig.update_xaxes(range=[0.5, Cn + 0.5], tickvals=list(range(1, Cn + 1)), ticktext=[f"Col {i}" for i in range(1, Cn + 1)],
                     showgrid=False, zeroline=False, side="top")
    fig.update_yaxes(range=[R + 0.5, 0.5], tickvals=list(range(1, R + 1)), ticktext=[f"Row {i}" for i in range(1, R + 1)],
                     showgrid=False, zeroline=False)
    style_fig(fig, "", "", "", height=height)
    fig.update_layout(margin=dict(l=60, r=20, t=40, b=20), legend=dict(orientation="h", y=-0.03, yanchor="top", x=0,
                                                                        font=dict(size=11)))
    return fig


def belief_heatmap(grid, title, scale=PROB_SCALE, label="Probability", height=360, truth=None, key_cells=True):
    robot_map = scale is ROBOT_SCALE  # the robot CAN be inside a hazard cell; the survivor cannot
    z = np.array(grid, dtype=float).copy()
    zm = z.copy()
    if not robot_map:
        for h in cfg.hazards:
            zm[h] = np.nan
    vmax = max(0.2, float(np.nanmax(zm)))
    fig = go.Figure(go.Heatmap(z=zm, x=[1, 2, 3, 4], y=[1, 2, 3, 4], zmin=0, zmax=vmax, colorscale=scale, xgap=3, ygap=3,
                               hoverongaps=False, customdata=np.vectorize(lambda v: f"{v:.1%}")(z),
                               hovertemplate="Cell (%{y},%{x}): %{customdata}<extra></extra>",
                               colorbar=dict(title=dict(text=label, font=dict(size=11, color=C["muted"])), tickformat=".0%",
                                             thickness=10, tickfont=dict(size=10, color=C["muted"]), outlinewidth=0)))
    for (r, c) in CELLS:
        if (r, c) in cfg.hazards and not robot_map:
            fig.add_annotation(x=c + 1, y=r + 1, text="⚠", showarrow=False, font=dict(color=C["coral"], size=15))
            fig.add_shape(type="rect", x0=c + .55, x1=c + 1.45, y0=r + .55, y1=r + 1.45,
                          line=dict(color=C["coral"], width=1.5), fillcolor="rgba(255,83,100,0.12)")
        else:
            if (r, c) in cfg.hazards:
                fig.add_shape(type="rect", x0=c + .55, x1=c + 1.45, y0=r + .55, y1=r + 1.45,
                              line=dict(color=C["coral"], width=1.5, dash="dot"))
                fig.add_annotation(x=c + 1.32, y=r + 0.68, text="⚠", showarrow=False, font=dict(color=C["coral"], size=11))
            v = z[r, c]
            fig.add_annotation(x=c + 1, y=r + 1, text=(f"{v:.0%}" if v >= 0.095 else f"{v:.1%}") if v >= 0.0005 else "0%", showarrow=False,
                               font=dict(family="JetBrains Mono, monospace", size=12.5,
                                         color=C["navy"] if v > 0.62 * vmax else C["ice"]))
    if truth is not None:
        fig.add_shape(type="rect", x0=truth[1] + .55, x1=truth[1] + 1.45, y0=truth[0] + .55, y1=truth[0] + 1.45,
                      line=dict(color=C["green"], width=2.5, dash="dash"))
    fig.update_xaxes(tickvals=[1, 2, 3, 4], ticktext=[f"Col {i}" for i in range(1, 5)], showgrid=False, side="bottom")
    fig.update_yaxes(tickvals=[1, 2, 3, 4], ticktext=[f"Row {i}" for i in range(1, 5)], autorange="reversed", showgrid=False)
    style_fig(fig, title, "", "", height=height, legend_bottom=False)
    fig.update_layout(margin=dict(l=55, r=10, t=52, b=36))
    return fig


def kpi(label, value, sub, icon_name, accent):
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-head">{icon(icon_name)}{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def kpi_row():
    acc = {"chip-success": C["green"], "chip-failed": C["coral"], "chip-active": C["cyan"],
           "chip-warn": C["amber"], "chip-standby": C["muted"]}[chip]
    rew = last["cumulative_reward"]
    html = '<div class="kpi-grid">' + "".join([
        kpi("Status", {"RESCUE COMPLETE": "Rescued", "MISSION FAILED": "Failed", "AUTONOMOUS RUN": "Running",
                       "PAUSED": "Paused", "STANDBY": "Standby"}[status], f"{policy_name}", "status", acc),
        kpi("Step", f"{step_no} / {cfg.max_steps}", f"{cfg.max_steps - step_no} steps remaining", "step", C["blue"]),
        kpi("Reward", f"{rew:+.0f} pts", f"last action {last['reward']:+.0f} pts", "reward",
            C["green"] if rew > 0 else (C["coral"] if rew < 0 else C["muted"])),
        (kpi("Confidence", "Found", f"survivor at {fmt_cell(runner.env.get_state_for_evaluation()['survivor'])}", "target", C["green"])
         if (done and rescued) else kpi("Confidence", f"{map_q_p:.0%}", f"most likely cell {fmt_cell(map_q)}", "target", C["amber"])),
        kpi("Entropy", f"{last['survivor_entropy']:.2f} bits",
            f"survivor · robot {last['robot_entropy']:.2f}", "entropy", C["cyan"]),
        kpi("Actions", f"{step_no}", f"{n_moves} moves · {n_scans} scans", "actions", C["blue"]),
    ]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def history_table() -> pd.DataFrame:
    rows = []
    for r in recs[1:]:
        rows.append({"Step": r["step"], "Action": r["action"], "Observation": r["observation"],
                     "Simulator outcome (unobserved)": r["outcome"], "Reward": r["reward"],
                     "Cumulative": r["cumulative_reward"], "Est. robot": fmt_cell(r["est_robot"]),
                     "Best survivor cell": f"{fmt_cell(r['est_survivor'])} · {r['est_survivor_prob']:.0%}",
                     "H(survivor) bits": r["survivor_entropy"]})
    df = pd.DataFrame(rows)
    if disclose and len(df):
        df.insert(4, "True robot [GT]", [fmt_cell(r["true_robot"]) for r in recs[1:]])
    return df


# ------------------------------------------------------------------------------------------
# Evaluation data (precomputed by the notebook, or re-run here with caching)
# ------------------------------------------------------------------------------------------
def _results_stamp() -> str:
    return "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in sorted(RESULTS.glob("*")) if p.is_file()) if RESULTS.exists() else ""


@st.cache_data(show_spinner=False)
def load_benchmark(stamp: str):
    out = {}
    try:
        out["summary"] = pd.read_csv(RESULTS / "evaluation_summary.csv")
        out["episodes"] = pd.read_json(RESULTS / "evaluation_episodes.json")
        out["fixed"] = pd.read_csv(RESULTS / "evaluation_fixed_summary.csv")
        out["sens"] = pd.read_csv(RESULTS / "sensitivity_results.csv")
        if (RESULTS / "evaluation_costed_scan_summary.csv").exists():
            out["costed"] = pd.read_csv(RESULTS / "evaluation_costed_scan_summary.csv")
    except (FileNotFoundError, ValueError):
        pass
    return out


@st.cache_data(show_spinner=False, max_entries=8)
def quick_eval(cfg_json: str, n_eps: int, sims: int, depth_: int, c_: float, roll: str):
    c = RescueConfig(**{k: (tuple(map(tuple, v)) if k in ("hazards", "sensor_table") else (tuple(v) if isinstance(v, list) else v))
                        for k, v in json.loads(cfg_json).items()})
    specs = [PolicySpec.of("pomcp", n_simulations=sims, horizon=depth_, exploration=c_, rollout=roll),
             PolicySpec.of("heuristic"), PolicySpec.of("qmdp"), PolicySpec.of("random"), PolicySpec.of("clairvoyant")]
    df = evaluate_policies(c, specs, n_episodes=n_eps, base_seed=2026, survivor_mode="random",
                           n_jobs=min(2, os.cpu_count() or 1))
    return df, summarise_evaluation(df)


bench = load_benchmark(_results_stamp())


SHORT = {"POMCP": "POMCP", "Risk-aware heuristic": "Heuristic", "QMDP": "QMDP", "Random": "Random",
         "Clairvoyant (reference)": "Clairvoyant<br>(reference)"}


def interval_bars(summary: pd.DataFrame, value, lo, hi, title, y_title, fmt=".0%", height=360, yrange=None, na_text="n/a"):
    order = [p for p in POLICY_COLORS if p in set(summary["Policy"])]
    s = summary.set_index("Policy").loc[order]
    xs = [SHORT[p] for p in order]
    vals = s[value].astype(float)
    top = np.where(vals >= 0, np.fmax(s[hi].astype(float), vals), np.fmin(s[lo].astype(float), vals))
    span = float(np.nanmax(np.r_[s[hi].astype(float), vals, 0.0]) - np.nanmin(np.r_[s[lo].astype(float), vals, 0.0])) or 1.0
    top = np.where(vals >= 0, top + 0.02 * span, top - 0.03 * span)  # keep labels clear of the error-bar caps
    fig = go.Figure(go.Bar(
        x=xs, y=vals.fillna(0), marker_color=[POLICY_COLORS[p] for p in order],
        marker_line=dict(color=[C["navy"]] * len(order), width=1), opacity=0.92, customdata=order,
        error_y=dict(type="data", symmetric=False, array=np.clip((s[hi] - vals).fillna(0), 0, None),
                     arrayminus=np.clip((vals - s[lo]).fillna(0), 0, None), color=C["ice"], thickness=1.4, width=6),
        hovertemplate="%{customdata}<br>%{y:" + fmt + "}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=np.nan_to_num(top, nan=0.0), mode="text",
                             text=[na_text if np.isnan(v) else format(v, fmt) for v in vals],
                             textposition=["top center" if (np.isnan(v) or v >= 0) else "bottom center" for v in vals],
                             textfont=dict(color=C["ice"], size=12, family="JetBrains Mono, monospace"), hoverinfo="skip"))
    style_fig(fig, title, "", y_title, height=height, legend_bottom=False)
    fig.update_xaxes(tickangle=0)
    if fmt.endswith("%"):
        fig.update_yaxes(tickformat=".0%", range=yrange or [0, 1.18])
    elif yrange:
        fig.update_yaxes(range=yrange)
    fig.update_layout(showlegend=False, margin=dict(l=64, r=20, t=56, b=50))
    return fig


# ==========================================================================================
# HEADER
# ==========================================================================================
st.markdown('<div class="hero"><div class="hero-kicker">Search &amp; rescue operations · partially observable environment</div>'
            '<div class="hero-title">RESCUEBOT <span>//</span> AUTONOMOUS MISSION CONTROL</div>'
            '<div class="hero-sub">A rescue robot reasons under uncertainty in a foggy 4×4 grid: it keeps an exact Bayesian belief over its own '
            'position and the survivor\'s, and chooses actions with Monte Carlo tree search (POMCP). Shared model: '
            '<code>rescuebot_core.py</code>.</div></div>', unsafe_allow_html=True)
if cfg_error:
    callout(f"⚠ Invalid environment settings ({cfg_error}). Default configuration in use.", "crit")
if policy_name == "POMCP" and rollout == "oracle":
    callout("POMCP is using the <b>oracle rollout</b>: its simulations read the <i>sampled</i> hidden robot and survivor cells. "
            "This is a non-deployable comparison setting; the default belief-tracked rollout uses the belief only.", "warn")
if policy_name == "Clairvoyant (reference)":
    callout("The <b>clairvoyant reference</b> reads the true robot and survivor positions. It is an idealised benchmark "
            "showing the value of perfect information, <b>not a deployable policy</b>.", "warn")


# ==========================================================================================
# 1. MISSION OVERVIEW
# ==========================================================================================
def page_overview():
    kpi_row()
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section("Tactical grid · survivor-belief intensity")
        show(grid_map(height=600), key="ov_map")
        callout(confidence_text(), "ok" if (done and rescued) else ("crit" if done else ""))
    with right:
        section("Mission brief")
        truth_line = ""
        if disclose:
            t = runner.env.get_state_for_evaluation()
            truth_line = (f'<div class="k">True robot <span class="tag tag-gt">GT</span></div><div class="v">{fmt_cell(t["robot"])}</div>'
                          f'<div class="k">True survivor <span class="tag tag-gt">GT</span></div><div class="v">{fmt_cell(t["survivor"])}</div>')
        st.markdown(
            f'<div class="panel"><h4>Configuration</h4><div class="kv">'
            f'<div class="k">Start cell</div><div class="v">{fmt_cell(cfg.start)}</div>'
            f'<div class="k">Hazards</div><div class="v">{", ".join(fmt_cell(h) for h in cfg.hazards) or "none"}</div>'
            f'<div class="k">Move success p</div><div class="v">{cfg.p_success:.2f}</div>'
            f'<div class="k">Sensor noise level</div><div class="v">{noise:.2f}</div>'
            f'<div class="k">Rewards</div><div class="v">+100 / −10 / −50</div>'
            f'<div class="k">Scan cost</div><div class="v">{cfg.scan_cost:.1f}</div>'
            f'<div class="k">Discount γ</div><div class="v">{cfg.gamma}</div>{truth_line}</div></div>', unsafe_allow_html=True)
        la = last["action"] or "—"
        obs = last["observation"]
        obs_html = f'<span class="obs-{obs.lower()}">{obs}</span>' if obs in OBS_COLORS else obs
        st.markdown(
            f'<div class="panel"><h4>Latest decision</h4><div class="kv">'
            f'<div class="k">Action</div><div class="v">{la}</div>'
            f'<div class="k">Heat reading</div><div class="v">{obs_html}</div>'
            f'<div class="k">Immediate reward</div><div class="v">{last["reward"]:+.0f} pts</div>'
            f'<div class="k">Robot estimate</div><div class="v">{fmt_cell(map_r)} · {map_r_p:.0%}</div>'
            f'<div class="k">Destination estimate</div><div class="v">{fmt_cell(map_q)} · {map_q_p:.0%}</div></div></div>',
            unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Step", icon=":material/skip_next:", disabled=done, width="stretch", key="ov_next", help="Execute next step"):
            do_step(); st.rerun()
        if b2.button("Reset", icon=":material/restart_alt:", width="stretch", key="ov_reset"):
            ss.runner = new_runner(); ss.playing = False; ss.plan_time = []; st.rerun()
    l2, r2 = st.columns([1, 1], gap="large")
    with l2:
        section("Robot localisation")
        show(belief_heatmap(rob_grid, "P(robot in cell)", ROBOT_SCALE, "P(robot)",
                            height=400, truth=runner.env.get_state_for_evaluation()["robot"] if disclose else None), key="ov_rob")
    with r2:
        section("Map legend & coordinate convention")
        st.markdown("""
* **Cell colour / percentage**: the robot's current probability that the survivor is in that cell (survivor marginal).
* **🤖 cyan square**: the robot's *most likely* position (from its belief), not necessarily where it really is.
* **Amber crosshair**: destination estimate, the most likely survivor cell.
* **Coloured ring**: last heat reading (blue = Cold, amber = Warm, red = Hot).
* **Dotted cyan line**: path of the robot's position estimate. **White line / ◆**: true path and position, **[ground truth]** only.
* **★ green**: true survivor, shown only in instructor mode or after the mission ends.
* Coordinates are one-based (row, column) from (1,1) top-left to (4,4) bottom-right; the code uses zero-based indices internally.
* **Robot map (left)**: probability of each robot cell. Without odometry, unobserved debris failures spread this distribution.
""")


# ==========================================================================================
# 2. LIVE ROBOT SIMULATION
# ==========================================================================================
def page_live():
    section("Mission controls")
    cols = st.columns(5)
    start = cols[0].button("Start", help="Start mission: autonomous play, one decision per autoplay interval", icon=":material/play_arrow:", type="primary", disabled=done or ss.playing,
                           width="stretch", key="lv_start")
    pause = cols[1].button("Pause", icon=":material/pause:", disabled=not ss.playing, width="stretch", key="lv_pause")
    nxt = cols[2].button("Step", help="Execute next step", icon=":material/skip_next:", disabled=done, width="stretch", key="lv_next")
    run_all = cols[3].button("Run all", help="Run complete mission (to rescue or the step limit)", icon=":material/fast_forward:", disabled=done, width="stretch", key="lv_all")
    reset = cols[4].button("Reset", help="Reset environment and belief", icon=":material/restart_alt:", width="stretch", key="lv_reset")
    if start:
        ss.playing = True; st.rerun()
    if pause:
        ss.playing = False; st.rerun()
    if nxt:
        ss.playing = False; do_step(); st.rerun()
    if run_all:
        ss.playing = False
        with st.spinner("Running mission to completion…"):
            while not runner.done:
                do_step()
        st.rerun()
    if reset:
        ss.runner = new_runner(); ss.playing = False; ss.plan_time = []; st.rerun()
    params = (f"sims {n_sims} · depth {depth} · c {c_ucb} · rollout {rollout}" if policy_name == "POMCP" else "no search parameters")
    st.markdown(f'<span class="tag">policy: {policy_name}</span><span class="tag">{params}</span>'
                f'<span class="tag">p = {p_success:.2f}</span><span class="tag">max steps {max_steps}</span>'
                f'<span class="tag">seeds env {env_seed} / planner {pol_seed}</span>'
                f'<span class="muted" style="color:#9BAEC5;font-size:.8rem"> · change parameters in the sidebar (the mission resets automatically)</span>',
                unsafe_allow_html=True)

    kpi_row()
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        section("Tactical grid")
        show(grid_map(height=540), key="lv_map")
    with right:
        section("Latest step")
        obs = last["observation"]
        st.markdown(
            f'<div class="panel"><h4>Step {step_no} telemetry</h4><div class="kv">'
            f'<div class="k">Selected action</div><div class="v">{last["action"] or "—"}</div>'
            f'<div class="k">Simulator outcome</div><div class="v">{last["outcome"]}</div>'
            f'<div class="k">Heat observation</div><div class="v"><span class="obs-{obs.lower()}">{obs}</span></div>'
            f'<div class="k">Immediate reward</div><div class="v">{last["reward"]:+.0f}</div>'
            f'<div class="k">Cumulative reward</div><div class="v">{last["cumulative_reward"]:+.0f}</div></div>'
            f'<p class="muted" style="margin-top:.5rem">The simulator outcome (moved / failed / blocked) is shown to you, '
            f'but the robot does not observe it.</p></div>', unsafe_allow_html=True)
        pt = ss.plan_time[-1] * 1000 if ss.plan_time else 0
        info = last["decision_info"]
        extra = ""
        if "q_values" in info and last["action"]:
            extra = (f'<div class="k">Planner estimate</div><div class="v">{info.get("estimated_return", max(info["q_values"].values())):+.1f}</div>')
        st.markdown(
            f'<div class="panel"><h4>Decision engine</h4><div class="kv">'
            f'<div class="k">Policy</div><div class="v">{policy_name}</div>'
            f'<div class="k">Decision time</div><div class="v">{pt:.0f} ms</div>{extra}'
            f'<div class="k">Hazard entries</div><div class="v">{sum(r["hazard_entry"] for r in recs)}</div></div>'
            f'<p class="muted" style="margin-top:.5rem">Ground-truth overlay: {"ON (display only)" if disclose else "OFF"}</p></div>',
            unsafe_allow_html=True)
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        section("Updated survivor belief")
        show(belief_heatmap(surv_grid, "P(survivor in cell)", height=400,
                            truth=runner.env.get_state_for_evaluation()["survivor"] if disclose else None), key="lv_surv")
    with b2:
        section("Updated robot-position belief")
        show(belief_heatmap(rob_grid, "P(robot in cell)", ROBOT_SCALE, "P(robot)", height=400,
                            truth=runner.env.get_state_for_evaluation()["robot"] if disclose else None), key="lv_rob")

    section("Action history")
    df = history_table()
    if len(df):
        st.dataframe(df.iloc[::-1], width="stretch", hide_index=True, height=min(420, 38 + 35 * len(df)),
                     column_config={"Reward": st.column_config.NumberColumn(format="%+d"),
                                    "Cumulative": st.column_config.NumberColumn(format="%+d"),
                                    "H(survivor) bits": st.column_config.NumberColumn(format="%.2f")})
    else:
        callout("No actions yet. Press <b>Start mission</b> for autonomous play, or <b>Execute next step</b> to advance one decision at a time.")

    if ss.playing and not runner.done:
        time.sleep(float(speed))
        do_step()
        if runner.done:
            ss.playing = False
        st.rerun()


# ==========================================================================================
# 3. BELIEF & SENSOR INTELLIGENCE
# ==========================================================================================
def expected_scan_entropy(joint: np.ndarray, noise_level: float):
    """Expected posterior survivor entropy after one Scan from the current belief (uses the core updater)."""
    c = cfg.with_(sensor_table=sensor_with_noise(noise_level))
    m = c.public_model()
    b = Belief(m, joint)
    exp_h = 0.0
    for o, p_o in zip(rb.HEAT_OBS, b.predicted_obs_distribution().values()):
        if p_o > 1e-12:
            exp_h += p_o * b.update(rb.SCAN, o).survivor_entropy()
    return exp_h


def page_belief():
    a, b_ = st.columns(2, gap="medium")
    truth = runner.env.get_state_for_evaluation() if disclose else None
    with a:
        section("A · Survivor probability heatmap")
        show(belief_heatmap(surv_grid, f"P(survivor in cell) at step {step_no}", height=400,
                            truth=truth["survivor"] if truth else None), key="bl_surv")
    with b_:
        section("B · Robot localisation heatmap")
        show(belief_heatmap(rob_grid, f"P(robot in cell) at step {step_no}", ROBOT_SCALE, "P(robot)", height=400,
                            truth=truth["robot"] if truth else None), key="bl_rob")
    callout("The two maps are <b>marginals</b> of one joint belief b(r, q): the survivor map sums over robot positions and the "
            "robot map sums over survivor positions. The hidden state is a single (robot, survivor) pair; the belief is the robot's "
            "probabilistic estimate of it." + (" Dashed green boxes mark the true cells [ground truth]." if truth else ""))

    section("C · Belief evolution")
    if step_no == 0:
        callout("Run at least one step to scrub through the belief history.")
    else:
        k = st.slider("Inspect belief after decision step", 0, step_no, step_no, key="bl_scrub")
        rk = recs[k]
        desc = "Initial belief" if k == 0 else f"After step {k}: {rk['action']} → {rk['observation'] if rk['action']=='Scan' else rk['outcome']}"
        e1, e2 = st.columns(2, gap="medium")
        with e1:
            show(belief_heatmap(rk["survivor_belief"], f"Survivor · {desc}", height=360,
                                truth=rk["true_survivor"] if disclose else None), key="bl_ev_s")
        with e2:
            show(belief_heatmap(rk["robot_belief"], f"Robot · {desc}", ROBOT_SCALE, "P(robot)", height=360,
                                truth=rk["true_robot"] if disclose else None), key="bl_ev_r")

    section("D · Entropy analysis")
    steps = [r["step"] for r in recs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=[r["survivor_entropy"] for r in recs], mode="lines+markers", name="Survivor Marginal Entropy",
                             line=dict(color=C["cyan"], width=3)))
    fig.add_trace(go.Scatter(x=steps, y=[r["robot_entropy"] for r in recs], mode="lines+markers", name="Robot Marginal Entropy",
                             line=dict(color="#A98BFF", width=2.4)))
    fig.add_trace(go.Scatter(x=steps, y=[r["joint_entropy"] for r in recs], mode="lines", name="Joint Belief Entropy",
                             line=dict(color=C["muted"], width=1.6, dash="dot")))
    for r in recs[1:]:
        if r["action"] == "Scan":
            fig.add_trace(go.Scatter(x=[r["step"]], y=[r["survivor_entropy"]], mode="markers", showlegend=False,
                                     marker=dict(color=OBS_COLORS[r["observation"]], size=12, line=dict(color=C["navy"], width=1)),
                                     hovertemplate=f"Scan → {r['observation']}<extra></extra>"))
    style_fig(fig, "Belief entropy over time (coloured dots = scan readings)", "Decision Step", "Belief Entropy (bits)", height=360)
    fig.update_xaxes(dtick=1 if step_no <= 15 else 2, range=[-0.3, max(step_no, 1) + 0.3])
    fig.update_yaxes(rangemode="tozero")
    show(fig, key="bl_ent")
    callout("Decreasing entropy generally indicates reduced uncertainty. Scans normally lower the survivor entropy. Unobserved "
            "movement raises the robot-position entropy, because the robot cannot tell a successful move from a debris failure.")

    section("E · Sensor intelligence")
    s1, s2 = st.columns([1, 1.15], gap="medium")
    with s1:
        scans = [r for r in recs[1:] if r["action"] == "Scan"]
        counts = {o: sum(r["observation"] == o for r in scans) for o in ("Cold", "Warm", "Hot")}
        fig = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), marker_color=[OBS_COLORS[o] for o in counts],
                               text=list(counts.values()), textposition="outside"))
        style_fig(fig, f"Observation history ({len(scans)} scans this mission)", "Heat reading", "Number of readings",
                  height=330, legend_bottom=False)
        top = max(3, max(counts.values()) * 1.3)
        fig.update_yaxes(range=[0, top], dtick=1 if top <= 8 else 2)
        show(fig, key="bl_obs")
        if scans:
            seq = " ".join(f'<span class="obs-{r["observation"].lower()}">●</span>' for r in scans)
            st.markdown(f'<div class="panel"><h4>Reading sequence</h4><p style="font-size:1.25rem;letter-spacing:.2em">{seq}</p>'
                        f'<p class="muted">Blue Cold · amber Warm · red Hot, in order of acquisition</p></div>', unsafe_allow_html=True)
    with s2:
        tab = np.array(cfg.sensor_table)
        e0, e1 = cfg.sensor_band_edges
        bands = [f"d ≤ {e0}", f"{e0+1} ≤ d ≤ {e1}", f"d ≥ {e1+1}"]
        fig = go.Figure()
        for k, o in enumerate(("Cold", "Warm", "Hot")):
            fig.add_trace(go.Bar(x=bands, y=tab[:, k], name=o, marker_color=OBS_COLORS[o],
                                 text=[f"{v:.0%}" for v in tab[:, k]], textposition="outside"))
        style_fig(fig, f"Observation likelihood Z(o | distance) · noise level {noise:.2f}",
                  "Manhattan distance between robot and survivor (cells)", "Probability of reading", height=360)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
        fig.update_layout(barmode="group")
        show(fig, key="bl_lik")
    levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    if done:
        callout("The <b>expected information gain</b> chart is shown while a mission is in progress: it evaluates the value of "
                "scanning <i>now</i>, from the current belief. Reset or start a new mission to see it.")
    else:
        cur_h = last["survivor_entropy"]
        exp = [expected_scan_entropy(belief.joint, lv) for lv in levels]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=levels, y=[cur_h - e for e in exp], mode="lines+markers", name="Expected information gain of one Scan",
                                 line=dict(color=C["cyan"], width=3), marker=dict(size=9),
                                 text=[f"{cur_h - e:.2f} bits" for e in exp], textposition="top center"))
        fig.add_vline(x=noise, line=dict(color=C["amber"], dash="dash"), annotation_text="current sensor",
                      annotation_font_color=C["amber"])
        style_fig(fig, "Sensor noise vs. information gained by one Scan now",
                  "Sensor noise level (0 = perfect · 1 = baseline · 2 = useless)", "Expected reduction in survivor entropy (bits)",
                  height=340)
        show(fig, key="bl_noise")
        callout("Expected information gain = H(current survivor belief) − Σₒ P(o | b) · H(belief after o). It is computed with the "
                "same Bayes updater the robot uses. Noisier sensors buy less certainty per scan, so more scans (or more moves) are "
                "needed before the robot can commit to a destination.")


# ==========================================================================================
# 4. POLICY & DECISION ANALYTICS
# ==========================================================================================
def page_policy():
    info = last["decision_info"]
    section("Latest decision · action-value estimates")
    if not last["action"]:
        callout("No decision yet. Execute at least one step to inspect the planner's reasoning.")
    elif "q_values" not in info:
        callout(f"The <b>{policy_name}</b> policy does not compute action values. Its latest choice was <b>{last['action']}</b>"
                + (f" ({info.get('reason')})." if info.get("reason") else ".")
                + " Select <b>POMCP</b> (or QMDP) in the sidebar to see value estimates.")
    else:
        q = info["q_values"]
        acts = list(q)
        chosen = last["action"]
        cols_ = [C["cyan"] if a == chosen else "rgba(52,120,246,0.55)" for a in acts]
        fig = go.Figure(go.Bar(x=acts, y=[q[a] for a in acts], marker_color=cols_,
                               text=[f"{q[a]:+.1f}" for a in acts], textposition="outside",
                               customdata=[info.get("visits", {}).get(a, 0) for a in acts],
                               hovertemplate="%{x}: %{y:+.2f} pts<br>visits %{customdata}<extra></extra>"))
        ymin, ymax = min(0, min(q.values())), max(0, max(q.values()))
        pad = 0.18 * (ymax - ymin + 1)
        style_fig(fig, f"Estimated value of each action at step {step_no} (chosen: {chosen})", "Available Action",
                  "Estimated Expected Return (reward points)", height=400, legend_bottom=False)
        fig.update_yaxes(range=[ymin - pad, ymax + pad])
        fig.update_layout(showlegend=False)
        k1, k2 = st.columns([1.6, 1], gap="large")
        with k1:
            show(fig, key="pl_q")
        with k2:
            visits = info.get("visits")
            vis_html = "".join(f'<div class="k">N(root, {a})</div><div class="v">{visits[a]}</div>' for a in acts) if visits else ""
            st.markdown(
                f'<div class="panel"><h4>Decision record</h4><div class="kv">'
                f'<div class="k">Action selected</div><div class="v">{chosen}</div>'
                f'<div class="k">Estimated policy return</div><div class="v">{info.get("estimated_return", q[chosen]):+.2f}</div>'
                f'<div class="k">Planning simulations</div><div class="v">{info.get("simulations", "—")}</div>'
                f'<div class="k">Search depth used</div><div class="v">{info.get("horizon", "—")} (of {info.get("configured_horizon", "—")})</div>'
                f'<div class="k">Steps remaining</div><div class="v">{info.get("steps_remaining", "—")}</div>'
                f'<div class="k">Rollout</div><div class="v">{info.get("rollout", "—")}</div>'
                f'<div class="k">Tree nodes</div><div class="v">{info.get("tree_nodes", "—")}</div>{vis_html}</div></div>',
                unsafe_allow_html=True)
        callout("These bars are <b>estimates of expected discounted return</b> from the planner's simulations, not rewards "
                "the robot has received. Actual rewards are in the reward-progression chart below.", "warn")

    hist_q = [(r["step"], r["decision_info"]["q_values"]) for r in recs[1:] if "q_values" in r["decision_info"]]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section("Action-value trajectories")
        if hist_q:
            fig = go.Figure()
            pal = {"Up": C["blue"], "Down": "#A98BFF", "Left": C["amber"], "Right": C["green"], "Scan": C["coral"]}
            for a in ACTION_NAMES.values():
                fig.add_trace(go.Scatter(x=[s for s, _ in hist_q], y=[qq[a] for _, qq in hist_q], mode="lines+markers",
                                         name=a, line=dict(color=pal[a], width=2)))
            style_fig(fig, "Q(b, a) estimates at each decision", "Decision step", "Estimated Expected Return (reward points)", height=360)
            show(fig, key="pl_qtraj")
        else:
            callout("Action-value history appears after POMCP or QMDP decisions.")
    with c2:
        section("Reward progression")
        steps = [r["step"] for r in recs]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=steps[1:], y=[r["reward"] for r in recs[1:]], name="Immediate reward",
                             marker_color=[C["green"] if r["reward"] > 0 else (C["coral"] if r["reward"] < -10 else "rgba(155,174,197,0.6)")
                                           for r in recs[1:]]))
        fig.add_trace(go.Scatter(x=steps, y=[r["cumulative_reward"] for r in recs], name="Cumulative reward",
                                 mode="lines+markers", line=dict(color=C["cyan"], width=3, shape="hv")))
        style_fig(fig, "Actual rewards received this mission", "Decision step", "Reward (points)", height=360)
        show(fig, key="pl_rew")

    section("Policy comparison across evaluation episodes")
    src = st.radio("Evaluation source", ["Notebook benchmark (default scenario, 100 episodes)",
                                         "Costed-scan extension (scan cost = 2, 50 episodes)",
                                         "Re-run with current sidebar settings"],
                   horizontal=True, key="pl_src", label_visibility="collapsed")
    summary = None
    if src.startswith("Costed"):
        if "costed" in bench:
            summary = bench["costed"]
            st.markdown('<span class="tag">source: results/evaluation_costed_scan_summary.csv</span>'
                        '<span class="tag">extension: each scan costs 2 points; assignment model is unchanged</span>',
                        unsafe_allow_html=True)
        else:
            callout("Costed-scan results not found. Run the notebook to generate them.", "warn")
    elif src.startswith("Notebook"):
        if "summary" in bench:
            summary = bench["summary"]
            st.markdown('<span class="tag">source: results/evaluation_summary.csv</span><span class="tag">survivor drawn from prior</span>'
                        '<span class="tag">common random numbers</span><span class="tag">95% CI: Wilson / bootstrap</span>',
                        unsafe_allow_html=True)
        else:
            callout("No precomputed benchmark found in <code>results/</code>. Run the notebook first, or use the re-run option.", "warn")
    else:
        n_eps = st.slider("Episodes per policy", 10, 60, 20, 10, key="pl_neps")
        if st.button("Run evaluation", icon=":material/science:", type="primary", key="pl_run"):
            ss.eval_req = (cfg.to_dict(), n_eps)
        if ss.get("eval_req") and ss.eval_req[0] == cfg.to_dict():
            with st.spinner(f"Evaluating 5 policies × {ss.eval_req[1]} episodes (cached per configuration)…"):
                _, summary = quick_eval(json.dumps(cfg.to_dict()), ss.eval_req[1], int(n_sims), int(depth), float(c_ucb), rollout)
        else:
            callout("Press <b>Run evaluation</b> to benchmark all policies with the current environment and POMCP settings. "
                    "Results are cached per configuration.")
    if summary is not None:
        g1, g2 = st.columns(2, gap="medium")
        with g1:
            show(interval_bars(summary, "Success rate", "success_lo", "success_hi", "Mission success rate (Wilson 95% CI)",
                               "Success rate"), key="pl_cmp_s")
            show(interval_bars(summary, "Mean steps", "steps_lo", "steps_hi", "Average number of steps (bootstrap 95% CI)",
                               "Steps per episode", fmt=".1f", yrange=[0, 36]), key="pl_cmp_st")
        with g2:
            lo = min(summary["reward_lo"].min() - 30, -30); hi = max(summary["reward_hi"].max() + 30, 30)
            show(interval_bars(summary, "Mean reward", "reward_lo", "reward_hi", "Average cumulative reward (bootstrap 95% CI)",
                               "Reward per episode (points)", fmt="+.1f", yrange=[lo, hi]), key="pl_cmp_r")
            s = summary.copy(); s["hz_lo"] = s["Mean hazard entries"]; s["hz_hi"] = s["Mean hazard entries"]
            show(interval_bars(s, "Mean hazard entries", "hz_lo", "hz_hi", "Hazard entries per episode",
                               "Mean hazard entries", fmt=".2f", yrange=[0, s["Mean hazard entries"].max() * 1.35 + 0.1]), key="pl_cmp_h")


# ==========================================================================================
# 5. PERFORMANCE & MISSION REPORT
# ==========================================================================================
def page_report():
    truth_s = runner.env.get_state_for_evaluation()["survivor"]  # report is evaluation; shown only after completion
    section("Mission report")
    if not done:
        callout(f"Mission in progress (step {step_no}/{cfg.max_steps}). The full report, including survivor-location accuracy, "
                "is finalised when the mission ends. Ground truth is withheld until then unless instructor mode is on.", "warn")
    pre = recs[-2] if (done and rescued and len(recs) >= 2) else last
    acc_txt = "—"
    if disclose:
        acc_txt = ("correct" if pre["est_survivor"] == truth_s else "incorrect") + f" {fmt_cell(pre['est_survivor'])} vs {fmt_cell(truth_s)}"
    outcome = "Success" if rescued else ("Failure" if done else "Running")
    st.markdown('<div class="kpi-grid">' + "".join([
        kpi("Outcome", outcome, f"policy {policy_name}", "status",
            C["green"] if rescued else (C["coral"] if done else C["amber"])),
        kpi("Steps", f"{step_no}", f"{n_moves} moves · {n_scans} scans", "step", C["blue"]),
        kpi("Reward", f"{last['cumulative_reward']:+.0f} pts", "undiscounted total", "reward", C["cyan"]),
        kpi("Hazards", f"{sum(r['hazard_entry'] for r in recs)}", "−50 points each", "actions", C["coral"]),
        kpi("Estimate", acc_txt.split(" ")[0].title() if disclose else "Hidden",
            ("best guess " + acc_txt.split(" ", 1)[1]) if disclose else "revealed at mission end",
            "target", C["amber"]),
        kpi("Entropy", f"{pre['survivor_entropy']:.2f} bits", "before contact" if rescued else "survivor belief",
            "entropy", C["cyan"]),
    ]) + "</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2, gap="medium")
    steps = [r["step"] for r in recs]
    with r1:
        fig = go.Figure(go.Scatter(x=steps, y=[r["cumulative_reward"] for r in recs], mode="lines+markers",
                                   line=dict(color=C["cyan"], width=3, shape="hv"), fill="tozeroy",
                                   fillcolor="rgba(0,213,255,0.08)", name="Cumulative reward"))
        style_fig(fig, "1 · Cumulative reward (this mission)", "Decision step", "Cumulative reward (points)", height=340,
                  legend_bottom=False)
        show(fig, key="rp_cum")
    with r2:
        comp = {k: sum(r[k] for r in recs[1:]) for k in ("move_cost", "hazard", "rescue", "scan_cost")}
        names = {"move_cost": "Move costs", "hazard": "Hazard penalties", "rescue": "Rescue reward", "scan_cost": "Scan costs"}
        colors = {"move_cost": C["muted"], "hazard": C["coral"], "rescue": C["green"], "scan_cost": C["amber"]}
        fig = go.Figure(go.Bar(y=[names[k] for k in comp], x=list(comp.values()), orientation="h",
                               marker_color=[colors[k] for k in comp], text=[f"{v:+.0f}" for v in comp.values()],
                               textposition="outside"))
        span = max(abs(v) for v in comp.values()) or 10
        style_fig(fig, f"2 · Reward components (net {sum(comp.values()):+.0f})", "Total contribution (points)", "", height=340,
                  legend_bottom=False)
        fig.update_xaxes(range=[-span * 1.35, span * 1.35])
        show(fig, key="rp_comp")

    section("Benchmark analytics")
    if "summary" not in bench:
        callout("Benchmark results not found in <code>results/</code>. Execute the notebook to generate them.", "warn")
        return
    summ, eps = bench["summary"], bench["episodes"]
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        show(interval_bars(summ, "Success rate", "success_lo", "success_hi", "3 · Mission completion rate", "Success rate",
                           height=380), key="rp_succ")
    with b2:
        show(interval_bars(summ.assign(a=summ["Steps to rescue (successes)"]), "Steps to rescue (successes)", "a", "a",
                           "4 · Steps to rescue (successful runs)", "Steps", fmt=".1f", height=380, yrange=[0, 32],
                           na_text="no rescues"), key="rp_steps")
    b3, e1 = st.columns(2, gap="medium")
    with b3:
        s = summ.copy(); s["a"] = s["Hazard-entry rate"]
        show(interval_bars(s, "Hazard-entry rate", "a", "a", "5 · Hazard-entry frequency",
                           "Share of episodes with ≥ 1 entry", height=380, yrange=[0, max(0.3, s["a"].max() * 1.3)]), key="rp_haz")

    with e1:
        fig = go.Figure()
        for pol in ["POMCP", "Risk-aware heuristic", "QMDP", "Random"]:
            tr = eps.loc[eps.policy == pol, "trace_survivor_entropy"]
            if not len(tr):
                continue
            L = max(len(t) for t in tr)
            mat = np.full((len(tr), L), np.nan)
            for i, t in enumerate(tr):
                mat[i, :len(t)] = t
            keep = (~np.isnan(mat)).sum(0) >= 10
            fig.add_trace(go.Scatter(x=np.arange(L)[keep], y=np.nanmean(mat, 0)[keep], name=f"{pol} (benchmark mean)",
                                     line=dict(color=POLICY_COLORS[pol], width=2.4)))
        fig.add_trace(go.Scatter(x=steps, y=[r["survivor_entropy"] for r in recs], name="This mission",
                                 line=dict(color=C["ice"], width=2, dash="dot"), mode="lines+markers"))
        style_fig(fig, "6 · Survivor-belief entropy over time", "Decision step", "Belief entropy (bits)", height=400)
        fig.update_layout(legend=dict(y=-0.3), margin=dict(b=130))
        show(fig, key="rp_ent")
    section("Sensitivity analysis")
    if True:
        sens = bench.get("sens")
        if sens is not None and len(sens):
            labels = {"p_success": "Movement success probability", "sensor_noise": "Sensor noise level",
                      "pomcp.n_simulations": "POMCP simulations", "pomcp.horizon": "POMCP search depth",
                      "scan_cost": "Scan cost", "pomcp.exploration": "UCB constant c", "rescue_reward": "Rescue reward"}
            avail = [p for p in labels if p in set(sens.parameter)]
            par = st.selectbox("7 · Sensitivity factor", avail, format_func=lambda p: labels[p], key="rp_sens")
            metric = st.radio("Metric", ["Success rate", "Mean reward"], horizontal=True, key="rp_metric")
            d = sens[sens.parameter == par]
            fig = go.Figure()
            lo_c, hi_c = ("success_lo", "success_hi") if metric == "Success rate" else ("reward_lo", "reward_hi")
            for pol, g in d.groupby("Policy", sort=False):
                xs = [str(v) for v in g["value"]]
                fig.add_trace(go.Scatter(x=xs, y=g[metric], name=pol, mode="lines+markers",
                                         line=dict(color=POLICY_COLORS[pol], width=2.6), marker=dict(size=9),
                                         error_y=dict(type="data", symmetric=False, array=np.clip(g[hi_c] - g[metric], 0, None),
                                                      arrayminus=np.clip(g[metric] - g[lo_c], 0, None), thickness=1.2)))
            style_fig(fig, f"Sensitivity: {metric.lower()} vs {labels[par].lower()}", labels[par],
                      "Success rate" if metric == "Success rate" else "Mean cumulative reward (points)", height=330)
            if metric == "Success rate":
                fig.update_yaxes(tickformat=".0%", range=[-0.03, 1.08])
            fig.update_xaxes(type="category")
            show(fig, key="rp_sens_fig")

    section("Automated interpretation")
    S = summ.set_index("Policy")
    lines = []
    if done:
        lines.append(f"This mission <b>{'succeeded' if rescued else 'failed'}</b> after {step_no} decisions with "
                     f"{last['cumulative_reward']:+.0f} points ({n_moves} moves, {n_scans} scans, "
                     f"{sum(r['hazard_entry'] for r in recs)} hazard entries).")
    if "POMCP" in S.index and "Risk-aware heuristic" in S.index:
        P, H = S.loc["POMCP"], S.loc["Risk-aware heuristic"]
        lines.append(f"In the benchmark ({int(P['Episodes'])} episodes per policy), POMCP rescued {P['Success rate']:.0%} "
                     f"(95% CI {P['Success 95% CI']}) with mean reward {P['Mean reward']:+.1f}, against {H['Success rate']:.0%} "
                     f"and {H['Mean reward']:+.1f} for the risk-aware heuristic.")
    if "QMDP" in S.index:
        Q = S.loc["QMDP"]
        lines.append(f"QMDP, which ignores the value of information, scanned {Q['Mean scans']:.1f} times per episode on average "
                     f"and succeeded in {Q['Success rate']:.0%} of episodes.")
    if "Clairvoyant (reference)" in S.index:
        Cv = S.loc["Clairvoyant (reference)"]
        dep = S.drop(index="Clairvoyant (reference)")
        best_name = dep["Mean reward"].idxmax(); best = dep["Mean reward"].max()
        lines.append(f"The perfect-information reference earns {Cv['Mean reward']:+.1f} points per mission. The gap of "
                     f"{Cv['Mean reward'] - best:.1f} points to the best deployable policy on reward ({best_name}, "
                     f"{dep.loc[best_name, 'Success rate']:.0%} success) is the cost of partial observability.")
        if best_name == "QMDP" and dep.loc["QMDP", "Success rate"] < 0.05:
            lines.append("Note the reward-design effect: QMDP scores well on reward only by <b>never moving</b>. Failing a mission "
                         "carries no penalty and scans are free, so 'do nothing' is worth 0 points, while a long, uncertain search "
                         "under γ = 0.95 is close to break-even. Compare success rates alongside reward.")
    st.markdown('<div class="panel">' + "".join(f"<p>• {l}</p>" for l in lines) + "</div>", unsafe_allow_html=True)


PAGES = {"Mission Overview": page_overview, "Live Robot Simulation": page_live,
         "Belief & Sensor Intelligence": page_belief, "Policy & Decision Analytics": page_policy,
         "Performance & Mission Report": page_report}
PAGES[page]()

# autoplay continues from any section
if ss.playing and not runner.done and page != "Live Robot Simulation":
    time.sleep(float(speed))
    do_step()
    if runner.done:
        ss.playing = False
    st.rerun()
