"""
Adaptive Traffic Signal Control — Streamlit Dashboard
======================================================
Provides a full GUI over all argparse parameters from the original
run_01.py / argparse.py, launches training / testing as background
subprocesses, streams live logs, and displays results.

Run with:
    streamlit run dashboard.py
"""

import os
import sys
import time
import subprocess
import threading
import pickle
import glob
import queue
import json

from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np

# ── optional plotting (graceful fallback if matplotlib missing) ──────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATSC — Deep RL Traffic Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark industrial / research-tool aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

/* ── global ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.3rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e2330;
}

/* ── main header ── */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #f1f5f9;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #0ea5e9;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 8px;
    letter-spacing: 0.04em;
}
.badge.green  { background: #10b981; }
.badge.amber  { background: #f59e0b; }
.badge.purple { background: #8b5cf6; }

/* ── metric cards ── */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
}
.metric-card .label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
}
.metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #38bdf8;
}
.metric-card .sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 2px;
}

/* ── log box ── */
.log-box {
    background: #080a0f;
    border: 1px solid #1a2332;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #7dd3fc;
    height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* ── section titles ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 5px;
    border-bottom: 1px solid #1e2330;
}

/* ── status pill ── */
.status-pill {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.status-idle    { background: #1e2330; color: #64748b; }
.status-running { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.status-done    { background: #1e3a5f; color: #38bdf8; border: 1px solid #0ea5e9; }
.status-error   { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }

/* ── command preview ── */
.cmd-preview {
    background: #080a0f;
    border: 1px solid #1a2332;
    border-left: 3px solid #38bdf8;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    color: #7dd3fc;
    word-break: break-all;
    margin-top: 0.5rem;
}

/* ── streamlit overrides ── */
.stButton > button {
    background: #0ea5e9;
    color: #fff;
    border: none;
    border-radius: 7px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.55rem 1.5rem;
    transition: background 0.2s;
    width: 100%;
}
.stButton > button:hover { background: #0284c7; }
.stButton > button:disabled { background: #1e2330; color: #475569; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

div[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
}

.stAlert { border-radius: 8px; }

/* ── divider ── */
hr { border-color: #1e2330; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ABSOLUTE BASE DIR  (so relative paths always resolve correctly)
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(rel_path: str) -> str:
    """Resolve a potentially-relative path against the script directory."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(_SCRIPT_DIR, rel_path)

def _scan_subdirs(parent: str) -> list[str]:
    """Return sorted list of immediate subdirectory names, or [] if parent doesn't exist."""
    p = _abs(parent)
    if not os.path.isdir(p):
        return []
    return sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)))


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "proc":        None,
        "log_lines":   [],
        "status":      "idle",   # idle | running | done | error
        "run_start":   None,
        "log_queue":   queue.Queue(),
        "reader_thread": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _stream_output(proc, q):
    """Background thread: read subprocess stdout/stderr into queue."""
    try:
        for line in iter(proc.stdout.readline, b""):
            q.put(line.decode("utf-8", errors="replace").rstrip())
        proc.stdout.close()
    except Exception as e:
        q.put(f"[reader error] {e}")


def _build_command(cfg: dict) -> list[str]:
    """Convert GUI config dict to a run_01.py command list."""
    cmd = [sys.executable, "run_01.py"]

    # Core
    cmd += ["-mode",   cfg["mode"]]
    cmd += ["-tsc",    cfg["tsc"]]
    cmd += ["-n",      str(cfg["n"])]
    cmd += ["-l",      str(cfg["l"])]
    cmd += ["-simlen", str(cfg["sim_len"])]
    cmd += ["-port",   str(cfg["port"])]

    # Simulation scenario
    if cfg["sim"] != "custom":
        cmd += ["-sim", cfg["sim"]]
    else:
        cmd += ["-netfp",   cfg["net_fp"]]
        cmd += ["-sumocfg", cfg["cfg_fp"]]

    # Flags
    if cfg["nogui"]:   cmd.append("-nogui")
    if cfg["save"]:    cmd.append("-save")
    if cfg["load"]:    cmd.append("-load")
    if cfg["load_replay"]: cmd.append("-load_replay")

    # Traffic params
    cmd += ["-scale",  str(cfg["scale"])]
    cmd += ["-demand", cfg["demand"]]
    cmd += ["-offset", str(cfg["offset"])]
    cmd += ["-gmin",   str(cfg["g_min"])]
    cmd += ["-y",      str(cfg["y"])]
    cmd += ["-r",      str(cfg["r"])]

    # Webster
    if cfg["tsc"] == "websters":
        cmd += ["-cmin",    str(cfg["c_min"])]
        cmd += ["-cmax",    str(cfg["c_max"])]
        cmd += ["-satflow", str(cfg["sat_flow"])]
        cmd += ["-f",       str(cfg["update_freq"])]

    # SOTL
    if cfg["tsc"] == "sotl":
        cmd += ["-theta", str(cfg["theta"])]
        cmd += ["-omega", str(cfg["omega"])]
        cmd += ["-mu",    str(cfg["mu"])]

    # RL
    if cfg["tsc"] in ("dqn", "ddpg", "ppo"):
        cmd += ["-eps",         str(cfg["eps"])]
        cmd += ["-nsteps",      str(cfg["nsteps"])]
        cmd += ["-nreplay",     str(cfg["nreplay"])]
        cmd += ["-batch",       str(cfg["batch"])]
        cmd += ["-gamma",       str(cfg["gamma"])]
        cmd += ["-updates",     str(cfg["updates"])]
        cmd += ["-target_freq", str(cfg["target_freq"])]
        cmd += ["-lr",          str(cfg["lr"])]
        cmd += ["-lre",         str(cfg["lre"])]
        cmd += ["-hidden_act",  cfg["hidden_act"]]
        cmd += ["-n_hidden",    str(cfg["n_hidden"])]
        cmd += ["-save_path",   cfg["save_path"]]
        cmd += ["-save_replay", cfg["save_replay"]]
        cmd += ["-save_t",      str(cfg["save_t"])]

    if cfg["tsc"] == "ddpg":
        cmd += ["-tau",  str(cfg["tau"])]
        cmd += ["-gmax", str(cfg["g_max"])]
        cmd += ["-lrc",  str(cfg["lrc"])]

    return cmd


def _start_run(cfg: dict):
    cmd = _build_command(cfg)
    st.session_state.log_lines = [
        f"▶ {datetime.now().strftime('%H:%M:%S')}  Starting run...",
        "$ " + " ".join(cmd),
        "─" * 60,
    ]
    st.session_state.status    = "running"
    st.session_state.run_start = time.time()
    st.session_state.log_queue = queue.Queue()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    st.session_state.proc = proc

    t = threading.Thread(
        target=_stream_output,
        args=(proc, st.session_state.log_queue),
        daemon=True,
    )
    t.start()
    st.session_state.reader_thread = t


def _stop_run():
    proc = st.session_state.proc
    if proc and proc.poll() is None:
        proc.terminate()
        st.session_state.log_lines.append("⛔ Process terminated by user.")
    st.session_state.status = "idle"
    st.session_state.proc   = None


def _drain_queue():
    """Pull all pending log lines from the reader thread queue."""
    q = st.session_state.log_queue
    new_lines = []
    while True:
        try:
            new_lines.append(q.get_nowait())
        except queue.Empty:
            break
    if new_lines:
        st.session_state.log_lines.extend(new_lines)

    # Check if process finished
    proc = st.session_state.proc
    if proc is not None and proc.poll() is not None:
        rc = proc.returncode
        st.session_state.status = "done" if rc == 0 else "error"
        st.session_state.proc   = None
        msg = (f"✅ Process finished  (exit code {rc})"
               if rc == 0 else
               f"❌ Process exited with code {rc}")
        st.session_state.log_lines.append("─" * 60)
        st.session_state.log_lines.append(msg)


def _load_pickle(fp):
    try:
        with open(fp, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _elapsed(start):
    if start is None:
        return "—"
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60}s"


def _status_html(status):
    labels = {
        "idle":    ("IDLE",    "status-idle"),
        "running": ("RUNNING", "status-running"),
        "done":    ("COMPLETE","status-done"),
        "error":   ("ERROR",   "status-error"),
    }
    txt, cls = labels.get(status, ("UNKNOWN", "status-idle"))
    return f'<span class="status-pill {cls}">{txt}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🚦 Adaptive Traffic Signal Control</h1>
  <p>Deep Reinforcement Learning with SUMO &nbsp;·&nbsp; BITS Pilani Dissertation &nbsp;·&nbsp; May 2026</p>
  <span class="badge">DQN</span>
  <span class="badge purple">PPO</span>
  <span class="badge green">MARL</span>
  <span class="badge amber">SUMO</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — ALL ARGPARSE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── Run settings ──────────────────────────────────────────────────────────
    st.markdown("### Run Settings")

    mode = st.selectbox(
        "Mode",
        ["train", "test"],
        help="train: actors generate experience + learner trains. "
             "test: load weights and run one evaluation episode."
    )
    tsc = st.selectbox(
        "Traffic Signal Controller",
        ["websters", "sotl", "maxpressure", "uniform", "dqn", "ddpg", "ppo"],
        index=4,
        help="Algorithm used to control each intersection."
    )
    sim_scenario = st.selectbox(
        "Simulation Scenario",
        ["double", "single"],
        help="Predefined SUMO network scenario."
    )
    net_fp = f"networks/{sim_scenario}.net.xml"
    cfg_fp = f"networks/{sim_scenario}.sumocfg"

    # ── Process settings ──────────────────────────────────────────────────────
    st.markdown("### Process Settings")

    n_actors = st.slider(
        "Sim processes (actors)",
        min_value=1, max_value=max(2, os.cpu_count() - 1),
        value=min(4, max(1, os.cpu_count() - 2)),
        help="Number of parallel SUMO simulations generating experiences."
    )

    default_learners = 0 if (mode == "test" or tsc in ("websters","sotl","maxpressure","uniform")) else 1
    n_learners = st.slider(
        "Learner processes",
        min_value=0, max_value=4,
        value=default_learners,
        help="0 for traditional TSC or test mode. ≥1 for RL training."
    )

    sim_len = st.number_input(
        "Simulation length (steps/s)",
        min_value=300, max_value=86400,
        value=10800, step=300,
        help="Total SUMO simulation time in seconds."
    )
    port = st.number_input(
        "Base TraCI port",
        min_value=1024, max_value=65000,
        value=9000,
        help="Each actor uses port + actor_index."
    )

    # ── Flags ─────────────────────────────────────────────────────────────────
    st.markdown("### Flags")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        nogui        = st.checkbox("No GUI",       value=True,  help="Run SUMO headless.")
        save_weights = st.checkbox("Save weights", value=True,  help="Save NN weights periodically.")
    with col_f2:
        load_weights = st.checkbox("Load weights", value=(mode == "test"), help="Load pre-trained weights.")
        load_replay  = st.checkbox("Load replay",  value=False, help="Load saved experience replay.")

    # ── Traffic generation ────────────────────────────────────────────────────
    st.markdown("### Traffic Generation")

    demand = st.selectbox(
        "Demand pattern",
        ["dynamic", "single"],
        help="dynamic: sine-wave varying traffic. single: one vehicle at a time."
    )
    scale  = st.slider(
        "Vehicle scale",
        min_value=0.5, max_value=3.0,
        value=1.4, step=0.1,
        help="Multiplier for vehicle generation rate."
    )
    offset = st.slider(
        "Max start offset (fraction)",
        min_value=0.0, max_value=0.5,
        value=0.25, step=0.05,
        help="Actors are staggered up to this fraction of sim_len."
    )

    # ── Signal timing ─────────────────────────────────────────────────────────
    st.markdown("### Signal Timing")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        g_min = st.number_input("Min green (s)", min_value=1,  max_value=60,  value=5)
    with col_t2:
        y_t   = st.number_input("Yellow (s)",    min_value=1,  max_value=10,  value=2)
    with col_t3:
        r_t   = st.number_input("All-red (s)",   min_value=0,  max_value=10,  value=3)

    # ── Webster's specific ────────────────────────────────────────────────────
    if tsc == "websters":
        st.markdown("### Webster's Parameters")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            c_min    = st.number_input("Min cycle (s)", min_value=20,  max_value=300, value=60)
            sat_flow = st.number_input("Sat flow (veh/s)", min_value=0.1, max_value=1.0,
                                       value=0.38, step=0.01, format="%.2f")
        with col_w2:
            c_max       = st.number_input("Max cycle (s)", min_value=60, max_value=600, value=180)
            update_freq = st.number_input("Update freq (s)", min_value=60, max_value=3600, value=900)
    else:
        c_min = 60; c_max = 180; sat_flow = 0.38; update_freq = 900

    # ── SOTL specific ─────────────────────────────────────────────────────────
    if tsc == "sotl":
        st.markdown("### SOTL Parameters")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1: theta = st.number_input("Theta", min_value=1, max_value=200, value=45)
        with col_s2: omega = st.number_input("Omega", min_value=1, max_value=20,  value=1)
        with col_s3: mu_v  = st.number_input("Mu",    min_value=1, max_value=20,  value=3)
    else:
        theta = 45; omega = 1; mu_v = 3

    # ── RL parameters ─────────────────────────────────────────────────────────
    if tsc in ("dqn", "ddpg", "ppo"):
        st.markdown("### RL Hyperparameters")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            eps         = st.number_input("Epsilon (ε)",    min_value=0.001, max_value=1.0,
                                          value=0.01, step=0.001, format="%.3f")
            gamma       = st.number_input("Gamma (γ)",      min_value=0.5,   max_value=0.9999,
                                          value=0.99, step=0.001, format="%.3f")
            batch       = st.number_input("Batch size",     min_value=8,     max_value=512,  value=32)
            nsteps      = st.number_input("N-step returns", min_value=1,     max_value=20,   value=1)
        with col_r2:
            updates     = st.number_input("Total updates",  min_value=100,   max_value=100000, value=10000, step=500)
            nreplay     = st.number_input("Replay buffer",  min_value=1000,  max_value=200000, value=10000, step=1000)
            target_freq = st.number_input("Target freq",    min_value=1,     max_value=500,    value=50)

        st.markdown("### Neural Network")
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            lr          = st.number_input("Learning rate (actor/DQN)", min_value=1e-6, max_value=1e-2,
                                          value=1e-4, step=1e-5, format="%.6f")
            lre         = st.number_input("Optimizer epsilon",        min_value=1e-10, max_value=1e-5,
                                          value=1e-8, step=1e-9, format="%.2e")
        with col_n2:
            hidden_act  = st.selectbox("Hidden activation", ["elu", "relu", "tanh", "sigmoid"], index=0)
            n_hidden    = st.number_input("Hidden layers scale", min_value=1, max_value=10, value=3)

        save_path   = st.text_input("Model save path",  value="Outputs/saved_models")
        save_replay = st.text_input("Replay save path", value="saved_replays")
        save_t      = st.number_input("Save interval (s)", min_value=30, max_value=3600, value=120)

        if tsc == "ddpg":
            st.markdown("### DDPG Extras")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1: tau  = st.number_input("Tau (τ)",    min_value=0.0001, max_value=0.1, value=0.005, format="%.4f")
            with col_d2: g_max = st.number_input("Max green", min_value=10,    max_value=120,  value=30)
            with col_d3: lrc  = st.number_input("Critic LR",  min_value=1e-5,  max_value=1e-2, value=1e-3, format="%.5f")
        else:
            tau = 0.005; g_max = 30; lrc = 0.001
    else:
        eps = 0.01; gamma = 0.99; batch = 32; nsteps = 1
        updates = 10000; nreplay = 10000; target_freq = 50
        lr = 1e-4; lre = 1e-8; hidden_act = "elu"; n_hidden = 3
        save_path = "Outputs/saved_models"; save_replay = "saved_replays"; save_t = 120
        tau = 0.005; g_max = 30; lrc = 0.001

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE CONFIG DICT
# ─────────────────────────────────────────────────────────────────────────────
cfg = dict(
    mode=mode, tsc=tsc,
    sim=sim_scenario, net_fp=net_fp, cfg_fp=cfg_fp,
    n=n_actors, l=n_learners, sim_len=sim_len, port=port,
    nogui=nogui, save=save_weights, load=load_weights, load_replay=load_replay,
    demand=demand, scale=scale, offset=offset,
    g_min=g_min, y=y_t, r=r_t,
    c_min=c_min, c_max=c_max, sat_flow=sat_flow, update_freq=update_freq,
    theta=theta, omega=omega, mu=mu_v,
    eps=eps, gamma=gamma, batch=batch, nsteps=nsteps,
    updates=updates, nreplay=nreplay, target_freq=target_freq,
    lr=lr, lre=lre, hidden_act=hidden_act, n_hidden=n_hidden,
    save_path=save_path, save_replay=save_replay, save_t=save_t,
    tau=tau, g_max=g_max, lrc=lrc,
)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_run, tab_results, tab_compare, tab_benchmark, tab_logs, tab_about = st.tabs([
    "▶  Run",
    "📊  Results",
    "⚖️  Compare",
    "🏁  Test & Benchmark",
    "📋  Live Logs",
    "ℹ️  About",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN
# ═════════════════════════════════════════════════════════════════════════════
with tab_run:

    # Status row
    status = st.session_state.status
    elapsed = _elapsed(st.session_state.run_start)
    log_count = len(st.session_state.log_lines)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.markdown(f"**Status:** {_status_html(status)}", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='label'>Mode</div>"
                    f"<div class='value' style='font-size:1rem'>{mode.upper()}</div></div>",
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='label'>Algorithm</div>"
                    f"<div class='value' style='font-size:1rem'>{tsc.upper()}</div></div>",
                    unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='label'>Elapsed</div>"
                    f"<div class='value' style='font-size:1rem'>{elapsed}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Command Preview</div>", unsafe_allow_html=True)
    cmd_str = " ".join(_build_command(cfg))
    st.markdown(f"<div class='cmd-preview'>{cmd_str}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Process Control</div>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 3])
    with col_btn1:
        run_disabled = (status == "running")
        if st.button("🚀  Start Run", disabled=run_disabled):
            _start_run(cfg)
            st.rerun()

    with col_btn2:
        stop_disabled = (status != "running")
        if st.button("⛔  Stop Run", disabled=stop_disabled):
            _stop_run()
            st.rerun()

    with col_btn3:
        if st.button("🔄  Refresh Status"):
            _drain_queue()
            st.rerun()

    # Quick config summary
    st.markdown("<div class='section-title'>Current Configuration Summary</div>",
                unsafe_allow_html=True)

    sum_cols = st.columns(4)
    summaries = [
        ("Scenario",   sim_scenario.upper()),
        ("Actors",     str(n_actors)),
        ("Learners",   str(n_learners)),
        ("Sim length", f"{sim_len}s"),
        ("GUI",        "Off" if nogui else "On"),
        ("Demand",     demand),
        ("Scale",      str(scale)),
        ("Offset",     str(offset)),
    ]
    for i, (lbl, val) in enumerate(summaries):
        with sum_cols[i % 4]:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='label'>{lbl}</div>"
                f"<div class='value' style='font-size:1.1rem;color:#94a3b8'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    if tsc in ("dqn", "ddpg", "ppo"):
        st.markdown("<div class='section-title'>RL Hyperparameter Summary</div>",
                    unsafe_allow_html=True)
        rl_cols = st.columns(4)
        rl_items = [
            ("Epsilon",     str(eps)),
            ("Gamma",       str(gamma)),
            ("Batch",       str(batch)),
            ("Replay buf",  f"{nreplay:,}"),
            ("Updates",     f"{updates:,}"),
            ("LR",          f"{lr:.0e}"),
            ("N-steps",     str(nsteps)),
            ("Target freq", str(target_freq)),
        ]
        for i, (lbl, val) in enumerate(rl_items):
            with rl_cols[i % 4]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='label'>{lbl}</div>"
                    f"<div class='value' style='font-size:1.1rem;color:#a78bfa'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # Live refresh while running
    if status == "running":
        _drain_queue()
        log_snippet = st.session_state.log_lines[-6:] if st.session_state.log_lines else []
        if log_snippet:
            st.markdown("<div class='section-title'>Latest Output</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='log-box'>" +
                "\n".join(log_snippet) +
                "</div>",
                unsafe_allow_html=True
            )
        time.sleep(2)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("<div class='section-title'>Load Results from Metrics Directory</div>",
                unsafe_allow_html=True)

    def _on_res_root_change():
        """When metrics root changes, clear dependent selections."""
        for k in ("res_tsc", "res_metric"):
            st.session_state.pop(k, None)

    def _on_res_tsc_change():
        """When controller changes, clear metric selection."""
        st.session_state.pop("res_metric", None)

    metrics_root = st.text_input(
        "Metrics root directory", value="Outputs/metrics",
        key="res_root", on_change=_on_res_root_change,
    )
    _metrics_root_abs = _abs(metrics_root)

    # Detect available controllers from disk
    _all_controllers = ["websters", "sotl", "maxpressure", "uniform", "dqn", "ddpg", "ppo"]
    _detected_ctrls = [d for d in _scan_subdirs(metrics_root) if d in _all_controllers]
    _ctrl_options = _detected_ctrls if _detected_ctrls else _all_controllers
    if "res_tsc" in st.session_state and st.session_state.res_tsc not in _ctrl_options:
        del st.session_state["res_tsc"]
    sel_tsc = st.selectbox(
        "Controller to inspect", _ctrl_options,
        key="res_tsc", on_change=_on_res_tsc_change,
    )

    # Mode selector (train vs test) — only shown if mode subdirs exist
    _detected_subdirs = _scan_subdirs(os.path.join(metrics_root, sel_tsc))
    _known_modes = {"train", "test"}
    _has_mode_dirs = bool(_known_modes & set(_detected_subdirs))
    if _has_mode_dirs:
        _mode_options = [d for d in _detected_subdirs if d in _known_modes]
        sel_mode = st.selectbox("Mode", _mode_options, key="res_mode")
        _metric_base = os.path.join(metrics_root, sel_tsc, sel_mode)
    else:
        sel_mode = None
        _metric_base = os.path.join(metrics_root, sel_tsc)

    # Detect available metrics for the selected controller/mode
    _fallback_metrics = ["delay", "emission", "queue", "speed", "throughput", "traveltime", "waitingtime"]
    _detected_metrics = _scan_subdirs(_metric_base)
    _metric_options = _detected_metrics if _detected_metrics else _fallback_metrics
    if "res_metric" in st.session_state and st.session_state.res_metric not in _metric_options:
        del st.session_state["res_metric"]
    sel_metric = st.selectbox("Metric", _metric_options, key="res_metric")

    # Show what was detected
    st.caption(f"📁 Resolved path: `{_metrics_root_abs}` — "
               f"{len(_ctrl_options)} controller(s), {len(_metric_options)} metric(s)")

    if st.button("📂  Load Results"):
        pattern = os.path.join(_abs(_metric_base), sel_metric, "**", "*.p")
        files   = sorted(glob.glob(pattern, recursive=True))

        if not files:
            st.warning(f"No `.p` files found under  {pattern}")
        else:
            st.success(f"Found {len(files)} result file(s).")
            all_data = {}
            for fp in files:
                d = _load_pickle(fp)
                if d is not None:
                    intersection = Path(fp).parent.name
                    if intersection not in all_data:
                        all_data[intersection] = []
                    if isinstance(d, list):
                        all_data[intersection].extend(d)
                    else:
                        all_data[intersection].append(d)

            # Helper: extract numeric series from values that may be dicts (e.g. emission)
            def _to_numeric(vals):
                """Convert a list of values to numeric. Dicts like {'co2':x, 'fuel':y} use first key."""
                out = []
                for v in vals:
                    if isinstance(v, (int, float)):
                        out.append(v)
                    elif isinstance(v, dict):
                        # use first numeric value from dict
                        for dv in v.values():
                            if isinstance(dv, (int, float)):
                                out.append(dv)
                                break
                return out

            if all_data:
                # For emission dicts, let user pick which sub-metric to display
                _emission_key = None
                _sample = next(iter(all_data.values()), [])
                if _sample and isinstance(_sample[0], dict):
                    _dict_keys = list(_sample[0].keys())
                    _emission_key = st.selectbox(
                        "Sub-metric", _dict_keys, key="res_submetric",
                    )

                def _extract(vals):
                    if _emission_key is not None:
                        return [v[_emission_key] for v in vals
                                if isinstance(v, dict) and _emission_key in v]
                    return _to_numeric(vals)

                st.markdown("<div class='section-title'>Summary Statistics</div>",
                            unsafe_allow_html=True)
                stat_cols = st.columns(min(4, len(all_data)))
                for idx, (inter, vals) in enumerate(all_data.items()):
                    numeric = _extract(vals)
                    if numeric:
                        with stat_cols[idx % 4]:
                            mean_v = np.mean(numeric)
                            std_v  = np.std(numeric)
                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<div class='label'>{inter}</div>"
                                f"<div class='value'>{mean_v:.1f}</div>"
                                f"<div class='sub'>± {std_v:.1f} std  ·  n={len(numeric)}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                if HAS_MPL:
                    st.markdown("<div class='section-title'>Time-series Plot</div>",
                                unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#111827")
                    ax.set_facecolor("#0d0f14")
                    colors = ["#38bdf8", "#34d399", "#a78bfa", "#f59e0b",
                              "#f472b6", "#fb923c"]
                    for i, (inter, vals) in enumerate(all_data.items()):
                        numeric = _extract(vals)
                        if numeric:
                            ax.plot(numeric, color=colors[i % len(colors)],
                                    linewidth=1.4, label=inter, alpha=0.9)
                    ax.set_xlabel("Timestep", color="#64748b", fontsize=9)
                    ax.set_ylabel(sel_metric.capitalize(), color="#64748b", fontsize=9)
                    ax.tick_params(colors="#475569", labelsize=8)
                    ax.spines[:].set_color("#1e2330")
                    if len(all_data) <= 6:
                        ax.legend(facecolor="#111827", edgecolor="#1e2330",
                                  labelcolor="#94a3b8", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Install matplotlib (`pip install matplotlib`) to enable plots.")

    # Summary CSV viewer
    st.markdown("<div class='section-title'>Simulation Summary (CSV)</div>",
                unsafe_allow_html=True)
    _results_dir = _abs(f"Outputs/results/{sel_tsc}")
    csv_files = sorted(
        glob.glob(os.path.join(_results_dir, "*.csv")) +
        glob.glob(os.path.join(_results_dir, "**", "*.csv"), recursive=True)
    )
    # deduplicate (overlapping globs)
    csv_files = sorted(set(csv_files))
    if csv_files:
        sel_csv = st.selectbox("Select CSV file", csv_files)
        try:
            import pandas as pd
            df = pd.read_csv(sel_csv)
            st.dataframe(df, use_container_width=True)

            if "avg_travel_time" in df.columns:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"<div class='metric-card'><div class='label'>Mean travel time</div>"
                        f"<div class='value'>{df['avg_travel_time'].mean():.1f}s</div></div>",
                        unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f"<div class='metric-card'><div class='label'>Best episode</div>"
                        f"<div class='value'>{df['avg_travel_time'].min():.1f}s</div></div>",
                        unsafe_allow_html=True)
                with c3:
                    st.markdown(
                        f"<div class='metric-card'><div class='label'>Episodes logged</div>"
                        f"<div class='value'>{len(df)}</div></div>",
                        unsafe_allow_html=True)
        except ImportError:
            # fallback without pandas
            rows = []
            with open(sel_csv) as f:
                for line in f:
                    rows.append(line.strip().split(","))
            if len(rows) > 1:
                st.text("Header: " + ", ".join(rows[0]))
                for r in rows[1:]:
                    st.text(", ".join(r))
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
    else:
        st.info(f"No CSV files found under Outputs/results/{sel_tsc}/. Run a simulation first.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ═════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("<div class='section-title'>Algorithm Comparison</div>",
                unsafe_allow_html=True)
    st.info(
        "Load result files for multiple controllers to compare them side-by-side. "
        "Each controller should have results saved under  `Outputs/metrics/<tsc>/`."
    )

    def _on_cmp_root_change():
        for k in ("cmp_metric",):
            st.session_state.pop(k, None)

    compare_root = st.text_input(
        "Metrics root", value="Outputs/metrics",
        key="cmp_root", on_change=_on_cmp_root_change,
    )
    _compare_root_abs = _abs(compare_root)

    # Mode selector for comparison
    _cmp_mode = st.selectbox("Mode", ["train", "test", "(legacy/all)"], key="cmp_mode")

    # Detect available metrics across all controllers on disk
    _cmp_fallback_metrics = ["delay", "emission", "queue", "speed", "throughput", "traveltime", "waitingtime"]
    _cmp_all_metrics = set()
    for _ctrl_name in _scan_subdirs(compare_root):
        if _cmp_mode == "(legacy/all)":
            # Look directly under controller (old structure)
            for _m in _scan_subdirs(os.path.join(compare_root, _ctrl_name)):
                if _m not in ("train", "test"):
                    _cmp_all_metrics.add(_m)
        else:
            for _m in _scan_subdirs(os.path.join(compare_root, _ctrl_name, _cmp_mode)):
                _cmp_all_metrics.add(_m)
            # Also check legacy structure
            for _m in _scan_subdirs(os.path.join(compare_root, _ctrl_name)):
                if _m not in ("train", "test"):
                    _cmp_all_metrics.add(_m)
    _cmp_metric_options = sorted(_cmp_all_metrics) if _cmp_all_metrics else _cmp_fallback_metrics
    if "cmp_metric" in st.session_state and st.session_state.cmp_metric not in _cmp_metric_options:
        del st.session_state["cmp_metric"]
    compare_metric = st.selectbox("Metric to compare", _cmp_metric_options, key="cmp_metric")

    # Detect available controllers from disk
    _cmp_all_controllers = ["websters", "sotl", "maxpressure", "uniform", "dqn", "ddpg", "ppo"]
    _cmp_detected = [d for d in _scan_subdirs(compare_root) if d in _cmp_all_controllers]
    _cmp_ctrl_options = _cmp_detected if _cmp_detected else _cmp_all_controllers
    controllers_sel = st.multiselect(
        "Controllers to compare",
        _cmp_ctrl_options,
        default=[c for c in ["websters", "dqn"] if c in _cmp_ctrl_options],
    )

    st.caption(f"📁 Resolved path: `{_compare_root_abs}` — "
               f"{len(_cmp_ctrl_options)} controller(s), {len(_cmp_metric_options)} metric(s)")

    if st.button("📊  Compare") and controllers_sel:
        summary = {}
        for ctrl in controllers_sel:
            # Search both new structure (with mode) and legacy structure (without mode)
            patterns = []
            if _cmp_mode == "(legacy/all)":
                patterns.append(os.path.join(_compare_root_abs, ctrl, compare_metric, "**", "*.p"))
            else:
                patterns.append(os.path.join(_compare_root_abs, ctrl, _cmp_mode, compare_metric, "**", "*.p"))
                # Also search legacy structure as fallback
                patterns.append(os.path.join(_compare_root_abs, ctrl, compare_metric, "**", "*.p"))
            files = []
            for pat in patterns:
                files.extend(glob.glob(pat, recursive=True))
            files = sorted(set(files))  # deduplicate
            all_vals = []
            for fp in files:
                d = _load_pickle(fp)
                if d and isinstance(d, list):
                    for v in d:
                        if isinstance(v, (int, float)):
                            all_vals.append(v)
                        elif isinstance(v, dict):
                            # emission dicts: sum co2 + fuel or use co2 as representative
                            if 'co2' in v:
                                all_vals.append(v['co2'])
                            else:
                                for dv in v.values():
                                    if isinstance(dv, (int, float)):
                                        all_vals.append(dv)
                                        break
            if all_vals:
                summary[ctrl] = {
                    "mean":   np.mean(all_vals),
                    "std":    np.std(all_vals),
                    "min":    np.min(all_vals),
                    "max":    np.max(all_vals),
                    "n":      len(all_vals),
                }

        if not summary:
            st.warning("No data found. Make sure simulations have been run and results saved.")
        else:
            # Metrics table
            st.markdown("<div class='section-title'>Summary Table</div>",
                        unsafe_allow_html=True)
            col_headers = st.columns([2, 1, 1, 1, 1, 1])
            headers = ["Controller", "Mean", "Std", "Min", "Max", "N samples"]
            for col, hdr in zip(col_headers, headers):
                col.markdown(f"**{hdr}**")

            best_mean = min(v["mean"] for v in summary.values())
            for ctrl, stats in summary.items():
                cols = st.columns([2, 1, 1, 1, 1, 1])
                highlight = "🏆 " if stats["mean"] == best_mean else ""
                cols[0].markdown(f"{highlight}**{ctrl}**")
                cols[1].markdown(f"`{stats['mean']:.1f}`")
                cols[2].markdown(f"`{stats['std']:.1f}`")
                cols[3].markdown(f"`{stats['min']:.1f}`")
                cols[4].markdown(f"`{stats['max']:.1f}`")
                cols[5].markdown(f"`{stats['n']:,}`")

            # Bar chart
            if HAS_MPL:
                st.markdown("<div class='section-title'>Mean Comparison Chart</div>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#111827")
                ax.set_facecolor("#0d0f14")

                ctrl_names = list(summary.keys())
                means_arr  = [summary[c]["mean"] for c in ctrl_names]
                stds_arr   = [summary[c]["std"]  for c in ctrl_names]
                bar_colors = ["#38bdf8" if m == best_mean else "#1e3a5f"
                              for m in means_arr]

                bars = ax.bar(ctrl_names, means_arr, color=bar_colors,
                              yerr=stds_arr, capsize=4,
                              error_kw={"ecolor": "#475569", "elinewidth": 1})
                for bar, mean in zip(bars, means_arr):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{mean:.1f}", ha="center", va="bottom",
                            color="#94a3b8", fontsize=8)

                ax.set_ylabel(compare_metric.capitalize(), color="#64748b", fontsize=9)
                ax.tick_params(colors="#475569", labelsize=9)
                ax.spines[:].set_color("#1e2330")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # % improvement vs best traditional
            traditional = [c for c in summary if c in ("websters","sotl","maxpressure","uniform")]
            if traditional and any(c in summary for c in ("dqn","ddpg","ppo")):
                best_trad = min(summary[c]["mean"] for c in traditional if c in summary)
                st.markdown("<div class='section-title'>Improvement vs Best Traditional</div>",
                            unsafe_allow_html=True)
                imp_cols = st.columns(len([c for c in ("dqn","ddpg","ppo") if c in summary]))
                for idx, rl_ctrl in enumerate([c for c in ("dqn","ddpg","ppo") if c in summary]):
                    improvement = (best_trad - summary[rl_ctrl]["mean"]) / best_trad * 100
                    color = "#34d399" if improvement > 0 else "#f87171"
                    with imp_cols[idx]:
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div class='label'>{rl_ctrl.upper()} vs traditional</div>"
                            f"<div class='value' style='color:{color}'>"
                            f"{'▼' if improvement > 0 else '▲'} {abs(improvement):.1f}%</div>"
                            f"<div class='sub'>{compare_metric} {'reduction' if improvement > 0 else 'increase'}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3.5 — TEST & BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("<div class='section-title'>Run Tests &amp; Live Comparison</div>",
                unsafe_allow_html=True)
    st.markdown(
        "Run test mode for all controllers and compare metrics side-by-side. "
        "Results update live from `Outputs/metrics/`."
    )

    # ── Controller selection & test parameters ──
    _bm_col1, _bm_col2 = st.columns([1, 1])
    with _bm_col1:
        _bm_controllers = st.multiselect(
            "Controllers to test",
            ["websters", "sotl", "dqn", "ddpg", "ppo"],
            default=["websters", "sotl", "dqn", "ddpg", "ppo"],
            key="bm_controllers",
        )
    with _bm_col2:
        _bm_simlen = st.number_input("Sim length (s)", value=3600, min_value=600, step=600, key="bm_simlen")
        _bm_scale = st.number_input("Demand scale", value=1.4, min_value=0.5, max_value=3.0, step=0.1, key="bm_scale")

    _bm_col3, _bm_col4 = st.columns([1, 1])
    with _bm_col3:
        _bm_gmin = st.number_input("gmin (s)", value=5, min_value=1, max_value=30, key="bm_gmin")
        _bm_gmax = st.number_input("gmax (s)", value=30, min_value=5, max_value=60, key="bm_gmax")
    with _bm_col4:
        _bm_n_hidden = st.number_input("n_hidden", value=3, min_value=1, max_value=5, key="bm_nhidden")
        _bm_port = st.number_input("Base port", value=9000, min_value=5000, key="bm_port")

    # Run test button
    if st.button("🚀  Run All Tests", key="bm_run_btn"):
        _bm_progress = st.progress(0)
        _bm_status_text = st.empty()
        for idx, ctrl in enumerate(_bm_controllers):
            _bm_status_text.info(f"Testing **{ctrl.upper()}** ({idx+1}/{len(_bm_controllers)})...")
            test_cmd = (
                f"{sys.executable} run_01.py -mode test -tsc {ctrl}"
                f" -n 1 -sim double -nogui -simlen {_bm_simlen}"
                f" -port {_bm_port} -scale {_bm_scale} -demand dynamic -offset 0.25"
                f" -gmin {_bm_gmin} -gmax {_bm_gmax} -y 2 -r 3"
                f" -save_path Outputs/saved_models -n_hidden {_bm_n_hidden} -hidden_act elu"
            )
            if ctrl in ("dqn", "ddpg", "ppo"):
                test_cmd += " -load"
            result = subprocess.run(test_cmd, shell=True, cwd=_SCRIPT_DIR,
                                    capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                st.warning(f"⚠️ {ctrl} test failed (exit {result.returncode})")
            _bm_progress.progress((idx + 1) / len(_bm_controllers))
        _bm_status_text.success("✅ All tests completed!")
        st.rerun()

    # ── Load & display results ──
    st.markdown("<div class='section-title'>Comparison Results</div>", unsafe_allow_html=True)

    _bm_metrics_root = os.path.join(_SCRIPT_DIR, "Outputs", "metrics")
    _bm_metric_names = ["traveltime", "delay", "queue", "speed", "throughput", "waitingtime", "emission"]

    # Colour mapping
    _bm_colours = {
        "websters": "#F44336", "sotl": "#9C27B0",
        "dqn": "#4CAF50", "ddpg": "#2196F3", "ppo": "#FF9800",
    }
    _bm_labels = {
        "websters": "Webster's", "sotl": "SOTL",
        "dqn": "DQN", "ddpg": "DDPG", "ppo": "PPO",
    }

    def _bm_load_metric(tsc, metric, junction=None):
        """Load latest metric data for a controller."""
        if junction:
            paths_to_try = [
                os.path.join(_bm_metrics_root, tsc, "test", metric, junction),
                os.path.join(_bm_metrics_root, tsc, metric, junction),
            ]
        else:
            paths_to_try = [
                os.path.join(_bm_metrics_root, tsc, "test", metric),
                os.path.join(_bm_metrics_root, tsc, metric),
            ]
        for p in paths_to_try:
            if os.path.isdir(p):
                files = sorted(os.listdir(p))
                if files:
                    data = _load_pickle(os.path.join(p, files[-1]))
                    if data is not None:
                        return data
        return None

    # Detect which controllers have test data
    _bm_available = []
    for ctrl in ["websters", "sotl", "dqn", "ddpg", "ppo"]:
        tt = _bm_load_metric(ctrl, "traveltime")
        if tt is not None:
            _bm_available.append(ctrl)

    if not _bm_available:
        st.info("No test results found. Run tests first using the button above.")
    else:
        st.caption(f"Data available for: {', '.join([_bm_labels.get(c, c) for c in _bm_available])}")

        # ── TRAVEL TIME COMPARISON ──
        st.markdown("##### 🚗 Travel Time Distribution")
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#111827")
            ax.set_facecolor("#0d0f14")

            tt_data = {}
            for ctrl in _bm_available:
                tt = _bm_load_metric(ctrl, "traveltime")
                if tt is not None and isinstance(tt, list):
                    tt_data[ctrl] = [v for v in tt if isinstance(v, (int, float))]

            if tt_data:
                box_data = [tt_data[c] for c in tt_data]
                box_labels = [_bm_labels.get(c, c) for c in tt_data]
                box_colours = [_bm_colours.get(c, "#38bdf8") for c in tt_data]

                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                                widths=0.5, showfliers=True)
                for patch, colour in zip(bp["boxes"], box_colours):
                    patch.set_facecolor(colour)
                    patch.set_alpha(0.8)
                for median in bp["medians"]:
                    median.set_color("white")
                    median.set_linewidth(2)
                for flier, colour in zip(bp["fliers"], box_colours):
                    flier.set(marker="o", markerfacecolor="none",
                              markeredgecolor=colour, markersize=3, alpha=0.5)
                for attr in ["caps", "whiskers"]:
                    items = np.array(bp[attr]).reshape(len(box_colours), 2)
                    for item_pair, colour in zip(items, box_colours):
                        for item in item_pair:
                            item.set_color(colour)

                ax.set_ylabel("Travel Time (s)", color="#94a3b8", fontsize=10)
                ax.set_title("Travel Time Comparison", color="#e2e8f0", fontsize=12)
                ax.tick_params(colors="#94a3b8", labelsize=9)
                ax.grid(True, linestyle="--", alpha=0.2, color="#475569")
                ax.spines[:].set_color("#1e2330")

                # Annotate mean/std
                for i, ctrl in enumerate(tt_data):
                    mu = np.mean(tt_data[ctrl])
                    sigma = np.std(tt_data[ctrl])
                    ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                            f"μ={mu:.0f}\nσ={sigma:.0f}",
                            ha="center", va="top", fontsize=7,
                            color=box_colours[i],
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="#111827", edgecolor=box_colours[i], alpha=0.9))

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ── SUMMARY METRICS TABLE ──
        st.markdown("##### 📋 Full Metrics Summary")

        # Build summary data
        _bm_summary = {}
        _bm_junctions = ["gneJ0", "gneJ6"]
        for ctrl in _bm_available:
            _bm_summary[ctrl] = {}
            # Travel time
            tt = _bm_load_metric(ctrl, "traveltime")
            if tt and isinstance(tt, list):
                tt_vals = [v for v in tt if isinstance(v, (int, float))]
                _bm_summary[ctrl]["travel_time"] = np.mean(tt_vals) if tt_vals else None
            else:
                _bm_summary[ctrl]["travel_time"] = None
            # Per-junction metrics
            for junc in _bm_junctions:
                for metric in ["delay", "queue", "speed", "throughput", "waitingtime"]:
                    data = _bm_load_metric(ctrl, metric, junc)
                    if data is not None:
                        if isinstance(data, list):
                            vals = [v for v in data if isinstance(v, (int, float))]
                            _bm_summary[ctrl][f"{metric}_{junc}"] = np.mean(vals) if vals else None
                        else:
                            _bm_summary[ctrl][f"{metric}_{junc}"] = float(np.mean(data))
                    else:
                        _bm_summary[ctrl][f"{metric}_{junc}"] = None
                # Emission (dict format)
                em = _bm_load_metric(ctrl, "emission", junc)
                if em and isinstance(em, list) and isinstance(em[0], dict):
                    co2_total = sum(d.get("co2", 0) for d in em)
                    _bm_summary[ctrl][f"co2_{junc}"] = co2_total
                else:
                    _bm_summary[ctrl][f"co2_{junc}"] = None

        # Display as a table using streamlit
        _bm_display_metrics = [
            ("Travel Time (s)", "travel_time", None),
            ("Delay gneJ0", "delay_gneJ0", None),
            ("Delay gneJ6", "delay_gneJ6", None),
            ("Queue gneJ0", "queue_gneJ0", None),
            ("Speed gneJ0 (m/s)", "speed_gneJ0", None),
            ("Throughput gneJ0", "throughput_gneJ0", None),
            ("Waiting gneJ0", "waitingtime_gneJ0", None),
            ("CO₂ gneJ0", "co2_gneJ0", None),
        ]

        # Header row
        n_ctrls = len(_bm_available)
        header_cols = st.columns([2] + [1] * n_ctrls)
        header_cols[0].markdown("**Metric**")
        for i, ctrl in enumerate(_bm_available):
            header_cols[i + 1].markdown(f"**{_bm_labels.get(ctrl, ctrl)}**")

        # Data rows
        for display_name, key, _ in _bm_display_metrics:
            row_cols = st.columns([2] + [1] * n_ctrls)
            row_cols[0].markdown(display_name)
            # Find best value (lowest for most metrics, highest for speed/throughput)
            vals = []
            for ctrl in _bm_available:
                v = _bm_summary[ctrl].get(key)
                if v is not None:
                    vals.append((ctrl, v))
            if vals:
                if "speed" in key or "throughput" in key:
                    best_ctrl = max(vals, key=lambda x: x[1])[0]
                else:
                    best_ctrl = min(vals, key=lambda x: x[1])[0]
            else:
                best_ctrl = None

            for i, ctrl in enumerate(_bm_available):
                v = _bm_summary[ctrl].get(key)
                if v is not None:
                    if v > 10000:
                        text = f"{v:,.0f}"
                    elif v > 100:
                        text = f"{v:.1f}"
                    else:
                        text = f"{v:.3f}"
                    if ctrl == best_ctrl:
                        row_cols[i + 1].markdown(f"🏆 `{text}`")
                    else:
                        row_cols[i + 1].markdown(f"`{text}`")
                else:
                    row_cols[i + 1].markdown("—")

        # ── BAR CHARTS PER METRIC ──
        st.markdown("##### 📊 Per-Metric Bar Comparison")
        _bm_bar_metric = st.selectbox(
            "Select metric for bar chart",
            ["travel_time", "delay_gneJ0", "queue_gneJ0", "speed_gneJ0",
             "throughput_gneJ0", "waitingtime_gneJ0", "co2_gneJ0"],
            key="bm_bar_metric",
        )

        if HAS_MPL:
            fig2, ax2 = plt.subplots(figsize=(8, 3.5), facecolor="#111827")
            ax2.set_facecolor("#0d0f14")

            bar_ctrls = []
            bar_vals = []
            bar_cols = []
            for ctrl in _bm_available:
                v = _bm_summary[ctrl].get(_bm_bar_metric)
                if v is not None:
                    bar_ctrls.append(_bm_labels.get(ctrl, ctrl))
                    bar_vals.append(v)
                    bar_cols.append(_bm_colours.get(ctrl, "#38bdf8"))

            if bar_vals:
                # Highlight best
                if "speed" in _bm_bar_metric or "throughput" in _bm_bar_metric:
                    best_idx = int(np.argmax(bar_vals))
                else:
                    best_idx = int(np.argmin(bar_vals))
                edge_colors = ["#fbbf24" if i == best_idx else "#00000000" for i in range(len(bar_vals))]

                bars = ax2.bar(bar_ctrls, bar_vals, color=bar_cols, edgecolor=edge_colors, linewidth=2)
                for bar, val in zip(bars, bar_vals):
                    label = f"{val:,.0f}" if val > 1000 else f"{val:.2f}"
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             label, ha="center", va="bottom", color="#94a3b8", fontsize=8)

                metric_title = _bm_bar_metric.replace("_", " ").replace("gneJ0", "(J0)").title()
                ax2.set_ylabel(metric_title, color="#94a3b8", fontsize=9)
                ax2.set_title(f"{metric_title} by Controller", color="#e2e8f0", fontsize=11)
                ax2.tick_params(colors="#94a3b8", labelsize=9)
                ax2.grid(True, axis="y", linestyle="--", alpha=0.2, color="#475569")
                ax2.spines[:].set_color("#1e2330")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

        # ── DRL vs Traditional improvement ──
        _bm_traditional = [c for c in _bm_available if c in ("websters", "sotl")]
        _bm_rl = [c for c in _bm_available if c in ("dqn", "ddpg", "ppo")]
        if _bm_traditional and _bm_rl:
            st.markdown("##### 🎯 DRL Improvement over Traditional Methods")
            best_trad_tt = min(
                _bm_summary[c]["travel_time"] for c in _bm_traditional
                if _bm_summary[c].get("travel_time") is not None
            )
            imp_cols = st.columns(len(_bm_rl))
            for idx, ctrl in enumerate(_bm_rl):
                with imp_cols[idx]:
                    rl_tt = _bm_summary[ctrl].get("travel_time")
                    if rl_tt is not None:
                        improvement = (best_trad_tt - rl_tt) / best_trad_tt * 100
                        color = "#34d399" if improvement > 0 else "#f87171"
                        icon = "▼" if improvement > 0 else "▲"
                        word = "faster" if improvement > 0 else "slower"
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<div class='label'>{_bm_labels.get(ctrl, ctrl)}</div>"
                            f"<div class='value' style='color:{color}'>"
                            f"{icon} {abs(improvement):.1f}%</div>"
                            f"<div class='sub'>{word} than best traditional ({best_trad_tt:.0f}s)</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

        # ── DRL ranking ──
        if len(_bm_rl) > 1:
            st.markdown("##### 🥇 DRL Ranking (by Travel Time)")
            rl_ranked = sorted(
                [(c, _bm_summary[c]["travel_time"]) for c in _bm_rl
                 if _bm_summary[c].get("travel_time") is not None],
                key=lambda x: x[1]
            )
            for rank, (ctrl, tt) in enumerate(rl_ranked, 1):
                medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"{rank}."
                st.markdown(f"{medal} **{_bm_labels.get(ctrl, ctrl)}** — {tt:.1f}s mean travel time")

        # Auto-refresh while tests running
        if st.button("🔄  Refresh Results", key="bm_refresh"):
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE LOGS
# ═════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("<div class='section-title'>Process Output</div>", unsafe_allow_html=True)

    _drain_queue()

    col_l1, col_l2, col_l3 = st.columns([2, 1, 1])
    with col_l1:
        st.markdown(f"**{len(st.session_state.log_lines)}** lines captured  ·  "
                    f"Status: {_status_html(st.session_state.status)}",
                    unsafe_allow_html=True)
    with col_l2:
        if st.button("🔄  Refresh Logs"):
            _drain_queue()
            st.rerun()
    with col_l3:
        if st.button("🗑️  Clear Logs"):
            st.session_state.log_lines = []
            st.rerun()

    log_text = "\n".join(st.session_state.log_lines[-200:])  # last 200 lines
    st.markdown(
        f"<div class='log-box'>{log_text if log_text else 'No output yet. Start a run.'}</div>",
        unsafe_allow_html=True
    )

    if st.session_state.status == "running":
        time.sleep(3)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div class="metric-card" style="margin-bottom:1.2rem">
  <div class="label">Dissertation</div>
  <div class="value" style="font-size:1.1rem;color:#e2e8f0">
    Adaptive Traffic Signal Control using Deep Reinforcement Learning with SUMO
  </div>
  <div class="sub" style="margin-top:6px">
    BITS Pilani · MTech AI & ML · May 2026 · ID: 2023AD05075
  </div>
</div>
""", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### Algorithms")
        algo_info = {
            "Webster's":    "Fixed-cycle, formula-based timing. Baseline comparison.",
            "SOTL":         "Self-Organising Traffic Lights. Reacts to vehicle thresholds.",
            "Max Pressure": "Maximises intersection throughput pressure. Rule-based adaptive.",
            "Uniform":      "Equal-time round-robin phase cycling. Simplest baseline.",
            "DQN":          "Deep Q-Network. Learns optimal phase selection policy.",
            "DDPG":         "Deep Deterministic Policy Gradient. Continuous action (duration).",
            "PPO":          "Proximal Policy Optimization. Stable on-policy actor-critic.",
        }
        for algo, desc in algo_info.items():
            st.markdown(f"**{algo}** — {desc}")

    with col_a2:
        st.markdown("#### Key Files")
        file_info = {
            "run_01.py":                   "Entry point",
            "src/argparse.py":          "All CLI parameters",
            "src/distprocs.py":         "Process orchestration",
            "src/simproc.py":           "Actor simulation process",
            "src/learnerproc.py":       "Neural network trainer",
            "src/sumosim.py":           "SUMO + TraCI interface",
            "src/neuralnet.py":         "Keras model wrapper",
            "src/trafficmetrics.py":    "Delay / queue metrics",
            "src/vehiclegen.py":        "Dynamic vehicle generation",
            "src/networkdata.py":       "Road network parser",
        }
        for fp, role in file_info.items():
            st.markdown(f"`{fp}` — {role}")

    st.markdown("#### Objectives (from dissertation)")
    objectives = [
        "~20% reduction in average waiting time vs fixed-time control",
        "~20% reduction in congestion",
        "~20% improvement in traffic throughput",
        "Significant CO₂ emission and fuel consumption reduction",
        "Scalable MARL coordination across multiple intersections",
    ]
    for obj in objectives:
        st.markdown(f"- {obj}")

    st.markdown("#### How to use this dashboard")
    st.markdown("""
1. Set all parameters in the **sidebar** — they map 1-to-1 with the original `argparse.py` flags.
2. Go to **▶ Run** tab and click **Start Run** to launch `run_01.py` as a background process.
3. Watch live output in **📋 Live Logs** — click Refresh or it auto-refreshes while running.
4. After training, switch mode to **test**, enable **Load weights**, and run again.
5. View per-intersection metrics in **📊 Results**.
6. Compare DQN / PPO against Webster's etc. in **⚖️ Compare**.
    """)