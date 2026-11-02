# app.py
# ═══════════════════════════════════════════════════════════════════
#  How Much Data Does Your Model Really Need?
#  A VC-Dimension-Guided Data Budget Estimator
#  Subtitle: Bridging PAC Learning Theory with Empirical Practice
# ═══════════════════════════════════════════════════════════════════
#  Run:  streamlit run app.py
# ═══════════════════════════════════════════════════════════════════

import sys, os
# Ensure project root is on the path so `src` and `config` import cleanly
sys.path.insert(0, os.path.dirname(__file__))

import math
import io
import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_EPSILON, DEFAULT_DELTA,
    TRAIN_FRACTIONS, AVAILABLE_MODELS, BUILTIN_DATASETS,
    MIN_SAMPLES_REQUIRED, DEFAULT_N_SEEDS,
)
from src.data_loader import get_dataset, describe_dataset
from src.preprocessing import split_and_preprocess
from src.vc_dimension import estimate_vc_dimension
from src.pac_bound import calculate_theoretical_samples, calculate_agnostic_samples, get_pac_bounds
from src.learning_curve import (generate_learning_curve,
                                find_empirical_sample_requirement)
from src.curve_fitting import fit_power_law, predict_sample_size
from src.evaluation import run_full_experiment, build_comparison_dataframe
from src.visualization import (
    plot_learning_curve, plot_accuracy_curve, plot_train_vs_val,
    plot_power_law_fit, plot_comparison_bars, plot_multi_model_curves,
    plot_dataset_2d,
)
from src.report_generator import generate_html_report, save_report_to_disk

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="VC-Dimension Data Budget Estimator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for High-Tech Dark Theme ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hero Header */
.hero-banner {
    background: radial-gradient(circle at 10% 20%, rgba(20, 30, 48, 0.95) 0%, rgba(15, 23, 42, 0.98) 90%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 30%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    font-weight: 500;
    color: #94a3b8;
    margin: 0 0 1rem 0;
}
.hero-tags {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    align-items: center;
}
.hero-pill {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.28);
}

/* KPI Card Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.8rem;
}
.kpi-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: left;
    transition: transform 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.kpi-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
}

/* Stepper Workflow */
.stepper-container {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.8rem;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.1rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.step-item:last-child {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
}
.step-badge {
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.95rem;
    flex-shrink: 0;
}
.step-badge.done {
    background: #059669;
    color: white;
}
.step-badge.active {
    background: #0284c7;
    color: white;
    box-shadow: 0 0 12px rgba(2, 132, 199, 0.5);
}
.step-badge.pending {
    background: #1e293b;
    color: #64748b;
    border: 1px solid #334155;
}
.step-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #f1f5f9;
    margin-bottom: 0.15rem;
}
.step-desc {
    font-size: 0.82rem;
    color: #94a3b8;
}

/* Formula & Info Boxes */
.formula-box {
    background: #090d16;
    border-left: 4px solid #38bdf8;
    padding: 1.1rem 1.4rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: #e2e8f0;
    white-space: pre-wrap;
    margin: 1rem 0;
    line-height: 1.5;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.info-card {
    background: #131d31;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    color: #cbd5e1;
    font-size: 0.9rem;
    margin: 0.8rem 0;
    border: 1px solid #233554;
}
.success-card {
    background: rgba(16, 185, 129, 0.1);
    border-left: 4px solid #10b981;
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    color: #6ee7b7;
    font-size: 0.88rem;
    margin: 0.8rem 0;
    border-top: 1px solid rgba(16, 185, 129, 0.2);
    border-right: 1px solid rgba(16, 185, 129, 0.2);
    border-bottom: 1px solid rgba(16, 185, 129, 0.2);
}
.warning-card {
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #f59e0b;
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    color: #fcd34d;
    font-size: 0.88rem;
    margin: 0.8rem 0;
    border-top: 1px solid rgba(245, 158, 11, 0.2);
    border-right: 1px solid rgba(245, 158, 11, 0.2);
    border-bottom: 1px solid rgba(245, 158, 11, 0.2);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: #090e1a !important;
    border-right: 1px solid #1e293b;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "dataset": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "preprocessor": None,
        "experiment_results": [],
        "comparison_df": None,
        "epsilon": DEFAULT_EPSILON,
        "delta": DEFAULT_DELTA,
        "selected_models": ["Perceptron"],
        "dataset_name": None,
        "n_features": None,
        "ran_experiment": False,
        "n_seeds": DEFAULT_N_SEEDS,
        "pac_setting": "Realizable (Blumer et al.)",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-load default dataset on initial launch
    if st.session_state.get("dataset") is None:
        try:
            ds = get_dataset("Iris")
            X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds["X"], ds["y"])
            st.session_state.update({
                "dataset": ds,
                "X_train": X_tr,
                "X_test": X_te,
                "y_train": y_tr,
                "y_test": y_te,
                "preprocessor": prep,
                "dataset_name": ds["dataset_name"],
                "n_features": ds["n_features"],
                "experiment_results": [],
                "comparison_df": None,
                "ran_experiment": False,
            })
        except Exception:
            pass

_init_state()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Experiment Setup")
    st.caption("Configure dataset, PAC specs, and model families.")
    st.markdown("---")

    # ── 1. Dataset Selection ──────────────────────────────────────
    st.markdown("#### 1. Dataset Selection")
    dataset_choice = st.selectbox(
        "Choose dataset",
        BUILTIN_DATASETS + ["CSV Upload"],
        key="ds_choice",
        help="Select a benchmark dataset or upload your own CSV file.",
    )

    uploaded_file = None
    target_column = ""
    col_options = []

    if dataset_choice == "CSV Upload":
        uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"], key="csv_upload"
        )
        if uploaded_file is not None:
            try:
                preview_df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                col_options = list(preview_df.columns)
                target_column = st.selectbox(
                    "Target column (labels)", col_options, key="target_col"
                )
            except Exception:
                st.error("Could not preview CSV. Please verify formatting.")

    load_btn = st.button("📥 Load Dataset", key="btn_load", use_container_width=True)
    if load_btn:
        try:
            with st.spinner("Loading and preprocessing dataset..."):
                ds = get_dataset(dataset_choice, uploaded_file, target_column)
                X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(
                    ds["X"], ds["y"]
                )
                st.session_state.update({
                    "dataset": ds,
                    "X_train": X_tr,
                    "X_test": X_te,
                    "y_train": y_tr,
                    "y_test": y_te,
                    "preprocessor": prep,
                    "dataset_name": ds["dataset_name"],
                    "n_features": ds["n_features"],
                    "experiment_results": [],
                    "comparison_df": None,
                    "ran_experiment": False,
                })
                st.success(f"✓ Loaded: **{ds['dataset_name']}** ({ds['n_samples']} samples, {ds['n_features']} features)")
        except Exception as exc:
            st.error(f"Dataset error: {exc}")

    # Dataset status indicator
    if st.session_state["dataset"] is not None:
        ds_curr = st.session_state["dataset"]
        st.caption(f"🟢 Active: **{ds_curr['dataset_name']}** · {ds_curr['n_samples']} samples · {ds_curr['n_features']} features")
    else:
        st.caption("🟡 No dataset loaded yet. Click **Load Dataset** above.")

    st.markdown("---")

    # ── 2. PAC Learning Parameters ────────────────────────────────
    st.markdown("#### 2. PAC Parameters")
    epsilon = st.slider(
        "Target Error Tolerance (ε)",
        min_value=0.01,
        max_value=0.50,
        value=float(st.session_state["epsilon"]),
        step=0.01,
        help="Maximum acceptable generalization error rate (0 < ε < 1). Smaller ε requires more data.",
        key="sl_epsilon",
    )
    delta = st.slider(
        "Failure Probability (δ)",
        min_value=0.01,
        max_value=0.30,
        value=float(st.session_state["delta"]),
        step=0.01,
        help="Acceptable probability that the algorithm fails to achieve error ≤ ε (Confidence = 1 - δ).",
        key="sl_delta",
    )
    st.session_state["epsilon"] = epsilon
    st.session_state["delta"] = delta

    pac_setting = st.selectbox(
        "PAC Theoretical Setting",
        ["Realizable (Blumer et al.)", "Agnostic (Anthony & Bartlett)"],
        index=0 if st.session_state.get("pac_setting") == "Realizable (Blumer et al.)" else 1,
        key="pac_setting_select",
        help="Realizable assumes zero-noise target concept in H. Agnostic allows noisy labels and model misspecification.",
    )
    st.session_state["pac_setting"] = pac_setting

    st.markdown("---")

    # ── 3. Model Family Selection ─────────────────────────────────
    st.markdown("#### 3. Learning Models")
    selected_models = st.multiselect(
        "Select models to evaluate",
        AVAILABLE_MODELS,
        default=st.session_state.get("selected_models", ["Perceptron"]),
        key="ms_models",
        help="Select one or more classifiers to compare theoretical vs empirical sample complexity.",
    )
    st.session_state["selected_models"] = selected_models

    model_kwargs = {}
    if "MLP (Multilayer Perceptron)" in selected_models:
        with st.expander("⚙️ MLP Architecture", expanded=False):
            hl_str = st.text_input("Hidden layers (units)", "100", key="mlp_hl", help="Comma-separated layer sizes, e.g. 100 or 50,25")
            try:
                hl = tuple(int(x.strip()) for x in hl_str.split(",") if x.strip())
            except ValueError:
                hl = (100,)
                st.warning("Invalid layer format; using (100,).")
            model_kwargs["hidden_layer_sizes"] = hl

    if "AdaBoost" in selected_models:
        with st.expander("⚙️ AdaBoost Ensembles", expanded=False):
            n_est = st.slider("Number of Decision Stumps (T)", 10, 200, 50, 10, key="ada_n_est")
            model_kwargs["n_estimators"] = n_est

    # ── 4. Empirical Variance / Random Seeds ──────────────────────
    with st.expander("🎲 Empirical Variance (Error Bars)", expanded=False):
        n_seeds = st.slider(
            "Repeated random seeds",
            min_value=1,
            max_value=5,
            value=int(st.session_state["n_seeds"]),
            step=1,
            help="Average learning curve metrics over multiple random stratified subsets to compute empirical standard deviation (error bars).",
            key="sl_seeds",
        )
        st.session_state["n_seeds"] = n_seeds

    st.markdown("---")

    # ── 5. Run & Reset Actions ────────────────────────────────────
    run_disabled = (
        st.session_state["X_train"] is None
        or not selected_models
    )

    if run_disabled:
        if st.session_state["X_train"] is None:
            st.warning("👉 Load a dataset first to enable experiment execution.")
        elif not selected_models:
            st.warning("👉 Select at least one model to enable experiment.")

    if st.button("🚀 Run Experiment", disabled=run_disabled, key="btn_run", type="primary", use_container_width=True):
        results = []
        prog = st.progress(0, "Initiating experiment suite…")
        n_mods = len(selected_models)

        for i, mname in enumerate(selected_models):
            prog.progress(i / n_mods, f"Training & evaluating {mname} ({i+1}/{n_mods})…")
            try:
                r = run_full_experiment(
                    model_name=mname,
                    X_train=st.session_state["X_train"],
                    y_train=st.session_state["y_train"],
                    X_test=st.session_state["X_test"],
                    y_test=st.session_state["y_test"],
                    epsilon=epsilon,
                    delta=delta,
                    n_features=st.session_state["n_features"],
                    n_seeds=st.session_state["n_seeds"],
                    **model_kwargs,
                )
                results.append(r)
            except Exception as exc:
                st.error(f"Error executing {mname}: {exc}")
                st.code(traceback.format_exc())

        prog.progress(1.0, "Experiments completed successfully!")
        if results:
            cdf = build_comparison_dataframe(results)
            st.session_state["experiment_results"] = results
            st.session_state["comparison_df"] = cdf
            st.session_state["ran_experiment"] = True
            st.success(f"✓ Evaluated {len(results)} model(s) across {len(TRAIN_FRACTIONS)} sample fractions.")

    if st.button("🔄 Reset Environment", key="btn_reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        _init_state()
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# MAIN AREA: HERO & TABS
# ══════════════════════════════════════════════════════════════════

# Hero Header
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">How Much Data Does Your Model Really Need?</div>
  <div class="hero-subtitle">A VC-Dimension-Guided Data Budget Estimator — Bridging PAC Learning Theory with Empirical Practice</div>
  <div class="hero-tags">
    <span class="hero-pill">PAC Learning</span>
    <span class="hero-pill">Vapnik-Chervonenkis Dimension</span>
    <span class="hero-pill">Sample Complexity</span>
    <span class="hero-pill">Power-Law Learning Curves</span>
    <span class="hero-pill">Empirical Validation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_home, tab_data, tab_theory, tab_curves, tab_comparison, tab_about = st.tabs([
    "🏠 Home Dashboard",
    "📂 Dataset Inspector",
    "🔬 Theory & PAC Bounds",
    "📈 Learning Curves",
    "⚖️ Model Comparison",
    "📖 Documentation & Viva",
])


# ──────────────────────────────────────────────────────────────────
# TAB 1 – HOME DASHBOARD
# ──────────────────────────────────────────────────────────────────
with tab_home:
    st.markdown("### 📊 Active Experiment Overview")

    # Dynamic KPI Cards Row
    ds_active = st.session_state["dataset"]
    exp_results = st.session_state["experiment_results"]
    has_run = st.session_state["ran_experiment"] and len(exp_results) > 0

    # Theoretical bound preview
    est_theory_bound = "N/A"
    if ds_active is not None and st.session_state["selected_models"]:
        first_model = st.session_state["selected_models"][0]
        vc_est = estimate_vc_dimension(first_model, ds_active["n_features"])
        if st.session_state["pac_setting"].startswith("Realizable"):
            b = calculate_theoretical_samples(st.session_state["epsilon"], st.session_state["delta"], vc_est["vc_dim"])
        else:
            b = calculate_agnostic_samples(st.session_state["epsilon"], st.session_state["delta"], vc_est["vc_dim"])
        est_theory_bound = f"{b['m']:,}"

    # Empirical requirement preview
    emp_achieved_str = "Pending Run"
    if has_run:
        emp_counts = [r["empirical"]["sample_count"] for r in exp_results if r["empirical"]["sample_count"] is not None]
        if emp_counts:
            emp_achieved_str = f"{min(emp_counts):,} samples"
        else:
            emp_achieved_str = "Target not met"

    n_samples_str = f"{ds_active['n_samples']:,}" if ds_active else "None"
    n_features_str = f"{ds_active['n_features']} features" if ds_active else "—"
    n_train_str = f"{st.session_state['X_train'].shape[0]:,}" if st.session_state["X_train"] is not None else "0"
    active_ds_name = ds_active["dataset_name"] if ds_active else "No Dataset Loaded"
    selected_mods_str = ", ".join(st.session_state["selected_models"]) if st.session_state["selected_models"] else "None"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Active Dataset</div>
        <div class="kpi-value" style="font-size: 1.25rem;">{active_ds_name}</div>
        <div class="kpi-sub">{n_samples_str} total samples ({n_features_str})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Selected Model(s)</div>
        <div class="kpi-value" style="font-size: 1.15rem;">{len(st.session_state['selected_models'])} Selected</div>
        <div class="kpi-sub">{selected_mods_str[:28]}{'...' if len(selected_mods_str) > 28 else ''}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">PAC Specs (ε, δ)</div>
        <div class="kpi-value">ε={st.session_state['epsilon']:.2f}</div>
        <div class="kpi-sub">Confidence: {(1-st.session_state['delta'])*100:.0f}% (δ={st.session_state['delta']:.2f})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Theoretical PAC Bound (m)</div>
        <div class="kpi-value" style="color: #c084fc;">{est_theory_bound}</div>
        <div class="kpi-sub">{st.session_state['pac_setting'].split(' ')[0]} bound</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Actual Training Budget</div>
        <div class="kpi-value" style="color: #38bdf8;">{n_train_str}</div>
        <div class="kpi-sub">80% stratified train split</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Empirical Sample Need</div>
        <div class="kpi-value" style="color: #34d399;">{emp_achieved_str}</div>
        <div class="kpi-sub">Observed on validation set</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns([1.6, 1.0])

    with c_left:
        st.markdown("#### 🧭 Interactive Workflow")
        
        # Determine status of each step
        s1_done = ds_active is not None
        s2_done = True
        s3_done = len(st.session_state["selected_models"]) > 0
        s4_done = st.session_state["X_train"] is not None
        s5_done = has_run
        s6_done = has_run

        def badge_cls(done, active):
            if done: return "done", "✓"
            if active: return "active", "●"
            return "pending", "○"

        b1_cls, b1_sym = badge_cls(s1_done, not s1_done)
        b2_cls, b2_sym = badge_cls(s2_done, False)
        b3_cls, b3_sym = badge_cls(s3_done, s1_done and not s3_done)
        b4_cls, b4_sym = badge_cls(s4_done, s1_done and not s4_done)
        b5_cls, b5_sym = badge_cls(s5_done, s4_done and s3_done and not s5_done)
        b6_cls, b6_sym = badge_cls(s6_done, False)

        st.markdown(f"""
        <div class="stepper-container">
          <div class="step-item">
            <div class="step-badge {b1_cls}">{b1_sym}</div>
            <div>
              <div class="step-title">Step 1: Choose Dataset</div>
              <div class="step-desc">Select from Iris, Breast Cancer, Wine, Synthetic Linearly Separable, or Synthetic Non-Linear.</div>
            </div>
          </div>
          <div class="step-item">
            <div class="step-badge {b2_cls}">{b2_sym}</div>
            <div>
              <div class="step-title">Step 2: Configure PAC Parameters (ε, δ)</div>
              <div class="step-desc">Define target generalization error tolerance ε and acceptable risk probability δ.</div>
            </div>
          </div>
          <div class="step-item">
            <div class="step-badge {b3_cls}">{b3_sym}</div>
            <div>
              <div class="step-title">Step 3: Select Learning Model(s)</div>
              <div class="step-desc">Choose Perceptron (exact VC dim = n+1), MLP, or AdaBoost ensemble.</div>
            </div>
          </div>
          <div class="step-item">
            <div class="step-badge {b4_cls}">{b4_sym}</div>
            <div>
              <div class="step-title">Step 4: Load & Preprocess Dataset</div>
              <div class="step-desc">Click 'Load Dataset' to perform clean, leakage-free 80/20 train/test splitting and scaling.</div>
            </div>
          </div>
          <div class="step-item">
            <div class="step-badge {b5_cls}">{b5_sym}</div>
            <div>
              <div class="step-title">Step 5: Run Empirical Experiment</div>
              <div class="step-desc">Click 'Run Experiment' in the sidebar to generate learning curves across 10 sample fractions.</div>
            </div>
          </div>
          <div class="step-item">
            <div class="step-badge {b6_cls}">{b6_sym}</div>
            <div>
              <div class="step-title">Step 6: Analyze Theoretical vs Empirical Results</div>
              <div class="step-desc">Inspect learning curves, power-law fits, sample budgets, and understand why theory is conservative.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("#### 💡 The Core Problem")
        st.markdown(r"""
        <div class="info-card">
        <strong>The Fundamental Question:</strong><br>
        <em>"How many training samples do I really need to achieve target accuracy with high confidence?"</em>
        <br><br>
        Machine learning practitioners face two radically different answers:
        <ul style="margin: 0.5rem 0 0 1rem; padding: 0;">
          <li><strong>Theoretical PAC Bound:</strong> Worst-case mathematical guarantee under arbitrary adversarial distributions. Extremely conservative.</li>
          <li><strong>Empirical Learning Curve:</strong> Real performance measured on your dataset with observed sample sizes.</li>
          <li><strong>Power-Law Extrapolation:</strong> $error(m) = a \cdot m^{-b} + c$ fitting the empirical rate of progress.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        if not has_run:
            st.info("💡 **Quick Start:** Click **Load Dataset** in the sidebar, then click **🚀 Run Experiment** to begin.")
        else:
            st.success("✓ Experiment complete! Explore the **Theory**, **Learning Curves**, and **Comparison** tabs above.")


# ──────────────────────────────────────────────────────────────────
# TAB 2 – DATASET INSPECTOR
# ──────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### 📂 Dataset Exploration & Characteristics")

    if st.session_state["dataset"] is None:
        st.markdown("""
        <div class="warning-card">
        ⚠️ <strong>No Dataset Loaded:</strong> Open the sidebar via <strong>&gt;&gt;</strong> in the top-left to select datasets, or click below to immediately load the default dataset.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Load Default Dataset (Iris)", key="btn_inline_load_iris", use_container_width=True):
            ds = get_dataset("Iris")
            X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds["X"], ds["y"])
            st.session_state.update({
                "dataset": ds,
                "X_train": X_tr,
                "X_test": X_te,
                "y_train": y_tr,
                "y_test": y_te,
                "preprocessor": prep,
                "dataset_name": ds["dataset_name"],
                "n_features": ds["n_features"],
                "experiment_results": [],
                "comparison_df": None,
                "ran_experiment": False,
            })
            st.rerun()
    else:
        ds = st.session_state["dataset"]
        info = describe_dataset(ds)

        # Overview Metrics Row
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        m_c1.metric("Total Samples", f"{info['n_samples']:,}")
        m_c2.metric("Input Features", info["n_features"])
        m_c3.metric("Classes", info["n_classes"])
        m_c4.metric("Missing Values", info["missing_values"])
        m_c5.metric("Train / Test Split", f"{st.session_state['X_train'].shape[0]} / {st.session_state['X_test'].shape[0]}")

        st.markdown("---")

        col_viz, col_stats = st.columns([1.1, 1.0])

        with col_viz:
            st.markdown("#### 🎨 Feature Space Projection (First 2 Features)")
            try:
                fig_2d = plot_dataset_2d(
                    ds["X"].to_numpy(),
                    ds["y"].to_numpy(),
                    info["feature_names"],
                    dataset_name=ds["dataset_name"],
                )
                st.pyplot(fig_2d, use_container_width=True)
                st.caption(
                    "2D scatter plot colored by class label. Linearly separable datasets "
                    "allow linear classifiers (Perceptron) to achieve zero error, whereas "
                    "overlapping or non-linear geometries require non-linear models or agnostic PAC bounds."
                )
            except Exception as e:
                st.error(f"Could not render 2D projection: {e}")

        with col_stats:
            st.markdown("#### ⚖️ Class Distribution")
            class_data = []
            for cls_name, count in info["class_distribution"].items():
                pct = info["class_proportions"].get(cls_name, 0)
                class_data.append({"Class Label": str(cls_name), "Samples": count, "Percentage": f"{pct:.1f}%"})
            st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)

            st.markdown("#### 🛡️ Preprocessing & Data Leakage Prevention")
            st.markdown("""
            <div class="info-card">
            <strong>Data Integrity Guarantee:</strong><br>
            • Train/test split: <strong>80% training / 20% held-out test</strong> (stratified by class).<br>
            • Feature imputation (median) and standardization (zero mean, unit variance) are fit <strong>strictly on the training split</strong>.<br>
            • The test set is completely held out and never seen during preprocessor fitting or model training.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📋 Dataset Preview & Statistical Summary")
        preview_df = pd.concat([ds["X"].head(10), ds["y"].rename(info["target_name"]).head(10)], axis=1)
        st.dataframe(preview_df, use_container_width=True)

        with st.expander("📊 Detailed Feature Statistics (Descriptive)", expanded=False):
            st.dataframe(ds["X"].describe(), use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 – THEORY & PAC BOUNDS
# ──────────────────────────────────────────────────────────────────
with tab_theory:
    st.markdown("### 🔬 PAC Learning Theory & VC Dimension Foundations")

    st.markdown(r"""
    **Probably Approximately Correct (PAC) learning** is the foundational theoretical framework of statistical learning theory, introduced by Leslie Valiant (1984) and extended by Blumer, Ehrenfeucht, Haussler, and Warmuth (1989).
    
    Given:
    - **Target Error Tolerance (ε):** An accuracy target where true error $\text{err}(h) \le \epsilon$.
    - **Failure Probability (δ):** The probability of drawing an unrepresentative training set, where confidence is $\ge 1 - \delta$.
    - **VC Dimension (d):** The capacity / complexity of the hypothesis class $\mathcal{H}$.
    """)

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("#### 1. Realizable Setting (Blumer et al., 1989)")
        st.markdown(r"""
        Assumes the true concept $c^*$ is **contained in hypothesis class** $\mathcal{H}$ ($c^* \in \mathcal{H}$), meaning a hypothesis exists that can achieve **zero empirical training error**.
        """)
        st.markdown("""
<div class="formula-box">
m ≥ (1 / ε) × [ 4 × log₂(2/δ) + 8 × d × log₂(13/ε) ]

Key Properties:
• Sample complexity scales as O((d / ε) · log(1/ε) + (1 / ε) · log(1/δ))
• Linear dependency on 1/ε because zero-error hypotheses shrink rapidly
• Distribution-free: holds for ANY underlying data distribution D
</div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("#### 2. Agnostic Setting (Anthony & Bartlett, 1999)")
        st.markdown(r"""
        Applies when real-world data contains **label noise**, or the hypothesis class is **misspecified** ($c^* \notin \mathcal{H}$). We must guarantee uniform convergence of empirical to true error.
        """)
        st.markdown("""
<div class="formula-box">
m ≥ (8 / ε²) × [ 2 × d × ln(8e / ε) + ln(4 / δ) ]

Key Properties:
• Sample complexity scales as O((d / ε²) · log(1/ε) + (1 / ε²) · log(1/δ))
• Quadratic dependency on 1/ε² due to Hoeffding/Chernoff uniform bounds
• Guarantees R(ĥ) ≤ inf_{h∈H} R(h) + ε with probability ≥ 1 - δ
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 VC Dimension by Model Family")

    vc_rows = [
        {
            "Model Family": "Perceptron (Linear Classifier)",
            "VC Dimension (d)": "d = n + 1",
            "Classification Type": "Exact (points in general position)",
            "Mathematical Foundation": "An n-dimensional hyperplane can shatter at most n+1 affine-independent points. Theorem 3.1 (Vapnik, 1998).",
        },
        {
            "Model Family": "MLP (Neural Network)",
            "VC Dimension (d)": "d ≈ Σ(units_i × units_{i+1} + units_{i+1})",
            "Classification Type": "Approximate / Illustrative bound",
            "Mathematical Foundation": "Piecewise-linear networks have VC dimension bounded by O(W · log W), where W is total trainable weights. Anthony & Bartlett (1999).",
        },
        {
            "Model Family": "AdaBoost (Decision Stumps)",
            "VC Dimension (d)": "d ≈ T × d_base × log(T × d_base)",
            "Classification Type": "Ensemble upper bound",
            "Mathematical Foundation": "Ensembles of T decision stumps (d_base = 2n). Bounded by Schapire & Singer (1999).",
        },
    ]
    st.dataframe(pd.DataFrame(vc_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🧮 Interactive Sample Budget Calculator")

    eps_curr = st.session_state["epsilon"]
    dlt_curr = st.session_state["delta"]

    if st.session_state["dataset"] is not None and st.session_state["selected_models"]:
        curr_ds = st.session_state["dataset"]
        n_feat = curr_ds["n_features"]

        for m_name in st.session_state["selected_models"]:
            vc_info = estimate_vc_dimension(m_name, n_feat)
            bounds = get_pac_bounds(eps_curr, dlt_curr, vc_info["vc_dim"])
            real_m = bounds["realizable"]["m"]
            agno_m = bounds["agnostic"]["m"]

            with st.expander(f"📌 {m_name} — VC Dimension d = {vc_info['vc_dim']:,} ({vc_info['label']})", expanded=True):
                c_calc1, c_calc2 = st.columns(2)

                with c_calc1:
                    st.markdown("##### Realizable PAC Bound (Blumer et al.)")
                    st.markdown(f'<div class="formula-box">{bounds["realizable"]["substituted"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="success-card">Theoretical Bound: <strong>m ≥ {real_m:,} samples</strong></div>', unsafe_allow_html=True)

                with c_calc2:
                    st.markdown("##### Agnostic PAC Bound (Anthony & Bartlett)")
                    st.markdown(f'<div class="formula-box">{bounds["agnostic"]["substituted"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="warning-card">Agnostic Bound: <strong>m ≥ {agno_m:,} samples</strong></div>', unsafe_allow_html=True)

                st.caption(f"**Interpretation:** To mathematically guarantee true error ≤ {eps_curr:.2f} with {(1-dlt_curr)*100:.0f}% confidence under any distribution, PAC theory requires at least {real_m:,} samples in the realizable setting and {agno_m:,} samples in the noisy agnostic setting.")
    else:
        st.info("Load a dataset and select models in the sidebar to compute live substituted sample complexity bounds.")


# ──────────────────────────────────────────────────────────────────
# TAB 4 – LEARNING CURVES
# ──────────────────────────────────────────────────────────────────
with tab_curves:
    st.markdown("### 📈 Empirical Learning Curves & Sample Complexity")

    if not st.session_state["ran_experiment"]:
        st.markdown("""
        <div class="warning-card">
        ⚠️ <strong>Experiment Not Yet Run:</strong> Click <strong>🚀 Run Experiment</strong> in the sidebar to generate empirical learning curves.
        </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state["experiment_results"]
        eps = st.session_state["epsilon"]

        # Multi-model overlay if multiple models were run
        if len(results) > 1:
            st.markdown("#### 🌐 Multi-Model Validation Error Comparison")
            fig_multi = plot_multi_model_curves(results, eps)
            st.pyplot(fig_multi, use_container_width=True)
            st.markdown("---")

        for r in results:
            mname = r["model_name"]
            lc = r["learning_curve"]
            pac_m = r["pac_bound"].get("m")

            st.markdown(f"#### 🤖 {mname}")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Error Rate vs Training Samples (with ±1σ bounds)**")
                fig_err = plot_learning_curve(lc, eps, model_name=mname, theory_m=pac_m)
                st.pyplot(fig_err, use_container_width=True)
                st.caption("Training error (blue) and validation error (orange) with empirical standard deviation shaded across random seeds.")

            with c2:
                st.markdown("**Accuracy vs Training Samples**")
                fig_acc = plot_accuracy_curve(lc, eps, model_name=mname)
                st.pyplot(fig_acc, use_container_width=True)
                st.caption(f"Target accuracy threshold is 1 - ε = {1 - eps:.2f} ({(1 - eps)*100:.1f}%).")

            c3, c4 = st.columns(2)

            with c3:
                st.markdown("**Power-Law Curve Fit & Extrapolation**")
                fig_fit = plot_power_law_fit(lc, r["curve_fit"], r["prediction"], eps, model_name=mname)
                st.pyplot(fig_fit, use_container_width=True)
                st.caption("Power-law fit: error(m) = a · m^(-b) + c. Vertical line indicates predicted sample size m* needed to reach target error.")

            with c4:
                st.markdown("**Generalization Gap (Train vs Validation Error)**")
                if len(lc["train_errors"]) >= 3:
                    fig_tv = plot_train_vs_val(lc, model_name=mname)
                    st.pyplot(fig_tv, use_container_width=True)
                    st.caption("Points closer to the diagonal represent smaller generalization gaps.")

            # Diagnostic badges
            emp = r["empirical"]
            fit = r["curve_fit"]
            pred = r["prediction"]

            if emp["achieved"]:
                st.markdown(f'<div class="success-card">✅ <strong>Empirical Target Achieved:</strong> {emp["note"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning-card">⚠️ <strong>Empirical Target Not Achieved in Tested Range:</strong> {emp["note"]}</div>', unsafe_allow_html=True)

            if fit["success"] and pred["achieved"]:
                st.markdown(f'<div class="info-card">📐 <strong>Power-Law Prediction:</strong> {pred["note"]}</div>', unsafe_allow_html=True)

            # Raw Data Expander
            with st.expander(f"📋 Raw Learning Curve Data — {mname}", expanded=False):
                raw_dict = {
                    "Training Samples (m)": lc["sample_counts"],
                    "Train Error (Mean)": [f"{e:.4f}" for e in lc["train_errors"]],
                    "Val Error (Mean)": [f"{e:.4f}" for e in lc["val_errors"]],
                    "Train Error (Std)": [f"±{s:.4f}" for s in lc.get("train_errors_std", [0]*len(lc["train_errors"]))],
                    "Val Error (Std)": [f"±{s:.4f}" for s in lc.get("val_errors_std", [0]*len(lc["val_errors"]))],
                    "Train Accuracy (%)": [f"{a*100:.2f}%" for a in lc["train_accuracies"]],
                    "Val Accuracy (%)": [f"{a*100:.2f}%" for a in lc["val_accuracies"]],
                }
                st.dataframe(pd.DataFrame(raw_dict), use_container_width=True, hide_index=True)

            st.markdown("---")


# ──────────────────────────────────────────────────────────────────
# TAB 5 – MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────
with tab_comparison:
    st.markdown("### ⚖️ Theoretical vs Empirical vs Predicted Requirements")

    if not st.session_state["ran_experiment"]:
        st.markdown("""
        <div class="warning-card">
        ⚠️ <strong>No Experiment Data:</strong> Run an experiment from the sidebar to populate the comparison dashboard.
        </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state["experiment_results"]
        eps = st.session_state["epsilon"]
        dlt = st.session_state["delta"]

        proper_rows = []
        for r in results:
            pac_m = r["pac_bound"].get("m")
            emp_m = r["empirical"].get("sample_count")
            pred_m = r["prediction"].get("sample_count")
            ratio = None
            if pac_m and emp_m and emp_m > 0:
                ratio = pac_m / emp_m

            proper_rows.append({
                "Model": r["model_name"],
                "VC-Dim Estimate": r["vc_info"]["vc_dim"],
                "VC-Dim Label": r["vc_info"]["label"],
                "Theoretical Bound (m)": pac_m,
                "Empirical Requirement (m)": emp_m,
                "Predicted Requirement (m)": pred_m,
                "Final Train Error": r["final_train_err"],
                "Final Val Error": r["final_val_err"],
                "Final Val Accuracy (%)": r["final_val_acc"],
                "Target Error (ε)": eps,
                "Failure Prob (δ)": dlt,
                "Theory/Empirical Ratio": ratio,
            })

        # Summary Metrics
        st.markdown(f"**Target Error ε = {eps:.2f} · Risk Probability δ = {dlt:.2f}**")
        m_cols = st.columns(len(results))
        for idx, r in enumerate(results):
            with m_cols[idx]:
                val_acc = r["final_val_acc"]
                val_acc_str = f"{val_acc*100:.1f}%" if val_acc is not None else "N/A"
                st.metric(
                    r["model_name"],
                    val_acc_str,
                    f"Final Val Error: {r['final_val_err']:.3f}" if r["final_val_err"] is not None else "N/A"
                )

        st.markdown("---")

        # Comparison Table
        st.markdown("#### 📊 Master Comparison Summary Table")
        display_rows = []
        for row in proper_rows:
            def _fmt(v, is_ratio=False):
                if v is None:
                    return "Not achieved / N/A"
                if is_ratio:
                    return f"{v:.1f}×"
                if isinstance(v, float):
                    return f"{v:.4f}"
                if isinstance(v, int):
                    return f"{v:,}"
                return str(v)

            display_rows.append({
                "Model": row["Model"],
                "VC Dim (d)": f"{row['VC-Dim Estimate']:,}",
                "VC Type": row["VC-Dim Label"],
                "Theoretical PAC m": _fmt(row["Theoretical Bound (m)"]),
                "Empirical Observed m": _fmt(row["Empirical Requirement (m)"]),
                "Power-Law Predicted m*": _fmt(row["Predicted Requirement (m)"]),
                "Final Val Accuracy": f"{row['Final Val Accuracy (%)']*100:.2f}%" if row["Final Val Accuracy (%)"] else "N/A",
                "Theory / Empirical Gap": _fmt(row["Theory/Empirical Ratio"], is_ratio=True),
            })

        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        st.markdown(r"""
        <div class="info-card">
        <strong>Reading the Results:</strong><br>
        • <strong>Theoretical PAC m:</strong> Mathematical upper bound guaranteeing error ≤ ε under worst-case distribution. Holds across any distribution.<br>
        • <strong>Empirical Observed m:</strong> Smallest training size at which held-out validation error actually reached target error on this dataset.<br>
        • <strong>Power-Law Predicted m*:</strong> Sample size extrapolated from the fitted curve $error(m) = a \cdot m^{-b} + c$.<br>
        • <strong>Theory / Empirical Gap:</strong> The factor by which theory overestimates real data requirements (typically 10× to 1000× due to favorable geometric clustering in real datasets).
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        st.markdown("#### 📊 Sample Size Comparison (Log Scale)")
        fig_bars = plot_comparison_bars(proper_rows)
        st.pyplot(fig_bars, use_container_width=True)

        # Downloads & Reports
        st.markdown("---")
        st.markdown("#### ⬇️ Export Experiment Deliverables & Technical Reports")

        # Prepare figures for the HTML report
        report_figures = {}
        if fig_bars:
            report_figures["Sample Size Comparison (Log Scale)"] = fig_bars
        for r in results:
            m_name = r["model_name"]
            lc = r["learning_curve"]
            fig_lc = plot_learning_curve(lc, target_error=eps, model_name=m_name)
            report_figures[f"Learning Curve – {m_name}"] = fig_lc

        ds_info = describe_dataset(st.session_state["dataset"]) if "dataset" in st.session_state else {}
        bound_mode = st.session_state.get("pac_setting", "realizable")

        html_report = generate_html_report(
            results=results,
            dataset_name=st.session_state.get("dataset_name", "Dataset"),
            dataset_info=ds_info,
            epsilon=eps,
            delta=dlt,
            bound_setting=bound_mode,
            figures=report_figures,
        )

        c_d1, c_d2, c_d3, c_d4 = st.columns(4)

        with c_d1:
            st.download_button(
                "📥 Download Summary Table (CSV)",
                data=pd.DataFrame(display_rows).to_csv(index=False),
                file_name="vc_comparison_results.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_dl_csv",
            )

        with c_d2:
            lc_frames = []
            for r in results:
                lc = r["learning_curve"]
                tmp = pd.DataFrame({
                    "Model": r["model_name"],
                    "Samples": lc["sample_counts"],
                    "Train Error": lc["train_errors"],
                    "Val Error": lc["val_errors"],
                    "Train Accuracy": lc["train_accuracies"],
                    "Val Accuracy": lc["val_accuracies"],
                })
                lc_frames.append(tmp)
            if lc_frames:
                st.download_button(
                    "📥 Download Learning Curves (CSV)",
                    data=pd.concat(lc_frames, ignore_index=True).to_csv(index=False),
                    file_name="learning_curve_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_dl_lc",
                )

        with c_d3:
            summary_dict = {
                "dataset": st.session_state.get("dataset_name"),
                "epsilon": eps,
                "delta": dlt,
                "models": [
                    {
                        "name": r["model_name"],
                        "vc_dim": r["vc_info"]["vc_dim"],
                        "theoretical_m": r["pac_bound"].get("m"),
                        "empirical_m": r["empirical"].get("sample_count"),
                        "predicted_m": r["prediction"].get("sample_count"),
                        "final_val_accuracy": r["final_val_acc"],
                    }
                    for r in results
                ]
            }
            st.download_button(
                "📥 Download JSON Report",
                data=json.dumps(summary_dict, indent=2),
                file_name="experiment_summary.json",
                mime="application/json",
                use_container_width=True,
                key="btn_dl_json",
            )

        with c_d4:
            st.download_button(
                "📄 Download HTML / PDF Report",
                data=html_report,
                file_name=f"vc_report_{st.session_state.get('dataset_name', 'experiment').lower().replace(' ', '_')}.html",
                mime="text/html",
                use_container_width=True,
                key="btn_dl_html",
            )

        with st.expander("👁️ Preview Full Technical Report"):
            import streamlit.components.v1 as components
            components.html(html_report, height=650, scrolling=True)


# ──────────────────────────────────────────────────────────────────
# TAB 6 – DOCUMENTATION & VIVA
# ──────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### 📖 Project Documentation & Research Foundations")

    st.markdown(r"""
    #### Project Title
    **“How Much Data Does Your Model Really Need? A VC-Dimension-Guided Data Budget Estimator”**  
    *Subtitle: “Bridging PAC Learning Theory with Empirical Practice”*

    #### Abstract
    Determining training sample requirements is a foundational challenge in machine learning. Classical theoretical guarantees derived from Probably Approximately Correct (PAC) learning and Vapnik-Chervonenkis (VC) dimension theory provide worst-case sample complexity bounds. However, practitioners observe that models frequently converge with far less data. This micro project systematically compares theoretical PAC/VC sample bounds with empirical learning curves and power-law extrapolations across multiple model families and datasets, demystifying the theory-practice gap.

    ---

    #### Frequently Asked Viva / Interview Questions & Answers

    **Q1: What is the Vapnik-Chervonenkis (VC) Dimension?**  
    **Answer:** The VC dimension of a hypothesis class $\mathcal{H}$ is the maximum cardinality of a set of data points that $\mathcal{H}$ can shatter — that is, correctly classify under every possible binary labelling ($2^d$ combinations). It measures the intrinsic expressive capacity of a model class independently of any data distribution.

    **Q2: Why is the Perceptron VC dimension exactly $n + 1$?**  
    **Answer:** A linear hyperplane in $n$-dimensional Euclidean space $\mathbb{R}^n$ can shatter any set of $n + 1$ points in general position (affine independence). By Radon's Theorem, any set of $n + 2$ points can be partitioned into two subsets whose convex hulls intersect, preventing linear separation. Hence, $d_{VC} = n + 1$.

    **Q3: Why are theoretical PAC sample bounds so much larger than empirical requirements?**  
    **Answer:** PAC bounds are **distribution-free** and **worst-case**. They must hold even against an adversary selecting the most pathological data distribution possible. Real-world datasets exhibit cluster structures, margin separation, and manifold constraints that make learning substantially easier than the worst-case scenario.

    **Q4: What is the difference between Realizable and Agnostic PAC bounds?**  
    **Answer:**
    - In the **Realizable setting** (Blumer et al., 1989), we assume the true concept $c^* \in \mathcal{H}$ and zero training error is achievable. The bound scales with $O\left(\frac{1}{\epsilon}\right)$.
    - In the **Agnostic setting** (Anthony & Bartlett, 1999), labels can be noisy or the model misspecified ($c^* \notin \mathcal{H}$). To guarantee uniform convergence via Hoeffding bounds, the sample complexity scales with $O\left(\frac{1}{\epsilon^2}\right)$, requiring far more samples.

    **Q5: What is the power-law learning curve formulation and why is it used?**  
    **Answer:** Empirical learning performance commonly follows $\text{error}(m) = a \cdot m^{-b} + c$, where $a$ is the learning scale factor, $b$ is the learning rate exponent, and $c$ is the irreducible Bayes error. Fitting this power-law allows extrapolating sample requirements to unseen data budgets without exhaustive retraining.

    **Q6: How does this codebase prevent data leakage?**  
    **Answer:** Preprocessing pipelines (imputation and standardization) are fit strictly on the 80% training split and applied to the test split. The held-out test split is never involved in fitting or model selection.

    ---

    #### Academic References
    1. **Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. K. (1989).** Learnability and the Vapnik-Chervonenkis dimension. *Journal of the ACM, 36*(4), 929–965. [doi:10.1145/76359.76371](https://doi.org/10.1145/76359.76371)
    2. **Vapnik, V. N. (1998).** *Statistical Learning Theory*. John Wiley & Sons.
    3. **Valiant, L. G. (1984).** A theory of the learnable. *Communications of the ACM, 27*(11), 1134–1142.
    4. **Anthony, M., & Bartlett, P. L. (1999).** *Neural Network Learning: Theoretical Foundations*. Cambridge University Press.
    5. **Schapire, R. E., & Singer, Y. (1999).** Improved boosting algorithms using confidence-rated predictions. *Machine Learning, 37*(3), 297–336.
    """)
