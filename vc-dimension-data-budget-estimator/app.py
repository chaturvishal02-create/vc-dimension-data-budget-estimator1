# app.py
# ═══════════════════════════════════════════════════════════════════
#  DATA BUDGET STUDIO
#  "How Much Data Does Your Model Really Need?
#   A VC-Dimension-Guided Data Budget Estimator —
#   Bridging PAC Learning Theory with Empirical Practice"
# ═══════════════════════════════════════════════════════════════════
#  Run:  streamlit run app.py
# ═══════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import io
import math
import time
import importlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import config
try:
    importlib.reload(config)
except Exception:
    pass

from config import (
    DEFAULT_EPSILON, DEFAULT_DELTA, TRAIN_FRACTIONS,
    AVAILABLE_MODELS, BUILTIN_DATASETS, MIN_SAMPLES_REQUIRED, DEFAULT_N_SEEDS,
)

PRIMARY_MODELS = getattr(config, "PRIMARY_MODELS", [
    "Decision Tree", "Random Forest", "Support Vector Machine",
    "K-Nearest Neighbors", "Logistic Regression", "Linear Regression",
    "Naive Bayes", "Perceptron"
])
CLASSIFICATION_MODELS = getattr(config, "CLASSIFICATION_MODELS", [
    "Decision Tree", "Random Forest", "Support Vector Machine",
    "K-Nearest Neighbors", "Logistic Regression", "Naive Bayes", "Perceptron"
])
REGRESSION_MODELS = getattr(config, "REGRESSION_MODELS", [
    "Decision Tree", "Random Forest", "Support Vector Machine",
    "K-Nearest Neighbors", "Linear Regression", "Perceptron"
])
SUPPORTED_UPLOAD_EXTENSIONS = getattr(config, "SUPPORTED_UPLOAD_EXTENSIONS", ["csv", "xlsx", "xls", "json", "txt", "parquet"])

from src.data_loader import (
    get_dataset, describe_dataset, read_uploaded_file,
    prepare_custom_dataset, load_custom_dataset, detect_task_type,
    sanitize_dataframe_for_arrow,
)
from src.data_analyzer import analyze_dataframe, profile_target_variable
from src.preprocessing import split_and_preprocess
from src.vc_dimension import estimate_vc_dimension
from src.pac_learning import (
    calculate_theoretical_samples, calculate_agnostic_samples,
    calculate_regression_samples, get_pac_bounds,
    generate_epsilon_sensitivity_curve, generate_delta_sensitivity_curve,
    generate_vc_sensitivity_curve,
)
from src.model_training import build_model, train_and_evaluate_model
from src.learning_curve import generate_learning_curve, find_empirical_sample_requirement
from src.curve_fitting import fit_power_law, predict_sample_size
from src.data_budget import estimate_data_budget
from src.evaluation import run_full_experiment, build_comparison_dataframe
from src.visualizations import (
    plot_plotly_learning_curve, plot_plotly_generalization_gap,
    plot_plotly_budget_comparison, plot_plotly_model_leaderboard,
    plot_plotly_pac_sensitivity, plot_plotly_feature_projection,
    plot_plotly_power_law_fit,
)
from src.report_generator import (
    generate_pdf_report, generate_excel_report,
    generate_html_report, save_report_to_disk,
)
from src.utils import (
    format_number, format_percent, format_currency, inject_custom_css,
)

# ── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="DATA BUDGET STUDIO — VC Dimension & PAC Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── State Initialisation ─────────────────────────────────────────
def _init_state():
    defaults = {
        "dataset": None,
        "raw_df": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "preprocessor": None,
        "experiment_results": [],
        "comparison_df": None,
        "epsilon": DEFAULT_EPSILON,
        "delta": DEFAULT_DELTA,
        "selected_models": ["Decision Tree", "Random Forest", "Support Vector Machine"],
        "dataset_name": None,
        "n_features": None,
        "task_type": "classification",
        "ran_experiment": False,
        "n_seeds": DEFAULT_N_SEEDS,
        "pac_setting": "Realizable (Blumer et al.)",
        "cost_per_sample": 1.0,
        "theme_dark": True,
        "nav_page": "🏠 Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-load demo Iris dataset on first launch if empty
    if st.session_state.get("dataset") is None:
        try:
            ds = get_dataset("Iris")
            X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds["X"], ds["y"], task_type="classification")
            st.session_state.update({
                "dataset": ds,
                "raw_df": pd.concat([ds["X"], ds["y"]], axis=1),
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "preprocessor": prep,
                "dataset_name": ds["dataset_name"],
                "n_features": ds["n_features"],
                "task_type": ds.get("task_type", "classification"),
                "experiment_results": [],
                "comparison_df": None,
                "ran_experiment": False,
            })
        except Exception:
            pass

_init_state()

# Theme CSS
is_dark = st.session_state.get("theme_dark", True)
inject_custom_css(is_dark=is_dark)


# ── Central Autonomous Experiment Execution Engine ───────────────
def execute_complete_experiment(models_to_run=None):
    """
    Centralized One-Click Experiment Runner.
    Executes training across all subsets, computes VC dimensions,
    calculates PAC bounds, fits power-law curves, computes data budgets,
    and updates session state.
    """
    if st.session_state.get("X_train") is None or st.session_state.get("dataset") is None:
        st.error("Cannot run experiment: No active dataset loaded. Please upload or load a dataset first.")
        return False

    task_type = st.session_state.get("task_type", "classification")
    valid_models = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS

    if models_to_run is None:
        models_to_run = [m for m in st.session_state.get("selected_models", []) if m in valid_models]
        if not models_to_run:
            models_to_run = valid_models[:3]
            st.session_state["selected_models"] = models_to_run

    results = []
    n_mods = len(models_to_run)
    progress_bar = st.progress(0, f"Starting complete experiment suite for {n_mods} model(s)...")

    for i, mname in enumerate(models_to_run):
        progress_bar.progress((i) / n_mods, f"Training & analyzing {mname} ({i+1}/{n_mods})...")
        try:
            res = run_full_experiment(
                model_name=mname,
                X_train=st.session_state["X_train"],
                y_train=st.session_state["y_train"],
                X_test=st.session_state["X_test"],
                y_test=st.session_state["y_test"],
                epsilon=st.session_state["epsilon"],
                delta=st.session_state["delta"],
                n_features=st.session_state["n_features"],
                n_seeds=st.session_state["n_seeds"],
                task_type=task_type,
            )
            results.append(res)
        except Exception as exc:
            st.warning(f"Note on {mname}: {exc}")

    progress_bar.progress(1.0, "Complete experiment suite executed successfully!")
    time.sleep(0.3)
    progress_bar.empty()

    if results:
        st.session_state["experiment_results"] = results
        st.session_state["comparison_df"] = build_comparison_dataframe(results)
        st.session_state["ran_experiment"] = True
        return True
    return False


# ── Sidebar Navigation ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DATA BUDGET STUDIO")
    st.caption("PAC Learning Theory & Empirical Budget Estimator")

    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state["theme_dark"], key="theme_toggle")
    if theme_toggle != st.session_state["theme_dark"]:
        st.session_state["theme_dark"] = theme_toggle
        st.rerun()

    st.markdown("---")

    nav_options = [
        "🏠 Home",
        "📤 Dataset Upload",
        "🔍 Dataset Explorer",
        "🤖 Model Selection",
        "📐 VC Dimension Calculator",
        "🎯 PAC Sample Complexity",
        "📈 Learning Curve Analysis",
        "💰 Data Budget Estimator",
        "🏆 Model Comparison",
        "🔬 Mathematical Explanation",
        "📑 Download Reports",
    ]

    selected_nav = st.radio(
        "Platform Navigation",
        nav_options,
        index=nav_options.index(st.session_state.get("nav_page", "🏠 Home")),
        key="sidebar_nav_radio",
    )
    if selected_nav != st.session_state.get("nav_page"):
        st.session_state["nav_page"] = selected_nav
        st.rerun()

    nav_page = st.session_state["nav_page"]

    st.markdown("---")
    st.markdown("### ⚙️ PAC & Budget Settings")

    eps_val = st.slider(
        "Target Error (ε)",
        0.01, 0.40, float(st.session_state["epsilon"]), 0.01,
        help="Maximum acceptable generalization error tolerance (ε). Smaller ε requires more data.",
        key="sb_eps"
    )
    delta_val = st.slider(
        "Failure Probability (δ)",
        0.01, 0.30, float(st.session_state["delta"]), 0.01,
        help="Acceptable risk probability δ (Confidence = 1 - δ).",
        key="sb_delta"
    )
    cost_val = st.number_input(
        "Cost / Sample ($)",
        min_value=0.01, value=float(st.session_state["cost_per_sample"]), step=0.25,
        help="Acquisition or annotation cost per data sample.",
        key="sb_cost"
    )
    st.session_state["epsilon"] = eps_val
    st.session_state["delta"] = delta_val
    st.session_state["cost_per_sample"] = cost_val

    # Status Pill
    ds_curr = st.session_state.get("dataset")
    if ds_curr:
        status_dot = "🟢" if st.session_state.get("ran_experiment") else "🟡"
        st.caption(f"{status_dot} **{ds_curr['dataset_name']}** · {ds_curr['n_samples']:,} rows · {ds_curr['n_features']} cols")
        st.caption(f"Task: **{st.session_state['task_type'].capitalize()}** · Results: **{'Ready' if st.session_state.get('ran_experiment') else 'Not Run'}**")
    else:
        st.caption("🔴 No dataset loaded")


# ── Global Hero Banner ───────────────────────────────────────────
st.markdown(f"""
<div class="kpi-card" style="border-left: 5px solid #6366F1; margin-bottom: 1.2rem; background: {'#0F172A' if is_dark else '#F8FAFC'};">
  <div style="font-size: 1.65rem; font-weight: 800; color: {'#FFFFFF' if is_dark else '#0F172A'}; margin-bottom: 0.2rem;">
    How Much Data Does Your Model Really Need?
  </div>
  <div style="font-size: 0.92rem; font-weight: 500; color: {'#94A3B8' if is_dark else '#64748B'}; margin-bottom: 0.6rem;">
    A VC-Dimension-Guided Data Budget Estimator — Bridging PAC Learning Theory with Empirical Practice
  </div>
  <div style="display: flex; gap: 0.4rem; flex-wrap: wrap;">
    <span class="badge">PAC Learning</span>
    <span class="badge">VC Dimension</span>
    <span class="badge">Sample Complexity</span>
    <span class="badge">Power-Law Fit</span>
    <span class="badge">Data Budget Optimizer</span>
    <span class="badge">Plotly Visualizations</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 1. 🏠 HOME DASHBOARD
# ══════════════════════════════════════════════════════════════════
if nav_page == "🏠 Home":
    st.markdown("### 📊 Active Overview & Quick Actions")

    ds_active = st.session_state.get("dataset")
    exp_results = st.session_state.get("experiment_results", [])
    has_run = st.session_state.get("ran_experiment") and len(exp_results) > 0

    # Theoretical bound preview
    est_theory = "N/A"
    top_model = st.session_state.get("selected_models", ["Decision Tree"])[0] if st.session_state.get("selected_models") else "Decision Tree"
    if ds_active is not None:
        vc_est = estimate_vc_dimension(top_model, ds_active["n_features"])
        if st.session_state["task_type"] == "regression":
            b = calculate_regression_samples(st.session_state["epsilon"], st.session_state["delta"], vc_est["vc_dim"])
        elif st.session_state["pac_setting"].startswith("Realizable"):
            b = calculate_theoretical_samples(st.session_state["epsilon"], st.session_state["delta"], vc_est["vc_dim"])
        else:
            b = calculate_agnostic_samples(st.session_state["epsilon"], st.session_state["delta"], vc_est["vc_dim"])
        est_theory = f"{b['m']:,}"

    # Empirical preview
    emp_str = "Pending Run"
    best_acc_str = "Pending"
    best_model_name = top_model
    ratio_str = "Pending"

    if has_run:
        achieved_list = [r["empirical"]["sample_count"] for r in exp_results if r["empirical"]["sample_count"]]
        if achieved_list:
            emp_str = f"{min(achieved_list):,} samples"
        else:
            emp_str = "Extrapolating"

        best_score = 0.0
        for r in exp_results:
            sc = r.get("final_val_acc") or 0.0
            if sc > best_score:
                best_score = sc
                best_model_name = r["model_name"]
        best_acc_str = f"{best_score*100:.1f}%" if st.session_state["task_type"] == "classification" else f"R² = {best_score:.3f}"

        # Conservatism ratio
        for r in exp_results:
            p_m = r.get("pac_bound", {}).get("m")
            e_m = r.get("empirical", {}).get("sample_count")
            if p_m and e_m and e_m > 0:
                ratio_str = f"{p_m / e_m:.1f}×"
                break

    n_samp_str = f"{ds_active['n_samples']:,}" if ds_active else "None"
    n_feat_str = f"{ds_active['n_features']} features" if ds_active else "—"
    task_str = st.session_state["task_type"].capitalize()

    # KPI Grid
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Active Dataset</div>
        <div class="kpi-value" style="font-size: 1.2rem;">{ds_active['dataset_name'] if ds_active else 'No Dataset'}</div>
        <div class="kpi-sub">{n_samp_str} records · {n_feat_str} · {task_str}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">PAC Specs (ε, δ)</div>
        <div class="kpi-value">ε = {st.session_state['epsilon']:.2f}</div>
        <div class="kpi-sub">Confidence: {(1-st.session_state['delta'])*100:.0f}% (δ={st.session_state['delta']:.2f})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Theoretical PAC Bound</div>
        <div class="kpi-value" style="color: #A855F7;">{est_theory}</div>
        <div class="kpi-sub">Worst-case distribution-free bound</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Empirical Sample Need</div>
        <div class="kpi-value" style="color: #10B981;">{emp_str}</div>
        <div class="kpi-sub">Observed on validation set</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Top Performance</div>
        <div class="kpi-value" style="color: #38BDF8;">{best_acc_str}</div>
        <div class="kpi-sub">Top model: {best_model_name}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Action Central Bar
    st.markdown("---")
    qa_col1, qa_col2, qa_col3 = st.columns([1.5, 1.0, 1.0])

    with qa_col1:
        if st.button("⚡ START ANALYSIS & RUN COMPLETE EXPERIMENT", type="primary", use_container_width=True, key="home_run_exp"):
            with st.spinner("Executing full empirical experiment and data budget pipeline..."):
                success = execute_complete_experiment()
                if success:
                    st.success("✓ Complete experiment suite finished! Results are ready across all modules.")
                    st.rerun()

    with qa_col2:
        if st.button("📤 Upload New Dataset", use_container_width=True, key="home_btn_upload"):
            st.session_state["nav_page"] = "📤 Dataset Upload"
            st.rerun()

    with qa_col3:
        if st.button("🎲 Reset to Benchmark", use_container_width=True, key="home_btn_bench"):
            ds = get_dataset("Iris")
            X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds["X"], ds["y"], task_type="classification")
            st.session_state.update({
                "dataset": ds,
                "raw_df": pd.concat([ds["X"], ds["y"]], axis=1),
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "preprocessor": prep,
                "dataset_name": ds["dataset_name"],
                "n_features": ds["n_features"],
                "task_type": "classification",
                "experiment_results": [],
                "comparison_df": None,
                "ran_experiment": False,
            })
            st.success("Loaded benchmark dataset: Iris")
            st.rerun()

    st.markdown("---")

    # Workflow and Content
    c_left, c_right = st.columns([1.4, 1.0])

    with c_left:
        st.markdown("#### 🧭 Complete Guided Workflow")
        st.markdown("""
        <div class="info-card">
        <strong>Seamless 5-Stage Interconnected Workflow:</strong><br/>
        1. <strong>Dataset Upload:</strong> Ingest any CSV, Excel, JSON, TXT, or Parquet dataset with instant profiling.<br/>
        2. <strong>Model Selection:</strong> Choose from 8 standard ML model families tailored to your task.<br/>
        3. <strong>VC Dimension & PAC Studio:</strong> Calculate mathematical capacities, bounds, and distribution-free limits.<br/>
        4. <strong>Empirical Learning Curves:</strong> Train models on 10 progressive data subsets with error bars.<br/>
        5. <strong>Data Budget Estimator:</strong> Compare PAC theory against empirical needs and calculate financial data acquisition costs.
        </div>
        """, unsafe_allow_html=True)

        if has_run:
            st.markdown("#### 🏆 Latest Experiment Highlights")
            st.dataframe(st.session_state["comparison_df"], use_container_width=True, hide_index=True)

    with c_right:
        st.markdown("#### 💡 Theory vs Practice Gap")
        st.markdown(f"""
        <div class="info-card">
        <strong>The Fundamental Question:</strong><br/>
        <em>"How many training samples does a model really need to reach target accuracy?"</em>
        <br/><br/>
        • <strong>Theoretical PAC Bound:</strong> Worst-case mathematical limit under adversarial distributions. Often 10x to 10,000x larger than reality.<br/>
        • <strong>Empirical Learning Curve:</strong> Real observed performance on your dataset distribution.<br/>
        • <strong>Power-Law Fit:</strong> $error(m) = a \\cdot m^{{-b}} + c$ modeling the empirical rate of return.<br/><br/>
        <strong>Current Conservatism Gap:</strong> <span style="color: #A855F7; font-weight: 700;">{ratio_str}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 2. 📤 DATASET UPLOAD
# ══════════════════════════════════════════════════════════════════
elif nav_page == "📤 Dataset Upload":
    st.markdown("### 📤 Universal Dataset Upload & Automatic Inspection")
    st.markdown("Upload any dataset in **CSV, Excel (.xlsx, .xls), JSON, TXT (tabular), or Parquet (.parquet)** format.")

    u_col1, u_col2 = st.columns([1.2, 1.0])

    with u_col1:
        source_mode = st.radio(
            "Dataset Source",
            ["Upload Your Own File", "Use Built-in Benchmark"],
            horizontal=True,
            key="ds_source_mode"
        )

        uploaded_file = None
        preview_df = None

        if source_mode == "Upload Your Own File":
            uploaded_file = st.file_uploader(
                "Select Dataset File",
                type=SUPPORTED_UPLOAD_EXTENSIONS,
                help="Supports CSV (.csv), Excel (.xlsx, .xls), JSON (.json), TXT, and Parquet (.parquet)",
                key="univ_file_uploader",
            )
            if uploaded_file is not None:
                try:
                    preview_df = read_uploaded_file(uploaded_file)
                    st.session_state["raw_df"] = preview_df
                except Exception as exc:
                    st.error(f"Could not parse uploaded dataset: {exc}")
        else:
            benchmark_name = st.selectbox(
                "Choose Benchmark Dataset",
                BUILTIN_DATASETS,
                key="bench_select"
            )

    with u_col2:
        if preview_df is not None:
            analysis = analyze_dataframe(preview_df)
            st.markdown("#### 🎯 Automated Inspection & Task Setup")

            st.markdown(f"""
            <div class="kpi-card" style="padding: 12px; margin-bottom: 10px;">
              <strong>Detected:</strong> {analysis['n_rows']:,} rows × {analysis['n_cols']} columns<br/>
              <strong>Numeric:</strong> {len(analysis['numeric_cols'])} · <strong>Categorical:</strong> {len(analysis['categorical_cols'])}<br/>
              <strong>Missing Values:</strong> {analysis['total_missing']} · <strong>Duplicates:</strong> {analysis['duplicate_count']}
            </div>
            """, unsafe_allow_html=True)

            # Auto-suggest target
            candidates = analysis["target_candidates"]
            default_target = candidates[0] if (candidates and candidates[0] in preview_df.columns) else preview_df.columns[-1]
            target_col = st.selectbox(
                "Target Column (Auto-Suggested)",
                list(preview_df.columns),
                index=list(preview_df.columns).index(default_target),
                key="up_sel_target"
            )

            # Auto-detect task
            t_profile = profile_target_variable(preview_df, target_col)
            detected_task = t_profile.get("task_type", "classification")

            task_choice = st.radio(
                "Identified Task Type",
                ["Classification", "Regression"],
                index=0 if detected_task == "classification" else 1,
                horizontal=True,
                help="Automatically identified based on target variable uniqueness and type.",
                key="up_task_radio"
            )

            test_split = st.slider("Held-out Test Split", 0.10, 0.40, 0.20, 0.05, key="up_split_slider")

            if st.button("✅ Confirm & Prepare Dataset for Training", type="primary", use_container_width=True, key="btn_confirm_dataset"):
                try:
                    with st.spinner("Cleaning, encoding, and splitting dataset (zero data leakage)..."):
                        ds_custom = prepare_custom_dataset(
                            preview_df,
                            target_column=target_col,
                            dataset_name=uploaded_file.name if uploaded_file else "Custom Dataset",
                            task_type=task_choice.lower(),
                        )
                        X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(
                            ds_custom["X"], ds_custom["y"],
                            test_size=test_split,
                            task_type=task_choice.lower()
                        )
                        # Pre-select recommended models
                        rec_models = (
                            ["Decision Tree", "Random Forest", "Support Vector Machine", "Logistic Regression"]
                            if task_choice.lower() == "classification"
                            else ["Decision Tree", "Random Forest", "Support Vector Machine", "Linear Regression"]
                        )
                        st.session_state.update({
                            "dataset": ds_custom,
                            "raw_df": preview_df,
                            "X_train": X_tr, "X_test": X_te,
                            "y_train": y_tr, "y_test": y_te,
                            "preprocessor": prep,
                            "dataset_name": ds_custom["dataset_name"],
                            "n_features": ds_custom["n_features"],
                            "task_type": task_choice.lower(),
                            "selected_models": rec_models,
                            "experiment_results": [],
                            "comparison_df": None,
                            "ran_experiment": False,
                        })
                        st.success(f"✓ Dataset Ready! {ds_custom['dataset_name']} ({ds_custom['n_samples']:,} samples, {ds_custom['n_features']} features).")
                        st.session_state["nav_page"] = "🤖 Model Selection"
                        st.rerun()
                except Exception as exc:
                    st.error(f"Error preparing dataset: {exc}")

        elif source_mode == "Use Built-in Benchmark":
            st.markdown("#### ⚡ Quick Load Benchmark")
            st.caption("Standard benchmark datasets for immediate evaluation.")
            if st.button("📥 Load Benchmark Dataset", type="primary", use_container_width=True, key="btn_load_bench"):
                try:
                    ds = get_dataset(benchmark_name)
                    X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds["X"], ds["y"], task_type="classification")
                    st.session_state.update({
                        "dataset": ds,
                        "raw_df": pd.concat([ds["X"], ds["y"]], axis=1),
                        "X_train": X_tr, "X_test": X_te,
                        "y_train": y_tr, "y_test": y_te,
                        "preprocessor": prep,
                        "dataset_name": ds["dataset_name"],
                        "n_features": ds["n_features"],
                        "task_type": ds.get("task_type", "classification"),
                        "selected_models": ["Decision Tree", "Random Forest", "Support Vector Machine"],
                        "experiment_results": [],
                        "comparison_df": None,
                        "ran_experiment": False,
                    })
                    st.success(f"✓ Loaded {benchmark_name}!")
                    st.session_state["nav_page"] = "🤖 Model Selection"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error loading benchmark: {exc}")

    if preview_df is not None:
        st.markdown("---")
        st.markdown("#### 📋 First 10 Records Preview")
        st.dataframe(sanitize_dataframe_for_arrow(preview_df.head(10)), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 3. 🔍 DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════
elif nav_page == "🔍 Dataset Explorer":
    st.markdown("### 🔍 Dataset Exploration & Deep Profiling")

    ds = st.session_state.get("dataset")
    raw_df = st.session_state.get("raw_df")

    if ds is None or raw_df is None:
        st.warning("⚠️ No dataset loaded yet.")
        if st.button("📥 Load Demo Dataset (Iris)", key="exp_load_demo"):
            ds = get_dataset("Iris")
            X_tr, X_te, y_tr, y_te, prep = split_and_preprocess(ds["X"], ds["y"], task_type="classification")
            st.session_state.update({
                "dataset": ds,
                "raw_df": pd.concat([ds["X"], ds["y"]], axis=1),
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "preprocessor": prep,
                "dataset_name": ds["dataset_name"],
                "n_features": ds["n_features"],
                "task_type": "classification",
                "experiment_results": [],
                "comparison_df": None,
                "ran_experiment": False,
            })
            st.rerun()
    else:
        info = describe_dataset(ds)
        analysis = analyze_dataframe(raw_df)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Records", f"{info['n_samples']:,}")
        m2.metric("Features", info["n_features"])
        m3.metric("Missing Values", analysis["total_missing"])
        m4.metric("Duplicates", analysis["duplicate_count"])
        m5.metric("Task Type", st.session_state["task_type"].capitalize())
        m6.metric("Train / Test Split", f"{len(st.session_state['X_train']):,} / {len(st.session_state['X_test']):,}")

        st.markdown("---")
        c_proj, c_dist = st.columns([1.2, 1.0])

        with c_proj:
            st.markdown("#### 🎨 2D Feature Space Projection")
            fig_proj = plot_plotly_feature_projection(
                ds["X"].to_numpy(),
                ds["y"].to_numpy(),
                feature_names=info["feature_names"],
                task_type=st.session_state["task_type"],
                dataset_name=ds["dataset_name"],
                is_dark=is_dark,
            )
            st.plotly_chart(fig_proj, use_container_width=True)

        with c_dist:
            st.markdown("#### ⚖️ Target Variable Balance")
            if st.session_state["task_type"] == "classification":
                class_rows = []
                for k, v in info["class_distribution"].items():
                    pct = info["class_proportions"].get(k, 0.0)
                    class_rows.append({"Class": str(k), "Count": v, "Percentage": f"{pct:.1f}%"})
                st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
            else:
                s = pd.Series(ds["y"])
                st.write(s.describe().to_frame().T)

            st.markdown("#### 🛡️ Data Leakage Prevention")
            st.markdown("""
            <div class="success-card">
            <strong>Leak-Free Preprocessing:</strong><br/>
            • 80/20 train/test split executed before fitting transformers.<br/>
            • Missing value imputation & standardization fit <strong>strictly on training split</strong>.<br/>
            • Held-out test set is never observed during training.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Descriptive Statistics Summary")
        st.dataframe(ds["X"].describe(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 4. 🤖 MODEL SELECTION
# ══════════════════════════════════════════════════════════════════
elif nav_page == "🤖 Model Selection":
    st.markdown("### 🤖 Model Selection & Central Experiment Launcher")
    st.markdown("Select from the 8 supported model families tailored to your active task.")

    task = st.session_state.get("task_type", "classification")
    allowed_models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS

    col_m1, col_m2 = st.columns([1.2, 1.0])

    with col_m1:
        st.markdown("#### 1. Choose Models to Evaluate")
        selected = st.multiselect(
            "Selected Models",
            allowed_models,
            default=[m for m in st.session_state["selected_models"] if m in allowed_models] or allowed_models[:3],
            help="Only models appropriate for the active task type are shown.",
            key="ms_primary_models"
        )
        st.session_state["selected_models"] = selected

        st.markdown("#### 2. Model Architecture Parameters")
        if "Decision Tree" in selected:
            with st.expander("🌲 Decision Tree Hyperparameters", expanded=False):
                st.slider("Max Tree Depth (D)", 1, 10, 4, key="cfg_dt_d")
        if "Random Forest" in selected:
            with st.expander("🌳 Random Forest Hyperparameters", expanded=False):
                st.slider("Number of Trees (T)", 10, 150, 50, 10, key="cfg_rf_t")
                st.slider("Tree Depth", 2, 8, 4, key="cfg_rf_d")
        if "Support Vector Machine" in selected:
            with st.expander("📐 SVM Hyperparameters", expanded=False):
                st.selectbox("Kernel", ["linear", "rbf"], key="cfg_svm_k")
                st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.5, key="cfg_svm_c")
        if "K-Nearest Neighbors" in selected:
            with st.expander("📍 KNN Hyperparameters", expanded=False):
                st.slider("Neighbors (k)", 1, 15, 3, 2, key="cfg_knn_k")

        n_seeds = st.slider("Repeated Random Seeds (Error Bars)", 1, 5, int(st.session_state["n_seeds"]), key="cfg_seeds")
        st.session_state["n_seeds"] = n_seeds

    with col_m2:
        st.markdown("#### 3. Central One-Click Runner")
        st.markdown(f"""
        <div class="info-card">
        <strong>Experiment Summary:</strong><br/>
        • Dataset: <strong>{st.session_state['dataset_name']}</strong> ({st.session_state['n_features']} features)<br/>
        • Task: <strong>{task.capitalize()}</strong><br/>
        • Models: <strong>{len(selected)}</strong> selected ({', '.join(selected[:3])})<br/>
        • Fractions: 10 subsets (10% to 100%)<br/>
        • Repeated Seeds: <strong>{n_seeds}</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 RUN COMPLETE EXPERIMENT", type="primary", use_container_width=True, key="btn_run_experiment_direct"):
            with st.spinner("Running complete empirical learning curve and data budget pipeline..."):
                success = execute_complete_experiment(selected)
                if success:
                    st.success("✓ Experiment finished! Results populated across all dashboard views.")
                    st.session_state["nav_page"] = "💰 Data Budget Estimator"
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
# 5. 📐 VC DIMENSION CALCULATOR
# ══════════════════════════════════════════════════════════════════
elif nav_page == "📐 VC Dimension Calculator":
    st.markdown("### 📐 Vapnik-Chervonenkis (VC) Dimension Calculator")
    st.markdown("Compute model capacity and statistical expressiveness according to statistical learning theory.")

    vc_c1, vc_c2 = st.columns([1.1, 1.0])

    with vc_c1:
        calc_model = st.selectbox(
            "Select Model Family",
            PRIMARY_MODELS + ["Polynomial Classifier", "Finite Class"],
            index=0,
            key="vc_calc_model"
        )
        input_n = st.number_input(
            "Input Dimensionality (n features)",
            min_value=1,
            value=int(st.session_state.get("n_features") or 4),
            step=1,
            key="vc_calc_n"
        )

        model_params = {}
        if calc_model == "Polynomial Classifier":
            deg = st.slider("Polynomial Degree (k)", 1, 5, 2, key="vc_calc_deg")
            model_params["degree"] = deg
        elif calc_model == "Decision Tree":
            dt_d = st.slider("Max Tree Depth (D)", 1, 8, 3, key="vc_calc_tree_d")
            model_params["max_depth"] = dt_d
        elif calc_model == "Random Forest":
            rf_t = st.slider("Number of Trees (T)", 5, 100, 20, 5, key="vc_calc_rf_t")
            rf_d = st.slider("Tree Depth (D)", 1, 6, 3, key="vc_calc_rf_d")
            model_params["n_estimators"] = rf_t
            model_params["max_depth"] = rf_d
        elif calc_model == "Support Vector Machine":
            svm_k = st.radio("SVM Kernel", ["linear", "rbf"], horizontal=True, key="vc_calc_svm_k")
            model_params["kernel"] = svm_k
        elif calc_model == "Finite Class":
            card = st.number_input("Hypothesis Class Cardinality (|H|)", min_value=2, value=1024, step=100, key="vc_calc_card")
            model_params["cardinality"] = card

        vc_result = estimate_vc_dimension(calc_model, input_n, **model_params)

    with vc_c2:
        st.markdown("#### 🔬 Theoretical Capacity Result")
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #8B5CF6;">
          <div class="kpi-label">{vc_result['family']}</div>
          <div class="kpi-value" style="color: #8B5CF6;">{vc_result['vc_dim']:,}</div>
          <div class="kpi-sub">Confidence: <strong>{vc_result['confidence']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Classification Label:** `{vc_result['label']}`")
        st.markdown(f"""
        <div class="formula-box">
Formula:      {vc_result['formula']}
Substituted:  {vc_result['substituted']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📜 Mathematical Derivation")
        st.info(vc_result["explanation"])

        st.markdown("#### 🛡️ Theoretical Assumptions")
        for asm in vc_result.get("assumptions", []):
            st.markdown(f"• {asm}")


# ══════════════════════════════════════════════════════════════════
# 6. 🎯 PAC SAMPLE COMPLEXITY
# ══════════════════════════════════════════════════════════════════
elif nav_page == "🎯 PAC Sample Complexity":
    st.markdown("### 🎯 PAC Learning Sample Complexity Studio")
    st.markdown("Calculate distribution-free sample size lower bounds for probably approximately correct learning.")

    pac_c1, pac_c2 = st.columns([1.1, 1.0])

    with pac_c1:
        st.markdown("#### PAC Framework Parameters")
        p_eps = st.slider("Target Error Tolerance (ε)", 0.01, 0.40, float(st.session_state["epsilon"]), 0.01, key="pac_eps")
        p_delta = st.slider("Failure Risk (δ)", 0.01, 0.30, float(st.session_state["delta"]), 0.01, key="pac_delta")
        p_vc = st.number_input("VC Dimension (d)", min_value=1, value=int(st.session_state.get("n_features", 4) + 1), step=1, key="pac_vc_in")

        pac_mode = st.radio(
            "Theoretical PAC Setting",
            ["Realizable (Blumer et al., 1989)", "Agnostic (Anthony & Bartlett, 1999)", "Regression (Pollard, 1984)"],
            index=0,
            key="pac_studio_mode"
        )

        if "Realizable" in pac_mode:
            pac_res = calculate_theoretical_samples(p_eps, p_delta, p_vc)
        elif "Agnostic" in pac_mode:
            pac_res = calculate_agnostic_samples(p_eps, p_delta, p_vc)
        else:
            pac_res = calculate_regression_samples(p_eps, p_delta, p_vc)

    with pac_c2:
        st.markdown("#### Computed PAC Lower Bound")
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #A855F7;">
          <div class="kpi-label">{pac_res['setting_label']}</div>
          <div class="kpi-value" style="color: #A855F7;">{pac_res['m']:,} samples</div>
          <div class="kpi-sub">Confidence: {(1-p_delta)*100:.0f}% · Error ≤ {p_eps:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="formula-box">
{pac_res['substituted']}
        </div>
        """, unsafe_allow_html=True)

        st.info(pac_res["explanation"])

    st.markdown("---")
    st.markdown("#### 📈 PAC Parameter Sensitivity Analysis")

    sens_type = st.radio("Vary Parameter", ["Error Tolerance (ε)", "Failure Risk (δ)", "VC Dimension (d)"], horizontal=True, key="sens_param_choice")

    if sens_type == "Error Tolerance (ε)":
        sens_data = generate_epsilon_sensitivity_curve(p_delta, p_vc, eps_min=0.02, eps_max=0.35, steps=30)
        fig_sens = plot_plotly_pac_sensitivity(sens_data, param_type="epsilon", is_dark=is_dark)
    elif sens_type == "Failure Risk (δ)":
        sens_data = generate_delta_sensitivity_curve(p_eps, p_vc, delta_min=0.01, delta_max=0.30, steps=30)
        fig_sens = plot_plotly_pac_sensitivity(sens_data, param_type="delta", is_dark=is_dark)
    else:
        sens_data = generate_vc_sensitivity_curve(p_eps, p_delta, vc_min=1, vc_max=60, steps=30)
        fig_sens = plot_plotly_pac_sensitivity(sens_data, param_type="vc", is_dark=is_dark)

    st.plotly_chart(fig_sens, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 7. 📈 LEARNING CURVE ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif nav_page == "📈 Learning Curve Analysis":
    st.markdown("### 📈 Empirical Learning Curve Analysis")

    results = st.session_state.get("experiment_results", [])
    if not results:
        st.markdown("""
        <div class="info-card">
        ℹ️ <strong>No Experiment Run Yet:</strong> Click below to execute the complete empirical learning curve suite for your selected models across 10 progressive data fractions.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Run Complete Experiment Now", type="primary", key="lc_empty_run"):
            with st.spinner("Running experiment..."):
                execute_complete_experiment()
                st.rerun()
    else:
        model_names = [r["model_name"] for r in results]
        selected_m = st.selectbox("Inspect Model Learning Curve", model_names, key="lc_inspect_select")
        r_active = next(r for r in results if r["model_name"] == selected_m)
        lc = r_active["learning_curve"]

        # Main Plotly interactive curve
        fig_lc = plot_plotly_learning_curve(
            lc,
            target_error=st.session_state["epsilon"],
            model_name=selected_m,
            theory_m=r_active.get("pac_bound", {}).get("m"),
            is_dark=is_dark,
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        col_gap, col_data = st.columns([1.1, 1.0])

        with col_gap:
            fig_gap = plot_plotly_generalization_gap(lc, model_name=selected_m, is_dark=is_dark)
            st.plotly_chart(fig_gap, use_container_width=True)

        with col_data:
            st.markdown("#### 📋 Empirical Subsets Log")
            df_lc = pd.DataFrame({
                "Subset (m)": lc["sample_counts"],
                "Train Error": [f"{e:.4f}" for e in lc["train_errors"]],
                "Val Error": [f"{e:.4f}" for e in lc["val_errors"]],
                "Val Std (±1σ)": [f"{s:.4f}" for s in lc["val_errors_std"]],
                "Gen Gap": [f"{g:.4f}" for g in lc.get("generalization_gaps", [0]*len(lc["sample_counts"]))],
            })
            st.dataframe(df_lc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# 8. 💰 DATA BUDGET ESTIMATOR
# ══════════════════════════════════════════════════════════════════
elif nav_page == "💰 Data Budget Estimator":
    st.markdown("### 💰 Data Budget Estimator & Sample Requirement Synthesis")
    st.markdown("Compare distribution-free PAC theoretical bounds with observed learning curves and fitted extrapolations.")

    results = st.session_state.get("experiment_results", [])
    if not results:
        st.markdown("""
        <div class="info-card">
        ℹ️ <strong>No Experiment Run Yet:</strong> Click below to immediately run all models and estimate data budgets.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Run Complete Experiment Now", type="primary", key="budget_empty_run"):
            with st.spinner("Running experiment..."):
                execute_complete_experiment()
                st.rerun()
    else:
        m_names = [r["model_name"] for r in results]
        chosen_m = st.selectbox("Select Model for Budget Recommendation", m_names, key="budget_m_select")
        r_curr = next(r for r in results if r["model_name"] == chosen_m)

        theory_m = r_curr.get("pac_bound", {}).get("m") or 1000
        emp_achieved = r_curr.get("empirical", {}).get("achieved", False)
        emp_m = r_curr.get("empirical", {}).get("sample_count")
        curr_n = len(st.session_state.get("X_train", []))

        budget_summary = estimate_data_budget(
            theoretical_m=theory_m,
            empirical_achieved=emp_achieved,
            empirical_m=emp_m,
            current_dataset_size=curr_n,
            lc_result=r_curr["learning_curve"],
            target_error=st.session_state["epsilon"],
            cost_per_sample=st.session_state["cost_per_sample"],
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Theoretical PAC Bound", f"{budget_summary['theoretical_m']:,} samples", help="Distribution-free worst-case guarantee")
        b2.metric("Empirical Requirement", f"{budget_summary['recommended_empirical_m']:,}" if budget_summary['recommended_empirical_m'] else "Not Met", f"Source: {budget_summary['recommendation_source']}")
        b3.metric("Additional Samples Needed", f"{budget_summary['additional_samples_needed']:,}" if budget_summary['additional_samples_needed'] is not None else "N/A")
        b4.metric("Estimated Additional Budget", format_currency(budget_summary['additional_cost']), f"@ ${budget_summary['cost_per_sample']:.2f} / sample")

        # Recommendation Reliability Badge
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {budget_summary['reliability_color']}; padding: 12px; margin-top: 10px;">
          <strong>Recommendation Reliability:</strong> <span style="color: {budget_summary['reliability_color']}; font-weight: 700;">{budget_summary['reliability_rating']}</span><br/>
          <small style="color: {'#94A3B8' if is_dark else '#64748B'};">{budget_summary['reliability_desc']}</small>
        </div>
        """, unsafe_allow_html=True)

        if budget_summary["conservatism_ratio"]:
            st.info(f"💡 **Conservatism Gap:** The theoretical PAC bound is **{budget_summary['conservatism_ratio']}× larger** than what is empirically needed on your dataset.")

        if budget_summary["floor_warning"]:
            st.warning(f"⚠️ {budget_summary['floor_warning']}")

        # Power law fit chart
        fig_fit = plot_plotly_power_law_fit(
            r_curr["learning_curve"],
            budget_summary["power_law_fit"],
            budget_summary["prediction_result"],
            target_error=st.session_state["epsilon"],
            model_name=chosen_m,
            is_dark=is_dark,
        )
        st.plotly_chart(fig_fit, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 9. 🏆 MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif nav_page == "🏆 Model Comparison":
    st.markdown("### 🏆 Multi-Model Comparison & Capacity Analysis")

    results = st.session_state.get("experiment_results", [])
    if not results:
        st.markdown("""
        <div class="info-card">
        ℹ️ <strong>No Models Evaluated Yet:</strong> Click below to run a complete side-by-side comparison across all selected models.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Run Complete Experiment Now", type="primary", key="comp_empty_run"):
            with st.spinner("Running experiment..."):
                execute_complete_experiment()
                st.rerun()
    else:
        df_comp = build_comparison_dataframe(results)

        st.markdown("#### 📋 Executive Leaderboard")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Comparative Sample Requirements (Log Scale)")
        fig_comp_bar = plot_plotly_budget_comparison(
            df_comp.to_dict(orient="records"),
            current_dataset_size=len(st.session_state.get("X_train", [])),
            is_dark=is_dark,
        )
        st.plotly_chart(fig_comp_bar, use_container_width=True)

        st.markdown("#### ⚡ Performance Score Ranking")
        fig_leader = plot_plotly_model_leaderboard(df_comp, metric_col="Final Val Accuracy (%)", is_dark=is_dark)
        st.plotly_chart(fig_leader, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# 10. 🔬 MATHEMATICAL EXPLANATION
# ══════════════════════════════════════════════════════════════════
elif nav_page == "🔬 Mathematical Explanation":
    st.markdown("### 🔬 Mathematical Foundations & Research Theory")

    t1, t2, t3 = st.tabs(["PAC Learning Foundations", "Bias-Variance Tradeoff", "Synthetic Noise Robustness"])

    with t1:
        st.markdown(r"""
        #### Probably Approximately Correct (PAC) Framework
        Introduced by Leslie Valiant (1984) and extended by Blumer, Ehrenfeucht, Haussler, and Warmuth (1989).
        
        **Realizable Sample Complexity Bound:**
        $$m \ge \frac{1}{\epsilon} \left[ 4 \log_2\left(\frac{2}{\delta}\right) + 8 d \log_2\left(\frac{13}{\epsilon}\right) \right]$$
        
        **Agnostic Sample Complexity Bound (Anthony & Bartlett, 1999):**
        $$m \ge \frac{8}{\epsilon^2} \left[ 2 d \ln\left(\frac{8e}{\epsilon}\right) + \ln\left(\frac{4}{\delta}\right) \right]$$
        
        Where:
        - $d$ = Vapnik-Chervonenkis (VC) dimension of the hypothesis class $\mathcal{H}$
        - $\epsilon$ = Maximum generalization error tolerance ($0 < \epsilon < 1$)
        - $\delta$ = Acceptable failure probability ($0 < \delta < 1$), giving confidence $1 - \delta$
        """)

    with t2:
        st.markdown("""
        #### The Bias-Variance Dilemma & Model Capacity
        As model complexity (VC dimension $d$) increases:
        • **Approximation Error (Bias):** Shrinks because a richer hypothesis class can fit complex manifolds.<br/>
        • **Estimation Error (Variance):** Grows because higher capacity requires more training samples to ensure uniform convergence.
        """)

    with t3:
        st.markdown("#### 🧪 Synthetic Label Noise Robustness Simulator")
        noise_level = st.slider("Inject Label Noise (%)", 0, 30, 0, 5, key="synth_noise_slider")
        if noise_level > 0:
            st.info(f"Simulating {noise_level}% corrupted training labels. In the presence of noise, realizable PAC bounds are violated, necessitating Agnostic bounds scaling with 1/ε².")


# ══════════════════════════════════════════════════════════════════
# 11. 📑 DOWNLOAD REPORTS
# ══════════════════════════════════════════════════════════════════
elif nav_page == "📑 Download Reports":
    st.markdown("### 📑 Export Executive & Academic Reports")
    st.markdown("Generate and download comprehensive research documentation in **PDF, Excel, and HTML** formats.")

    results = st.session_state.get("experiment_results", [])
    ds_active = st.session_state.get("dataset")

    if not results:
        st.info("💡 You can download reports for the dataset directly, or click below to run models first for complete comparison data.")
        if st.button("🚀 Run Complete Experiment First", key="dl_empty_run"):
            execute_complete_experiment()
            st.rerun()

    col_pdf, col_xl, col_html = st.columns(3)

    # 1. PDF Report Download
    with col_pdf:
        st.markdown("#### 📄 Executive PDF Report")
        st.caption("Publication-ready styled PDF generated via ReportLab with tables, citations, and budget recommendations.")
        try:
            pdf_bytes = generate_pdf_report(
                results=results,
                dataset_name=ds_active["dataset_name"] if ds_active else "Experiment",
                dataset_info=describe_dataset(ds_active) if ds_active else {},
                epsilon=st.session_state["epsilon"],
                delta=st.session_state["delta"],
                bound_setting=st.session_state["pac_setting"],
                cost_per_sample=st.session_state["cost_per_sample"],
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"Data_Budget_Report_{int(time.time())}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="btn_dl_pdf",
            )
        except Exception as exc:
            st.error(f"Could not generate PDF: {exc}")

    # 2. Excel Workbook Download
    with col_xl:
        st.markdown("#### 📊 Excel Research Workbook")
        st.caption("Multi-sheet structured Excel workbook generated via OpenPyXL containing summary, evaluations, and learning curve logs.")
        try:
            xl_bytes = generate_excel_report(
                results=results,
                dataset_name=ds_active["dataset_name"] if ds_active else "Experiment",
                dataset_info=describe_dataset(ds_active) if ds_active else {},
                epsilon=st.session_state["epsilon"],
                delta=st.session_state["delta"],
                cost_per_sample=st.session_state["cost_per_sample"],
            )
            st.download_button(
                label="⬇️ Download Excel (.xlsx)",
                data=xl_bytes,
                file_name=f"Data_Budget_Data_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="btn_dl_xl",
            )
        except Exception as exc:
            st.error(f"Could not generate Excel: {exc}")

    # 3. HTML Report Download
    with col_html:
        st.markdown("#### 🌐 Interactive HTML Report")
        st.caption("Self-contained portable HTML document with embedded figures and instant print-to-PDF button.")
        try:
            html_str = generate_html_report(
                results=results,
                dataset_name=ds_active["dataset_name"] if ds_active else "Experiment",
                dataset_info=describe_dataset(ds_active) if ds_active else {},
                epsilon=st.session_state["epsilon"],
                delta=st.session_state["delta"],
                bound_setting=st.session_state["pac_setting"],
            )
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_str.encode("utf-8"),
                file_name=f"Data_Budget_Report_{int(time.time())}.html",
                mime="text/html",
                use_container_width=True,
                key="btn_dl_html",
            )
        except Exception as exc:
            st.error(f"Could not generate HTML: {exc}")
