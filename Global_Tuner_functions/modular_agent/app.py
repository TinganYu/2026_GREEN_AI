import streamlit as st
import subprocess
import json
import time
import os
import sys
import pandas as pd
from pathlib import Path

# --- Configuration ---
st.set_page_config(page_title="Green AI Orchestrator", layout="wide")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# --- Dynamically Load Available Tasks ---
sys.path.insert(0, str(ROOT_DIR))
try:
    from Evals import EVALUATOR_MAP
    AVAILABLE_TASKS = list(EVALUATOR_MAP.keys())
except ImportError:
    AVAILABLE_TASKS = ["gsm8k", "math", "humaneval", "mbpp", "mmlu", "hellaswag", "truthfulqa"]

def get_latest_exp_dir(model_id_str, task_str):
    results_dir = ROOT_DIR / "tuning_results"
    if not results_dir.exists():
        return None
    model_name = Path(model_id_str).name
    task_safe = task_str.replace(",", "_") if task_str else ""
    prefix = f"exp_{model_name}_{task_safe}"
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not exp_dirs:
        return None
    exp_dirs.sort(key=lambda x: x.name)
    return exp_dirs[-1]

def load_json(file_path):
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def read_log_tail(file_path, lines=150):
    if not file_path.exists():
        return "Waiting for logs..."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()
            if not content:
                return "Log file created, waiting for output..."
            return "".join(content[-lines:])
    except Exception as e:
        return f"Error reading log: {e}"

# --- STATE INITIALIZATION ---
if "process" not in st.session_state: st.session_state.process = None
if "run_status" not in st.session_state: st.session_state.run_status = "idle"
if "start_time" not in st.session_state: st.session_state.start_time = None
if "elapsed_time" not in st.session_state: st.session_state.elapsed_time = 0
if "log_file_handle" not in st.session_state: st.session_state.log_file_handle = None

# --- Sidebar UI: Parameters ---
st.sidebar.title("⚙️ Optimization Settings")
st.sidebar.header("General")
model_id = st.sidebar.text_input("Model ID", value="meta-llama/Llama-3.2-1B-Instruct")

selected_tasks = st.sidebar.multiselect(
    "Evaluation Tasks", 
    options=AVAILABLE_TASKS,
    default=["gsm8k"] if "gsm8k" in AVAILABLE_TASKS else []
)

if not selected_tasks:
    st.sidebar.warning("⚠️ Please select at least one task!")
    task = ""
else:
    task = ",".join(selected_tasks)

model_name = Path(model_id).name
num_samples = st.sidebar.number_input("Eval Samples per Task (0 = All)", min_value=0, value=0, step=10)
memory_type = st.sidebar.selectbox("LLM Memory Type", ["full", "window", "summary", "tool"])

st.sidebar.header("Stopping Conditions")
max_iterations = st.sidebar.number_input("Max Iterations (0 = infinite)", min_value=0, value=20)
max_time_hours = st.sidebar.number_input("Max Time Limit (Hours, 0 = infinite)", min_value=0.0, value=6.0, step=0.5)

st.sidebar.header("Specific Targets (Optional)")
target_vram = st.sidebar.slider("Target VRAM Reduction (%)", 0.0, 1.0, 0.0, step=0.05)
target_lat = st.sidebar.slider("Target Latency Reduction (%)", 0.0, 1.0, 0.0, step=0.05)
target_emit = st.sidebar.slider("Target Emissions Reduction (%)", 0.0, 1.0, 0.0, step=0.05)
target_acc_drop = st.sidebar.slider("Max Acc Drop (absolute pp)", 0.0, 0.5, 0.0, step=0.01)
patience = st.sidebar.number_input("Patience (Iterations)", min_value=1, value=2)

st.sidebar.header("Scoring Weights")
w_acc = st.sidebar.number_input("Accuracy Weight", value=3.0)
w_lat = st.sidebar.number_input("Latency Weight", value=1.0)
w_vram = st.sidebar.number_input("VRAM Weight", value=1.0)
w_emit = st.sidebar.number_input("Emissions Weight", value=1.0)

# --- Main UI ---
st.title("🌱 Green AI Compression Orchestrator")

col1, col2 = st.columns([1, 5])
with col1:
    start_btn = st.button("🚀 Start Tuning", use_container_width=True, type="primary")
with col2:
    stop_btn = st.button("🛑 Stop Process", use_container_width=True)

log_path = ROOT_DIR / "tuning_results" / "latest_run.log"

# --- Button Logic ---
if stop_btn and st.session_state.process:
    st.session_state.process.terminate()
    st.session_state.process = None
    st.session_state.run_status = "stopped"
    if st.session_state.start_time:
        st.session_state.elapsed_time = time.time() - st.session_state.start_time
    st.session_state.start_time = None
    if st.session_state.log_file_handle:
        st.session_state.log_file_handle.close()

if start_btn:
    if st.session_state.run_status == "running":
        st.warning("A tuning process is already running!")
    elif not task:
        st.error("❌ You must select at least one Evaluation Task!")
    else:
        (ROOT_DIR / "tuning_results").mkdir(exist_ok=True)
        st.session_state.log_file_handle = open(log_path, "w", encoding="utf-8")
        
        cmd = [
            "python", "Global_Tuner_functions/modular_agent/orchestrator.py",
            "--model_id", model_id, "--task", task, "--memory_type", memory_type,
            "--max_iterations", str(max_iterations), "--max_time_hours", str(max_time_hours),
            "--target_vram_pct", str(target_vram), "--target_lat_pct", str(target_lat),
            "--target_emit_pct", str(target_emit), "--target_acc_drop", str(target_acc_drop),
            "--patience", str(patience), "--acc_weight", str(w_acc), "--lat_weight", str(w_lat),
            "--vram_weight", str(w_vram), "--emit_weight", str(w_emit)
        ]
        if num_samples > 0:
            cmd.extend(["--num_samples", str(num_samples)])
            
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
            
        st.session_state.process = subprocess.Popen(
            cmd, 
            stdout=st.session_state.log_file_handle, 
            stderr=subprocess.STDOUT, 
            text=True,
            env=env
        )
        st.session_state.run_status = "running"
        st.session_state.start_time = time.time()
        st.session_state.elapsed_time = 0
        time.sleep(1)

# --- Background Process Monitoring ---
if st.session_state.process:
    ret_code = st.session_state.process.poll()
    if ret_code is not None:
        st.session_state.process = None
        if st.session_state.start_time:
            st.session_state.elapsed_time = time.time() - st.session_state.start_time
        st.session_state.start_time = None
        
        if st.session_state.log_file_handle:
            st.session_state.log_file_handle.close()
            
        if ret_code == 0:
            st.session_state.run_status = "success"
        else:
            st.session_state.run_status = f"crashed (Exit code {ret_code})"

# --- Fetch Data Before Rendering ---
latest_dir = get_latest_exp_dir(model_id, task)

is_new_run = False
if latest_dir and st.session_state.start_time:
    if latest_dir.stat().st_ctime >= (st.session_state.start_time - 5):
        is_new_run = True

if latest_dir and st.session_state.run_status == "idle":
    config_path = latest_dir / "experiment_config.json"
    if config_path.exists():
        exp_config = load_json(config_path) or {}
        history_samples = exp_config.get("num_samples")
        expected_samples = num_samples if num_samples > 0 else None
        if history_samples != expected_samples:
            latest_dir = None 

results = []
baseline = {}

# 1. Try to load from active experiment
if latest_dir:
    raw_res = load_json(latest_dir / "optimization_results.json")
    results = raw_res if raw_res is not None else []
    
    config_path = latest_dir / "experiment_config.json"
    if config_path.exists():
        exp_config = load_json(config_path) or {}
        baseline = exp_config.get("baseline", {})

# 2. SMART CACHE PREVIEW: Scan all files to find matching baselines (Just like orchestrator!)
if not baseline:
    expected_samples = num_samples if num_samples > 0 else None
    cache_dir = ROOT_DIR / "tuning_results" / "baselines"
    
    if cache_dir.exists():
        for cache_file in cache_dir.glob(f"{model_name}_*.json"):
            try:
                raw_cache = load_json(cache_file) or {}
                if raw_cache.get("num_samples") == expected_samples:
                    details = raw_cache.get("details", {})
                    # If the cache file contains all the tasks we requested, preview it!
                    if all(t in details for t in selected_tasks):
                        baseline = raw_cache
                        break
            except Exception:
                continue

# ==========================================
# UI SECTION 1: TOP LEVEL TRACKERS & STATUS
# ==========================================
st.markdown("### ⏱️ Live Progress")
prog_col1, prog_col2, prog_col3 = st.columns(3)

prog_col1.metric("Current Iteration", f"{len(results)} / {max_iterations if max_iterations > 0 else '∞'}")

current_elapsed = st.session_state.elapsed_time
if st.session_state.run_status == "running" and st.session_state.start_time:
    current_elapsed = time.time() - st.session_state.start_time

hours, rem = divmod(current_elapsed, 3600)
mins, secs = divmod(rem, 60)
prog_col2.metric("Elapsed Time", f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

if max_time_hours > 0:
    max_sec = max_time_hours * 3600
    progress_pct = min(current_elapsed / max_sec, 1.0)
    with prog_col3:
        st.write(f"**Time Limit:** {progress_pct*100:.1f}%")
        st.progress(progress_pct)

status_banner = st.empty()
if st.session_state.run_status == "success":
    status_banner.success("✅ Optimization Complete!")
elif "crashed" in st.session_state.run_status:
    status_banner.error(f"❌ Process {st.session_state.run_status}! Check the trace logs below.")
elif st.session_state.run_status == "stopped":
    status_banner.warning("🛑 Process Terminated by User.")
elif st.session_state.run_status == "running":
    if not latest_dir or not is_new_run:
        status_banner.info("⏳ Starting process and preparing files...")
    else:
        exp_config_path = latest_dir / "experiment_config.json"
        results_path = latest_dir / "optimization_results.json"
        if not exp_config_path.exists():
            if baseline:
                status_banner.info("✅ Cached Baseline found! Waiting for orchestrator to start Iteration 1...")
            else:
                status_banner.warning("⏳ Evaluating Baseline from scratch... (Check Terminal Log!)")
        else:
            raw_evals = load_json(results_path)
            safe_evals = raw_evals if raw_evals is not None else []
            if not results_path.exists() or len(safe_evals) == 0:
                status_banner.info("🤖 Baseline ready! LLM is generating Iteration 1...")
            else:
                status_banner.success("🔄 Optimization in progress. Plotting live data...")

# ==========================================
# UI SECTION 2: BASELINE PERFORMANCE
# ==========================================
if baseline:
    acc = float(baseline.get('accuracy') or 0.0)
    lat = float(baseline.get('latency')  or 0.0)
    vram = float(baseline.get('vram')    or 0.0)
    emit = float(baseline.get('emissions') or 0.0)
    
    st.markdown("### 📏 Baseline Performance (Score = 1.0000)")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Accuracy", f"{acc:.4f}")
    b_col2.metric("Latency", f"{lat:.4f} s")
    b_col3.metric("VRAM", f"{vram:.4f} GB")
    b_col4.metric("Emissions", f"{emit:.6f} kg")
else:
    if st.session_state.run_status != "idle":
        st.info("📏 Baseline metrics will appear here once evaluation completes.")
st.divider()

# ==========================================
# UI SECTION 3: SCROLLABLE TERMINAL LOG
# ==========================================
if st.session_state.run_status != "idle":
    with st.expander("🖥️ Live Terminal & Error Logs", expanded=("crashed" in st.session_state.run_status)):
        with st.container(height=350):
            st.code(read_log_tail(log_path, lines=150), language="bash")

# ==========================================
# UI SECTION 4: CHARTS & TABLES
# ==========================================
if results:
    df_data = []
    for r in results:
        cfg = r.get("config", {})
        metrics = r.get("metrics", {})
        df_data.append({
            "名稱": r.get("trial_name", "unknown"),
            "Mode": cfg.get("mode", "unknown"),
            "設定": json.dumps(cfg),
            "Score": metrics.get("score", 0.0),
            "Δ Accuracy %": metrics.get("acc_diff_pct", 0.0),
            "Δ Latency %": metrics.get("lat_red", 0.0) * 100,
            "Δ VRAM %": metrics.get("vram_red", 0.0) * 100,
            "Δ Emissions %": metrics.get("emit_red", 0.0) * 100,
            "Error": r.get("error", "None")
        })
    df = pd.DataFrame(df_data)

    tab1, tab2, tab3 = st.tabs(["📈 Score Timeline", "🎯 Pareto Explorer", "📊 Detailed Results Table"])
    
    with tab1:
        st.line_chart(df["Score"].rename("Optimization Score"))
        
    with tab2:
        st.markdown("##### 🔍 Dynamic Trade-off Explorer")
        metric_options = ["Δ Accuracy %", "Δ Latency %", "Δ VRAM %", "Δ Emissions %", "Score"]
        
        sel_col1, sel_col2, sel_col3 = st.columns(3)
        with sel_col1: x_axis = st.selectbox("X-Axis", metric_options, index=2)
        with sel_col2: y_axis = st.selectbox("Y-Axis", metric_options, index=0)
        with sel_col3: bubble_size = st.selectbox("Bubble Size", metric_options, index=4)

        st.scatter_chart(df, x=x_axis, y=y_axis, color="Mode", size=bubble_size, height=500)

    with tab3:
        sort_col = st.selectbox("Sort Table By:", df.columns.tolist(), index=df.columns.get_loc("Δ Accuracy %"))
        
        delta_df = df[["名稱", "Mode", "設定", "Score", 
                       "Δ Accuracy %", "Δ Latency %", "Δ VRAM %", "Δ Emissions %", "Error"]].sort_values(
            sort_col, ascending=False
        )

        styled_delta = delta_df.style
        styled_delta = styled_delta.background_gradient(subset=["Score"], cmap="RdYlGn")
        styled_delta = styled_delta.background_gradient(subset=["Δ Accuracy %"], cmap="RdYlGn")
        for col in ["Δ Latency %", "Δ VRAM %", "Δ Emissions %"]:
            styled_delta = styled_delta.background_gradient(subset=[col], cmap="RdYlGn")
        
        styled_delta = styled_delta.format({
            "Score": "{:.4f}", "Δ Accuracy %": "{:+.1f}%", "Δ Latency %": "{:+.1f}%",
            "Δ VRAM %": "{:+.1f}%", "Δ Emissions %": "{:+.1f}%",
        })
        
        st.dataframe(styled_delta, use_container_width=True, height=500)

# Trigger auto-rerun ONLY if process is actively running
if st.session_state.run_status == "running":
    time.sleep(3)
    st.rerun()