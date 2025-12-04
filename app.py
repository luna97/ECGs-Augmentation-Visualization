import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb
import yaml
import augmentations  # Imports your augmentations.py file
from typing import Any
import plotly.express as px
import os
import json
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 1. SETUP & CONFIG LOADING
# ==========================================

st.set_page_config(layout="wide", page_title="Multi-Stage ECG Augmentation")
CUSTOM_PATHS_FILE = "ecg_paths.json" # <--- FILE TO STORE YOUR PATHS

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

CONFIG = load_config()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_custom_paths():
    """Loads custom paths from JSON file."""
    if os.path.exists(CUSTOM_PATHS_FILE):
        with open(CUSTOM_PATHS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_custom_path(name, path):
    """Saves a new path to the JSON file."""
    current_paths = load_custom_paths()
    current_paths[name] = path
    with open(CUSTOM_PATHS_FILE, "w") as f:
        json.dump(current_paths, f, indent=4)

# The dialog function
@st.dialog("Add New ECG Record")
def show_add_file_dialog():
    st.write("Upload the .hea (header) file for the WFDB record.")
    
    # 1. Display Name Input
    new_name = st.text_input("Display Name", placeholder="e.g. Patient 001")
    
    # 2. File Uploader (Replaces the Tkinter logic)
    # Streamlit handles the file selection and upload from the user's machine
    uploaded_file = st.file_uploader(
        "Upload WFDB Header (.hea)", 
        type="hea", # Restrict to .hea files
        accept_multiple_files=False
    )

    # Note: We don't use st.text_input for the path anymore, 
    # as the path is managed by the uploaded_file object.
    
    # Buttons
    col_submit, col_cancel = st.columns([1, 1])
    
    if col_submit.button("Save & Load", type="primary"):
        if new_name and uploaded_file:
            
            # --- File Handling Logic ---
            
            # The uploaded_file is a BytesIO object, not a local path.
            # We must save it temporarily to a location the WFDB reader can access.
            
            # Create a unique temporary directory for this session or user
            # temp_dir = 'temp_uploads' # Better to use tempfile.mkdtemp() for production
            # os.makedirs(temp_dir, exist_ok=True)
            
            # Define the full temporary path
            # The filename is what the user uploaded (e.g., 'record_100.hea')
            # temp_path_full = os.path.join(temp_dir, uploaded_file.name)
            
            # 1. Save the .hea file content to the server's temp path
            # with open(temp_path_full, "wb") as f:
            #    f.write(uploaded_file.getbuffer())

            # 2. Derive the base path (without extension) for WFDB access
            # base_path = os.path.splitext(temp_path_full)[0]
            
            # Your original function to save the path (now the temp path)
            save_custom_path(new_name, uploaded_file)
            
            st.session_state['selected_sample_key'] = new_name
            st.toast("Record uploaded and saved successfully!")
            st.rerun() # Closes the dialog and updates the main app
            
        else:
            st.warning("Please fill in the Display Name and upload the .hea file.")
    
    if col_cancel.button("Cancel"):
        st.session_state['dialog_open'] = False
        st.rerun()

def get_real_ecg(file_name: str):
    """Loads ECG data using WFDB."""
    try:
        s, info = wfdb.rdsamp(file_name)
        t = np.arange(s.shape[0]) / info['fs']
        # print(f'leads: {info["sig_name"]}, fs: {info["fs"]}, shape: {s.shape}')
        return s, t, info['sig_name'], info['fs']
    except Exception as e:
        st.error(f"Error loading file {file_name}: {e}")
        # Return dummy data if file missing
        return np.zeros((1000, 12)), np.linspace(0, 2, 1000), [f"L{i}" for i in range(12)], 500

def instantiate_augmentation(name: str, params: dict, fs: int):
    """
    Dynamically instantiate the class from augmentations.py.
    Injects context-aware variables (fs, max_length) automatically.
    """
    if not hasattr(augmentations, name):
        st.error(f"Class '{name}' not found in augmentations.py")
        return None
    
    cls = getattr(augmentations, name)
    
    # 1. Create a copy of params to avoid modifying session state
    init_kwargs = params.copy()
    
    # 2. Inject context-aware parameters if the class expects them
    # We inspect the mapping based on common variable names used in your classes
    
    # Mapping of YAML/Class arg names -> Value from App
    # This assumes consistent naming in your augmentations.py classes
    context_map = {
        'fs': fs,
        'signal_fs': fs,
        'current_freq': fs,
        's_freq': fs,
        'max_length': fs * 10  # Arbitrary max length for cropping
    }

    # Only inject if the user hasn't manually provided it (though usually these aren't in YAML params)
    # We cheat slightly: if the class has these arguments in __init__, we pass them.
    # A cleaner way is to check the class signature, but for now we look at specific class needs:
    
    if name in ["RandomShiftBaselineWander"]:
        init_kwargs['signal_fs'] = fs
    elif name in ["Masking", "HighpassFilter", "LowpassFilter"]:
        init_kwargs['fs'] = fs
    elif name in ["RandomResample"]:
        init_kwargs['current_freq'] = fs
    elif name in ["RandomCrop"]:
        init_kwargs['max_length'] = fs * 10
    elif name in ["FrequencyShift"]:
        init_kwargs['s_freq'] = fs

    try:
        return cls(**init_kwargs)
    except TypeError as e:
        st.error(f"Error instantiating {name}: {e}. Check config.yaml params match class __init__.")
        return None

# ==========================================
# 3. STREAMLIT UI LOGIC
# ==========================================

st.title("🫀 Multi-Stage ECG Augmentation Pipeline")

# --- INITIALIZE SESSION STATE ---
if 'pipeline' not in st.session_state: st.session_state['pipeline'] = [] 
if 'seed' not in st.session_state: st.session_state['seed'] = 42

# --- SIDEBAR: PIPELINE BUILDER ---
with st.sidebar:
    st.header("1. Input Signal")
    
    if st.button("New Random Seed"):
        st.session_state['seed'] = np.random.randint(0, 1000)

    # --- FILE SELECTION LOGIC ---
    
    # 1. Define Default Paths
    default_options = {
        # "HEEDB ECG": "data/heedb/de_111189359_20161124145652_20161128164509",
        # "HEEDB ECG padded": "data/heedb/de_115966442_20050713160941_20050716091727",
        # "CODE-15% ECG": "data/code/1000010"
        "PTB-XL 01000": "data/ptb_xl/01000_hr",
        "PTB-XL 01154": "data/ptb_xl/01154_hr",
    }
    custom_options = load_custom_paths()
    all_options = {**default_options, **custom_options}
    
    # Prepare list for dropdown
    dropdown_keys = list(all_options.keys()) + ["➕ Add Local File..."]

    # Ensure session state key exists for the widget
    if 'selected_sample_key' not in st.session_state:
        st.session_state['selected_sample_key'] = dropdown_keys[0]

    # Select Box
    selected_option_key = st.selectbox(
        "Select Sample", 
        dropdown_keys,
        key='selected_sample_key' # Important for auto-selecting after save
    )
    
    # Logic: If special option selected, show popup
    if selected_option_key == "➕ Add Local File...":
        show_add_file_dialog()
        # Fallback values while dialog is open
        original_signal = np.zeros((1000, 12))
        t_orig = np.linspace(0, 1, 1000)
        lead_names = [f"L{i}" for i in range(12)]
        fs = 1000
    else:
        # Load the selected file
        file_path_to_load = all_options[selected_option_key]
        
        # Option to delete custom files
        if selected_option_key in custom_options:
            if st.button("🗑️ Delete from list"):
                del custom_options[selected_option_key]
                with open(CUSTOM_PATHS_FILE, "w") as f:
                    json.dump(custom_options, f, indent=4)
                # Reset selection to first default
                st.session_state['selected_sample_key'] = list(default_options.keys())[0]
                st.rerun()

        original_signal, t_orig, lead_names, fs = get_real_ecg(file_path_to_load)

    st.divider()
    
    # ... (Rest of sidebar: Pipeline Builder, etc. remains identical) ...
    # PASTE THE REST OF THE PIPELINE LOGIC FROM PREVIOUS STEPS HERE
    # Ensure you keep the Pipeline logic, the Loop, and the Plotting logic.
    
    st.header("2. Pipeline Builder")
    # ... [Previous Pipeline Logic Code] ...
    
    available_augs = list(CONFIG['augmentations'].keys())
    c1, c2 = st.columns([2, 1])
    selected_aug_to_add = c1.selectbox("Select Augmentation", available_augs)
    
    if c2.button("Add"):
        aug_config = CONFIG['augmentations'][selected_aug_to_add]['params']
        defaults = {k: v['default'] for k, v in aug_config.items()}
        st.session_state['pipeline'].append({'name': selected_aug_to_add, 'params': defaults})
        st.rerun()
        
    # [Iterate Pipeline UI Code ...]
    indices_to_remove = []
    for i, item in enumerate(st.session_state['pipeline']):
        name = item['name']
        aug_def = CONFIG['augmentations'][name]
        with st.expander(f"{i+1}. {name}", expanded=True):
             # [Sliders Logic Code ...]
             for param_key, param_info in aug_def['params'].items():
                if param_info.get('hidden', False): continue
                label = param_info.get('label', param_key)
                widget_key = f"{name}_{i}_{param_key}"
                
                if param_info['type'] == 'float':
                    item['params'][param_key] = st.slider(label, float(param_info['min']), float(param_info['max']), float(item['params'][param_key]), key=widget_key)
                elif param_info['type'] == 'int':
                    item['params'][param_key] = st.slider(label, int(param_info['min']), int(param_info['max']), int(item['params'][param_key]), step=1, key=widget_key)
                elif param_info['type'] == 'bool':
                    item['params'][param_key] = st.checkbox(label, value=bool(item['params'][param_key]), key=widget_key)
                    
             if st.button(f"Remove", key=f"rem_{i}"): indices_to_remove.append(i)

    if indices_to_remove:
        for index in sorted(indices_to_remove, reverse=True): del st.session_state['pipeline'][index]
        st.rerun()

    if st.session_state['pipeline']:
        col_clr, col_run = st.columns(2)
        if col_clr.button("Clear All"):
            st.session_state['pipeline'] = []
            st.rerun()
        if col_run.button("Refresh / Re-Roll"):
            st.session_state['seed'] = np.random.randint(0, 100000)
            st.rerun()

# --- MAIN PROCESS & VISUALIZATION ---

# 1. Pipeline Processing
history = {}
history["Original"] = (t_orig, original_signal)
sig_np = original_signal.copy()

# Set random seeds for reproducibility
np.random.seed(st.session_state['seed'])
import torch
torch.manual_seed(st.session_state['seed'])

for i, item in enumerate(st.session_state['pipeline']):
    name = item['name']
    params = item['params']
    
    # Factory instantiation
    aug_layer = instantiate_augmentation(name, params, fs)
    
    if aug_layer:
        if hasattr(aug_layer, 'training'):
            aug_layer.training = True
            
        try:
            sig_np = aug_layer(sig_np)
            
            # Recalculate time if length changed
            new_len = sig_np.shape[0] if not torch.is_tensor(sig_np) else sig_np.shape[0]
            t_current = np.linspace(0, t_orig[-1], new_len) # Approximate time stretch mapping
            
            # Handle Tensor -> Numpy conversion for plotting
            if torch.is_tensor(sig_np):
                plot_data = sig_np.detach().cpu().numpy()
            else:
                plot_data = sig_np

            history[f"Step {i+1}: {name}"] = (t_current, plot_data)
            
        except Exception as e:
            st.error(f"Pipeline Error at {name}: {e}")
            break

# 2. Plotting Logic
st.subheader("ECG Visualization")

# Lead Ordering Logic (Interleaved)
num_rows, num_cols = 6, 2
midpoint = len(lead_names) // 2
col1_names = lead_names[:midpoint]
col2_names = lead_names[midpoint:]
new_lead_names = []
for n1, n2 in zip(col1_names, col2_names):
    new_lead_names.extend([n1, n2])

fig = make_subplots(
    rows=6, cols=2, 
    shared_xaxes=True, 
    vertical_spacing=0.03,
    subplot_titles=new_lead_names,
)

colors = px.colors.qualitative.Plotly
# ["#D7D3D3", '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

for layer_name in history.keys():
    t_plot, sig_plot = history[layer_name]
    
    # Style
    if layer_name == "Original":
        color = 'rgba(255, 255, 255, 0.9)'
        width = 1.5
    else:
        c_idx = list(history.keys()).index(layer_name) % len(colors)
        color = colors[c_idx]
        width = 1.5

    for lead_i in range(12): # Assume 12 leads
        row = (lead_i % 6) + 1
        col = (lead_i // 6) + 1
        
        # Safety check if signal has fewer leads than expected
        if lead_i < sig_plot.shape[1]:
            y_val = sig_plot[:, lead_i]
            
            fig.add_trace(go.Scatter(
                y=y_val,
                mode='lines',
                name=layer_name,
                line=dict(color=color, width=width),
                showlegend=(lead_i == 0),
                legendgroup=layer_name
            ), row=row, col=col)

fig.update_layout(
    height=1000, 
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, groupclick="togglegroup"),
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)