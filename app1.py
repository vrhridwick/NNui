import streamlit as st
import graphviz
import pandas as pd
import numpy as np
import time
from services.payload_builder import build_training_payload
from services.validators import validate_full_config
import json
import os
# Assuming computation_inspector exists in your local directory
try:
    from computation_inspector import render_computation_inspector
except ImportError:
    # Fallback if file is missing (for safety)
    def render_computation_inspector(**kwargs):
        st.warning("computation_inspector.py not found. Computation view disabled.")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="LucidNN", layout="wide", page_icon="🧠")

# --- TITLE SECTION ---
st.title("LucidNN 🧠")
st.caption("Interactive Neural Network Visualization Tool")
st.markdown("---")

# --- SESSION STATE INITIALIZATION ---
# 1. New Sidebar State
if 'layers' not in st.session_state:
    st.session_state.layers = [{"id": 0, "neurons": 3}]
if 'layer_counter' not in st.session_state:
    st.session_state.layer_counter = 0

# 2. Existing Visualizer State
if 'hidden_layers' not in st.session_state:
    st.session_state.hidden_layers = [3] 
if 'network_data' not in st.session_state:
    st.session_state.network_data = {} 
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = {} 
if 'output_history' not in st.session_state:
    st.session_state.output_history = [] 
if 'targets' not in st.session_state:
    st.session_state.targets = [] 

# --- HELPER FUNCTIONS ---
def get_topology(inputs, hidden, outputs):
    return [inputs] + hidden + [outputs]

def init_neuron_data(layer_idx, neuron_idx, num_prev_neurons):
    key = f"L{layer_idx}_N{neuron_idx}"
    if key not in st.session_state.network_data or \
       len(st.session_state.network_data[key]['weights']) != num_prev_neurons:
        
        st.session_state.network_data[key] = {
            "bias": np.random.uniform(-0.5, 0.5),
            "weights": [np.random.uniform(-1, 1) for _ in range(num_prev_neurons)]
        }
        
        if st.session_state.trained:
             st.session_state.trained = False
             st.session_state.training_history = {}
             st.toast(f"Network architecture changed. Model reset.", icon="⚠️")
    return key

def calculate_stats(topology):
    total_layers = len(topology)
    total_neurons = sum(topology)
    total_connections = 0
    for i in range(len(topology) - 1):
        total_connections += topology[i] * topology[i+1]
    return total_layers, total_neurons, total_connections

# --- DIALOG: SET WEIGHTS & BIAS ---
@st.dialog("Set Weights & Bias")
def open_neuron_editor(layer_idx, neuron_idx, prev_layer_size):
    key = init_neuron_data(layer_idx, neuron_idx, prev_layer_size)
    data = st.session_state.network_data[key]

    st.subheader(f"Editing: Hidden Layer {layer_idx}, Neuron {neuron_idx+1}")
    
    # Bias
    new_bias = st.number_input("Bias", value=float(data['bias']), step=0.01, key=f"bias_{key}")
    
    st.markdown("---")
    st.markdown(f"**Weights (from previous layer: {prev_layer_size} inputs)**")
    
    # Weights
    new_weights = []
    cols = st.columns(3)
    for i in range(prev_layer_size):
        with cols[i % 3]:
            current_w_val = float(data['weights'][i])
            w = st.number_input(f"W_{i+1}", value=current_w_val, step=0.01, key=f"w_{key}_{i}")
            new_weights.append(w)
            
    if st.button("🎲 Randomize Values"):
        st.session_state.network_data[key]['bias'] = np.random.uniform(-1, 1)
        st.session_state.network_data[key]['weights'] = [np.random.uniform(-1, 1) for _ in range(prev_layer_size)]
        st.rerun()

    if st.button("Save Changes", type="primary"):
        st.session_state.network_data[key]['bias'] = new_bias
        st.session_state.network_data[key]['weights'] = new_weights
        st.rerun()

# --- SIDEBAR CONFIGURATION (UPDATED) ---
with st.sidebar:
    st.header("Model Configuration")
    
    tab_arch, tab_hyper, tab_train = st.tabs(["Architecture", "Hyperparameters", "Training"])

    # --- TAB 1: ARCHITECTURE ---
    with tab_arch:
        input_nodes = st.number_input("Input Nodes", min_value=1, value=2, step=1)
        output_nodes = st.number_input("Output Nodes", min_value=1, value=1, step=1)
        
        st.markdown("---")
        st.subheader("Hidden Layers")
        
        if st.button("➕ Add Hidden Layer", use_container_width=True):
            st.session_state.layer_counter += 1
            st.session_state.layers.append({
                "id": st.session_state.layer_counter, 
                "neurons": 3
            })
            st.rerun()

        layers_to_remove = []
        for i, layer in enumerate(st.session_state.layers):
            st.markdown(f"**Layer {i+1}**")
            col1, col2 = st.columns([4, 1])
            with col1:
                new_val = st.number_input(
                    label="Neurons",
                    min_value=1,
                    value=layer['neurons'],
                    step=1,
                    key=f"layer_neurons_{layer['id']}",
                    label_visibility="collapsed"
                )
                st.session_state.layers[i]['neurons'] = new_val
            with col2:
                if st.button("✖", key=f"del_{layer['id']}", help="Delete this layer"):
                    layers_to_remove.append(i)

        if layers_to_remove:
            for index in sorted(layers_to_remove, reverse=True):
                del st.session_state.layers[index]
            st.rerun()

    # --- TAB 2: HYPERPARAMETERS ---
    with tab_hyper:
        st.subheader("Hyperparameters")
        activation = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Softmax"])
        loss_fn = st.selectbox("Loss Function", ["Mean Squared Error (MSE)", "Binary Cross-Entropy", "Categorical Cross-Entropy", "Hinge Loss"])
        regularization = st.selectbox("Regularization", ["None", "L1 (Lasso)", "L2 (Ridge)", "Dropout"])
        
        if regularization == "Dropout":
            st.slider("Dropout Rate", 0.0, 1.0, 0.2)

    # --- TAB 3: TRAINING ---
    with tab_train:
        st.subheader("Training Config")
        epochs = st.slider("Epochs", min_value=10, max_value=1000, step=10, value=100)
        learning_rate = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f")

# --- COMPATIBILITY BRIDGE ---
# This maps the variables from your new sidebar structure to the variables 
# the existing visualization logic expects.
st.session_state.hidden_layers = [layer["neurons"] for layer in st.session_state.layers]
activ_func = activation
loss_func = loss_fn
epochs_setting = epochs

# Recalculate topology based on new sidebar inputs
topology = get_topology(input_nodes, st.session_state.hidden_layers, output_nodes)
prev_layer_size = input_nodes
# print("Hidden layers:", st.session_state.layers)
# print("Input nodes:", input_nodes)
# print("Output nodes:", output_nodes)
# Hidden layers
for layer_idx, neurons in enumerate(st.session_state.hidden_layers, start=1):
    for neuron_idx in range(neurons):
        init_neuron_data(layer_idx, neuron_idx, prev_layer_size)
    prev_layer_size = neurons

# Output layer
output_layer_idx = len(st.session_state.hidden_layers) + 1
for neuron_idx in range(output_nodes):
    init_neuron_data(output_layer_idx, neuron_idx, prev_layer_size)
# --- MAIN PAGE LAYOUT ---
col_viz, col_interact = st.columns([3, 2])

# --- LEFT COLUMN: VISUALIZATION ---
with col_viz:
    st.subheader("Network Architecture")
    
    # Calculate stats for display
    t_layers, t_neurons, t_conns = calculate_stats(topology)
    s1, s2, s3 = st.columns(3)
    s1.metric("Layers", t_layers)
    s2.metric("Neurons", t_neurons)
    s3.metric("Connections", t_conns)

    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', splines='line', bgcolor='transparent')
    
    for l_idx, count in enumerate(topology):
        with graph.subgraph(name=f'cluster_{l_idx}') as c:
            c.attr(color='white', label=f'Layer {l_idx}')
            
            if l_idx == 0:
                color = '#FFCCCC' # Light Red
                label_prefix = 'x'
            elif l_idx == len(topology)-1:
                color = '#CCFFCC' # Light Green
                label_prefix = 'y'
            else:
                color = '#FFFFCC' # Light Yellow
                label_prefix = 'N'
            
            for n_idx in range(count):
                node_label = f"{label_prefix}{n_idx+1}"
                c.node(f'{l_idx}_{n_idx}', 
                       label=node_label, 
                       shape='circle', 
                       style='filled', 
                       fillcolor=color, 
                       color='black', 
                       fontcolor='black', 
                       width='0.6', 
                       fixedsize='true')

    for l_idx in range(len(topology) - 1):
        for n1 in range(topology[l_idx]):
            for n2 in range(topology[l_idx+1]):
                graph.edge(f'{l_idx}_{n1}', f'{l_idx+1}_{n2}', color='black')

    st.graphviz_chart(graph, use_container_width=True)

# --- RIGHT COLUMN: INTERACTION ---
with col_interact:
    st.subheader("Neuron Details")
    
    neuron_options = []
    for l in range(1, len(topology)): 
        layer_type = "Output" if l == len(topology)-1 else f"Hidden {l}"
        for n in range(topology[l]):
            neuron_options.append(f"Layer {l} ({layer_type}) - Neuron {n+1}")
            
    selected_neuron_str = st.selectbox("Select a Neuron to Inspect:", neuron_options)
    
    if selected_neuron_str:
        parts = selected_neuron_str.split(' ')
        l_idx = int(parts[1])
        n_idx = int(parts[-1]) - 1
        prev_layer_size = topology[l_idx - 1]
        
        key = init_neuron_data(l_idx, n_idx, prev_layer_size)
        curr_data = st.session_state.network_data[key]
        
        st.markdown(f"**Current Bias:** `{curr_data['bias']:.4f}`")
        
        with st.expander("View Weights", expanded=True):
            w_df = pd.DataFrame(curr_data['weights'], columns=["Weight Value"])
            if len(curr_data['weights']) == prev_layer_size:
                w_df.index = [f"Connection from Layer {l_idx-1} Neuron {i+1}" for i in range(prev_layer_size)]
            else:
                w_df.index = [f"Input {i+1}" for i in range(len(curr_data['weights']))]
            st.dataframe(w_df, use_container_width=True)

        if not st.session_state.trained:
            if st.button("🛠️ Edit Weights & Bias"):
                open_neuron_editor(l_idx, n_idx, prev_layer_size)
        
        else:
            st.info(f"Average Weight Over {epochs_setting} Epochs")
            
            if key in st.session_state.training_history:
                history_data = st.session_state.training_history[key]
                avg_weights = [np.mean(epoch_weights) for epoch_weights in history_data]
                
                chart_data = pd.DataFrame({
                    "Epoch": range(len(avg_weights)),
                    "Avg Weight": avg_weights
                })
                
                st.line_chart(chart_data, x="Epoch", y="Avg Weight", height=250)

# --- BOTTOM SECTION: TRAINING & RESULTS ---
st.markdown("---")
# st.write(st.session_state.network_data)


    
if not st.session_state.trained:
    if st.button("Train Model", type="primary"):

        payload = build_training_payload(
        input_nodes=input_nodes,
        hidden_layers=st.session_state.layers,
        output_nodes=output_nodes,
        activation=activation,
        epochs=epochs,
        learning_rate=learning_rate,
        # training_inputs=training_inputs,
        # training_targets=training_targets,
        network_data=st.session_state.network_data
        )
        os.makedirs("config", exist_ok=True)
        with open("config/config.json", "w") as f:
            json.dump(payload, f, indent=4)

        st.success("config.json generated!")

        with st.spinner(f"Training for {epochs_setting} epochs..."):
            time.sleep(1.0) 
            
            # --- SIMULATION START ---
            st.session_state.training_history = {}
            st.session_state.output_history = []
            
            st.session_state.targets = [round(np.random.uniform(0.1, 0.9), 4) for _ in range(output_nodes)]
            
            output_hist = []
            
            for epoch in range(epochs_setting + 1):
                progress = 1 - (0.95 ** epoch)
                
                # A. Generate Output Predictions
                epoch_preds = []
                for t in st.session_state.targets:
                    noise = (np.random.normal(0, 0.5) * (1 - progress)) + (0.5 * (1-progress))
                    pred = t + noise
                    epoch_preds.append(pred)
                output_hist.append(epoch_preds)

                # B. Generate Weights History
                for l in range(1, len(topology)):
                    for n in range(topology[l]):
                        k = f"L{l}_N{n}"
                        if k not in st.session_state.training_history:
                            st.session_state.training_history[k] = []
                        
                        prev_size = topology[l-1]
                        base_weights = st.session_state.network_data.get(k, {}).get('weights', [0]*prev_size)
                        current_weights = [w + np.random.normal(0, 0.01 * epoch) for w in base_weights]
                        st.session_state.training_history[k].append(current_weights)

            st.session_state.output_history = output_hist
            st.session_state.trained = True
            st.rerun()

else:
    c_reset, c_slider = st.columns([1, 4])
    
    if c_reset.button("Reset Model"):
        st.session_state.trained = False
        st.session_state.network_data = {}
        st.rerun()

    with c_slider:
        curr_epoch = st.slider("Epoch Timeline", 0, epochs_setting, epochs_setting)
        
        if len(st.session_state.output_history) > curr_epoch:
            current_preds = st.session_state.output_history[curr_epoch]
            targets = st.session_state.targets
            mse = np.mean([(t - p)**2 for t, p in zip(targets, current_preds)])
            st.metric(label=f"Total Error ({loss_func}) at Epoch {curr_epoch}", value=f"{mse:.5f}")

    # --- EXPECTED vs ACTUAL TABLE ---
    st.subheader(f"Output Comparison at Epoch {curr_epoch}")
    
    if len(st.session_state.output_history) > curr_epoch:
        current_preds = st.session_state.output_history[curr_epoch]
        targets = st.session_state.targets
        
        comparison_data = []
        for i, (pred, target) in enumerate(zip(current_preds, targets)):
            comparison_data.append({
                "Output Neuron": f"y{i+1}",
                "Expected (Target)": f"{target:.4f}",
                "Actual (Predicted)": f"{pred:.4f}",
                "Error (Diff)": f"{abs(target - pred):.4f}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # --- ERROR GRAPH ---
    st.subheader("Total Error vs Epoch")
    
    mse_history = []
    for preds in st.session_state.output_history:
        mse = np.mean([(t - p)**2 for t, p in zip(st.session_state.targets, preds)])
        mse_history.append(mse)

    loss_df = pd.DataFrame({
        "Epoch": range(len(mse_history)),
        "Error": mse_history
    })
    
    st.line_chart(loss_df, x="Epoch", y="Error", height=250)

    # --- WEIGHT SUMMARY ---
    st.subheader("Layer-wise Weight Summary (Final Epoch)")
    summary_data = []
    for l in range(1, len(topology)):
        for n in range(topology[l]):
            key = f"L{l}_N{n}"
            if key in st.session_state.training_history:
                final_weights = st.session_state.training_history[key][-1]
                curr_bias = st.session_state.network_data.get(key, {}).get('bias', 0.0)
                
                summary_data.append({
                    "Layer": l,
                    "Neuron": n+1,
                    "Avg Wt": round(np.mean(final_weights), 4),
                    "Min Wt": round(np.min(final_weights), 4),
                    "Max Wt": round(np.max(final_weights), 4),
                    "Bias": round(curr_bias, 4)
                })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    st.markdown("---")

st.markdown("---")
st.subheader("🔍 Detailed Computation View")

if st.session_state.trained:
    input_vector = [1.0] * topology[0]
    render_computation_inspector(
        topology=topology,
        network_data=st.session_state.network_data,
        input_vector=input_vector,
        activation_fn=activ_func,
        epoch_idx=curr_epoch
    )
else:
    st.info("Train the model to inspect computations.")
