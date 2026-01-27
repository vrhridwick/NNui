import streamlit as st
import graphviz
import pandas as pd
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="LucidNN", layout="wide", page_icon="🧠")

# --- TITLE SECTION ---
st.title("LucidNN 🧠")
st.caption("Interactive Neural Network Visualization Tool")
st.markdown("---")

# --- SESSION STATE MANAGEMENT ---
if 'hidden_layers' not in st.session_state:
    st.session_state.hidden_layers = [3] 

if 'network_data' not in st.session_state:
    st.session_state.network_data = {} 
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = {} # Weights history
if 'output_history' not in st.session_state:
    st.session_state.output_history = []   # Outputs history (Actual vs Expected)
if 'targets' not in st.session_state:
    st.session_state.targets = []          # The "Expected" values

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

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("Network Config")
    
    # 1. ARCHITECTURE
    st.subheader("Architecture")
    
    col_label, col_add = st.columns([3, 1])
    col_label.write("**Hidden Layers**")
    if col_add.button("➕"): 
        st.session_state.hidden_layers.append(3) 
    
    layers_to_remove = []
    for i, n in enumerate(st.session_state.hidden_layers):
        c1, c2 = st.columns([4, 1])
        st.session_state.hidden_layers[i] = c1.number_input(f"Layer {i+1} Neurons", 1, 10, n, key=f"h_{i}")
        if c2.button("X", key=f"rm_{i}"): 
            layers_to_remove.append(i)
    
    for i in sorted(layers_to_remove, reverse=True):
        st.session_state.hidden_layers.pop(i)
        st.rerun()
        
    st.markdown("---")
    input_nodes = st.number_input("Input Features", 1, 10, 2)
    output_nodes = st.number_input("Output Classes", 1, 10, 2)

    # 2. HYPERPARAMETERS
    st.markdown("---")
    st.subheader("Hyperparameters")
    activ_func = st.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "Softmax"])
    loss_func = st.selectbox("Loss Function", ["Mean Squared Error (MSE)", "Cross Entropy", "Hinge Loss"])
    epochs_setting = st.number_input("Number of Epochs", 10, 1000, 100)

    # 3. STATS
    st.markdown("---")
    st.subheader("Network Stats")
    topology = get_topology(input_nodes, st.session_state.hidden_layers, output_nodes)
    t_layers, t_neurons, t_conns = calculate_stats(topology)
    st.metric("Total Layers", t_layers)
    st.metric("Total Neurons", t_neurons)
    st.metric("Total Connections", t_conns)


# --- MAIN PAGE LAYOUT ---
col_viz, col_interact = st.columns([3, 2])

# --- LEFT COLUMN: VISUALIZATION ---
with col_viz:
    st.subheader("Network Architecture")
    
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

if not st.session_state.trained:
    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training for {epochs_setting} epochs..."):
            time.sleep(1.0) 
            
            # --- SIMULATION START ---
            st.session_state.training_history = {}
            st.session_state.output_history = []
            
            # 1. Generate "Truth" Targets (random between 0 and 1)
            # We simulate that the network tries to reach these values
            st.session_state.targets = [round(np.random.uniform(0.1, 0.9), 4) for _ in range(output_nodes)]
            
            # 2. Simulate Training Loop
            output_hist = []
            
            for epoch in range(epochs_setting + 1):
                # Calculate simulated progress factor (0.0 to 1.0)
                # Network gets better as epoch increases
                progress = 1 - (0.95 ** epoch) # Converges to 1
                
                # A. Generate Output Predictions for this epoch
                epoch_preds = []
                for t in st.session_state.targets:
                    # Current prediction = Target + Noise
                    # Noise decreases as progress increases
                    noise = (np.random.normal(0, 0.5) * (1 - progress)) + (0.5 * (1-progress))
                    pred = t + noise
                    epoch_preds.append(pred)
                output_hist.append(epoch_preds)

                # B. Generate Weights History (Random Walk)
                for l in range(1, len(topology)):
                    for n in range(topology[l]):
                        k = f"L{l}_N{n}"
                        if k not in st.session_state.training_history:
                            st.session_state.training_history[k] = []
                        
                        prev_size = topology[l-1]
                        # Use existing weights or random start
                        base_weights = st.session_state.network_data.get(k, {}).get('weights', [0]*prev_size)
                        
                        # Perturb weights slightly based on epoch
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
        # SLIDER
        curr_epoch = st.slider("Epoch Timeline", 0, epochs_setting, epochs_setting)
        
        # Calculate Total Error for this specific epoch
        # (Consistent with the table below)
        if len(st.session_state.output_history) > curr_epoch:
            current_preds = st.session_state.output_history[curr_epoch]
            targets = st.session_state.targets
            # MSE Calculation
            mse = np.mean([(t - p)**2 for t, p in zip(targets, current_preds)])
            st.metric(label=f"Total Error (MSE) at Epoch {curr_epoch}", value=f"{mse:.5f}")

    # --- EXPECTED vs ACTUAL TABLE (Requested Feature) ---
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
    
    # Calculate MSE for all epochs to plot graph
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