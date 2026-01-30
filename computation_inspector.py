import streamlit as st
import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

ACTIVATIONS = {
    "ReLU": relu,
    "Sigmoid": sigmoid
}

def render_computation_inspector(
    topology,
    network_data,
    input_vector,
    activation_fn,
    epoch_idx
):
    st.markdown("## 🧮 Computation Inspector")

    # --- TOGGLE BUTTONS ---
    mode = st.radio(
        "",
        ["Forward Propagation", "Gradient Calculation", "Parameter Updates"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Forward Propagation":
        render_forward_propagation(
            topology,
            network_data,
            input_vector,
            activation_fn,
            epoch_idx
        )

    elif mode == "Gradient Calculation":
        st.info("Gradient computation will appear here (Phase 2)")

    elif mode == "Parameter Updates":
        st.info("Parameter update computation will appear here (Phase 2)")


def render_forward_propagation(
    topology,
    network_data,
    input_vector,
    activation_fn,
    epoch_idx
):
    st.subheader("Forward Propagation")

    A_prev = np.array(input_vector).reshape(-1, 1)

    act_fn = ACTIVATIONS.get(activation_fn, relu)

    for layer_idx in range(1, len(topology)):
        st.markdown(f"### 🔹 Layer {layer_idx} Computation")

        weights = []
        biases = []

        for neuron_idx in range(topology[layer_idx]):
            key = f"L{layer_idx}_N{neuron_idx}"
            neuron = network_data.get(key)

            if neuron:
                weights.append(neuron["weights"])
                biases.append(neuron["bias"])
            else:
                weights.append([0] * topology[layer_idx - 1])
                biases.append(0)

        W = np.array(weights)
        b = np.array(biases).reshape(-1, 1)

        Z = W @ A_prev + b
        A = act_fn(Z)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Input (A⁽ˡ⁻¹⁾)**")
            st.dataframe(pd.DataFrame(A_prev))

        with col2:
            st.markdown("**Weights (W⁽ˡ⁾)**")
            st.dataframe(pd.DataFrame(W))

            st.markdown("**Biases (b⁽ˡ⁾)**")
            st.dataframe(pd.DataFrame(b))

        with col3:
            st.markdown("**Z = W·A + b**")
            st.dataframe(pd.DataFrame(Z))

            st.markdown("**Activation A = f(Z)**")
            st.dataframe(pd.DataFrame(A))

        st.markdown(
            "**Equation:**  \n"
            "`A_prev × W + b = Z → A`"
        )

        A_prev = A
