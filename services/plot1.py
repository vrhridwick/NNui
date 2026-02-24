import json
import streamlit as st
import pandas as pd
import numpy as np


def load_training_result(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def display_training_results(data):

    history = data["history"]

    epochs = [item["epoch"] for item in history]
    losses = [item["error"] for item in history]

    accuracies = [
        1 - abs(item["actual_output"][0] - item["expected_output"][0])
        for item in history
    ]

    # --- RESET + SLIDER ROW ---
    c1, c2 = st.columns([1, 4])

    with c1:
        if st.button("Reset View"):
            st.rerun()

    with c2:
        selected_epoch = st.slider(
            "Epoch Timeline",
            min_value=1,
            max_value=len(history),
            value=len(history)
        )

    epoch_data = history[selected_epoch - 1]

    # --- METRIC DISPLAY ---
    st.metric(
        label=f"Loss at Epoch {selected_epoch}",
        value=f"{epoch_data['error']:.5f}"
    )

    # --- OUTPUT COMPARISON ---
    st.subheader(f"Output Comparison (Epoch {selected_epoch})")

    comparison = []
    for i, (pred, target) in enumerate(
        zip(epoch_data["actual_output"], epoch_data["expected_output"])
    ):
        comparison.append({
            "Output Neuron": f"y{i+1}",
            "Expected": round(target, 4),
            "Predicted": round(pred, 4),
            "Error": round(abs(target - pred), 4)
        })

    st.dataframe(
        pd.DataFrame(comparison),
        use_container_width=True,
        hide_index=True
    )

    # --- LOSS GRAPH ---
    st.subheader("Loss vs Epoch")

    loss_df = pd.DataFrame({
        "Epoch": epochs,
        "Loss": losses
    })

    st.line_chart(loss_df, x="Epoch", y="Loss", height=250)

    # --- ACCURACY GRAPH ---
    st.subheader("Accuracy vs Epoch")

    acc_df = pd.DataFrame({
        "Epoch": epochs,
        "Accuracy": accuracies
    })

    st.line_chart(acc_df, x="Epoch", y="Accuracy", height=250)

    st.markdown("---")