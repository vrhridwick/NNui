def build_training_payload(
    input_nodes,
    hidden_layers,
    output_nodes,
    activation,
    epochs,
    learning_rate,
    # training_inputs=None,
    # training_targets=None,
    network_data
):
    return {
        "type": "INIT_NETWORK",
        "network": {
            "input_size": input_nodes,
            "hidden_layers": [
                {
                    "neurons": layer,
                    "activation": activation.lower()
                }
                for layer in hidden_layers
            ],
            "output_layer": {
                "neurons": output_nodes,
                "activation": activation.lower()
            }
        },
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate
        },
        # "training_data": {
        #     "inputs": training_inputs,
        #     "targets": training_targets
        # },
        "initial_state": network_data
    }
# def collect_network_config(state):
#     return {
#         "input_size": state.input_nodes,
#         "hidden_layers": [
#             {
#                 "neurons": layer_size,
#                 "activation": state.activation.lower()
#             }
#             for layer_size in state.layers
#         ],
#         "output_layer": {
#             "neurons": state.output_nodes,
#             "activation": state.activation.lower()
#         }
#     }


# def collect_hyperparameters(state):
#     return {
#         "epochs": state.epochs,
#         "learning_rate": state.learning_rate
#     }


# def collect_training_data(state):
#     return {
#         "inputs": state.training_inputs,
#         "targets": state.training_targets
#     }


# def collect_initial_state(state):
#     return state.network_data


# def build_training_payload(state):
#     return {
#         "type": "INIT_NETWORK",
#         "network": collect_network_config(state),
#         "hyperparameters": collect_hyperparameters(state),
#         "training_data": collect_training_data(state),
#         "initial_state": collect_initial_state(state)
#     }