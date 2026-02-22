def validate_network_structure(state):
    if state.input_nodes <= 0:
        return False, "Input layer must have at least 1 neuron"

    if state.output_nodes <= 0:
        return False, "Output layer must have at least 1 neuron"

    if any(layer <= 0 for layer in state.layers):
        return False, "Hidden layers must have at least 1 neuron"

    return True, None


def validate_training_data(state):
    if len(state.training_inputs) != state.input_nodes:
        return False, "Training inputs must match input size"

    if len(state.training_targets) != state.output_nodes:
        return False, "Targets must match output size"

    return True, None


def validate_full_config(state):
    checks = [
        validate_network_structure(state),
        validate_training_data(state)
    ]

    for valid, message in checks:
        if not valid:
            return False, message

    return True, None