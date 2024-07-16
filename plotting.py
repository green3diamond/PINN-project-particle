import torch
import numpy as np
import matplotlib.pyplot as plt

# Visualization functions
def plot_positions(positions, title):
    x = positions[:, 0]
    y = positions[:, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.grid(True)
    plt.show()

def plot_predicted_trajectories(model, initial_state, true_positions, title, axisPlt, steps=300):
    model.eval()
    predicted_positions = []
    state = initial_state.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        for _ in range(steps):
            predicted_state = model(state)
            predicted_positions.append(predicted_state[:, 2:].numpy())
            state = predicted_state  # Use the output as the next input

    predicted_positions = np.concatenate(predicted_positions, axis=0)

    axisPlt.plot(true_positions[:,0], true_positions[:, 1], label='True Trajectory', marker='o')
    axisPlt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted Trajectory', marker='x')
    axisPlt.set_title(title)
    axisPlt.set_xlabel('x position')
    axisPlt.set_ylabel('y position')
    axisPlt.legend()
    axisPlt.grid(True)

    return predicted_positions

def plot_predicted_trajectories_LSTM(model, initial_state, true_positions, title, axisPlt, steps=300):
    model.eval()
    predicted_positions = []
    state = initial_state.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        for _ in range(steps):
            predicted_state = model(state)
            predicted_positions.append(predicted_state[:,0, 2:].numpy())
            state = predicted_state  # Use the output as the next input

    predicted_positions = np.concatenate(predicted_positions, axis=0)

    axisPlt.plot(true_positions[:,0], true_positions[:, 1], label='True Trajectory', marker='o')
    axisPlt.plot(predicted_positions[:, 0], predicted_positions[:, 1], label='Predicted Trajectory', marker='x')
    axisPlt.set_title(title)
    axisPlt.set_xlabel('x position')
    axisPlt.set_ylabel('y position')
    axisPlt.legend()
    axisPlt.grid(True)

    return predicted_positions

# Prepare input data for plotting predicted trajectories
def extract_initial_state_and_true_positions(dataloader, steps=300):
    true_positions=torch.empty((0, 2))
    for input_sequences, output_sequences in dataloader:
        initial_state = input_sequences[0]
        break
    for input_sequences, output_sequences in dataloader:
        true_positions = torch.cat((true_positions, output_sequences[:, 2:]))
    return initial_state, true_positions[:steps,:]

def extract_initial_state_and_true_positions_LSTM(dataloader, steps=300):
    true_positions=torch.empty((0, 2))
    for input_sequences, output_sequences in dataloader:
        initial_state = input_sequences[0]
        break
    for input_sequences, output_sequences in dataloader:
        true_positions = torch.cat((true_positions, output_sequences[:,0, 2:]))
    return initial_state, true_positions[:steps,:]
