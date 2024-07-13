import torch
import numpy as np

# Load data from text file
def load_data(file_path):
    data = np.loadtxt(file_path)
    velocities = data[:, :2]
    positions = data[:, 2:]
    return velocities, positions

def get_dataloaders(batch_size, train_path, test_path):
    train_velocities, train_positions = load_data(train_path)
    test_velocities, test_positions = load_data(test_path)

    def create_pairs(velocities, positions):
        input_states = np.hstack((velocities[:-1], positions[:-1]))
        output_states = np.hstack((velocities[1:], positions[1:]))
        return input_states, output_states

    train_input_states, train_output_states = create_pairs(train_velocities, train_positions)
    test_input_states, test_output_states = create_pairs(test_velocities, test_positions)

    # Convert data to PyTorch tensors
    train_input_states = torch.tensor(train_input_states, dtype=torch.float32)
    train_output_states = torch.tensor(train_output_states, dtype=torch.float32)
    test_input_states = torch.tensor(test_input_states, dtype=torch.float32)
    test_output_states = torch.tensor(test_output_states, dtype=torch.float32)

    # Create dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(train_input_states, train_output_states)
    test_dataset = torch.utils.data.TensorDataset(test_input_states, test_output_states)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

