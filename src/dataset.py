import torch
import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path)
    velocities = data[:, :2]
    positions = data[:, 2:]
    return velocities, positions

def create_pairs(velocities, positions, seq_length=1):
    input_states = np.hstack((velocities[:-1], positions[:-1]))
    output_states = np.hstack((velocities[1:], positions[1:]))
    return input_states, output_states

def get_dataloaders(batch_size, train_path, test_path, seq_length=16):
    train_velocities, train_positions = load_data(train_path)
    test_velocities, test_positions = load_data(test_path)
    
    train_input_sequences, train_output_sequences = create_pairs(train_velocities, train_positions, seq_length)
    test_input_sequences, test_output_sequences = create_pairs(test_velocities, test_positions, seq_length)
    
    train_input_sequences = torch.tensor(train_input_sequences, dtype=torch.float32)
    train_output_sequences = torch.tensor(train_output_sequences, dtype=torch.float32)
    test_input_sequences = torch.tensor(test_input_sequences, dtype=torch.float32)
    test_output_sequences = torch.tensor(test_output_sequences, dtype=torch.float32)
    
    train_dataset = torch.utils.data.TensorDataset(train_input_sequences, train_output_sequences)
    test_dataset = torch.utils.data.TensorDataset(test_input_sequences, test_output_sequences)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_pairs_LSTM(velocities, positions, seq_length=16):
    input_sequences = []
    output_sequences = []
        
    for i in range(len(velocities) - seq_length):
        input_seq = np.hstack((velocities[i:i+seq_length], positions[i:i+seq_length]))
        output_seq = np.hstack((velocities[i+1:i+seq_length+1], positions[i+1:i+seq_length+1]))
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    
    return np.array(input_sequences), np.array(output_sequences)

def get_dataloaders_LSTM(batch_size, train_path, test_path, seq_length=16):
    train_velocities, train_positions = load_data(train_path)
    test_velocities, test_positions = load_data(test_path)
    
    train_input_sequences, train_output_sequences = create_pairs_LSTM(train_velocities, train_positions, seq_length)
    test_input_sequences, test_output_sequences = create_pairs_LSTM(test_velocities, test_positions, seq_length)
    
    train_input_sequences = torch.tensor(train_input_sequences, dtype=torch.float32)
    train_output_sequences = torch.tensor(train_output_sequences, dtype=torch.float32)
    test_input_sequences = torch.tensor(test_input_sequences, dtype=torch.float32)
    test_output_sequences = torch.tensor(test_output_sequences, dtype=torch.float32)
    
    train_dataset = torch.utils.data.TensorDataset(train_input_sequences, train_output_sequences)
    test_dataset = torch.utils.data.TensorDataset(test_input_sequences, test_output_sequences)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

