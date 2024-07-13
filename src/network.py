import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RecurrentNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=4, num_layers=4):
        super(RecurrentNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_states):
        # Initial hidden state
        h_0 = torch.zeros(self.num_layers, input_states.size(0), self.hidden_size)
        out, _ = self.rnn(input_states, h_0)
        out = self.fc(out)
        return out

