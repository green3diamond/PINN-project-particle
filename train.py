import torch
import torch.nn as nn
from src.network import FeedForwardNN
from torch import optim
from src.dataset import get_dataloaders

from tqdm.auto import tqdm

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    progress_bar = tqdm(range(num_epochs), desc='Training Progress', leave=True)
    for epoch in progress_bar:
        running_loss = 0.0
        for input_states, output_states in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_states)
            loss = criterion(outputs, output_states)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * input_states.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')



def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_states, output_states in test_loader:
            outputs = model(input_states)
            loss = criterion(outputs, output_states)
            total_loss += loss.item() * input_states.size(0)
    
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss}')


if __name__ == '__main__':
    model = FeedForwardNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, test_loader = get_dataloaders(64, "data_lorentz/train.txt", "data_lorentz/test.txt", 1)
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    evaluate_model(model, test_loader, criterion)