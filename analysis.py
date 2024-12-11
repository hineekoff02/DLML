import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import ParameterGrid
import re

# Custom Dataset
class DarkMatterDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.data = []
        self.load_data()

    def load_data(self):
        def load_data(self, only_noise=True):
        pattern = r"output_(\d+)_(\d+)\.h5"
        for file in self.file_list:
            match = re.match(pattern, file)
            if match:
                # Check if the file should be processed based on the 'only_noise' flag
                if only_noise and 'noise' not in file:
                    continue
                if not only_noise and 'noise' in file:
                    continue

                size = int(match.group(1))
                seed = int(match.group(2))
                with h5py.File(file, 'r') as h5file:
                    energies = h5file['energies'][:]
                    times = h5file['times'][:]
                    traces_NR = h5file['traces_NR'][:]
                    traces_ER = h5file['traces_ER'][:]
                    for i in range(len(traces_NR)):
                        self.data.append((energies[i], times[i], traces_NR[i]))
                
                    for i in range(len(traces_ER)):
                        self.data.append((energies[i], times[i], traces_ER[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, position = self.data[idx]
        return torch.tensor(waveform, dtype=torch.float32), torch.tensor(position, dtype=torch.float32)

# Transformer Model for Temporal Encoding
class WaveformTransformer(nn.Module):
    def __init__(self, input_dim=54, embed_dim=128, num_heads=4, num_layers=3):
        super(WaveformTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers
        )
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embedding(x)  # Shape: [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, embed_dim]
        x = self.transformer(x, x)  # Self-attention
        x = x.permute(1, 0, 2)  # Back to [batch_size, seq_len, embed_dim]
        x = x.mean(dim=1)  # Global pooling
        x = self.fc(x)  # Shape: [batch_size, embed_dim]
        return x

# GNN for Spatial Understanding
class PositionGNN(nn.Module):
    def __init__(self, node_features=128, output_dim=3):
        super(PositionGNN, self).__init__()
        self.conv1 = nn.Linear(node_features, node_features)
        self.conv2 = nn.Linear(node_features, node_features)
        self.fc = nn.Linear(node_features, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.fc(x)
        return x

# Combined Model
class DarkMatterPositionModel(nn.Module):
    def __init__(self):
        super(DarkMatterPositionModel, self).__init__()
        self.transformer = WaveformTransformer()
        self.gnn = PositionGNN()

    def forward(self, x):
        transformer_output = self.transformer(x)
        position_output = self.gnn(transformer_output)
        return position_output

# Training Loop with Validation
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for waveforms, positions in train_loader:
            optimizer.zero_grad()
            predictions = model(waveforms)
            loss = criterion(predictions, positions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = validate_model(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss}")

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for waveforms, positions in val_loader:
            predictions = model(waveforms)
            loss = criterion(predictions, positions)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Hyperparameter Tuning
def hyperparameter_tuning(dataset, param_grid, val_split=0.2, epochs=10):
    best_params = None
    best_val_loss = float('inf')

    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        model = DarkMatterPositionModel()
        train_size = int(len(dataset) * (1 - val_split))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        train_model(model, train_loader, val_loader, epochs=params['epochs'], lr=params['lr'])
        val_loss = validate_model(model, val_loader, nn.MSELoss())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

    print(f"Best params: {best_params} with Validation Loss: {best_val_loss}")
    return best_params

# Example Usage
if __name__ == "__main__":
    # Replace with actual file paths
    file_list = ["output_50_1334465.h5", "output_noise_1056707.h5"]
    dataset = DarkMatterDataset(file_list)

    # Hyperparameter grid
    param_grid = {
        'batch_size': [16, 32],
        'lr': [1e-3, 1e-4],
        'epochs': [5, 10]
    }

    best_params = hyperparameter_tuning(dataset, param_grid)

    # Train final model with best params
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    final_model = DarkMatterPositionModel()
    train_model(final_model, train_loader, val_loader, epochs=best_params['epochs'], lr=best_params['lr'])
