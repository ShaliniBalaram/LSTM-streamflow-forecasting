import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTMModel(nn.Module):
    """
    LSTM model for streamflow forecasting
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, device='cpu', 
                patience=10, checkpoint_dir=None):
    """
    Train the LSTM model
    
    Args:
        model: The LSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to use for training ('cpu' or 'cuda')
        patience: Number of epochs to wait for improvement before early stopping
        checkpoint_dir: Directory to save model checkpoints
        
    Returns:
        Trained model and training history
    """
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Create checkpoint directory if it doesn't exist
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            
            # Save the best model
            if checkpoint_dir is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    if checkpoint_dir is not None:
        plt.savefig(os.path.join(checkpoint_dir, 'training_history.png'))
    
    return model, history

def evaluate_model(model, test_loader, device='cpu', target_scaler=None):
    """
    Evaluate the trained model on the test set
    
    Args:
        model: Trained LSTM model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        target_scaler: Scaler to inverse transform predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Inverse transform if scaler is provided
    if target_scaler is not None:
        all_preds = target_scaler.inverse_transform(all_preds)
        all_targets = target_scaler.inverse_transform(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results
