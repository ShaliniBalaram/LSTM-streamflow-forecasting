import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class StreamflowDataset(Dataset):
    """
    Custom Dataset for loading streamflow forecasting data
    """
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def create_dataloaders(data_dict, batch_size=32):
    """
    Create DataLoaders from preprocessed data dictionary
    
    Args:
        data_dict: Dictionary containing preprocessed data (x_train, y_train, etc.)
        batch_size: Batch size for the DataLoaders
        
    Returns:
        Dictionary containing DataLoaders
    """
    train_dataset = StreamflowDataset(data_dict['x_train'], data_dict['y_train'])
    val_dataset = StreamflowDataset(data_dict['x_val'], data_dict['y_val'])
    test_dataset = StreamflowDataset(data_dict['x_test'], data_dict['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }

def load_processed_data(data_dir=None):
    """
    Load processed data from files
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    return {
        'x_train': x_train, 'y_train': y_train,
        'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test
    }

def plot_predictions(predictions, targets, output_dir=None, title='Streamflow Predictions'):
    """
    Plot model predictions against actual targets
    
    Args:
        predictions: Model predictions
        targets: Actual target values
        output_dir: Directory to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Streamflow')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'predictions.png'))
    
    plt.close()

def save_metrics(metrics, output_dir=None):
    """
    Save evaluation metrics to a file
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save the metrics
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean Squared Error (MSE): {metrics['mse']:.6f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}\n")
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}\n")
        f.write(f"R-squared (R²): {metrics['r2']:.6f}\n")
    
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.txt')}")
