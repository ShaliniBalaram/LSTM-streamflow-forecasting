import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data():
    """
    Load rainfall and streamflow data from CSV files
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    rainfall_path = os.path.join(data_dir, 'Abusu_rainfall.csv')
    streamflow_path = os.path.join(data_dir, 'Abusu_streamflow.csv')
    
    rainfall_data = pd.read_csv(rainfall_path, parse_dates=['date'])
    streamflow_data = pd.read_csv(streamflow_path, parse_dates=['date'])
    
    # Merge datasets on date
    merged_data = pd.merge(rainfall_data, streamflow_data, on='date', suffixes=('_rainfall', '_streamflow'))
    
    return merged_data

def preprocess_data(data, sequence_length=7, train_ratio=0.7, val_ratio=0.15):
    """
    Preprocess data for LSTM model:
    1. Create sequences of input data and target values
    2. Scale data
    3. Split into training, validation, and test sets
    
    Args:
        data: Merged dataframe containing rainfall and streamflow data
        sequence_length: Number of time steps to use for input sequences
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Scaled data and scalers for inverse transformation
    """
    # Extract features and target
    features = data[['rainfall']].values
    target = data[['streamflow']].values
    
    # Scale features and target
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    # Create sequences
    x, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        x.append(scaled_features[i:i+sequence_length])
        y.append(scaled_target[i+sequence_length])
    
    x, y = np.array(x), np.array(y)
    
    # Split into training, validation, and test sets
    train_size = int(len(x) * train_ratio)
    val_size = int(len(x) * val_ratio)
    
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[train_size+val_size:], y[train_size+val_size:]
    
    return {
        'x_train': x_train, 'y_train': y_train,
        'x_val': x_val, 'y_val': y_val,
        'x_test': x_test, 'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }

def save_processed_data(processed_data, output_dir=None):
    """
    Save processed data to files for later use
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, 'x_train.npy'), processed_data['x_train'])
    np.save(os.path.join(output_dir, 'y_train.npy'), processed_data['y_train'])
    np.save(os.path.join(output_dir, 'x_val.npy'), processed_data['x_val'])
    np.save(os.path.join(output_dir, 'y_val.npy'), processed_data['y_val'])
    np.save(os.path.join(output_dir, 'x_test.npy'), processed_data['x_test'])
    np.save(os.path.join(output_dir, 'y_test.npy'), processed_data['y_test'])
    
    # We cannot directly save the scalers as numpy arrays
    # They will be saved in the model checkpoint or pickle files later
    
    print(f"Processed data saved to {output_dir}")

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    
    print("Preprocessing data...")
    processed_data = preprocess_data(data)
    
    print("Saving processed data...")
    save_processed_data(processed_data)
    
    print("Done!")
