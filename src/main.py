#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for LSTM-based streamflow forecasting.
This script orchestrates the entire process of data preprocessing,
model training, evaluation, and result visualization.
"""

import os
import argparse
import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Import modules from src directory
from preprocess import load_data, preprocess_data, save_processed_data
from model import LSTMModel, train_model, evaluate_model
from utils import create_dataloaders, load_processed_data, plot_predictions, save_metrics

def parse_arguments():
    """
    Parse command line arguments for the streamflow forecasting pipeline.
    """
    parser = argparse.ArgumentParser(description='LSTM Streamflow Forecasting')
    
    # Mode parameters
    parser.add_argument('--mode', type=str, default='train_eval',
                        choices=['train_eval', 'predict'],
                        help='Mode of operation: train_eval (train and evaluate) or predict (make predictions only)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the input data')
    parser.add_argument('--processed_data_dir', type=str, default='data/processed',
                        help='Directory to save/load processed data')
    parser.add_argument('--reprocess', action='store_true',
                        help='Reprocess data even if processed data exists')
    parser.add_argument('--sequence_length', type=int, default=7,
                        help='Length of input sequences (days)')
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden units in LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--no_train', action='store_true',
                        help='Skip training and load saved model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                        help='Path to saved model checkpoint to load')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation (cuda or cpu)')
    
    return parser.parse_args()

def prepare_directories(args):
    """
    Create necessary directories for output files.
    """
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

def prepare_data(args):
    """
    Load and preprocess data, or load already processed data.
    """
    processed_data_exists = os.path.exists(os.path.join(args.processed_data_dir, 'x_train.npy'))
    
    if processed_data_exists and not args.reprocess:
        print("Loading preprocessed data...")
        data_dict = load_processed_data(args.processed_data_dir)
        
        # We need to recreate the scalers
        # Load original data to fit the scalers (not ideal but necessary)
        print("Recreating scalers...")
        raw_data = load_data()
        features = raw_data[['rainfall']].values
        target = raw_data[['streamflow']].values
        
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        feature_scaler.fit(features)
        target_scaler.fit(target)
        
        data_dict['feature_scaler'] = feature_scaler
        data_dict['target_scaler'] = target_scaler
    else:
        print("Loading and preprocessing data...")
        raw_data = load_data()
        data_dict = preprocess_data(raw_data, sequence_length=args.sequence_length)
        
        # Save processed data
        print("Saving processed data...")
        save_processed_data(data_dict, args.processed_data_dir)
    
    return data_dict

def load_best_model(model_path, model_class, device, hidden_size=64, num_layers=2, dropout=0.2):
    """
    Load the best model checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        model_class: The model class to instantiate
        device: Device to load the model on (cpu or cuda)
        hidden_size: Hidden size for the LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        
    Returns:
        Loaded model instance
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize the model with the same architecture
    model = model_class(
        input_size=1,  # Will be overridden by the loaded state dict
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def create_or_load_model(args, input_size, output_size):
    """
    Create a new model or load a saved model.
    """
    # Load model if specified
    if args.no_train and args.model_path:
        print(f"Loading saved model from {args.model_path}...")
        model = load_best_model(
            model_path=args.model_path,
            model_class=LSTMModel,
            device=args.device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        # Create new model
        model = LSTMModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=output_size,
            dropout=args.dropout
        )
    
    return model

def save_scalers(scalers, output_dir):
    """
    Save the feature and target scalers using pickle.
    """
    feature_scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    target_scaler_path = os.path.join(output_dir, 'target_scaler.pkl')
    
    with open(feature_scaler_path, 'wb') as f:
        pickle.dump(scalers['feature_scaler'], f)
    
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(scalers['target_scaler'], f)
    
    print(f"Scalers saved to {output_dir}")

def predict_streamflow(args, data_dict, dataloaders, model):
    """
    Make predictions using a trained model.
    """
    print("\nMaking predictions with trained model...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predict_dir = os.path.join(args.output_dir, f"predictions_{timestamp}")
    os.makedirs(predict_dir, exist_ok=True)
    
    # Make predictions on test set
    with torch.no_grad():
        for inputs, targets in dataloaders['test_loader']:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            
            # Store predictions and targets
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)
    
    # Inverse transform predictions and targets
    test_predictions = data_dict['target_scaler'].inverse_transform(test_predictions)
    test_targets = data_dict['target_scaler'].inverse_transform(test_targets)
    
    # Plot and save predictions
    plot_predictions(
        predictions=test_predictions.flatten(),
        targets=test_targets.flatten(),
        output_dir=predict_dir,
        title='LSTM Streamflow Predictions'
    )
    
    # Calculate metrics
    mse = mean_squared_error(test_targets, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': test_predictions,
        'targets': test_targets
    }
    
    # Print metrics
    print("\nPrediction Metrics:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R²: {metrics['r2']:.6f}")
    
    # Save metrics
    save_metrics(metrics, predict_dir)
    
    print(f"\nPrediction results saved to {predict_dir}")

def run_pipeline(args=None):
    """
    Main function to run the complete streamflow forecasting pipeline.
    """
    # Parse arguments if not provided
    if args is None:
        args = parse_arguments()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Check if CUDA is available if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Create directories
    prepare_directories(args)
    
    # Prepare data
    data_dict = prepare_data(args)
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(data_dict, batch_size=args.batch_size)
    
    # Define model dimensions
    input_size = data_dict['x_train'].shape[2]  # Number of features
    output_size = data_dict['y_train'].shape[1]  # Number of output variables
        
    # Handle prediction mode
    if args.mode == 'predict':
        # Load model
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model file not found: {args.model_path}")
            
        model = load_best_model(
            model_path=args.model_path,
            model_class=LSTMModel,
            device=args.device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        
        # Make predictions
        predict_streamflow(args, data_dict, dataloaders, model)
        return
    
    # Training and evaluation mode
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set model checkpoint directory
    model_checkpoint_dir = os.path.join(args.model_dir, f"run_{timestamp}")
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    # Create or load model
    model = create_or_load_model(args, input_size, output_size)
    
    # Save scalers for later use
    save_scalers({
        'feature_scaler': data_dict['feature_scaler'],
        'target_scaler': data_dict['target_scaler']
    }, model_checkpoint_dir)
    
    # Train model if not skipping training
    if not args.no_train:
        print("\nTraining model...")
        trained_model, history = train_model(
            model=model,
            train_loader=dataloaders['train_loader'],
            val_loader=dataloaders['val_loader'],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            patience=args.patience,
            checkpoint_dir=model_checkpoint_dir
        )
        
        # Save training history plot
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'training_history.png'))
        plt.close()
    else:
        trained_model = model
    
    # Evaluate model on test data
    print("\nEvaluating model...")
    eval_results = evaluate_model(
        model=trained_model,
        test_loader=dataloaders['test_loader'],
        device=args.device,
        target_scaler=data_dict['target_scaler']
    )
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"MSE: {eval_results['mse']:.6f}")
    print(f"RMSE: {eval_results['rmse']:.6f}")
    print(f"MAE: {eval_results['mae']:.6f}")
    print(f"R²: {eval_results['r2']:.6f}")
    
    # Save evaluation metrics
    save_metrics(eval_results, run_dir)
    
    # Plot and save predictions
    print("\nPlotting predictions...")
    plot_predictions(
        predictions=eval_results['predictions'].flatten(),
        targets=eval_results['targets'].flatten(),
        output_dir=run_dir,
        title='Streamflow Forecast vs Actual'
    )
    
    print(f"\nResults saved to {run_dir}")
    print("Streamflow forecasting completed successfully!")

if __name__ == "__main__":
    run_pipeline()
