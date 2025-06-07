# LSTM Streamflow Forecasting

## Author
- **Name:** Shalini Balaram
- **Email:** shalinib0204@gmail.com
- **Institute:** Indian Institute of Technology Madras (IITM)

## Project Overview
This project uses LSTM (Long Short-Term Memory) neural networks to forecast streamflow based on historical rainfall and streamflow data for the Abusu station.

## Project Structure

- `data/`: Contains rainfall and streamflow data for Abusu station and catchment attributes
  - `Abusu_rainfall.csv`: Historical rainfall data
  - `Abusu_streamflow.csv`: Historical streamflow data
  - `Catchments.Attributes.xlsx`: Catchment attributes
  - `processed/`: Directory for preprocessed data (created automatically)

- `src/`: Source code
  - `preprocess.py`: Data preprocessing utilities
  - `model.py`: LSTM model definition
  - `utils.py`: Utility functions
  - `main.py`: Main implementation of the forecasting pipeline

- `run.py`: Entry point script to run the forecasting pipeline
- `requirements.txt`: List of required Python packages
- `models/`: Directory for model checkpoints (created automatically)
- `results/`: Directory for evaluation results (created automatically)

## Getting Started

1. Install the required packages:
```
pip install -r requirements.txt
```

2. Run the forecasting pipeline with default settings:
```
python -m src.run
```

## How to Use the Script

### Basic Usage

The main script has been placed in the `src` directory and can be run using the Python module notation:

```bash
# Run with default settings (training and evaluation mode)
python -m src.run
```

The script will:
1. Load and preprocess the data from the `data` directory
2. Create a train/validation/test split of the data
3. Train an LSTM model on the training data
4. Evaluate the model on the test data
5. Save the model, plots, and metrics to timestamped directories

### Available Modes

The script supports two main modes of operation:

```bash
# Training and evaluation mode (default)
python -m src.run --mode train_eval

# Prediction mode (requires a trained model)
python -m src.run --mode predict --model_path models/run_YYYYMMDD_HHMMSS/best_model.pt
```

### Customizing Data Parameters

```bash
# Change the sequence length (default: 7)
python -m src.run --sequence_length 14

# Force reprocessing of data even if processed data exists
python -m src.run --reprocess
```

### Customizing Model Architecture

```bash
# Change the model architecture
python -m src.run --hidden_size 128 --num_layers 3 --dropout 0.3
```

### Customizing Training Parameters

```bash
# Modify training hyperparameters
python -m src.run --batch_size 64 --epochs 150 --learning_rate 0.0005 --patience 15
```

### Output and Reproducibility

```bash
# Change output directories
python -m src.run --output_dir my_results --model_dir my_models

# Set random seed for reproducibility
python -m src.run --seed 42
```

### Hardware Configuration

```bash
# Use GPU if available
python -m src.run --device cuda
```

### Getting Help

For a complete list of options, run:
```bash
python -m src.run --help
```

### Example Workflow

A typical workflow might look like this:

```bash
# 1. Train a model with custom parameters
python -m src.run --hidden_size 128 --sequence_length 14 --epochs 200 --learning_rate 0.0005 --seed 42

# 2. Use the trained model to make predictions
python -m src.run --mode predict --model_path models/run_YYYYMMDD_HHMMSS/best_model.pt
```

Replace `YYYYMMDD_HHMMSS` with the actual timestamp of your training run.
