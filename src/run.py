#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point script for LSTM streamflow forecasting application.
This script simply imports the pipeline from the src module and runs it.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main functionality from src.main
from src.main import run_pipeline

if __name__ == "__main__":
    # Run the pipeline with command line arguments
    run_pipeline()
