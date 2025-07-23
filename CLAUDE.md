# CLAUDE.md - Deep Learning Water Level Prediction Project

## Project Overview
This is a deep learning project for water level prediction using time series data from hydrological sensors. The codebase implements both Transformer and LSTM models for multi-step water level forecasting (240 steps/24 hours ahead).

**CRITICAL**: This codebase has undergone major refactoring to fix severe data leakage issues. Always use the leak-free pipeline (`data_pipeline.py` and `train_leak_free.py`) instead of the legacy functions in `utils.py`.

## Architecture

### Core Components
1. **Models** (`models/`)
   - `transformer.py`: Transformer model with encoder-only optimization
   - `lstm.py`: LSTM model with bidirectional support

2. **Data Pipeline** (`data_pipeline.py`)
   - Leak-free time series data processing
   - Chronological train/val/test splitting
   - Gap detection and interpolation
   - Temporal feature engineering
   - Non-overlapping test sequences

3. **Training Scripts**
   - `train_leak_free.py`: Production-ready training script
   - `training.ipynb`: Jupyter notebook for experimentation (contains legacy code)

4. **Utilities** (`utils.py`)
   - Model creation and parameter counting
   - Training/evaluation loops
   - Checkpoint management
   - **WARNING**: `prepare_data_for_model()` has data leakage - use `data_pipeline.py` instead

5. **Testing** (`test_data_integrity.py`)
   - Comprehensive test suite for data pipeline integrity
   - Leak detection tests
   - Production readiness checks

## Critical Issues Fixed

### Data Leakage (Previously Present)
1. **Scaler Leakage**: MinMaxScaler was fit on entire dataset before splitting
2. **Overlapping Windows**: Test sequences had 99.86% overlap (stride=1 vs pred_horizon=240)
3. **Random Splitting**: Used random train/test split instead of chronological
4. **Validation Set Misuse**: Used test set for early stopping

### Solutions Implemented
1. **Proper Splitting**: Chronological split with buffer zones
2. **Correct Scaling**: Fit scalers only on training data
3. **Non-overlapping Test**: Test sequences use stride=prediction_horizon
4. **Three-way Split**: Separate train/val/test sets

## Common Commands

### Training
```bash
# Train transformer model (recommended)
python train_leak_free.py --model-type transformer --epochs 100 --batch-size 32

# Train LSTM model
python train_leak_free.py --model-type lstm --epochs 100 --batch-size 64

# Custom data splits
python train_leak_free.py --train-end 2024-12-01 --val-end 2025-02-01
```

### Testing
```bash
# Run all data integrity tests
python test_data_integrity.py

# Compare old vs new pipeline (demonstrates leakage impact)
python leak_comparison_demo.py
```

### Linting and Type Checking
```bash
# If using Python linting (add to project if needed)
# pip install ruff
# ruff check .

# Type checking (if mypy is added)
# pip install mypy
# mypy .
```

## Data Format

### Input Data
- **Format**: Parquet files in `data/` directory
- **Columns**: 
  - `collect_time`: Timestamp
  - `value`: Water level measurement
  - `sensor_id`: Sensor identifier
- **Frequency**: 6-minute intervals (may have gaps)

### Processed Features
- **Temporal Features**:
  - `hour_sin`, `hour_cos`: Cyclical hour encoding
  - `dow_sin`, `dow_cos`: Cyclical day-of-week encoding
  - `tide_sin`, `tide_cos`: Tidal cycle encoding (12.42 hours)
  - `time_since_gap`: Time since last data gap
  - `gap_filled`: Boolean indicator for interpolated values

## Model Configurations

### Transformer (Recommended)
```python
{
    'hidden_dim': 128,
    'num_heads': 8,
    'dim_feedforward': 2048,
    'num_layers_enc': 6,
    'dropout': 0.1,
    'encoder_only': True  # 50% memory reduction
}
```

### LSTM
```python
{
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'bidirectional': False
}
```

## Key Parameters
- **Sequence Length**: 720 (72 hours of history)
- **Prediction Horizon**: 240 (24 hours ahead)
- **Train Stride**: seq_len // 4 (75% overlap for more samples)
- **Test Stride**: pred_horizon (0% overlap for realistic evaluation)

## Directory Structure
```
.
├── data/
│   ├── raw/           # Original parquet files
│   ├── processed/     # Processed datasets
│   └── models/        # Saved models and pipelines
├── models/
│   ├── transformer.py
│   └── lstm.py
├── model_checkpoints/
│   └── transformer/   # Saved checkpoints
├── data_pipeline.py   # Main data processing pipeline
├── train_leak_free.py # Production training script
├── utils.py          # Utility functions (use cautiously)
├── test_data_integrity.py # Test suite
└── requirements.txt  # Dependencies
```

## Best Practices

### Data Processing
1. **Always use chronological splitting** for time series
2. **Fit scalers only on training data**
3. **Use non-overlapping windows for test set**
4. **Handle gaps before creating sequences**
5. **Add temporal features** for better predictions

### Model Training
1. **Use validation set for early stopping** (not test set)
2. **Monitor for overfitting** with train/val curves
3. **Save best model based on validation loss**
4. **Use learning rate scheduling**

### Production Deployment
1. **Save the entire pipeline** (including scalers)
2. **Test for data leakage** before deployment
3. **Validate on truly future data**
4. **Monitor prediction drift** in production

## Common Pitfalls to Avoid
1. **Don't use test set for any training decisions**
2. **Don't create overlapping test sequences**
3. **Don't fit preprocessors on full dataset**
4. **Don't ignore data gaps** - they need proper handling
5. **Don't use the old `prepare_data_for_model()` function**

## Debugging Tips
1. **Check sequence shapes**: Ensure (batch, seq_len, features)
2. **Verify no data leakage**: Run `test_data_integrity.py`
3. **Monitor GPU memory**: Transformer uses ~50% less in encoder-only mode
4. **Check temporal alignment**: Test sequences shouldn't overlap with training period

## Future Improvements
1. Add weather data as external features
2. Implement attention visualization
3. Add uncertainty quantification
4. Experiment with ensemble methods
5. Add real-time inference pipeline

## Contact
For questions about the data pipeline refactoring and leak prevention, refer to the extensive comments in `data_pipeline.py` and the test cases in `test_data_integrity.py`.