"""
Demonstration of data leakage impact on model performance.

This script compares the old (leaky) pipeline with the new (leak-free) pipeline
to show the dramatic difference in reported vs actual performance.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from data_pipeline import TimeSeriesDataPipeline
from utils import prepare_data_for_model, create_model, count_parameters


def evaluate_old_pipeline(df, seq_len=720, pred_horizon=240, test_size=0.2):
    """Evaluate using the old (leaky) pipeline."""
    print("\n" + "="*60)
    print("EVALUATING OLD (LEAKY) PIPELINE")
    print("="*60)
    
    # Old approach: prepare_data_for_model with all its issues
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_model(
        df, seq_len=seq_len, test_size=test_size, 
        prediction_horizon=pred_horizon
    )
    
    print(f"\nOld pipeline statistics:")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Overlap ratio: {seq_len/pred_horizon:.1f}x (sequences are {(seq_len-1)/seq_len*100:.1f}% identical)")
    
    # Quick model evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create simple LSTM for testing
    model = create_model('lstm', input_size=1, output_size=pred_horizon, 
                        device=device, seq_len=seq_len).to(device)
    
    # Train for a few epochs
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Use small batch for demo
        batch_size = 32
        idx = np.random.choice(len(X_train), batch_size)
        batch_X = X_train[idx].to(device)
        batch_y = y_train[idx].to(device)
        
        outputs = model(batch_X)
        if batch_y.dim() == 3:
            batch_y = batch_y.squeeze(-1)
            
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        # Test on subset for speed
        test_pred = model(X_test[:100].to(device)).cpu().numpy()
        test_true = y_test[:100].numpy()
        
        if test_true.ndim == 3:
            test_true = test_true.squeeze(-1)
            
    # Metrics on first prediction step
    mae = mean_absolute_error(test_true[:, 0], test_pred[:, 0])
    mse = mean_squared_error(test_true[:, 0], test_pred[:, 0])
    r2 = r2_score(test_true[:, 0], test_pred[:, 0])
    
    print(f"\nOLD PIPELINE RESULTS (appear great due to leakage):")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mae, mse, r2, test_pred, test_true


def evaluate_new_pipeline(df, seq_len=720, pred_horizon=240):
    """Evaluate using the new (leak-free) pipeline."""
    print("\n" + "="*60)
    print("EVALUATING NEW (LEAK-FREE) PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = TimeSeriesDataPipeline()
    
    # Process data properly
    gaps = pipeline.detect_gaps(df)
    df_filled, gap_mask = pipeline.fill_gaps(df)
    df_features = pipeline.add_temporal_features(df_filled)
    
    # Proper chronological split
    train_end = pd.to_datetime('2024-12-01')
    val_end = pd.to_datetime('2025-02-01')
    
    train_df, val_df, test_df = pipeline.chronological_split(
        df_features, train_end.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d')
    )
    
    # Scale properly
    feature_cols = ['value']
    scalers = pipeline.fit_scalers(train_df, feature_cols)
    train_scaled = pipeline.transform_features(train_df, feature_cols)
    test_scaled = pipeline.transform_features(test_df, feature_cols)
    
    # Create sequences with NO OVERLAP in test
    X_train, y_train = pipeline.create_sequences(
        train_scaled, seq_len, pred_horizon, stride=seq_len//4
    )
    X_test, y_test = pipeline.create_sequences(
        test_scaled, seq_len, pred_horizon, stride=pred_horizon  # No overlap!
    )
    
    print(f"\nNew pipeline statistics:")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)} (non-overlapping)")
    print(f"Train period: {train_df['collect_time'].min()} to {train_df['collect_time'].max()}")
    print(f"Test period: {test_df['collect_time'].min()} to {test_df['collect_time'].max()}")
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Same model architecture
    model = create_model('lstm', input_size=1, output_size=pred_horizon,
                        device=device, seq_len=seq_len).to(device)
    
    # Train
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        batch_size = min(32, len(X_train))
        idx = np.random.choice(len(X_train), batch_size)
        batch_X = X_train[idx]
        batch_y = y_train[idx]
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).cpu().numpy()
        test_true = y_test.cpu().numpy()
    
    # Metrics
    mae = mean_absolute_error(test_true[:, 0], test_pred[:, 0])
    mse = mean_squared_error(test_true[:, 0], test_pred[:, 0])
    r2 = r2_score(test_true[:, 0], test_pred[:, 0])
    
    print(f"\nNEW PIPELINE RESULTS (realistic performance):")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return mae, mse, r2, test_pred, test_true, scalers['value']


def plot_comparison(old_metrics, new_metrics):
    """Plot comparison of old vs new pipeline metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['MAE', 'MSE', 'R²']
    old_values = old_metrics
    new_values = new_metrics
    
    x = np.arange(len(metrics))
    width = 0.35
    
    for i, (ax, metric, old_val, new_val) in enumerate(
        zip(axes, metrics, old_values, new_values)
    ):
        bars1 = ax.bar(x - width/2, [old_val], width, label='Old (Leaky)', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, [new_val], width, label='New (Leak-free)', color='green', alpha=0.7)
        
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks([0])
        ax.set_xticklabels([metric])
        ax.legend()
        
        # Add value labels
        ax.bar_label(bars1, fmt='%.4f')
        ax.bar_label(bars2, fmt='%.4f')
        
        # Add percentage change
        if metric != 'R²':
            pct_change = ((new_val - old_val) / old_val) * 100
            ax.text(0, max(old_val, new_val) * 1.1, 
                   f'{pct_change:+.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('Data Leakage Impact: Old vs New Pipeline', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    """Run the comparison demo."""
    print("WATER LEVEL PREDICTION: DATA LEAKAGE DEMONSTRATION")
    print("=" * 60)
    print("This demo shows how data leakage creates falsely optimistic results")
    print("and why proper time series splitting is critical.\n")
    
    # Load sample data
    try:
        df = pd.read_parquet('data/1836026195.parquet')
        print(f"Loaded data: {len(df)} samples")
        print(f"Date range: {df['collect_time'].min()} to {df['collect_time'].max()}")
    except FileNotFoundError:
        print("Error: Data file not found. Please ensure data/1836026195.parquet exists.")
        return
    
    # Run both pipelines
    old_mae, old_mse, old_r2, _, _ = evaluate_old_pipeline(df)
    new_mae, new_mse, new_r2, _, _, _ = evaluate_new_pipeline(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: The Impact of Data Leakage")
    print("="*60)
    print(f"\nOld Pipeline (with leakage):")
    print(f"  - Overlapping windows in train/test")
    print(f"  - Scaler fitted on entire dataset")
    print(f"  - Random train/test split")
    print(f"  → Artificially low error: MAE = {old_mae:.4f}")
    
    print(f"\nNew Pipeline (leak-free):")
    print(f"  - Non-overlapping test windows")
    print(f"  - Scaler fitted on training only")
    print(f"  - Chronological split with gaps")
    print(f"  → Realistic error: MAE = {new_mae:.4f}")
    
    print(f"\nDifference: {((new_mae - old_mae) / old_mae * 100):.1f}% higher (realistic) error")
    print("\nThis demonstrates why the old pipeline would fail in production!")
    
    # Plot comparison
    plot_comparison([old_mae, old_mse, old_r2], [new_mae, new_mse, new_r2])


if __name__ == "__main__":
    main()