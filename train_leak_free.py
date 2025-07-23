"""
Leak-free training script for water level prediction models.

This script demonstrates the proper use of the new data pipeline to train
models without data leakage.
"""

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
import logging

from data_pipeline import TimeSeriesDataPipeline, create_time_series_cv_splits
from utils import (
    create_model, count_parameters, setup_training,
    train_single_epoch, evaluate_model, save_checkpoint,
    create_checkpoint_dir, plot_training_history, plot_error_analysis
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_validation(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_type: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    device: torch.device = torch.device('cpu'),
    early_stopping_patience: int = 15
):
    """
    Train model with proper validation set for early stopping.
    
    Uses validation set instead of test set for all model selection decisions.
    """
    # Setup training
    criterion, optimizer, scheduler = setup_training(model, lr)
    checkpoint_dir = create_checkpoint_dir(model_type)
    
    # Tracking
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    logger.info(f"Training {model_type} model:")
    logger.info(f"  Train samples: {len(X_train)}")
    logger.info(f"  Val samples: {len(X_val)}")
    logger.info(f"  Test samples: {len(X_test)}")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_single_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_mse = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model based on validation performance
            checkpoint_path, _ = save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss,
                train_losses, val_losses, checkpoint_dir, model_type
            )
            
            # Also evaluate on test set (but don't use for decisions)
            test_loss, test_mse = evaluate_model(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            
            logger.info(
                f"Epoch {epoch:3d} - Train: {train_loss:.6f}, "
                f"Val: {val_loss:.6f}, Test: {test_loss:.6f} [BEST]"
            )
        else:
            patience_counter += 1
            logger.info(
                f"Epoch {epoch:3d} - Train: {train_loss:.6f}, "
                f"Val: {val_loss:.6f} (patience: {patience_counter}/{early_stopping_patience})"
            )
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
    # Final test evaluation
    logger.info("\nFinal evaluation on test set:")
    test_loss, test_mse = evaluate_model(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.6f}, Test MSE: {test_mse:.6f}")
    
    return train_losses, val_losses, test_losses


def main():
    """Main training pipeline with leak-free data processing."""
    parser = argparse.ArgumentParser(description='Train water level prediction model')
    parser.add_argument('--data-file', type=str, default='data/1836026195.parquet',
                       help='Path to data file')
    parser.add_argument('--model-type', type=str, default='transformer',
                       choices=['transformer', 'lstm'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=720,
                       help='Sequence length')
    parser.add_argument('--pred-horizon', type=int, default=240,
                       help='Prediction horizon')
    parser.add_argument('--train-end', type=str, default='2024-12-01',
                       help='End date for training set')
    parser.add_argument('--val-end', type=str, default='2025-02-01',
                       help='End date for validation set')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize pipeline
    pipeline = TimeSeriesDataPipeline()
    
    # Load and process data
    logger.info("Loading data...")
    df = pipeline.load_raw_data(args.data_file)
    
    # Detect and fill gaps
    logger.info("Processing gaps...")
    gaps = pipeline.detect_gaps(df)
    df_filled, gap_mask = pipeline.fill_gaps(df)
    
    # Add temporal features
    logger.info("Adding temporal features...")
    df_features = pipeline.add_temporal_features(df_filled)
    
    # Chronological split
    logger.info("Creating chronological splits...")
    train_df, val_df, test_df = pipeline.chronological_split(
        df_features, args.train_end, args.val_end
    )
    
    # Define features to use
    feature_cols = ['value', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
                   'tide_sin', 'tide_cos']
    
    # Fit scalers on training data only
    logger.info("Fitting scalers on training data...")
    scalers = pipeline.fit_scalers(train_df, feature_cols)
    
    # Transform all sets
    train_scaled = pipeline.transform_features(train_df, feature_cols)
    val_scaled = pipeline.transform_features(val_df, feature_cols)
    test_scaled = pipeline.transform_features(test_df, feature_cols)
    
    # Create sequences with proper windowing
    logger.info("Creating sequences...")
    # Training: small overlap for more samples
    X_train, y_train = pipeline.create_sequences(
        train_scaled, args.seq_len, args.pred_horizon, 
        stride=args.seq_len // 4, features=feature_cols
    )
    
    # Validation: moderate overlap
    X_val, y_val = pipeline.create_sequences(
        val_scaled, args.seq_len, args.pred_horizon,
        stride=args.seq_len // 2, features=feature_cols
    )
    
    # Test: NO overlap to simulate production
    X_test, y_test = pipeline.create_sequences(
        test_scaled, args.seq_len, args.pred_horizon,
        stride=args.pred_horizon, features=feature_cols
    )
    
    # Run leak test
    logger.info("Running data leakage test...")
    no_leak = pipeline.run_leak_test(X_train, y_train, X_test, y_test)
    if not no_leak:
        logger.error("Data leakage detected! Aborting training.")
        return
        
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(
        args.model_type,
        input_size=len(feature_cols),
        output_size=args.pred_horizon,
        device=device,
        seq_len=args.seq_len
    ).to(device)
    
    param_stats = count_parameters(model)
    logger.info(f"Model parameters: {param_stats['total']:,}")
    
    # Train with validation
    logger.info("Starting training...")
    train_losses, val_losses, test_losses = train_with_validation(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        args.model_type, args.epochs, args.batch_size, args.lr, device
    )
    
    # Save pipeline for inference
    pipeline_path = os.path.join('data/models', f'pipeline_{args.model_type}.pkl')
    pipeline.save_pipeline(pipeline_path)
    logger.info(f"Saved pipeline to {pipeline_path}")
    
    # Plot results
    plot_training_history(train_losses, val_losses, evaluation_frequency=1)
    
    logger.info("Training completed successfully!")
    

if __name__ == "__main__":
    main()