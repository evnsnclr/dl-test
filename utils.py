"""
Water Level Prediction Utilities

This module contains utility functions for training and evaluating water level prediction models
using both Transformer and LSTM architectures.

Modules:
    - Data preparation and preprocessing
    - Model creation and configuration
    - Training and evaluation functions
    - Checkpoint management
    - Visualization utilities
"""

import os
import glob
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats

# Import model classes
from models.transformer import TransformerWaterLevelPrediction
from models.lstm import LSTMFloodDetection


# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================

DEFAULT_MODEL_CONFIGS = {
    'transformer': {
        'class': TransformerWaterLevelPrediction,
        'params': {
            'hidden_dim': 128,
            'num_heads': 8,
            'dim_feedforward': 512,
            'num_layers_enc': 4,
            'num_layers_dec': 4,
            'dropout': 0.1,
        }
    },
    'lstm': {
        'class': LSTMFloodDetection,
        'params': {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
            'bidirectional': True
        }
    }
}


# =====================================================================
# DATA PREPARATION
# =====================================================================

def prepare_data_for_model(
    df: pd.DataFrame, 
    seq_len: int = 60, 
    test_size: float = 0.2, 
    prediction_horizon: int = 1,
    time_column: str = 'collect_time',
    value_column: str = 'value',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Prepare pandas DataFrame for time series prediction models.
    
    Args:
        df: DataFrame with time series data
        seq_len: Sequence length for model input
        test_size: Fraction of data for testing
        prediction_horizon: Number of future steps to predict
        time_column: Name of time column for sorting
        value_column: Name of value column to predict
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Validate inputs
    if len(df) < seq_len + prediction_horizon:
        raise ValueError(f"Data length ({len(df)}) must be at least seq_len + prediction_horizon ({seq_len + prediction_horizon})")
    
    # Handle time column - support both 'seconds' and 'collect_time'
    if time_column not in df.columns:
        if 'collect_time' in df.columns:
            time_column = 'collect_time'
        elif 'seconds' in df.columns:
            time_column = 'seconds'
        else:
            raise KeyError(f"Neither '{time_column}' nor 'collect_time' nor 'seconds' found in DataFrame columns: {list(df.columns)}")
    
    # Sort by time
    df = df.sort_values(time_column).reset_index(drop=True)
    
    # Extract values and check for NaN
    values = df[value_column].values.reshape(-1, 1)
    if np.isnan(values).any():
        print(f"Warning: Found {np.isnan(values).sum()} NaN values in {value_column}")
        # Remove NaN values
        values = values[~np.isnan(values).flatten()].reshape(-1, 1)
    
    # Split data FIRST to avoid data leakage
    split_idx = int(len(values) * (1 - test_size))
    train_values = values[:split_idx]
    test_values = values[split_idx:]
    
    # Fit scaler only on training data to prevent data leakage
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)  # Only transform test data
    
    # Recombine for sequence creation (maintaining the split point)
    scaled_values = np.vstack([train_scaled, test_scaled])
    
    # Create sequences
    X_src, y_tgt = [], []
    
    for i in range(seq_len, len(scaled_values) - prediction_horizon + 1):
        # Source sequence (input)
        X_src.append(scaled_values[i-seq_len:i])
        
        # Target sequence (what we want to predict)
        y_tgt.append(scaled_values[i:i+prediction_horizon])
    
    X_src = np.array(X_src)
    y_tgt = np.array(y_tgt)
    
    print(f"Data shapes - X_src: {X_src.shape}, y_tgt: {y_tgt.shape}")
    
    # Find the correct split index for sequences
    # We need to find which sequences belong to train vs test based on their last timestamp
    sequence_split_idx = 0
    for i in range(len(X_src)):
        # Check if this sequence's prediction target goes beyond the original train/test split
        if i + seq_len + prediction_horizon > split_idx:
            sequence_split_idx = i
            break
    
    if sequence_split_idx == 0:
        # If no proper split point found, use proportional split
        sequence_split_idx = int(len(X_src) * (1 - test_size))
    
    X_train_src = X_src[:sequence_split_idx]
    X_test_src = X_src[sequence_split_idx:]
    y_train_tgt = y_tgt[:sequence_split_idx]
    y_test_tgt = y_tgt[sequence_split_idx:]
    
    print(f"Train/Test split: {len(X_train_src)} train sequences, {len(X_test_src)} test sequences")
    
    # Convert to tensors
    X_train_src = torch.FloatTensor(X_train_src)  
    X_test_src = torch.FloatTensor(X_test_src)  
    y_train_tgt = torch.FloatTensor(y_train_tgt) 
    y_test_tgt = torch.FloatTensor(y_test_tgt)
    
    return X_train_src, X_test_src, y_train_tgt, y_test_tgt, scaler


# =====================================================================
# MODEL CREATION
# =====================================================================

def create_model(
    model_type: str,
    input_size: int = 1,
    output_size: int = 1,
    device: torch.device = torch.device('cpu'),
    seq_len: Optional[int] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> Union[TransformerWaterLevelPrediction, LSTMFloodDetection]:
    """
    Factory function to create model based on type.
    
    Args:
        model_type: Type of model ('transformer' or 'lstm')
        input_size: Number of input features
        output_size: Number of output features
        device: Device to place model on
        seq_len: Sequence length (required for transformer)
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Created model instance
    """
    if model_type not in DEFAULT_MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(DEFAULT_MODEL_CONFIGS.keys())}")
    
    # Validate device
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
    
    # Validate input parameters
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    
    # Get default configuration
    config = DEFAULT_MODEL_CONFIGS[model_type].copy()
    model_class = config['class']
    model_params = config['params'].copy()
    
    # Add required parameters
    model_params['input_size'] = input_size
    model_params['output_size'] = output_size
    model_params['device'] = device
    
    # Add sequence length for transformer
    if model_type == 'transformer':
        if seq_len is None:
            raise ValueError("seq_len is required for transformer model")
        model_params['max_length'] = seq_len
    
    # Override with custom config if provided
    if custom_config:
        model_params.update(custom_config)
    
    print(f"\nCreating {model_type.upper()} model with configuration:")
    for key, value in model_params.items():
        if key != 'device':
            print(f"  {key}: {value}")
    
    try:
        model = model_class(**model_params)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to create {model_type} model: {str(e)}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'size_mb': total_params * 4 / 1024 / 1024
    }


# =====================================================================
# TRAINING SETUP
# =====================================================================

def setup_training(
    model: nn.Module,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-9
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Setup training components (optimizer, scheduler, criterion).
    
    Args:
        model: Model to train
        lr: Learning rate
        weight_decay: Weight decay for regularization
        betas: Adam beta parameters
        eps: Adam epsilon parameter
        
    Returns:
        Tuple of (criterion, optimizer, scheduler)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        betas=betas, 
        eps=eps, 
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    return criterion, optimizer, scheduler


def create_checkpoint_dir(model_type: str = 'transformer', base_dir: str = 'model_checkpoints') -> str:
    """
    Create checkpoint directory structure.
    
    Args:
        model_type: Type of model
        base_dir: Base directory for checkpoints
        
    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir = os.path.join(base_dir, model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_loss: float,
    test_loss: float,
    train_losses: List[float],
    test_losses: List[float],
    checkpoint_dir: str,
    model_type: str = 'transformer',
    additional_info: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        train_loss: Training loss
        test_loss: Test loss
        train_losses: History of training losses
        test_losses: History of test losses
        checkpoint_dir: Directory to save checkpoint
        model_type: Type of model
        additional_info: Additional information to save
        
    Returns:
        Tuple of (checkpoint_path, checkpoint_dict)
    """
    checkpoint = {
        'model_type': model_type,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'test_losses': test_losses,
    }
    
    # Save model-specific configuration
    if model_type == 'transformer' and hasattr(model, 'input_size'):
        checkpoint['model_params'] = {
            'input_size': model.input_size,
            'output_size': model.output_size,
            'hidden_dim': model.hidden_dim,
            'max_length': model.max_length
        }
    elif model_type == 'lstm' and hasattr(model, 'input_size'):
        checkpoint['model_params'] = {
            'input_size': model.input_size,
            'output_size': model.output_size,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'bidirectional': model.bidirectional
        }
    
    # Add additional info if provided
    if additional_info:
        checkpoint.update(additional_info)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_type}_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path, checkpoint


# =====================================================================
# TRAINING FUNCTIONS
# =====================================================================

def train_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train model for a single epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_train_loss = 0
    num_batches = 0
    nan_count = 0
    
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        try:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Check for NaN in inputs
            if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                print(f"Warning: NaN detected in batch {batch_idx}, skipping...")
                nan_count += 1
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print(f"Warning: NaN in model outputs at batch {batch_idx}, skipping...")
                nan_count += 1
                continue
            
            # Handle multi-step prediction dimensions
            if batch_y.dim() == 3 and batch_y.shape[-1] == 1:
                batch_y = batch_y.squeeze(-1)
            
            loss = criterion(outputs, batch_y)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}, skipping...")
                nan_count += 1
                continue
            
            loss.backward()
            
            # Check gradient norms before clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"Warning: Invalid gradients at batch {batch_idx}, skipping update...")
                optimizer.zero_grad()
                nan_count += 1
                continue
            
            optimizer.step()
            epoch_train_loss += loss.item()
            num_batches += 1
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            if "out of memory" in str(e):
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                raise e
            continue
    
    if num_batches == 0:
        raise ValueError(f"No valid batches in epoch! {nan_count} batches had NaN/inf values.")
    
    if nan_count > 0:
        print(f"Warning: {nan_count} batches skipped due to NaN/inf values")
    
    return epoch_train_loss / num_batches


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Tuple of (average_loss, average_mse)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)

            # Reshape if needed
            if y_batch.dim() == 3 and y_batch.shape[-1] == 1:
                y_batch = y_batch.squeeze(-1)

            loss = criterion(preds, y_batch)
            mse = torch.mean((preds - y_batch) ** 2).item()
            
            total_loss += loss.item() * X_batch.size(0)
            total_mse += mse * X_batch.size(0)
            total_samples += X_batch.size(0)
            
            # Clean up GPU memory
            del X_batch, y_batch, preds, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    
    return avg_loss, avg_mse


def train_water_level_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model_type: str = 'transformer',
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    evaluation_frequency: int = 10,
    device: torch.device = torch.device('cpu'),
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
    early_stopping_patience: int = 20,
    min_delta: float = 1e-6
) -> Tuple[List[float], List[float]]:
    """
    Main training function for water level prediction models.
    
    Args:
        model: Model to train
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data
        model_type: Type of model ('transformer' or 'lstm')
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        evaluation_frequency: How often to evaluate on test set
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        verbose: Whether to print progress
        early_stopping_patience: Epochs to wait before early stopping
        min_delta: Minimum change to qualify as an improvement
        
    Returns:
        Tuple of (train_losses, test_losses)
    """
    # Setup training components
    criterion, optimizer, scheduler = setup_training(model, lr)
    
    if checkpoint_dir is None:
        checkpoint_dir = create_checkpoint_dir(model_type)
    
    # Initialize tracking variables
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    early_stopping_counter = 0
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if verbose:
        print(f"Training {model_type.upper()} model on device: {device}")
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Test data: {X_test.shape[0]} samples")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        print(f"Model will be evaluated every {evaluation_frequency} epochs")
    
    # Main training loop
    for epoch in range(epochs):
        # Train for one epoch
        avg_train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_train_loss)
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Evaluate periodically
        if epoch % evaluation_frequency == 0:
            test_loss, test_mse = evaluate_model(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            
            # Clear GPU cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save checkpoint
            checkpoint_path, checkpoint = save_checkpoint(
                model, optimizer, scheduler, epoch, avg_train_loss, test_loss, 
                train_losses, test_losses, checkpoint_dir, model_type
            )
            
            if verbose:
                print(f'Epoch {epoch:3d}, Train Loss: {avg_train_loss:.6f}, '
                      f'Test Loss: {test_loss:.6f}, Test MSE: {test_mse:.6f}, '
                      f'LR: {current_lr:.2e} [Checkpoint Saved]')
            
            # Save best model and check for early stopping
            if test_loss < best_test_loss - min_delta:
                best_test_loss = test_loss
                early_stopping_counter = 0
                best_checkpoint_path = os.path.join(checkpoint_dir, f'best_{model_type}_model.pth')
                torch.save(checkpoint, best_checkpoint_path)
                if verbose:
                    print(f'    → New best model saved! (Test Loss: {best_test_loss:.6f})')
            else:
                early_stopping_counter += 1
                if verbose and early_stopping_counter > 0:
                    print(f'    → No improvement for {early_stopping_counter} evaluations')
            
            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                if verbose:
                    print(f'\n*** Early stopping triggered after {epoch} epochs ***')
                    print(f'Best test loss: {best_test_loss:.6f}')
                break
        else:
            # Just show training loss
            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:3d}, Train Loss: {avg_train_loss:.6f}, LR: {current_lr:.2e}')
    
    # Final evaluation (only if not already done)
    current_epoch = min(epoch, epochs - 1)  # Handle early stopping case
    
    # Check if we need to do final evaluation
    if len(test_losses) == 0 or (current_epoch + 1) % evaluation_frequency != 0:
        if verbose:
            print(f"\nRunning final evaluation...")
        
        final_test_loss, final_test_mse = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(final_test_loss)
        
        if verbose:
            print(f'Final Test Loss: {final_test_loss:.6f}, Test MSE: {final_test_mse:.6f}')
    else:
        # Use the last evaluated test loss
        final_test_loss = test_losses[-1]
        final_test_mse = test_losses[-1]  # Using loss as MSE approximation
    
    # Save final model
    final_checkpoint_path, final_checkpoint = save_checkpoint(
        model, optimizer, scheduler, current_epoch, avg_train_loss, final_test_loss, 
        train_losses, test_losses, checkpoint_dir, model_type
    )
    
    final_path = os.path.join(checkpoint_dir, f'final_{model_type}_model.pth')
    torch.save(final_checkpoint, final_path)
    
    if verbose:
        print(f'\nTraining completed!')
        print(f'Final model saved to: {final_path}')
        print(f'Best model saved to: {os.path.join(checkpoint_dir, f"best_{model_type}_model.pth")}')
    
    return train_losses, test_losses


# =====================================================================
# CHECKPOINT MANAGEMENT
# =====================================================================

def load_checkpoint(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu'),
    create_model_fn: Optional[callable] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into (optional)
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        device: Device to load model on
        create_model_fn: Function to create model if not provided
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect model type
    model_type = checkpoint.get('model_type', 'transformer')
    print(f"Detected model type: {model_type}")
    
    # Create model if not provided
    if model is None:
        if create_model_fn:
            model = create_model_fn(model_type).to(device)
        else:
            # Try to recreate from saved params
            model_params = checkpoint.get('model_params', {})
            model = create_model(
                model_type,
                input_size=model_params.get('input_size', 1),
                output_size=model_params.get('output_size', 1),
                device=device,
                seq_len=model_params.get('max_length')
            ).to(device)
        print(f"Created new {model_type} model instance")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model state loaded from epoch {checkpoint['epoch']}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    print(f"Checkpoint info:")
    print(f"  Model Type: {model_type}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train Loss: {checkpoint['train_loss']:.6f}")
    print(f"  Test Loss: {checkpoint['test_loss']:.6f}")
    
    return model, checkpoint


def list_checkpoints(
    model_type: Optional[str] = None,
    base_dir: str = 'model_checkpoints'
) -> List[str]:
    """
    List all available checkpoints.
    
    Args:
        model_type: Optional model type filter
        base_dir: Base directory for checkpoints
        
    Returns:
        List of checkpoint file paths
    """
    if model_type:
        checkpoint_dirs = [os.path.join(base_dir, model_type)]
    else:
        # List all model type directories
        checkpoint_dirs = []
        if os.path.exists(base_dir):
            for dir_name in os.listdir(base_dir):
                full_path = os.path.join(base_dir, dir_name)
                if os.path.isdir(full_path):
                    checkpoint_dirs.append(full_path)
    
    all_checkpoints = []
    
    for checkpoint_dir in checkpoint_dirs:
        if not os.path.exists(checkpoint_dir):
            continue
            
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
        checkpoint_files.sort()
        
        if checkpoint_files:
            print(f"\nAvailable checkpoints in {checkpoint_dir}:")
            for i, file_path in enumerate(checkpoint_files):
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {i+1}. {filename} ({file_size:.2f} MB)")
                all_checkpoints.append(file_path)
    
    return all_checkpoints


def resume_training(
    checkpoint_path: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    additional_epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    evaluation_frequency: int = 10,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Resume training from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data
        additional_epochs: Additional epochs to train
        lr: Learning rate
        batch_size: Batch size
        evaluation_frequency: Evaluation frequency
        device: Device to train on
        
    Returns:
        Tuple of (model, total_train_losses, total_test_losses)
    """
    # Load checkpoint and model
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    model_type = checkpoint.get('model_type', 'transformer')
    
    # Create new optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Load optimizer and scheduler states
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    previous_train_losses = checkpoint.get('train_losses', [])
    previous_test_losses = checkpoint.get('test_losses', [])
    
    print(f"\nResuming {model_type} training from epoch {start_epoch}")
    print(f"Training for {additional_epochs} more epochs...")
    
    # Continue training
    new_train_losses, new_test_losses = train_water_level_model(
        model, X_train, y_train, X_test, y_test,
        model_type=model_type,
        epochs=additional_epochs,
        lr=lr,
        batch_size=batch_size,
        evaluation_frequency=evaluation_frequency,
        device=device
    )
    
    # Combine loss histories
    total_train_losses = previous_train_losses + new_train_losses
    total_test_losses = previous_test_losses + new_test_losses
    
    return model, total_train_losses, total_test_losses


# =====================================================================
# MEMORY MANAGEMENT UTILITIES
# =====================================================================

def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with memory info in MB
    """
    if device.type != 'cuda':
        return {'available': float('inf'), 'total': float('inf'), 'used': 0, 'percent': 0}
    
    try:
        # Get memory info in bytes
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        available = total - reserved
        
        # Convert to MB
        return {
            'total': total / 1024 / 1024,
            'reserved': reserved / 1024 / 1024,
            'allocated': allocated / 1024 / 1024,
            'available': available / 1024 / 1024,
            'percent': (reserved / total) * 100
        }
    except:
        return {'available': float('inf'), 'total': float('inf'), 'used': 0, 'percent': 0}


def calculate_optimal_batch_size(
    model_size_mb: float,
    data_point_size_mb: float,
    device: torch.device,
    safety_factor: float = 0.8
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        model_size_mb: Size of model in MB
        data_point_size_mb: Size of single data point in MB
        device: Device to check memory for
        safety_factor: Fraction of available memory to use
        
    Returns:
        Optimal batch size
    """
    if device.type != 'cuda':
        return 32  # Default CPU batch size
    
    memory_info = get_gpu_memory_info(device)
    available_mb = memory_info['available'] * safety_factor
    
    # Reserve memory for model and gradients (3x model size for safety)
    available_for_data = available_mb - (3 * model_size_mb)
    
    if available_for_data <= 0:
        return 1  # Minimum batch size
    
    # Calculate batch size
    batch_size = int(available_for_data / data_point_size_mb)
    
    # Clamp to reasonable range
    return max(1, min(batch_size, 512))


# =====================================================================
# PREDICTION AND EVALUATION
# =====================================================================

def batched_inference(
    model: nn.Module,
    X_data: torch.Tensor,
    batch_size: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Perform inference in batches to manage memory.
    
    Args:
        model: Model to use for inference
        X_data: Input data
        batch_size: Batch size for inference (auto-calculated if None)
        device: Device to run inference on
        
    Returns:
        Predictions tensor
    """
    model.eval()
    
    # Auto-calculate batch size if not provided
    if batch_size is None:
        # Estimate sizes
        model_size_mb = count_parameters(model)['size_mb']
        data_point_size_mb = X_data[0].numel() * 4 / 1024 / 1024  # 4 bytes per float
        batch_size = calculate_optimal_batch_size(model_size_mb, data_point_size_mb, device)
        print(f"Auto-calculated batch size: {batch_size}")
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            try:
                batch = X_data[i:i+batch_size].to(device)
                pred_batch = model(batch)
                predictions.append(pred_batch.cpu())
                
                # Clean up
                del batch, pred_batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Try with smaller batch size
                    if batch_size > 1:
                        print(f"OOM with batch size {batch_size}, reducing to {batch_size // 2}")
                        torch.cuda.empty_cache()
                        return batched_inference(model, X_data, batch_size // 2, device)
                    else:
                        raise e
                else:
                    raise e
    
    return torch.cat(predictions, dim=0)


def create_continuous_predictions(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    scaler: MinMaxScaler,
    seq_len: int,
    prediction_horizon: int,
    device: torch.device = torch.device('cpu')
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Create continuous predictions from non-overlapping sequences.
    
    Args:
        model: Trained model
        X_test: Test input data
        y_test: Test target data
        scaler: Scaler used for data normalization
        seq_len: Sequence length
        prediction_horizon: Prediction horizon
        device: Device to run predictions on
        
    Returns:
        Tuple of (predictions_unscaled, true_values_unscaled) or None
    """
    model.eval()
    
    # Use non-overlapping sequences
    step_size = prediction_horizon
    
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for i in range(0, len(X_test) - 1, step_size):
            if i + 1 >= len(X_test):
                break
                
            # Get input sequence
            X_batch = X_test[i:i+1].to(device)
            y_batch = y_test[i:i+1]
            
            # Make prediction
            pred = model(X_batch)
            
            # Store predictions and true values
            predictions.append(pred.cpu().numpy())
            
            # Handle different y_test shapes
            if y_batch.dim() == 3:
                y_flat = y_batch.squeeze(-1)
            else:
                y_flat = y_batch
            true_values.append(y_flat.numpy())
            
            # Clear GPU memory
            del X_batch, pred
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    if not predictions:
        print("No predictions generated")
        return None
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    
    # Flatten for inverse transform
    pred_flat = predictions.reshape(-1, 1)
    true_flat = true_values.reshape(-1, 1)
    
    # Inverse transform
    pred_unscaled = scaler.inverse_transform(pred_flat).flatten()
    true_unscaled = scaler.inverse_transform(true_flat).flatten()
    
    return pred_unscaled, true_unscaled


def calculate_metrics(
    predictions: np.ndarray,
    true_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: Predicted values
        true_values: True values
        
    Returns:
        Dictionary of metrics
    """
    errors = predictions - true_values
    
    mse = np.mean(errors**2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((true_values - np.mean(true_values))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(np.abs(errors))
    }


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_training_history(
    train_losses: List[float],
    test_losses: List[float],
    evaluation_frequency: int = 10,
    figsize: Tuple[int, int] = (20, 5)
) -> None:
    """
    Plot training history.
    
    Args:
        train_losses: Training losses
        test_losses: Test losses
        evaluation_frequency: How often test evaluation was done
        figsize: Figure size
    """
    epochs = len(train_losses)
    
    plt.figure(figsize=figsize)
    
    # Loss curves
    plt.subplot(1, 2, 1)
    train_epochs = list(range(len(train_losses)))
    plt.plot(train_epochs, train_losses, label='Train Loss', alpha=0.8)
    
    # Plot test loss only at evaluation epochs
    test_epochs = []
    for i in range(0, epochs, evaluation_frequency):
        test_epochs.append(i)
    if (epochs - 1) % evaluation_frequency != 0:
        test_epochs.append(epochs - 1)
    
    test_epochs = test_epochs[:len(test_losses)]
    plt.plot(test_epochs, test_losses, label='Test Loss', alpha=0.8, marker='o', markersize=4)
    
    plt.title(f'Training History (Test evaluated every {evaluation_frequency} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Training vs Testing MSE
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_losses, label='Training MSE', alpha=0.8, color='blue')
    plt.plot(test_epochs, test_losses, label='Testing MSE', alpha=0.8, color='red', marker='o', markersize=4)
    plt.title('Training vs Testing MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()


def plot_predictions(
    predictions: np.ndarray,
    true_values: np.ndarray,
    time_step: float = 0.1,
    n_points: int = 500,
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Plot predictions vs true values.
    
    Args:
        predictions: Predicted values
        true_values: True values
        time_step: Time step between points (in hours)
        n_points: Number of points to plot
        figsize: Figure size
    """
    n_plot = min(n_points, len(predictions))
    time_hours = np.arange(n_plot) * time_step
    
    plt.figure(figsize=figsize)
    
    # Time series plot
    plt.subplot(1, 2, 1)
    plt.plot(time_hours, true_values[:n_plot], label='Actual', linewidth=2, alpha=0.8, color='blue')
    plt.plot(time_hours, predictions[:n_plot], label='Predicted', linewidth=2, alpha=0.8, color='red')
    plt.title(f'Predictions vs Actual ({n_plot} points)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Water Level (ft)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(true_values, predictions, alpha=0.5, s=8)
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    plt.xlabel('Actual Water Level (ft)')
    plt.ylabel('Predicted Water Level (ft)')
    plt.title('Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Calculate and display R²
    metrics = calculate_metrics(predictions, true_values)
    plt.text(0.05, 0.95, f'R² = {metrics["r2"]:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_error_analysis(
    predictions: np.ndarray,
    true_values: np.ndarray,
    prediction_horizon: int = 240,
    figsize: Tuple[int, int] = (20, 10)
) -> Dict[str, Any]:
    """
    Comprehensive error analysis visualization.
    
    Args:
        predictions: Predicted values
        true_values: True values
        prediction_horizon: Prediction horizon for sequence analysis
        figsize: Figure size
        
    Returns:
        Dictionary containing metrics and analysis results
    """
    errors = predictions - true_values
    metrics = calculate_metrics(predictions, true_values)
    
    plt.figure(figsize=figsize)
    
    # Error distribution
    plt.subplot(2, 3, 1)
    plt.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.4f} ft')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Prediction Error (ft)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error over time
    plt.subplot(2, 3, 2)
    time_hours = np.arange(len(errors)) * 0.1
    plt.plot(time_hours[:min(1000, len(errors))], errors[:min(1000, len(errors))], 
             alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Time (hours)')
    plt.ylabel('Error (ft)')
    plt.title('Prediction Error Over Time')
    plt.grid(True, alpha=0.3)
    
    # Error by sequence position
    plt.subplot(2, 3, 3)
    if len(errors) >= prediction_horizon:
        error_by_position = [[] for _ in range(prediction_horizon)]
        for i, error in enumerate(errors):
            pos = i % prediction_horizon
            error_by_position[pos].append(error)
        
        mean_errors_by_pos = [np.mean(pos_errors) if pos_errors else 0 
                              for pos_errors in error_by_position]
        
        plot_positions = list(range(min(50, prediction_horizon)))
        plot_errors = mean_errors_by_pos[:min(50, prediction_horizon)]
        
        plt.bar(plot_positions, plot_errors, alpha=0.7, color='purple')
        plt.xlabel('Position within sequence')
        plt.ylabel('Mean Prediction Error (ft)')
        plt.title('Error by Sequence Position')
        plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(2, 3, 4)
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal Distribution)')
    plt.grid(True, alpha=0.3)
    
    # Residuals vs predicted
    plt.subplot(2, 3, 5)
    plt.scatter(predictions, errors, alpha=0.5, s=8)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Values (ft)')
    plt.ylabel('Residuals (ft)')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    stats_text = f"""Model Performance Metrics
    
MSE:  {metrics['mse']:.6f} ft²
RMSE: {metrics['rmse']:.6f} ft  
MAE:  {metrics['mae']:.6f} ft
R²:   {metrics['r2']:.4f}

Error Statistics:
Mean Error: {metrics['mean_error']:.6f} ft
Std Error:  {metrics['std_error']:.6f} ft
Max Error:  {metrics['max_error']:.6f} ft

Data Range:
Min: {true_values.min():.2f} ft
Max: {true_values.max():.2f} ft
Range: {true_values.max() - true_values.min():.2f} ft"""
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    return metrics


# =====================================================================
# BACKWARD COMPATIBILITY
# =====================================================================

def prepare_data_for_transformer(*args, **kwargs):
    """Backward compatibility wrapper for prepare_data_for_model."""
    return prepare_data_for_model(*args, **kwargs)


def train_transformer_flood_detection(*args, **kwargs):
    """Backward compatibility wrapper for train_water_level_model."""
    if 'model_type' not in kwargs:
        kwargs['model_type'] = 'transformer'
    return train_water_level_model(*args, **kwargs)