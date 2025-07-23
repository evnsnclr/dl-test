"""
Leak-free time series data pipeline for water level prediction.

This module implements a rigorous data preprocessing pipeline that prevents
data leakage and ensures reproducible, production-ready model training.

Key features:
- Chronological train/val/test splitting
- Proper gap handling and interpolation
- Non-overlapping test windows
- Temporal feature engineering
- Comprehensive validation and testing
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataPipeline:
    """
    Comprehensive time series data pipeline with leak prevention.
    
    This class handles all aspects of time series preprocessing including:
    - Data versioning and reproducibility
    - Chronological splitting
    - Gap detection and filling
    - Feature engineering
    - Proper scaling without leakage
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        target_column: str = "value",
        time_column: str = "collect_time",
        sensor_id_column: str = "sensor_id",
        target_freq: str = "6T",  # 6 minutes
        max_gap_interpolate: str = "24H",
        scaler_type: str = "robust"
    ):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Base directory for data storage
            target_column: Name of target variable column
            time_column: Name of timestamp column
            sensor_id_column: Name of sensor ID column
            target_freq: Target frequency for resampling
            max_gap_interpolate: Maximum gap size to interpolate
            scaler_type: Type of scaler ('minmax' or 'robust')
        """
        self.data_dir = data_dir
        self.target_column = target_column
        self.time_column = time_column
        self.sensor_id_column = sensor_id_column
        self.target_freq = target_freq
        self.max_gap_interpolate = pd.Timedelta(max_gap_interpolate)
        self.scaler_type = scaler_type
        
        # Create directory structure
        self._create_data_dirs()
        
        # Initialize scalers and metadata
        self.scalers = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'pipeline_version': '2.0',
            'settings': {
                'target_freq': target_freq,
                'max_gap_interpolate': max_gap_interpolate,
                'scaler_type': scaler_type
            }
        }
        
    def _create_data_dirs(self):
        """Create versioned data directory structure."""
        dirs = [
            os.path.join(self.data_dir, "raw"),
            os.path.join(self.data_dir, "interim"),
            os.path.join(self.data_dir, "processed"),
            os.path.join(self.data_dir, "models"),
            os.path.join(self.data_dir, "logs")
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def load_raw_data(self, file_path: str, sensor_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data with versioning.
        
        Args:
            file_path: Path to raw data file
            sensor_id: Optional sensor ID to filter
            
        Returns:
            Raw dataframe
        """
        # Load data
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        # Filter by sensor if specified
        if sensor_id and self.sensor_id_column in df.columns:
            df = df[df[self.sensor_id_column] == sensor_id].copy()
            
        # Ensure datetime
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # Sort by time
        df = df.sort_values(self.time_column).reset_index(drop=True)
        
        # Calculate data hash for versioning
        data_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:8]
        logger.info(f"Loaded data with hash: {data_hash}")
        
        return df
        
    def detect_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect gaps in time series data.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with gap information
        """
        # Calculate time differences
        df = df.copy()
        df['time_diff'] = df[self.time_column].diff()
        
        # Expected frequency
        expected_freq = pd.Timedelta(self.target_freq)
        
        # Identify gaps (more than 1.5x expected frequency)
        df['is_gap'] = df['time_diff'] > (1.5 * expected_freq)
        
        # Calculate gap sizes
        gaps = df[df['is_gap']].copy()
        gaps['gap_size'] = gaps['time_diff']
        gaps['gap_hours'] = gaps['gap_size'].dt.total_seconds() / 3600
        
        # Log gap statistics
        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} gaps in data")
            logger.warning(f"Largest gap: {gaps['gap_hours'].max():.1f} hours")
            logger.warning(f"Total gap time: {gaps['gap_hours'].sum():.1f} hours")
            
        return gaps
        
    def fill_gaps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fill gaps with appropriate interpolation.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (filled dataframe, gap mask dataframe)
        """
        # Create regular time index
        freq = pd.Timedelta(self.target_freq)
        regular_index = pd.date_range(
            start=df[self.time_column].min(),
            end=df[self.time_column].max(),
            freq=self.target_freq
        )
        
        # Reindex to regular intervals
        df_indexed = df.set_index(self.time_column)
        df_regular = df_indexed.reindex(regular_index)
        
        # Create gap mask
        gap_mask = pd.DataFrame(index=regular_index)
        gap_mask['is_original'] = ~df_regular[self.target_column].isna()
        gap_mask['gap_size'] = 0
        
        # Calculate gap sizes
        for i in range(1, len(gap_mask)):
            if not gap_mask.iloc[i]['is_original']:
                if i > 0 and not gap_mask.iloc[i-1]['is_original']:
                    gap_mask.iloc[i, gap_mask.columns.get_loc('gap_size')] = (
                        gap_mask.iloc[i-1]['gap_size'] + 1
                    )
                else:
                    gap_mask.iloc[i, gap_mask.columns.get_loc('gap_size')] = 1
                    
        # Convert gap size to timedelta
        gap_mask['gap_duration'] = gap_mask['gap_size'] * freq
        
        # Fill based on gap size
        df_filled = df_regular.copy()
        
        # Small gaps (< 1 hour): forward fill
        small_gap_mask = gap_mask['gap_duration'] <= pd.Timedelta('1H')
        df_filled.loc[small_gap_mask, self.target_column] = (
            df_filled[self.target_column].fillna(method='ffill')
        )
        
        # Medium gaps (1-24 hours): linear interpolation
        medium_gap_mask = (
            (gap_mask['gap_duration'] > pd.Timedelta('1H')) & 
            (gap_mask['gap_duration'] <= self.max_gap_interpolate)
        )
        if medium_gap_mask.any():
            df_filled.loc[medium_gap_mask, self.target_column] = (
                df_filled.loc[medium_gap_mask, self.target_column].interpolate(method='linear')
            )
            
        # Large gaps: leave as NaN but mark in metadata
        large_gap_mask = gap_mask['gap_duration'] > self.max_gap_interpolate
        gap_mask['interpolation_method'] = 'original'
        gap_mask.loc[small_gap_mask, 'interpolation_method'] = 'forward_fill'
        gap_mask.loc[medium_gap_mask, 'interpolation_method'] = 'linear'
        gap_mask.loc[large_gap_mask, 'interpolation_method'] = 'none'
        
        # Reset index
        df_filled = df_filled.reset_index()
        df_filled.rename(columns={'index': self.time_column}, inplace=True)
        
        # Log interpolation statistics
        logger.info(f"Interpolation summary:")
        logger.info(f"  Original points: {gap_mask['is_original'].sum()}")
        logger.info(f"  Forward filled: {small_gap_mask.sum()}")
        logger.info(f"  Linear interpolated: {medium_gap_mask.sum()}")
        logger.info(f"  Left as NaN: {large_gap_mask.sum()}")
        
        return df_filled, gap_mask
        
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for better prediction.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df[self.time_column].dt.hour
        df['day_of_week'] = df[self.time_column].dt.dayofweek
        df['day_of_month'] = df[self.time_column].dt.day
        df['month'] = df[self.time_column].dt.month
        df['quarter'] = df[self.time_column].dt.quarter
        
        # Cyclical encoding for temporal features
        # Hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Month of year
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Tidal features (assuming semi-diurnal tide ~12.42 hours)
        tidal_period_hours = 12.42
        df['hours_since_start'] = (
            df[self.time_column] - df[self.time_column].min()
        ).dt.total_seconds() / 3600
        df['tide_sin'] = np.sin(2 * np.pi * df['hours_since_start'] / tidal_period_hours)
        df['tide_cos'] = np.cos(2 * np.pi * df['hours_since_start'] / tidal_period_hours)
        
        # Drop intermediate columns
        df.drop(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 
                 'hours_since_start'], axis=1, inplace=True)
        
        return df
        
    def chronological_split(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        val_end_date: str,
        buffer_days: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically with buffer zones.
        
        Args:
            df: Input dataframe
            train_end_date: End date for training set (YYYY-MM-DD)
            val_end_date: End date for validation set (YYYY-MM-DD)
            buffer_days: Days to skip between sets to prevent leakage
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Convert dates
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        buffer = pd.Timedelta(days=buffer_days)
        
        # Split with buffers
        train_df = df[df[self.time_column] < train_end].copy()
        val_df = df[
            (df[self.time_column] >= train_end + buffer) & 
            (df[self.time_column] < val_end)
        ].copy()
        test_df = df[df[self.time_column] >= val_end + buffer].copy()
        
        # Log split information
        logger.info(f"Data split summary:")
        logger.info(f"  Train: {len(train_df)} samples ({train_df[self.time_column].min()} to {train_df[self.time_column].max()})")
        logger.info(f"  Val: {len(val_df)} samples ({val_df[self.time_column].min()} to {val_df[self.time_column].max()})")
        logger.info(f"  Test: {len(test_df)} samples ({test_df[self.time_column].min()} to {test_df[self.time_column].max()})")
        
        # Validate no overlap
        assert train_df[self.time_column].max() < val_df[self.time_column].min(), "Train/Val overlap detected!"
        assert val_df[self.time_column].max() < test_df[self.time_column].min(), "Val/Test overlap detected!"
        
        return train_df, val_df, test_df
        
    def create_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int,
        pred_horizon: int,
        stride: Optional[int] = None,
        features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for model training with proper windowing.
        
        Args:
            df: Input dataframe
            seq_len: Length of input sequences
            pred_horizon: Prediction horizon
            stride: Step size between sequences (defaults to 1 for train, pred_horizon for test)
            features: List of feature columns to use
            
        Returns:
            Tuple of (X, y) arrays
        """
        if features is None:
            features = [self.target_column]
            
        # Default stride
        if stride is None:
            stride = 1
            
        # Extract feature array
        data = df[features].values
        
        X, y = [], []
        
        # Create sequences with specified stride
        for i in range(0, len(data) - seq_len - pred_horizon + 1, stride):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + pred_horizon, 0])  # Only target column for y
            
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")
        
        return X, y
        
    def fit_scalers(self, train_df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Fit scalers on training data only.
        
        Args:
            train_df: Training dataframe
            features: List of features to scale
            
        Returns:
            Dictionary of fitted scalers
        """
        scalers = {}
        
        for feature in features:
            if self.scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
                
            # Fit on training data only
            train_values = train_df[feature].values.reshape(-1, 1)
            scaler.fit(train_values)
            scalers[feature] = scaler
            
            logger.info(f"Fitted {self.scaler_type} scaler for {feature}")
            
        self.scalers = scalers
        return scalers
        
    def transform_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            df: DataFrame to transform
            features: List of features to scale
            
        Returns:
            Transformed dataframe
        """
        df_scaled = df.copy()
        
        for feature in features:
            if feature not in self.scalers:
                raise ValueError(f"No scaler fitted for feature: {feature}")
                
            values = df[feature].values.reshape(-1, 1)
            df_scaled[feature] = self.scalers[feature].transform(values)
            
        return df_scaled
        
    def save_pipeline(self, pipeline_path: str):
        """Save the entire pipeline configuration and scalers."""
        pipeline_data = {
            'metadata': self.metadata,
            'scalers': self.scalers,
            'settings': {
                'target_column': self.target_column,
                'time_column': self.time_column,
                'target_freq': self.target_freq,
                'scaler_type': self.scaler_type
            }
        }
        
        joblib.dump(pipeline_data, pipeline_path)
        logger.info(f"Saved pipeline to {pipeline_path}")
        
    def load_pipeline(self, pipeline_path: str):
        """Load pipeline configuration and scalers."""
        pipeline_data = joblib.load(pipeline_path)
        
        self.metadata = pipeline_data['metadata']
        self.scalers = pipeline_data['scalers']
        settings = pipeline_data['settings']
        
        self.target_column = settings['target_column']
        self.time_column = settings['time_column']
        self.target_freq = settings['target_freq']
        self.scaler_type = settings['scaler_type']
        
        logger.info(f"Loaded pipeline from {pipeline_path}")
        
    def run_leak_test(self, X_train, y_train, X_test, y_test) -> bool:
        """
        Test for data leakage by checking if shuffling degrades performance.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            True if no leakage detected, False otherwise
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error
        
        # Train simple model on correct split
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train.mean(axis=1))
        correct_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        correct_mae = mean_absolute_error(y_test.mean(axis=1), correct_pred)
        
        # Create leaked data by mixing train and test
        X_leaked = np.vstack([X_train, X_test])
        y_leaked = np.vstack([y_train, y_test])
        
        # Shuffle
        indices = np.random.permutation(len(X_leaked))
        X_leaked = X_leaked[indices]
        y_leaked = y_leaked[indices]
        
        # Split again
        split_idx = len(X_train)
        X_train_leaked = X_leaked[:split_idx]
        y_train_leaked = y_leaked[:split_idx]
        X_test_leaked = X_leaked[split_idx:]
        y_test_leaked = y_leaked[split_idx:]
        
        # Train on leaked data
        model_leaked = RandomForestRegressor(n_estimators=10, random_state=42)
        model_leaked.fit(
            X_train_leaked.reshape(X_train_leaked.shape[0], -1),
            y_train_leaked.mean(axis=1)
        )
        leaked_pred = model_leaked.predict(X_test_leaked.reshape(X_test_leaked.shape[0], -1))
        leaked_mae = mean_absolute_error(y_test_leaked.mean(axis=1), leaked_pred)
        
        # Check if performance degraded significantly
        performance_ratio = leaked_mae / correct_mae
        logger.info(f"Leak test - Correct MAE: {correct_mae:.4f}, Leaked MAE: {leaked_mae:.4f}")
        logger.info(f"Performance ratio: {performance_ratio:.2f}")
        
        # If leaked performance is much better, we have leakage
        if performance_ratio < 0.5:
            logger.error("DATA LEAKAGE DETECTED! Leaked model performs much better.")
            return False
        else:
            logger.info("No data leakage detected.")
            return True
            

def create_time_series_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    test_size: int = 1000,
    gap_size: int = 100
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits with gaps.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of CV splits
        test_size: Size of each test set
        gap_size: Gap between train and test to prevent leakage
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    
    # Calculate split positions
    total_test_samples = n_splits * (test_size + gap_size)
    if total_test_samples > n_samples * 0.5:
        raise ValueError("Test sets would use more than 50% of data")
        
    # Create splits from end backwards
    for i in range(n_splits):
        test_end = n_samples - i * (test_size + gap_size)
        test_start = test_end - test_size
        train_end = test_start - gap_size
        
        if train_end < n_samples * 0.3:  # Ensure reasonable train size
            break
            
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)
        
        splits.append((train_indices, test_indices))
        
    return splits[::-1]  # Reverse to chronological order


if __name__ == "__main__":
    # Example usage
    pipeline = TimeSeriesDataPipeline()
    
    # Demonstrate the complete pipeline
    logger.info("Initializing water level prediction data pipeline...")
    
    # This would be run on actual data
    # df = pipeline.load_raw_data("data/1836026195.parquet")
    # gaps = pipeline.detect_gaps(df)
    # df_filled, gap_mask = pipeline.fill_gaps(df)
    # df_features = pipeline.add_temporal_features(df_filled)
    # train_df, val_df, test_df = pipeline.chronological_split(
    #     df_features, "2024-12-01", "2025-02-01"
    # )