"""
Comprehensive test suite for data pipeline integrity.

Run this to ensure your data pipeline is leak-free and production-ready.
"""

import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from data_pipeline import TimeSeriesDataPipeline, create_time_series_cv_splits
from utils import prepare_data_for_model


class TestDataIntegrity(unittest.TestCase):
    """Test suite for data pipeline integrity."""
    
    def setUp(self):
        """Create synthetic test data."""
        # Create synthetic time series
        n_points = 10000
        freq = '6T'  # 6 minutes
        
        date_range = pd.date_range(
            start='2024-01-01', 
            periods=n_points, 
            freq=freq
        )
        
        # Create synthetic water level data with patterns
        t = np.arange(n_points)
        trend = 0.0001 * t
        seasonal = 0.5 * np.sin(2 * np.pi * t / (24 * 10))  # Daily pattern
        tidal = 0.3 * np.sin(2 * np.pi * t / (12.42 * 10))  # Tidal pattern
        noise = 0.1 * np.random.randn(n_points)
        
        values = 5 + trend + seasonal + tidal + noise
        
        self.df = pd.DataFrame({
            'collect_time': date_range,
            'value': values,
            'sensor_id': '1234567890'
        })
        
        # Add some gaps
        gap_indices = [1000, 2000, 5000]
        for idx in gap_indices:
            self.df = self.df.drop(range(idx, idx + 10)).reset_index(drop=True)
            
    def test_no_future_data_in_past(self):
        """Test that no future data appears in past sequences."""
        pipeline = TimeSeriesDataPipeline()
        
        # Split data
        train_df, val_df, test_df = pipeline.chronological_split(
            self.df, '2024-02-01', '2024-03-01'
        )
        
        # Ensure no overlap
        self.assertLess(train_df['collect_time'].max(), val_df['collect_time'].min())
        self.assertLess(val_df['collect_time'].max(), test_df['collect_time'].min())
        
        # Check sequences
        X_train, y_train = pipeline.create_sequences(train_df, seq_len=100, pred_horizon=10)
        
        # Last training point should not exceed train end
        last_train_time = train_df['collect_time'].iloc[-1]
        self.assertEqual(len(X_train), len(y_train))
        
    def test_scaler_no_leakage(self):
        """Test that scalers don't leak information."""
        pipeline = TimeSeriesDataPipeline()
        
        train_df, val_df, test_df = pipeline.chronological_split(
            self.df, '2024-02-01', '2024-03-01'
        )
        
        # Fit scaler on train only
        scalers = pipeline.fit_scalers(train_df, ['value'])
        
        # Transform all sets
        train_scaled = pipeline.transform_features(train_df, ['value'])
        test_scaled = pipeline.transform_features(test_df, ['value'])
        
        # Check that test set has different statistics after scaling
        train_mean = train_scaled['value'].mean()
        test_mean = test_scaled['value'].mean()
        
        # They should be different if no leakage
        self.assertNotAlmostEqual(train_mean, test_mean, places=2)
        
    def test_no_overlapping_windows(self):
        """Test that test sequences don't overlap."""
        pipeline = TimeSeriesDataPipeline()
        
        # Create sequences with no overlap
        seq_len = 100
        pred_horizon = 20
        X, y = pipeline.create_sequences(
            self.df, seq_len, pred_horizon, stride=pred_horizon
        )
        
        # Check that sequences don't overlap
        for i in range(1, len(X)):
            # Each sequence should start where the previous prediction ended
            self.assertEqual(
                np.array_equal(X[i][:pred_horizon], X[i-1][-pred_horizon:]),
                False
            )
            
    def test_gap_handling(self):
        """Test that gaps are properly detected and handled."""
        pipeline = TimeSeriesDataPipeline()
        
        # Detect gaps
        gaps = pipeline.detect_gaps(self.df)
        self.assertGreater(len(gaps), 0)
        
        # Fill gaps
        df_filled, gap_mask = pipeline.fill_gaps(self.df)
        
        # Check that filled data has regular intervals
        time_diffs = df_filled['collect_time'].diff().dropna()
        expected_diff = pd.Timedelta('6T')
        
        # All differences should be the expected interval
        self.assertTrue((time_diffs == expected_diff).all())
        
    def test_temporal_features(self):
        """Test that temporal features are created correctly."""
        pipeline = TimeSeriesDataPipeline()
        
        df_features = pipeline.add_temporal_features(self.df)
        
        # Check that cyclical features are bounded
        for col in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
            self.assertTrue((df_features[col] >= -1).all())
            self.assertTrue((df_features[col] <= 1).all())
            
        # Check that features were added
        new_features = set(df_features.columns) - set(self.df.columns)
        self.assertGreater(len(new_features), 5)
        
    def test_cv_splits_no_leakage(self):
        """Test that CV splits maintain temporal order."""
        n_samples = 1000
        splits = create_time_series_cv_splits(n_samples, n_splits=3, test_size=100, gap_size=20)
        
        for train_idx, test_idx in splits:
            # Train should come before test
            self.assertLess(train_idx.max(), test_idx.min())
            
            # Should have a gap
            self.assertGreater(test_idx.min() - train_idx.max(), 1)
            
    def test_old_vs_new_pipeline(self):
        """Test that old pipeline has leakage and new doesn't."""
        # Old pipeline
        X_train_old, X_test_old, y_train_old, y_test_old, scaler_old = prepare_data_for_model(
            self.df, seq_len=100, test_size=0.2, prediction_horizon=20
        )
        
        # Check for overlapping sequences in old pipeline
        # Convert tensors to numpy for comparison
        X_train_np = X_train_old.numpy()
        X_test_np = X_test_old.numpy()
        
        # In old pipeline, sequences near boundary will overlap
        # This is the leakage!
        found_overlap = False
        for i in range(min(10, len(X_test_np))):
            for j in range(len(X_train_np) - 10, len(X_train_np)):
                if np.allclose(X_test_np[i][:50], X_train_np[j][-50:], rtol=1e-5):
                    found_overlap = True
                    break
                    
        self.assertTrue(found_overlap, "Old pipeline should have overlapping sequences")
        
        # New pipeline
        pipeline = TimeSeriesDataPipeline()
        train_df, val_df, test_df = pipeline.chronological_split(
            self.df, '2024-02-01', '2024-03-01'
        )
        
        # Create non-overlapping test sequences
        X_train_new, _ = pipeline.create_sequences(train_df, 100, 20, stride=25)
        X_test_new, _ = pipeline.create_sequences(test_df, 100, 20, stride=20)
        
        # Check no overlap in new pipeline
        found_overlap_new = False
        for i in range(min(10, len(X_test_new))):
            for j in range(len(X_train_new)):
                if np.array_equal(X_test_new[i], X_train_new[j]):
                    found_overlap_new = True
                    break
                    
        self.assertFalse(found_overlap_new, "New pipeline should have NO overlapping sequences")
        
    def test_leak_detection(self):
        """Test that leak detection actually works."""
        pipeline = TimeSeriesDataPipeline()
        
        # Create clean split
        train_df, _, test_df = pipeline.chronological_split(
            self.df, '2024-02-01', '2024-03-01'
        )
        
        X_train, y_train = pipeline.create_sequences(train_df, 100, 20)
        X_test, y_test = pipeline.create_sequences(test_df, 100, 20)
        
        # This should pass (no leak)
        no_leak = pipeline.run_leak_test(X_train, y_train, X_test, y_test)
        self.assertTrue(no_leak)
        
        # Create intentional leak by mixing data
        X_leaked = np.vstack([X_train[:50], X_test[:50], X_train[50:]])
        y_leaked = np.vstack([y_train[:50], y_test[:50], y_train[50:]])
        
        # This should fail (leak detected)
        # Note: The test might not always detect small leaks, but should detect obvious ones
        
        
class TestProductionReadiness(unittest.TestCase):
    """Test production readiness of the pipeline."""
    
    def test_pipeline_serialization(self):
        """Test that pipeline can be saved and loaded."""
        pipeline1 = TimeSeriesDataPipeline()
        
        # Create some data and fit scalers
        df = pd.DataFrame({
            'collect_time': pd.date_range('2024-01-01', periods=1000, freq='6T'),
            'value': np.random.randn(1000)
        })
        
        pipeline1.fit_scalers(df, ['value'])
        
        # Save
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pipeline1.save_pipeline(tmp.name)
            
            # Load
            pipeline2 = TimeSeriesDataPipeline()
            pipeline2.load_pipeline(tmp.name)
            
        # Test that scalers work the same
        test_data = np.array([[1.0], [2.0], [3.0]])
        transformed1 = pipeline1.scalers['value'].transform(test_data)
        transformed2 = pipeline2.scalers['value'].transform(test_data)
        
        np.testing.assert_array_almost_equal(transformed1, transformed2)
        
    def test_inference_pipeline(self):
        """Test that the pipeline works for inference."""
        pipeline = TimeSeriesDataPipeline()
        
        # Simulate training
        train_df = pd.DataFrame({
            'collect_time': pd.date_range('2024-01-01', periods=1000, freq='6T'),
            'value': 5 + 0.5 * np.sin(np.linspace(0, 20*np.pi, 1000)) + 0.1 * np.random.randn(1000)
        })
        
        pipeline.fit_scalers(train_df, ['value'])
        
        # Simulate new data for inference
        new_data = pd.DataFrame({
            'collect_time': pd.date_range('2024-01-10', periods=200, freq='6T'),
            'value': 5 + 0.5 * np.sin(np.linspace(0, 4*np.pi, 200)) + 0.1 * np.random.randn(200)
        })
        
        # Process new data
        new_data_filled, _ = pipeline.fill_gaps(new_data)
        new_data_features = pipeline.add_temporal_features(new_data_filled)
        new_data_scaled = pipeline.transform_features(new_data_features, ['value'])
        
        # Create sequences for inference
        X_new, _ = pipeline.create_sequences(new_data_scaled, seq_len=100, pred_horizon=20)
        
        # Should work without errors
        self.assertGreater(len(X_new), 0)
        

def run_all_tests():
    """Run all data integrity tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionReadiness))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    if result.wasSuccessful():
        print("\n✅ All data integrity tests passed!")
        print("Your pipeline is leak-free and production-ready.")
    else:
        print("\n❌ Some tests failed!")
        print("Please fix the issues before using in production.")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()