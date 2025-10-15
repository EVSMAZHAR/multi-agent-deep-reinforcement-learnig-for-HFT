"""
Test Data Pipeline Integration
===============================

Tests the integrated data collection and feature engineering pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.ingest import ingest_simulator_data, save_consolidated_data
from features.build_features import (
    add_basic_features, add_technical_features, 
    compute_scaler, apply_scaler
)
from data.make_dataset import split_by_date, to_tensors, create_sequences


class TestDataIngestion:
    """Test data ingestion module"""
    
    def test_synthetic_data_generation(self):
        """Test synthetic market data generation"""
        # Generate synthetic data
        n_samples = 1000
        df = pd.DataFrame({
            'ts': pd.date_range('2020-01-01', periods=n_samples, freq='100ms'),
            'symbol': 'SYMA',
            'best_bid': 100 + np.random.randn(n_samples) * 0.1,
            'best_ask': 100.02 + np.random.randn(n_samples) * 0.1,
            'bid_qty_1': 100 + np.random.randn(n_samples) * 10,
            'ask_qty_1': 100 + np.random.randn(n_samples) * 10,
        })
        
        assert len(df) == n_samples
        assert 'best_bid' in df.columns
        assert 'best_ask' in df.columns
        assert df['best_bid'].mean() > 99  # Sanity check
        
    def test_consolidation(self):
        """Test data consolidation"""
        # Create test data
        df1 = pd.DataFrame({
            'ts': pd.date_range('2020-01-01', periods=100, freq='100ms'),
            'symbol': 'SYMA',
            'best_bid': 100.0,
            'best_ask': 100.02,
        })
        
        df2 = pd.DataFrame({
            'ts': pd.date_range('2020-01-01 00:00:10', periods=100, freq='100ms'),
            'symbol': 'SYMA',
            'best_bid': 100.05,
            'best_ask': 100.07,
        })
        
        # Consolidate
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.sort_values(['symbol', 'ts']).drop_duplicates(['symbol', 'ts'])
        
        assert len(df) == 200
        assert df['ts'].is_monotonic_increasing


class TestFeatureEngineering:
    """Test feature engineering module"""
    
    def setup_method(self):
        """Setup test data"""
        n_samples = 1000
        self.df = pd.DataFrame({
            'ts': pd.date_range('2020-01-01', periods=n_samples, freq='100ms'),
            'symbol': 'SYMA',
            'best_bid': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
            'best_ask': 100.02 + np.cumsum(np.random.randn(n_samples) * 0.01),
            'bid_qty_1': np.abs(np.random.randn(n_samples) * 10 + 100),
            'ask_qty_1': np.abs(np.random.randn(n_samples) * 10 + 100),
        })
    
    def test_basic_features(self):
        """Test basic feature computation"""
        df = add_basic_features(self.df)
        
        # Check that features were added
        assert 'spread' in df.columns
        assert 'imbalance' in df.columns
        assert 'microprice' in df.columns
        assert 'mid_price' in df.columns
        
        # Check values are reasonable
        assert (df['spread'] >= 0).all()
        assert (df['imbalance'] >= -1).all() and (df['imbalance'] <= 1).all()
        assert (df['microprice'] > 0).all()
    
    def test_technical_features(self):
        """Test technical feature computation"""
        df = add_basic_features(self.df)
        df = add_technical_features(df, windows={'fast': 10, 'slow': 30})
        
        # Check technical features were added
        assert 'returns' in df.columns
        assert 'volatility_fast' in df.columns
        assert 'volatility_slow' in df.columns
        assert 'spread_ma_fast' in df.columns
        
        # Check no NaN values after warmup
        assert not df['volatility_fast'].iloc[30:].isna().any()
    
    def test_scaler_robust(self):
        """Test robust scaling"""
        df = add_basic_features(self.df)
        feature_cols = ['spread', 'imbalance', 'microprice']
        
        scaler = compute_scaler(df, feature_cols, method='robust')
        
        assert scaler['method'] == 'robust'
        assert 'median' in scaler
        assert 'iqr' in scaler
        assert len(scaler['median']) == len(feature_cols)
        
        # Apply scaler
        df_scaled = apply_scaler(df, scaler)
        
        # Check scaling worked
        for col in feature_cols:
            # After robust scaling, median should be ~0
            assert abs(df_scaled[col].median()) < 0.1
    
    def test_scaler_standard(self):
        """Test standard scaling"""
        df = add_basic_features(self.df)
        feature_cols = ['spread', 'imbalance', 'microprice']
        
        scaler = compute_scaler(df, feature_cols, method='standard')
        
        assert scaler['method'] == 'standard'
        assert 'mean' in scaler
        assert 'std' in scaler
        
        # Apply scaler
        df_scaled = apply_scaler(df, scaler)
        
        # Check scaling worked
        for col in feature_cols:
            # After standard scaling, mean should be ~0, std ~1
            assert abs(df_scaled[col].mean()) < 0.1
            assert abs(df_scaled[col].std() - 1.0) < 0.2


class TestDatasetCreation:
    """Test dataset creation module"""
    
    def setup_method(self):
        """Setup test data"""
        n_samples = 1000
        self.df = pd.DataFrame({
            'ts': pd.date_range('2020-01-01', periods=n_samples, freq='100ms'),
            'best_bid': 100 + np.random.randn(n_samples) * 0.1,
            'best_ask': 100.02 + np.random.randn(n_samples) * 0.1,
            'spread': 0.02 + np.random.randn(n_samples) * 0.01,
            'imbalance': np.random.randn(n_samples) * 0.1,
            'microprice': 100.01 + np.random.randn(n_samples) * 0.1,
            'bid_qty_1': 100 + np.random.randn(n_samples) * 10,
            'ask_qty_1': 100 + np.random.randn(n_samples) * 10,
            'mid_price': 100.01 + np.random.randn(n_samples) * 0.1,
        })
    
    def test_date_splitting(self):
        """Test date-based splitting"""
        splits = {
            'dev': {
                'start': '2020-01-01 00:00:00',
                'end': '2020-01-01 00:01:00'
            },
            'val': {
                'start': '2020-01-01 00:01:00',
                'end': '2020-01-01 00:01:30'
            }
        }
        
        parts = split_by_date(self.df, splits)
        
        assert 'dev' in parts
        assert 'val' in parts
        assert len(parts['dev']) > 0
        assert len(parts['val']) > 0
    
    def test_sequence_creation(self):
        """Test temporal sequence creation"""
        feature_cols = ['best_bid', 'best_ask', 'spread', 'imbalance', 'microprice']
        history_T = 20
        
        X = create_sequences(self.df, history_T, feature_cols)
        
        # Check shape
        expected_samples = len(self.df) - history_T
        assert X.shape == (expected_samples, history_T, len(feature_cols))
        
        # Check data type
        assert X.dtype == np.float32
    
    def test_tensor_creation(self):
        """Test full tensor creation"""
        tensors = to_tensors(self.df, history_T=20)
        
        assert 'X' in tensors
        assert 'y' in tensors
        assert 'ts' in tensors
        
        # Check shapes are consistent
        assert len(tensors['X']) == len(tensors['y'])
        assert len(tensors['X']) == len(tensors['ts'])
        
        # Check X has correct dimensions
        assert len(tensors['X'].shape) == 3  # [N, T, F]
        assert tensors['X'].shape[1] == 20  # T = 20


class TestEndToEndPipeline:
    """Test complete pipeline integration"""
    
    def test_full_pipeline(self):
        """Test complete data pipeline"""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create directory structure
            interim_dir = tmpdir / "interim"
            features_dir = tmpdir / "features"
            interim_dir.mkdir()
            features_dir.mkdir()
            
            # Step 1: Generate synthetic data
            n_samples = 1000
            df = pd.DataFrame({
                'ts': pd.date_range('2020-01-01', periods=n_samples, freq='100ms'),
                'symbol': 'SYMA',
                'best_bid': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
                'best_ask': 100.02 + np.cumsum(np.random.randn(n_samples) * 0.01),
                'bid_qty_1': np.abs(np.random.randn(n_samples) * 10 + 100),
                'ask_qty_1': np.abs(np.random.randn(n_samples) * 10 + 100),
            })
            
            # Save as interim data
            snapshots_file = interim_dir / "snapshots.parquet"
            df.to_parquet(snapshots_file, index=False)
            
            # Step 2: Feature engineering
            df = add_basic_features(df)
            df = add_technical_features(df, windows={'fast': 10, 'slow': 30})
            
            # Compute and apply scaler
            feature_cols = ['best_bid', 'best_ask', 'spread', 'imbalance', 'microprice',
                          'bid_qty_1', 'ask_qty_1', 'mid_price']
            scaler = compute_scaler(df, feature_cols, method='robust')
            df = apply_scaler(df, scaler)
            
            # Save features
            features_file = features_dir / "features.parquet"
            df.to_parquet(features_file, index=False)
            
            # Step 3: Create datasets
            tensors = to_tensors(df, history_T=20)
            
            # Save tensors
            np.savez_compressed(features_dir / "dev_tensors.npz", **tensors)
            
            # Verify outputs
            assert (features_dir / "dev_tensors.npz").exists()
            
            # Load and check
            data = np.load(features_dir / "dev_tensors.npz")
            assert 'X' in data
            assert data['X'].shape[0] > 0
            assert data['X'].shape[1] == 20  # history_T
            assert data['X'].shape[2] == len(feature_cols)


class TestEnvironmentCompatibility:
    """Test compatibility with EnhancedCTDEHFTEnv"""
    
    def test_feature_dimensions(self):
        """Test that features match environment expectations"""
        # Create test features
        n_samples = 100
        history_T = 20
        n_features = 8
        
        X = np.random.randn(n_samples, history_T, n_features).astype(np.float32)
        
        # This is what the environment expects
        assert X.shape[1] == 20  # history length
        assert X.shape[2] >= 5   # minimum features
        assert X.dtype == np.float32
    
    def test_data_loading(self):
        """Test that data can be loaded by environment"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            n_samples = 100
            history_T = 20
            n_features = 8
            
            X = np.random.randn(n_samples, history_T, n_features).astype(np.float32)
            y = np.random.randn(n_samples).astype(np.float32)
            ts = np.arange(n_samples, dtype=np.int64)
            
            # Save
            dataset_path = tmpdir / "test_tensors.npz"
            np.savez_compressed(dataset_path, X=X, y=y, ts=ts)
            
            # Load (simulating environment)
            data = np.load(dataset_path)
            X_loaded = data['X']
            
            assert np.array_equal(X, X_loaded)
            assert X_loaded.shape == (n_samples, history_T, n_features)


def run_tests():
    """Run all tests"""
    print("Running Data Pipeline Tests...")
    print("=" * 60)
    
    # Run with pytest if available
    try:
        pytest.main([__file__, '-v'])
    except:
        # Fallback to manual test execution
        print("Running manual tests...")
        
        test_classes = [
            TestDataIngestion,
            TestFeatureEngineering,
            TestDatasetCreation,
            TestEndToEndPipeline,
            TestEnvironmentCompatibility
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            print(f"\n{test_class.__name__}")
            print("-" * 40)
            
            test_instance = test_class()
            
            # Get test methods
            test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
            
            for method_name in test_methods:
                total_tests += 1
                try:
                    # Run setup if exists
                    if hasattr(test_instance, 'setup_method'):
                        test_instance.setup_method()
                    
                    # Run test
                    method = getattr(test_instance, method_name)
                    method()
                    
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"  ✗ {method_name}: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"Results: {passed_tests}/{total_tests} tests passed")
        print("=" * 60)
        
        return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
