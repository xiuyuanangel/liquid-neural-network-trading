"""
测试数据处理器
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processor import KlineProcessor, TradingDataset, create_data_loaders


class TestKlineProcessor:
    """测试K线处理器"""
    
    def test_klines_to_dataframe(self):
        """测试K线转DataFrame"""
        processor = KlineProcessor()
        
        klines = [
            {'id': 1704067200, 'open': '100', 'high': '105', 'low': '98', 'close': '102', 'vol': '1000'},
            {'id': 1704067260, 'open': '102', 'high': '107', 'low': '101', 'close': '105', 'vol': '1200'}
        ]
        
        df = processor.klines_to_dataframe(klines)
        
        assert len(df) == 2
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'close' in df.columns
        assert df['close'].dtype == np.float64
    
    def test_calculate_technical_indicators(self):
        """测试技术指标计算"""
        processor = KlineProcessor()
        
        # 生成测试数据
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 101,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        df_with_indicators = processor.calculate_technical_indicators(df)
        
        # 检查是否添加了指标
        assert 'returns' in df_with_indicators.columns
        assert 'sma_5' in df_with_indicators.columns
        assert 'rsi' in df_with_indicators.columns
        assert 'macd' in df_with_indicators.columns
        assert 'bb_upper' in df_with_indicators.columns
    
    def test_create_training_samples(self):
        """测试创建训练样本"""
        processor = KlineProcessor(window_size=10, prediction_horizon=5)
        
        # 生成测试数据
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        df = pd.DataFrame({
            'open': np.linspace(100, 200, 100),
            'high': np.linspace(102, 202, 100),
            'low': np.linspace(98, 198, 100),
            'close': np.linspace(101, 201, 100),
            'volume': np.ones(100) * 1000
        }, index=dates)
        
        X, y = processor.create_training_samples(df)
        
        # 检查形状
        assert X.shape[0] == len(y)
        assert X.shape[1] == 10  # window_size
        assert X.shape[2] == 5   # feature_columns
        
        # 检查标签
        assert set(np.unique(y)).issubset({0, 1})
    
    def test_normalize_features(self):
        """测试特征归一化"""
        processor = KlineProcessor()
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        normalized, stats = processor.normalize_features(df, method='zscore')
        
        # 检查归一化结果
        assert np.abs(normalized['feature1'].mean()) < 0.1
        assert np.abs(normalized['feature1'].std() - 1.0) < 0.1
        
        # 检查统计信息
        assert 'feature1' in stats
        assert 'mean' in stats['feature1']
        assert 'std' in stats['feature1']


class TestTradingDataset:
    """测试交易数据集"""
    
    def test_dataset_length(self):
        """测试数据集长度"""
        X = np.random.randn(100, 10, 5)
        y = np.random.randint(0, 2, 100)
        
        dataset = TradingDataset(X, y)
        
        assert len(dataset) == 100
    
    def test_dataset_getitem(self):
        """测试数据集获取项"""
        X = np.random.randn(100, 10, 5)
        y = np.random.randint(0, 2, 100)
        
        dataset = TradingDataset(X, y)
        
        x_item, y_item = dataset[0]
        
        assert x_item.shape == (10, 5)
        assert y_item.item() in [0, 1]
    
    def test_create_data_loaders(self):
        """测试创建数据加载器"""
        X = np.random.randn(1000, 10, 5)
        y = np.random.randint(0, 2, 1000)
        
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, train_ratio=0.8, val_ratio=0.1, batch_size=32
        )
        
        # 检查批次
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] <= 32
        assert batch_x.shape[1:] == (10, 5)