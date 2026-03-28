"""
数据预处理模块
处理K线数据，生成训练样本
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class KlineProcessor:
    """K线数据处理器"""
    
    def __init__(
        self,
        window_size: int = 86400,  # 60天 * 24小时 * 60分钟 (假设使用1分钟数据)
        prediction_horizon: int = 10,  # 10分钟后
        feature_columns: List[str] = None
    ):
        """
        初始化数据处理器
        
        Args:
            window_size: 输入窗口大小（分钟数）
            prediction_horizon: 预测时间（分钟）
            feature_columns: 使用的特征列
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        
        if feature_columns is None:
            self.feature_columns = ['open', 'high', 'low', 'close', 'vol']
        else:
            self.feature_columns = feature_columns
    
    def klines_to_dataframe(self, klines: List[Dict]) -> pd.DataFrame:
        """
        将K线数据转换为DataFrame
        
        Args:
            klines: K线数据列表
            
        Returns:
            DataFrame
        """
        if not klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines)
        
        # 确保列名正确
        column_mapping = {
            'id': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount',
            'count': 'count'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 转换时间戳
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('datetime')
        
        # 转换数值类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 排序
        df = df.sort_index()
        
        # 删除重复项
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: K线DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        df = df.copy()
        
        # 价格变化率
        df['returns'] = df['close'].pct_change()
        
        # 简单移动平均
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # 指数移动平均
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 波动率
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 价格位置（在高低区间中的位置）
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = 'zscore'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        特征归一化
        
        Args:
            df: DataFrame
            method: 归一化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            归一化后的DataFrame和统计信息
        """
        df = df.copy()
        stats = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                stats[col] = {'mean': mean, 'std': std}
                
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val
                if range_val > 0:
                    df[col] = (df[col] - min_val) / range_val
                stats[col] = {'min': min_val, 'max': max_val}
                
            elif method == 'robust':
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr
                stats[col] = {'median': median, 'iqr': iqr}
        
        return df, stats
    
    def create_training_samples(
        self,
        df: pd.DataFrame,
        feature_columns: List[str] = None,
        stride: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建训练样本
        
        Args:
            df: 处理后的DataFrame
            feature_columns: 特征列
            
        Returns:
            X: 输入数据 [num_samples, window_size, num_features]
            y: 标签 [num_samples]
        """
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        # 确保所有特征列存在
        available_cols = [col for col in feature_columns if col in df.columns]
        if not available_cols:
            raise ValueError(f"没有可用的特征列。可用列: {df.columns.tolist()}")
        
        # 填充缺失值
        df = df.ffill().bfill().fillna(0)
        
        X = []
        y = []
        
        # 需要的数据点数量
        total_needed = self.window_size + self.prediction_horizon
        
        if len(df) < total_needed:
            raise ValueError(f"数据点不足。需要 {total_needed}，实际 {len(df)}")
        
        # 滑动窗口创建样本（stride控制步长，减少样本数降低内存）
        for i in range(0, len(df) - total_needed + 1, stride):
            # 输入窗口
            window_data = df.iloc[i:i + self.window_size][available_cols].values
            
            # 预测目标：prediction_horizon分钟后的价格变化
            current_price = df.iloc[i + self.window_size - 1]['close']
            future_price = df.iloc[i + self.window_size + self.prediction_horizon - 1]['close']
            
            # 计算涨跌 (1: 涨, 0: 跌)
            price_change = (future_price - current_price) / current_price
            label = 1 if price_change > 0 else 0
            
            X.append(window_data)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"创建训练样本: X.shape={X.shape}, y.shape={y.shape}")
        print(f"类别分布: 涨={np.sum(y==1)}, 跌={np.sum(y==0)}")
        
        return X, y
    
    def prepare_data(
        self,
        klines: List[Dict],
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        完整的数据准备流程
        
        Args:
            klines: K线数据
            normalize: 是否归一化
            
        Returns:
            X, y, 统计信息
        """
        # 转换为DataFrame
        df = self.klines_to_dataframe(klines)
        
        if len(df) < self.window_size + self.prediction_horizon:
            raise ValueError(f"数据不足。需要至少 {self.window_size + self.prediction_horizon} 条数据")
        
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        
        # 归一化
        stats = {}
        if normalize:
            df, stats = self.normalize_features(df, method='zscore')
        
        # 创建训练样本
        X, y = self.create_training_samples(df)
        
        return X, y, stats


class TradingDataset(Dataset):
    """交易数据集（使用float16降低内存占用）"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).half()
        self.y = torch.LongTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        X: 输入数据
        y: 标签
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        训练、验证、测试数据加载器
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # 划分数据集
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"数据集划分: 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
    
    # 创建数据集
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    test_dataset = TradingDataset(X_test, y_test)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_recent_window_data(
    klines: List[Dict],
    window_size: int = 86400,
    processor: Optional[KlineProcessor] = None
) -> np.ndarray:
    """
    获取最近窗口数据用于预测
    
    Args:
        klines: K线数据
        window_size: 窗口大小
        processor: 数据处理器
        
    Returns:
        输入数据 [1, window_size, num_features]
    """
    if processor is None:
        processor = KlineProcessor(window_size=window_size)
    
    # 处理数据
    df = processor.klines_to_dataframe(klines)
    df = processor.calculate_technical_indicators(df)
    df, _ = processor.normalize_features(df, method='zscore')
    
    # 获取最近window_size条数据
    recent_data = df.iloc[-window_size:]
    
    # 提取特征
    feature_columns = processor.feature_columns
    available_cols = [col for col in feature_columns if col in recent_data.columns]
    
    if len(available_cols) < len(feature_columns):
        print(f"警告: 部分特征列缺失，使用可用列: {available_cols}")
    
    # 填充缺失值
    recent_data = recent_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 转换为numpy
    X = recent_data[available_cols].values
    
    # 添加batch维度
    X = X[np.newaxis, ...]
    
    return X