"""
主程序入口
整合数据获取、模型训练和预测
"""
import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import HuobiAPI
from data_cache import DataCache
from data_processor import KlineProcessor, create_data_loaders, get_recent_window_data
from lnn_model import LNNPredictor, LNNTrainer


# 配置日志
Path('logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """交易系统主类"""
    
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        window_days: int = 60,
        prediction_minutes: int = 10,
        model_dir: str = "models",
        data_cache_dir: str = "data/cache"
    ):
        """
        初始化交易系统
        
        Args:
            symbol: 交易对
            window_days: 数据窗口天数
            prediction_minutes: 预测时间（分钟）
            model_dir: 模型保存目录
            data_cache_dir: 数据缓存目录
        """
        self.symbol = symbol
        self.window_days = window_days
        self.prediction_minutes = prediction_minutes
        
        # 创建目录
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        # 火币K线数据API为公开接口，不需要API Key认证
        self.api = HuobiAPI()
        self.cache = DataCache(cache_dir=data_cache_dir)
        
        # 数据处理器
        # 60天 * 24小时 * 60分钟 = 86400分钟
        self.processor = KlineProcessor(
            window_size=window_days * 24 * 60,
            prediction_horizon=prediction_minutes
        )
        
        # 模型
        self.model: Optional[LNNPredictor] = None
        self.trainer: Optional[LNNTrainer] = None
        
        logger.info(f"交易系统初始化完成: {symbol}, {window_days}天窗口, {prediction_minutes}分钟预测")
    
    def fetch_data(self) -> list:
        """获取数据（带缓存）"""
        logger.info(f"获取 {self.symbol} 数据...")
        
        # 使用缓存获取数据
        data = self.cache.get_data_with_cache(
            symbol=self.symbol,
            period="1min",
            start_time=datetime.now() - __import__('datetime').timedelta(days=self.window_days + 1),
            end_time=datetime.now(),
            fetch_func=self.api.get_klines_in_range
        )
        
        logger.info(f"获取到 {len(data)} 条数据")
        return data
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 10
    ):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            patience: 早停耐心值
        """
        # 获取数据
        klines = self.fetch_data()
        
        if len(klines) < self.processor.window_size + self.processor.prediction_horizon:
            raise ValueError("数据不足，无法训练")
        
        # 准备数据
        logger.info("准备训练数据...")
        X, y, stats = self.processor.prepare_data(klines, normalize=True)
        
        # 保存统计信息
        stats_path = self.model_dir / "normalization_stats.json"
        with open(stats_path, 'w') as f:
            # 转换numpy类型为Python类型
            stats_serializable = {}
            for k, v in stats.items():
                stats_serializable[k] = {kk: float(vv) if hasattr(vv, 'item') else vv 
                                         for kk, vv in v.items()}
            json.dump(stats_serializable, f, indent=2)
        logger.info(f"统计信息已保存到 {stats_path}")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, batch_size=batch_size
        )
        
        # 创建模型
        num_features = X.shape[2]
        self.model = LNNPredictor(
            input_features=num_features,
            seq_length=self.processor.window_size,
            hidden_size=128,
            num_layers=3,
            num_classes=2,
            dropout=0.3
        )
        
        # 创建训练器
        self.trainer = LNNTrainer(
            model=self.model,
            learning_rate=learning_rate
        )
        
        # 训练
        logger.info("开始训练...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_losses = []
            train_accs = []
            
            for batch_x, batch_y in train_loader:
                loss, acc = self.trainer.train_step(batch_x, batch_y)
                train_losses.append(loss)
                train_accs.append(acc)
            
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)
            
            # 验证
            val_losses = []
            val_accs = []
            
            for batch_x, batch_y in val_loader:
                loss, acc = self.trainer.validate(batch_x, batch_y)
                val_losses.append(loss)
                val_accs.append(acc)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)
            
            # 学习率调度
            self.trainer.scheduler.step(avg_val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}"
            )
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                model_path = self.model_dir / "best_model.pth"
                self.trainer.save_model(str(model_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停于 epoch {epoch+1}")
                    break
        
        # 测试
        logger.info("测试模型...")
        test_losses = []
        test_accs = []
        
        for batch_x, batch_y in test_loader:
            loss, acc = self.trainer.validate(batch_x, batch_y)
            test_losses.append(loss)
            test_accs.append(acc)
        
        logger.info(
            f"测试结果 - Loss: {np.mean(test_losses):.4f}, Acc: {np.mean(test_accs):.4f}"
        )
    
    def predict(self) -> dict:
        """
        预测
        
        Returns:
            预测结果
        """
        # 加载模型
        if self.model is None:
            model_path = self.model_dir / "best_model.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 需要先获取数据来确定特征数
            klines = self.fetch_data()
            X_recent = get_recent_window_data(klines, self.processor.window_size, self.processor)
            num_features = X_recent.shape[2]
            
            self.model = LNNPredictor(
                input_features=num_features,
                seq_length=self.processor.window_size,
                hidden_size=128,
                num_layers=3,
                num_classes=2,
                dropout=0.3
            )
            
            self.trainer = LNNTrainer(model=self.model)
            self.trainer.load_model(str(model_path))
        
        # 获取最新数据
        klines = self.fetch_data()
        X_recent = get_recent_window_data(klines, self.processor.window_size, self.processor)
        
        # 预测
        X_tensor = torch.FloatTensor(X_recent)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
        
        # 当前价格
        current_price = klines[-1]['close']
        
        result = {
            'symbol': self.symbol,
            'current_price': float(current_price),
            'prediction': '涨' if pred.item() == 1 else '跌',
            'up_probability': float(probs[0][1]),
            'down_probability': float(probs[0][0]),
            'prediction_horizon_minutes': self.prediction_minutes,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"预测结果: {result}")
        
        # 保存预测结果
        pred_path = self.log_dir / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(pred_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='液态神经网络交易预测系统')
    parser.add_argument('--symbol', type=str, default='BTC-USDT', help='交易对')
    parser.add_argument('--days', type=int, default=60, help='训练数据天数')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='运行模式')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    
    args = parser.parse_args()
    
    # 创建系统
    system = TradingSystem(
        symbol=args.symbol,
        window_days=args.days
    )
    
    if args.mode == 'train':
        system.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    elif args.mode == 'predict':
        result = system.predict()
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()