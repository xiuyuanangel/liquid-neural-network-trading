"""
液态神经网络(Liquid Neural Network, LNN)模型
基于常微分方程(ODE)的连续时间神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LiquidNeuron(nn.Module):
    """
    液态神经元
    实现连续时间动态系统的神经元
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        tau: float = 1.0,
        dt: float = 0.1
    ):
        """
        初始化液态神经元
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            tau: 时间常数
            dt: 时间步长
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau
        self.dt = dt
        
        # 输入权重
        self.input_weights = nn.Linear(input_size, hidden_size)
        
        # 循环权重（自连接）
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 液态时间常数（可学习参数）
        self.tau_log = nn.Parameter(torch.ones(hidden_size) * np.log(tau))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # Xavier初始化
        nn.init.xavier_uniform_(self.input_weights.weight)
        nn.init.zeros_(self.input_weights.bias)
        
        # 正交初始化循环权重
        nn.init.orthogonal_(self.recurrent_weights.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播（欧拉方法离散化）
        
        Args:
            x: 输入 [batch_size, input_size]
            h: 当前隐藏状态 [batch_size, hidden_size]
            
        Returns:
            新的隐藏状态 [batch_size, hidden_size]
        """
        # 计算输入和循环连接
        input_contrib = self.input_weights(x)
        rec_contrib = self.recurrent_weights(h)
        
        # 计算动态变化（使用tanh激活）
        dh = torch.tanh(input_contrib + rec_contrib)
        
        # 应用时间常数（可学习）
        tau = torch.exp(self.tau_log)
        
        # 欧拉离散化更新
        h_new = h + (self.dt / tau) * (dh - h)
        
        return h_new


class LiquidNeuralNetwork(nn.Module):
    """
    液态神经网络
    多层液态神经元组成的网络
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2,
        tau: float = 1.0,
        dt: float = 0.1
    ):
        """
        初始化液态神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: 液态层数
            output_size: 输出维度（分类数）
            dropout: Dropout概率
            tau: 时间常数
            dt: 时间步长
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 液态层
        self.liquid_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.liquid_layers.append(
                LiquidNeuron(layer_input_size, hidden_size, tau, dt)
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            hidden: 初始隐藏状态元组
            
        Returns:
            输出 [batch_size, output_size]
            最终隐藏状态元组
        """
        batch_size, seq_len, _ = x.shape
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = tuple(
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            )
        
        # 逐时间步处理
        current_hidden = list(hidden)
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            
            # 通过每一层液态层
            for layer_idx, layer in enumerate(self.liquid_layers):
                if layer_idx == 0:
                    layer_input = x_t
                else:
                    layer_input = current_hidden[layer_idx - 1]
                
                current_hidden[layer_idx] = layer(layer_input, current_hidden[layer_idx])
        
        # 使用最后一层的最终隐藏状态
        final_hidden = current_hidden[-1]
        
        # Dropout
        final_hidden = self.dropout(final_hidden)
        
        # 输出
        output = self.output_layer(final_hidden)
        
        return output, tuple(current_hidden)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """初始化隐藏状态"""
        return tuple(
            torch.zeros(batch_size, self.hidden_size, device=device)
            for _ in range(self.num_layers)
        )


class LNNPredictor(nn.Module):
    """
    基于LNN的价格预测模型
    """
    
    def __init__(
        self,
        input_features: int = 5,  # OHLCV
        seq_length: int = 1440,   # 60天 * 24小时 (假设使用小时数据)
        hidden_size: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,     # 涨/跌
        dropout: float = 0.3
    ):
        """
        初始化预测模型
        
        Args:
            input_features: 输入特征数（开高低收量）
            seq_length: 序列长度
            hidden_size: 隐藏层维度
            num_layers: 层数
            num_classes: 分类数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.seq_length = seq_length
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU()
        )
        
        # 液态神经网络
        self.lnn = LiquidNeuralNetwork(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,
            dropout=dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, seq_length, input_features]
            
        Returns:
            预测 logits [batch_size, num_classes]
        """
        batch_size, seq_len, features = x.shape
        
        # 特征提取
        x_flat = x.view(-1, features)
        features_extracted = self.feature_extractor(x_flat)
        features_extracted = features_extracted.view(batch_size, seq_len, -1)
        
        # LNN处理
        lnn_out, _ = self.lnn(features_extracted)
        
        # 分类
        logits = self.classifier(lnn_out)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            x: 输入
            
        Returns:
            概率分布 [batch_size, num_classes]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入
            
        Returns:
            预测类别 [batch_size]
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


class LNNTrainer:
    """LNN模型训练器"""
    
    def __init__(
        self,
        model: LNNPredictor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[float, float]:
        """
        单步训练
        
        Args:
            x: 输入
            y: 标签
            
        Returns:
            损失和准确率
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        # 前向传播
        logits = self.model(x)
        loss = self.criterion(logits, y)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 计算准确率
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == y).float().mean().item()
        
        return loss.item(), accuracy
    
    def validate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[float, float]:
        """
        验证
        
        Args:
            x: 输入
            y: 标签
            
        Returns:
            损失和准确率
        """
        self.model.eval()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == y).float().mean().item()
        
        return loss.item(), accuracy
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"模型已从 {path} 加载")