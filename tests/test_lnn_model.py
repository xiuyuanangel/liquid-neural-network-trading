"""
测试LNN模型
"""
import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lnn_model import LiquidNeuron, LiquidNeuralNetwork, LNNPredictor, LNNTrainer


class TestLiquidNeuron:
    """测试液态神经元"""
    
    def test_init(self):
        """测试初始化"""
        neuron = LiquidNeuron(input_size=10, hidden_size=20)
        
        assert neuron.input_size == 10
        assert neuron.hidden_size == 20
    
    def test_forward(self):
        """测试前向传播"""
        neuron = LiquidNeuron(input_size=10, hidden_size=20)
        
        batch_size = 4
        x = torch.randn(batch_size, 10)
        h = torch.zeros(batch_size, 20)
        
        h_new = neuron(x, h)
        
        assert h_new.shape == (batch_size, 20)


class TestLiquidNeuralNetwork:
    """测试液态神经网络"""
    
    def test_init(self):
        """测试初始化"""
        lnn = LiquidNeuralNetwork(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=2
        )
        
        assert lnn.input_size == 10
        assert lnn.hidden_size == 32
        assert lnn.num_layers == 2
        assert len(lnn.liquid_layers) == 2
    
    def test_forward(self):
        """测试前向传播"""
        lnn = LiquidNeuralNetwork(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            output_size=2
        )
        
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 10)
        
        output, hidden = lnn(x)
        
        assert output.shape == (batch_size, 2)
        assert len(hidden) == 2
        assert hidden[0].shape == (batch_size, 32)
    
    def test_init_hidden(self):
        """测试隐藏状态初始化"""
        lnn = LiquidNeuralNetwork(
            input_size=10,
            hidden_size=32,
            num_layers=2
        )
        
        batch_size = 4
        device = torch.device('cpu')
        
        hidden = lnn.init_hidden(batch_size, device)
        
        assert len(hidden) == 2
        assert hidden[0].shape == (batch_size, 32)
        assert hidden[1].shape == (batch_size, 32)


class TestLNNPredictor:
    """测试LNN预测器"""
    
    def test_init(self):
        """测试初始化"""
        model = LNNPredictor(
            input_features=5,
            seq_length=60,
            hidden_size=64,
            num_layers=2
        )
        
        assert model.seq_length == 60
    
    def test_forward(self):
        """测试前向传播"""
        model = LNNPredictor(
            input_features=5,
            seq_length=60,
            hidden_size=64,
            num_layers=2
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 60, 5)
        
        logits = model(x)
        
        assert logits.shape == (batch_size, 2)
    
    def test_predict_proba(self):
        """测试概率预测"""
        model = LNNPredictor(
            input_features=5,
            seq_length=60,
            hidden_size=64
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 60, 5)
        
        probs = model.predict_proba(x)
        
        assert probs.shape == (batch_size, 2)
        # 检查概率和为1
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    def test_predict(self):
        """测试类别预测"""
        model = LNNPredictor(
            input_features=5,
            seq_length=60,
            hidden_size=64
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 60, 5)
        
        preds = model.predict(x)
        
        assert preds.shape == (batch_size,)
        assert torch.all((preds == 0) | (preds == 1))


class TestLNNTrainer:
    """测试LNN训练器"""
    
    def test_init(self):
        """测试初始化"""
        model = LNNPredictor(input_features=5, seq_length=60)
        trainer = LNNTrainer(model=model, learning_rate=1e-3)
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
    
    def test_train_step(self):
        """测试训练步骤"""
        model = LNNPredictor(input_features=5, seq_length=60)
        trainer = LNNTrainer(model=model, learning_rate=1e-3)
        
        batch_size = 4
        x = torch.randn(batch_size, 60, 5)
        y = torch.randint(0, 2, (batch_size,))
        
        loss, acc = trainer.train_step(x, y)
        
        assert isinstance(loss, float)
        assert 0 <= acc <= 1
    
    def test_validate(self):
        """测试验证"""
        model = LNNPredictor(input_features=5, seq_length=60)
        trainer = LNNTrainer(model=model)
        
        batch_size = 4
        x = torch.randn(batch_size, 60, 5)
        y = torch.randint(0, 2, (batch_size,))
        
        loss, acc = trainer.validate(x, y)
        
        assert isinstance(loss, float)
        assert 0 <= acc <= 1