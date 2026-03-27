# 液态神经网络量化交易项目

基于液态神经网络(Liquid Neural Network, LNN)的加密货币价格预测系统。

## 功能特性

- 从火币API获取永续合约K线数据（公开接口，无需API Key）
- 本地数据缓存，避免重复请求
- 液态神经网络模型架构
- 60天窗口数据输入，预测10分钟后涨跌
- GitHub Actions自动化运行

## 项目结构

```
.
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions配置
├── data/
│   └── cache/              # 本地数据缓存
├── models/                 # 训练好的模型
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py     # 数据获取模块
│   ├── data_cache.py       # 数据缓存管理
│   ├── lnn_model.py        # 液态神经网络模型
│   ├── data_processor.py   # 数据预处理
│   └── main.py             # 主程序入口
├── tests/                  # 测试文件
├── requirements.txt        # Python依赖
└── README.md              # 项目说明
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 本地运行

```bash
# 训练模式
python src/main.py --symbol BTC-USDT --days 60 --mode train

# 预测模式
python src/main.py --symbol BTC-USDT --mode predict
```

### GitHub Actions运行

1. GitHub Actions配置（无需API Key）
   - 火币K线数据API为公开接口，不需要配置API Key

2. 手动触发工作流或等待定时任务

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| symbol | 交易对 | BTC-USDT |
| days | 训练数据天数 | 60 |
| mode | 运行模式(train/predict) | train |
| window_size | 输入窗口大小(分钟) | 86400 (60天) |
| prediction_horizon | 预测时间(分钟) | 10 |

## 液态神经网络简介

液态神经网络是一种新型的神经网络架构，具有以下特点：
- 连续时间动态系统
- 可解释性强
- 参数效率高
- 适合时间序列预测

## 许可证

MIT License