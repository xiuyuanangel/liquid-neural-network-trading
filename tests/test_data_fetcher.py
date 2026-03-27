"""
测试数据获取模块
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetcher import HuobiAPI
from data_cache import DataCache


class TestHuobiAPI:
    """测试火币API"""
    
    def test_init(self):
        """测试初始化"""
        api = HuobiAPI()
        assert api.api_key is None
        assert api.secret_key is None
        
        api2 = HuobiAPI(api_key="test_key", secret_key="test_secret")
        assert api2.api_key == "test_key"
        assert api2.secret_key == "test_secret"
    
    def test_get_kline_data_params(self):
        """测试K线数据参数构造"""
        api = HuobiAPI()
        
        # 测试基本参数
        params = {
            'contract_code': 'BTC-USDT',
            'period': '1min',
            'size': 100
        }
        
        assert params['contract_code'] == 'BTC-USDT'
        assert params['period'] == '1min'
        assert params['size'] == 100


class TestDataCache:
    """测试数据缓存"""
    
    def test_init(self, tmp_path):
        """测试初始化"""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))
        assert cache.cache_dir.exists()
    
    def test_cache_key_generation(self, tmp_path):
        """测试缓存键生成"""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        key1 = cache._generate_cache_key("BTC-USDT", "1min", start, end)
        key2 = cache._generate_cache_key("BTC-USDT", "1min", start, end)
        key3 = cache._generate_cache_key("ETH-USDT", "1min", start, end)
        
        # 相同参数应生成相同键
        assert key1 == key2
        # 不同参数应生成不同键
        assert key1 != key3
    
    def test_save_and_load_cache(self, tmp_path):
        """测试缓存保存和加载"""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))
        
        # 测试数据
        test_data = [
            {'id': 1, 'open': 100, 'close': 101},
            {'id': 2, 'open': 101, 'close': 102}
        ]
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        # 保存
        cache.save_cached_data("BTC-USDT", "1min", start, end, test_data)
        
        # 加载
        loaded = cache.get_cached_data("BTC-USDT", "1min", start, end)
        
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]['id'] == 1
    
    def test_merge_data(self, tmp_path):
        """测试数据合并去重"""
        cache = DataCache(cache_dir=str(tmp_path / "cache"))
        
        existing = [
            {'id': 1, 'open': 100},
            {'id': 2, 'open': 101}
        ]
        
        new = [
            {'id': 2, 'open': 101},  # 重复
            {'id': 3, 'open': 102}
        ]
        
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)
        
        # 先保存现有数据
        cache.save_cached_data("BTC-USDT", "1min", start, end, existing)
        
        # 合并新数据
        merged = cache.merge_with_cache("BTC-USDT", "1min", new)
        
        assert len(merged) == 3
        ids = [item['id'] for item in merged]
        assert sorted(ids) == [1, 2, 3]