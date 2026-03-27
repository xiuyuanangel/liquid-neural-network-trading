"""
数据缓存管理模块
支持本地文件缓存，避免重复API请求
"""
import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存文件路径
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _generate_cache_key(
        self,
        symbol: str,
        period: str,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """
        生成缓存键
        
        Args:
            symbol: 交易对
            period: K线周期
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            缓存键
        """
        key_str = f"{symbol}_{period}_{start_time.isoformat()}_{end_time.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get_cached_data(
        self,
        symbol: str,
        period: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[List[Dict]]:
        """
        获取缓存数据
        
        Args:
            symbol: 交易对
            period: K线周期
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            缓存的数据列表，如果不存在或已过期则返回None
        """
        cache_key = self._generate_cache_key(symbol, period, start_time, end_time)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None
            
        # 检查缓存是否过期（默认缓存1小时）
        cache_info = self.metadata.get(cache_key, {})
        cached_time = datetime.fromisoformat(cache_info.get('cached_time', '2000-01-01'))
        
        # 对于历史数据，缓存24小时；对于最近数据，缓存1小时
        now = datetime.now()
        if end_time < now - timedelta(hours=1):
            # 历史数据，缓存24小时
            if now - cached_time > timedelta(hours=24):
                return None
        else:
            # 最近数据，缓存1小时
            if now - cached_time > timedelta(hours=1):
                return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"从缓存加载数据: {symbol} {period} ({len(data)} 条)")
            return data
        except Exception as e:
            print(f"读取缓存失败: {e}")
            return None
    
    def save_cached_data(
        self,
        symbol: str,
        period: str,
        start_time: datetime,
        end_time: datetime,
        data: List[Dict]
    ):
        """
        保存数据到缓存
        
        Args:
            symbol: 交易对
            period: K线周期
            start_time: 开始时间
            end_time: 结束时间
            data: 要缓存的数据
        """
        if not data:
            return
            
        cache_key = self._generate_cache_key(symbol, period, start_time, end_time)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            # 保存数据
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # 更新元数据
            self.metadata[cache_key] = {
                'symbol': symbol,
                'period': period,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'cached_time': datetime.now().isoformat(),
                'data_count': len(data),
                'file_size': cache_file.stat().st_size
            }
            self._save_metadata()
            
            print(f"数据已缓存: {symbol} {period} ({len(data)} 条)")
            
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def get_data_with_cache(
        self,
        symbol: str,
        period: str,
        start_time: datetime,
        end_time: datetime,
        fetch_func
    ) -> List[Dict]:
        """
        带缓存的数据获取
        
        Args:
            symbol: 交易对
            period: K线周期
            start_time: 开始时间
            end_time: 结束时间
            fetch_func: 数据获取函数
            
        Returns:
            数据列表
        """
        # 尝试从缓存获取
        cached_data = self.get_cached_data(symbol, period, start_time, end_time)
        if cached_data is not None:
            return cached_data
        
        # 从API获取
        print(f"从API获取数据: {symbol} {period}")
        data = fetch_func(symbol, start_time, end_time, period)
        
        # 保存到缓存
        if data:
            self.save_cached_data(symbol, period, start_time, end_time, data)
        
        return data
    
    def merge_with_cache(
        self,
        symbol: str,
        period: str,
        new_data: List[Dict]
    ) -> List[Dict]:
        """
        将新数据与缓存合并，去重
        
        Args:
            symbol: 交易对
            period: K线周期
            new_data: 新获取的数据
            
        Returns:
            合并后的数据列表
        """
        if not new_data:
            return []
        
        # 获取时间范围
        timestamps = [item['id'] for item in new_data]
        start_time = datetime.fromtimestamp(min(timestamps))
        end_time = datetime.fromtimestamp(max(timestamps))
        
        # 尝试获取现有缓存
        cache_key = self._generate_cache_key(symbol, period, start_time, end_time)
        cache_file = self._get_cache_file_path(cache_key)
        
        existing_data = []
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    existing_data = pickle.load(f)
            except Exception:
                pass
        
        # 合并并去重
        all_data = existing_data + new_data
        seen_ids = set()
        unique_data = []
        
        for item in sorted(all_data, key=lambda x: x['id']):
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique_data.append(item)
        
        # 更新缓存
        if unique_data:
            self.save_cached_data(symbol, period, start_time, end_time, unique_data)
        
        return unique_data
    
    def clear_expired_cache(self, max_age_hours: int = 168):
        """
        清理过期缓存
        
        Args:
            max_age_hours: 最大缓存时间（小时），默认7天
        """
        now = datetime.now()
        expired_keys = []
        
        for cache_key, info in list(self.metadata.items()):
            cached_time = datetime.fromisoformat(info.get('cached_time', '2000-01-01'))
            if now - cached_time > timedelta(hours=max_age_hours):
                cache_file = self._get_cache_file_path(cache_key)
                if cache_file.exists():
                    cache_file.unlink()
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.metadata[key]
        
        self._save_metadata()
        
        if expired_keys:
            print(f"清理了 {len(expired_keys)} 个过期缓存文件")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_size = 0
        total_files = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            total_size += cache_file.stat().st_size
            total_files += 1
        
        return {
            'total_files': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir)
        }