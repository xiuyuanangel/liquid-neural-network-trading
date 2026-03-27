"""
火币API数据获取模块
支持永续合约K线数据获取

API文档: https://www.htx.com/zh-cn/opend/newApiPages/?id=8cb73746-77b5-11ed-9966-0242ac110003
说明: 获取K线数据为公开接口，不需要API Key认证
"""
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json


class HuobiAPI:
    """火币API客户端 - 获取K线数据为公开接口，无需API Key"""
    
    # API endpoints
    REST_API_URL = "https://api.hbdm.com"
    
    def __init__(self):
        """
        初始化火币API客户端
        获取K线数据为公开接口，不需要API Key认证
        """
        self.session = requests.Session()
        # 设置请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0'
        })
    
    def get_kline_data(
        self,
        symbol: str,
        period: str = "1min",
        size: Optional[int] = None,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None
    ) -> List[Dict]:
        """
        获取K线数据
        
        接口说明:
        - 公开接口，不需要API Key认证
        - 同一IP所有业务总共1秒最多800个请求
        - 参数规则: size与from&to必填其一；如果填写from，也要填写to
        - 如果size、from、to均填写，会忽略from、to参数
        
        Args:
            symbol: 合约代码或合约标识，如 "BTC-USDT"(永续), "BTC-USDT-CW"(当周)
            period: K线周期，支持 1min, 5min, 15min, 30min, 60min, 4hour, 1day, 1week, 1mon
            size: 获取数据条数，默认150，最大2000
            from_time: 开始时间戳(秒)，10位时间戳
            to_time: 结束时间戳(秒)，10位时间戳
            
        Returns:
            K线数据列表，按时间升序排列
        """
        endpoint = "/linear-swap-ex/market/history/kline"
        
        params = {
            'contract_code': symbol,
            'period': period
        }
        
        # 根据API规则设置参数
        # 如果提供了from和to，优先使用时间范围（避免size与from/to同时使用时from/to被忽略的问题）
        if from_time is not None and to_time is not None:
            params['from'] = from_time
            params['to'] = to_time
            # 不设置size，让API返回时间范围内的所有数据（最多2000条）
        elif size is not None:
            params['size'] = min(max(size, 1), 2000)  # 限制在1-2000之间
        else:
            # 默认获取150条
            params['size'] = 150
            
        url = f"{self.REST_API_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'ok':
                err_msg = data.get('err-msg', '未知错误')
                raise Exception(f"API错误: {err_msg}")
            
            result = data.get('data', [])
            # 确保按时间升序排列
            result = sorted(result, key=lambda x: x.get('id', 0))
            return result
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求失败: {str(e)}")
    
    def get_klines_in_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        period: str = "1min"
    ) -> List[Dict]:
        """
        获取指定时间范围内的所有K线数据
        自动分页获取，正确处理超过2000条数据的情况，避免重复获取
        
        策略:
        1. 使用from/to参数按时间范围获取
        2. 每次获取最多2000条
        3. 根据返回数据的最早时间，调整下一次查询的结束时间
        4. 确保不重复获取数据（下一次的to = 本次最早时间 - 1个周期）
        
        Args:
            symbol: 交易对
            start_time: 开始时间
            end_time: 结束时间
            period: K线周期
            
        Returns:
            K线数据列表，按时间升序排列，无重复
        """
        all_data = []
        
        # 根据周期确定时间间隔（秒）
        period_seconds = self._get_period_seconds(period)
        
        # 将datetime转换为秒级时间戳（API要求10位时间戳）
        # 对齐到周期边界（API要求from/to为K线周期的整数倍）
        start_ts = (int(start_time.timestamp()) // period_seconds) * period_seconds
        end_ts = (int(end_time.timestamp()) // period_seconds) * period_seconds
        
        current_to = end_ts
        
        while current_to > start_ts:
            try:
                # 使用from/to参数获取数据
                # API会返回[from, to]范围内的数据，最多2000条
                data = self.get_kline_data(
                    symbol=symbol,
                    period=period,
                    from_time=start_ts,
                    to_time=current_to
                )
                
                if not data:
                    break
                
                # 将数据添加到总列表
                all_data.extend(data)
                
                # 获取本次返回数据的最早时间戳
                earliest_ts = min(int(item['id']) for item in data)
                
                # 如果最早时间戳已经小于等于开始时间，说明已经获取完所有数据
                if earliest_ts <= start_ts:
                    break
                
                # 如果最早时间戳等于当前结束时间，说明可能只有一条数据，避免死循环
                if earliest_ts >= current_to:
                    break
                
                # 更新下一次查询的结束时间为本次最早时间的前一个周期
                # 这样可以确保不重复获取数据
                current_to = earliest_ts - period_seconds
                
                # 如果下一次查询的结束时间早于开始时间，结束循环
                if current_to <= start_ts:
                    break
                
                # 频率限制：同一IP所有业务1秒最多800个请求
                # 保守起见，每请求间隔0.05秒（每秒20个请求）
                time.sleep(0.05)
                
            except Exception as e:
                print(f"获取数据出错: {e}")
                time.sleep(1)
                continue
        
        # 去重并按时间排序
        seen_ids = set()
        unique_data = []
        for item in sorted(all_data, key=lambda x: x.get('id', 0)):
            item_id = item.get('id')
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_data.append(item)
        
        print(f"成功获取 {len(unique_data)} 条K线数据（去重后）")
        return unique_data
    
    def _get_period_seconds(self, period: str) -> int:
        """
        获取周期对应的秒数
        
        Args:
            period: K线周期
            
        Returns:
            周期秒数
        """
        period_map = {
            '1min': 60,
            '5min': 300,
            '15min': 900,
            '30min': 1800,
            '60min': 3600,
            '4hour': 14400,
            '1day': 86400,
            '1week': 604800,
            '1mon': 2592000  # 按30天计算
        }
        return period_map.get(period, 60)
    
    def get_recent_klines(
        self,
        symbol: str,
        days: int = 60,
        period: str = "1min"
    ) -> List[Dict]:
        """
        获取最近N天的K线数据
        
        Args:
            symbol: 交易对，如 "BTC-USDT"
            days: 天数
            period: K线周期
            
        Returns:
            K线数据列表，按时间升序排列
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        return self.get_klines_in_range(symbol, start_time, end_time, period)
    
    def get_contract_types(self) -> Dict[str, str]:
        """
        获取合约类型说明
        
        Returns:
            合约类型字典
        """
        return {
            '永续合约': 'BTC-USDT',
            '当周合约': 'BTC-USDT-CW',
            '次周合约': 'BTC-USDT-NW',
            '当季合约': 'BTC-USDT-CQ',
            '次季合约': 'BTC-USDT-NQ',
            '交割合约': 'BTC-USDT-210625'  # 示例：2021年6月25日交割
        }