"""测试脚本 - 验证修复后的分页获取功能"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_fetcher import HuobiAPI
from datetime import datetime, timedelta

api = HuobiAPI()

# 测试：获取最近2小时的1min K线（少量数据，快速验证）
print("="*60)
print("测试1: 获取最近2小时的1min K线")
print("="*60)
data = api.get_recent_klines("BTC-USDT", days=0, period="1min")
# 手动调用 get_klines_in_range 测试2小时
data = api.get_klines_in_range(
    "BTC-USDT",
    datetime.now() - timedelta(hours=2),
    datetime.now(),
    "1min"
)
print(f"获取到 {len(data)} 条数据")

# 测试：模拟实际训练场景，获取最近3天的1min K线
print("\n" + "="*60)
print("测试2: 获取最近3天的1min K线")
print("="*60)
data = api.get_klines_in_range(
    "BTC-USDT",
    datetime.now() - timedelta(days=3),
    datetime.now(),
    "1min"
)
print(f"获取到 {len(data)} 条数据")
if data:
    print(f"时间范围: {datetime.fromtimestamp(data[0]['id'])} ~ {datetime.fromtimestamp(data[-1]['id'])}")

print("\n测试完成")
