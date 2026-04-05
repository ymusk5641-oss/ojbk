import os
import sys
import threading
import datetime
import re
import csv
import traceback
from functools import wraps
from config import CFG,BASE_DIR 

import queue

class HunterShield:
    """
    [修复版] 终极安全护盾
    修复：引入 queue 队列单线程消费，彻底杜绝瞬间并发异常引发的底层线程爆炸 (OOM)。
    """
    _lock = threading.Lock()
    _dump_path = os.path.join(BASE_DIR, "hunter_crash_dump.txt")
    
    # 新增队列与状态锁
    _log_queue = queue.Queue()
    _worker_started = False

    @classmethod
    def _log_worker(cls):
        """后台单线程消费者，依次处理积压异常"""
        while True:
            log_content = cls._log_queue.get()
            try:
                # 文件写入交由单线程排队执行，不再需要文件锁
                with open(cls._dump_path, 'a', encoding='utf-8') as f:
                    f.write(log_content)
            except:
                pass 
            cls._log_queue.task_done()

    @classmethod
    def record(cls, context, exc):
        """核心记录逻辑：基于队列的非阻塞收集"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=4))
        log_content = f"[{timestamp}] 💥 {context}\n{tb_str}{'-'*50}\n"
        
        # 仅将数据塞入队列，极速返回，绝对不阻塞
        cls._log_queue.put(log_content)
        
        # 懒加载启动消费守护线程（全局仅一个）
        if not cls._worker_started:
            with cls._lock:
                if not cls._worker_started:
                    threading.Thread(target=cls._log_worker, daemon=True).start()
                    cls._worker_started = True


# ==========================================
# 武器 1：函数装饰器 (用于替换整个函数的 try-except)
# ==========================================
def safe_catch(context="Unknown", default_ret=None):
    """
    用法: 
    @safe_catch(context="网络请求", default_ret=None)
    def get_fresh(self, url): ...
    """
    def decorator(func):
        @wraps(func) # 必须保留原函数的元信息，防止并发池取不到函数名
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                HunterShield.record(f"{context} -> {func.__name__}", e)
                return default_ret
        return wrapper
    return decorator

# ==========================================
# 武器 2：上下文管理器 (用于替换代码块内的 try-except)
# ==========================================
class catch_silently:
    """
    用法:
    with catch_silently(context="资金流转换"):
        flow_val = float(f62)
    """
    def __init__(self, context="Unknown"):
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            HunterShield.record(self.context, exc_val)
        return True # 返回 True 代表异常已被内部消化，不要再向外抛出了


class LoggerSystem:
    """
    [核心组件] 全维测日志系统 (V5.7 - Schema驱动版)
    升级记录:
    1. [架构升级] 接入 CORE_FEATURE_SCHEMA，实现因子落盘的自动化与标准化。
    2. [安全] 采用延迟导入解决 Circular Import 问题。
    3. [格式] 统一保留 4 位小数，兼顾精度与存储体积。
    """
    def __init__(self):
        # 定义各类日志文件的绝对路径
        self.summary_file = os.path.join(BASE_DIR, "hunter_summary.txt")
        self.debug_file = os.path.join(BASE_DIR, "hunter_debug.log")
        self.trace_file = os.path.join(BASE_DIR, "hunter_ai_trace.log")
        self.factor_file = os.path.join(BASE_DIR, "hunter_factors.csv")
        self.journal_file = os.path.join(BASE_DIR, "hunter_journal.csv")
        self.mining_file = os.path.join(BASE_DIR, "hunter_mining.log") 
        
        self.lock = threading.Lock()
        
        # 确保日志目录存在
        if not os.path.exists(BASE_DIR):
            try: os.makedirs(BASE_DIR)
            except: pass

        # 日志轮转机制
        try:
            if os.path.exists(self.debug_file) and os.path.getsize(self.debug_file) > 5 * 1024 * 1024:
                ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                os.rename(self.debug_file, f"{self.debug_file}.bak_{ts}")
        except: pass

        # 初始化日志文件头
        try:
            with open(self.debug_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n[BOOT V405 PRO] {datetime.datetime.now()}\n{'='*60}\n")
            
            # [核心升级] 动态生成表头，与 Schema 100% 对齐
            if not os.path.exists(self.factor_file):
                # 基础固定列
                headers = ["timestamp", "symbol", "name", "price", "final_score", "ai_score", "trend_tag"]
                # 动态扩展列 (直接读取宪法配置)
                headers.extend(CFG.CORE_FEATURE_SCHEMA)
                
                with open(self.factor_file, "w", newline='', encoding="utf-8-sig") as f:
                    csv.writer(f).writerow(headers)
            
            if not os.path.exists(self.journal_file):
                with open(self.journal_file, "w", newline='', encoding="utf-8-sig") as f:
                    csv.writer(f).writerow([
                        "Time", "Symbol", "Name", "Price", "Score", "Phase", 
                        "RSRS", "Breadth", "PE", "Bias20", "Flow", 
                        "Reason", "Status", "Duration", "Result", 
                        "TrendTag", "WinRate", "VWAP_Bias",
                        "MarketVal", "Profit", "TotalAssets"
                    ])
                    
            if not os.path.exists(self.mining_file):
                with open(self.mining_file, "w", encoding="utf-8") as f:
                    f.write(f"=== Hunter Data Mining Log Created: {datetime.datetime.now()} ===\n")

        except Exception as e:
            print(f"[Logger Init Error] {e}")

    def log_ui(self, message):
        """记录界面摘要"""
        try:
            msg_str = str(message)
            clean_msg = re.sub(r'\[/?(?:color|b|i|sup|sub).*?\]', '', msg_str)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with self.lock:
                with open(self.summary_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] {clean_msg}\n")
        except Exception as e:
            print(f"[LoggerUI Error] {e}")

    def log_debug(self, tag, content):
        """记录详细调试信息"""
        with self.lock:
            try:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                with open(self.debug_file, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] [{tag:<15}] {str(content)}\n")
            except: pass

    def log_trace(self, content):
        """记录 AI 思维链"""
        with self.lock:
            try:
                with open(self.trace_file, "a", encoding="utf-8") as f:
                    f.write(f"{content}\n")
            except: pass

    def log_factors(self, data_dict):
        """
        [标准落盘] 全维特征写入 (自动化 Schema 版)
        优势: 永远不会漏写因子，永远不会列名错位。
        """
        with self.lock:
            try:
                if not os.path.exists(self.factor_file):
                    self.__init__() 

                with open(self.factor_file, "a", newline='', encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 辅助函数：安全转浮点
                    def to_f(key, default=0.0):
                        try:
                            val = data_dict.get(key, default)
                            return float(val)
                        except:
                            return float(default)

                    # 1. 组装基础元数据 (固定顺序)
                    row = [
                        ts, 
                        data_dict.get('symbol', ''), 
                        data_dict.get('name', ''), 
                        to_f('price'),
                        f"{to_f('final_score'):.1f}", 
                        f"{to_f('ai_score'):.1f}", 
                        data_dict.get('trend_desc', '-')
                    ]
                    
                      # 2. 动态组装 41 维特征 (自动化)
                    for feat in CFG.CORE_FEATURE_SCHEMA:
                        val = to_f(feat)
                        # 统一保留 4 位小数，足以覆盖 RSRS/Amihud 等精密指标，
                        # 同时对于 Volume 这种大数也能接受 (10000.0000)
                        row.append(f"{val:.4f}")

                    writer.writerow(row)
            except Exception as e:
                try:
                    with open(self.debug_file, "a", encoding="utf-8") as f:
                        f.write(f"[LOG_ERR] Failed to write factors: {str(e)}\n")
                except: pass
                # [核心修复] 必须把异常往上抛，不能自己悄悄咽下去！
                raise Exception(f"文件写入被拒: {e}")


    def log_exception(self, context_tag, e, extra_info=""):
        """记录异常堆栈"""
        with self.lock:
            try:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                tb_str = traceback.format_exc()
                msg = (
                    f"\n{'!'*60}\n[{ts}] [CRITICAL ERROR] @ {context_tag}\n"
                    f"Message: {str(e)}\nContext: {extra_info}\n"
                    f"{'-'*20} Stack Trace {'-'*20}\n{tb_str}\n{'!'*60}\n"
                )
                with open(self.debug_file, "a", encoding="utf-8") as f:
                    f.write(msg)
                print(msg) 
            except: pass


# 实例化
RECORDER = LoggerSystem()


# ==================== 2. 北京时钟 ====================

class BeijingClock:
    """
    [基础组件] 时间管理
    负责处理时区转换（UTC -> UTC+8），判断交易日以及当前的交易阶段（盘前、盘中、盘后）。
    新增 is_market_time 用于判断实时交易活跃时段。
    """
    @staticmethod
    def now():
        """获取当前的北京时间"""
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        return utc_now.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    
    @staticmethod
    def now_str(): 
        return BeijingClock.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def is_trading_day(target_date=None): 
        """判断是否为工作日 (剔除周末与法定节假日)"""
        dt = target_date if target_date else BeijingClock.now()
        if dt.weekday() >= 5: 
            return False
            
        # [修复] 使用通用节假日规则，而非硬编码特定年份日期
        # 元旦、清明、劳动节、端午、中秋、国庆固定日期 + 动态春节
        year = dt.year
        fixed_holidays = ['0101', '0501', '0502', '0503', '0504', '0505',
                          '1001', '1002', '1003', '1004', '1005', '1006', '1007']
        # 春节(农历正月初一前后约7天，按年份估算)
        spring_festival_map = {
            2025: ['0128', '0129', '0130', '0131', '0201', '0202', '0203', '0204'],
            2026: ['0215', '0216', '0217', '0218', '0219', '0220', '0221', '0222'],
            2027: ['0205', '0206', '0207', '0208', '0209', '0210', '0211', '0212'],
            2028: ['0125', '0126', '0127', '0128', '0129', '0130', '0131', '0201'],
            2029: ['0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219'],
            2030: ['0201', '0202', '0203', '0204', '0205', '0206', '0207', '0208'],
        }
        spring_holidays = spring_festival_map.get(year, [])
        # 清明(通常4月4-6日，取常见范围)
        qingming = ['0404', '0405', '0406']
        # 端午(农历五月初五，大致6月)
        duanwu_map = {2025: '0531', 2026: '0619', 2027: '0609', 2028: '0528', 2029: '0616', 2030: '0605'}
        duanwu = [duanwu_map.get(year, '0610')]
        # 中秋(农历八月十五，大致9-10月)
        zhongqiu_map = {2025: '1006', 2026: '0925', 2027: '0915', 2028: '1003', 2029: '0922', 2030: '0912'}
        zhongqiu = [zhongqiu_map.get(year, '0917')]
        holidays = fixed_holidays + spring_holidays + qingming + duanwu + zhongqiu
        if dt.strftime("%m%d") in holidays: 
            return False
        return True


    @staticmethod
    def is_market_time():
        """
        [专项优化] 判断当前是否处于 A 股盘中交易时段 (9:15 - 15:05)
        用于决定是否强制绕过本地缓存，直接拉取最新的复权价格。
        """
        if not BeijingClock.is_trading_day(): return False
        now_time = BeijingClock.now().time()
        # 涵盖竞价到收盘后的一小段时间
        # 排除午休时段 11:30-13:00
        return (datetime.time(9, 15) <= now_time <= datetime.time(11, 30) or
                datetime.time(13, 0) <= now_time <= datetime.time(15, 5))

    @staticmethod
    def get_phase(force_phase=None):
        """
        判断当前的交易阶段：
        OPEN: 09:15 ~ 11:30
        MID:  13:00 ~ 14:30
        TAIL: 14:30 ~ 15:00
        POST: 其他时间 (盘后/休市)
        """
        if force_phase: return force_phase
        if not BeijingClock.is_trading_day(): return "POST"
        
        now_time = BeijingClock.now().time()
        if datetime.time(9, 15) <= now_time < datetime.time(11, 30): return "OPEN"
        elif datetime.time(13, 0) <= now_time < datetime.time(14, 30): return "MID"
        elif datetime.time(14, 30) <= now_time < datetime.time(15, 0): return "TAIL"
        else: return "POST"

    @staticmethod
    def get_trading_days(start_ts, end_ts):
        """计算两个时间戳之间的实际交易日天数（剔除周末与节假日）"""
        tz_beijing = datetime.timezone(datetime.timedelta(hours=8))
        start_dt = datetime.datetime.fromtimestamp(start_ts, tz=tz_beijing)
        end_dt = datetime.datetime.fromtimestamp(end_ts, tz=tz_beijing)
        days = 0
        current = start_dt
        while current < end_dt:
            if BeijingClock.is_trading_day(current):
                days += 1
            current += datetime.timedelta(days=1)
        return days

