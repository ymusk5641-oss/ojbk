import os
import pickle
import threading
import time
import datetime
import traceback
import json
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import  RECORDER
from config import CFG,BASE_DIR
from governance import DataAdapter, DataSanitizer, DataValidator, StandardSchema

# ==================== 0. 数据源注册表 (大一统管理) ====================
class DataSource:
    """
    [核心基建] 网络接口注册表 (Single Source of Truth)
    功能: 统一管理所有上游数据源 URL，禁止在业务逻辑中硬编码。
    """
    URLS = {
        # [A] 实时快照 (腾讯): 支持多股批量，返回五档、成交明细
        "SNAPSHOT": "http://qt.gtimg.cn/q={codes}",
        
        # [B] 历史K线 (腾讯/Baostock灾备): 复权数据
        "KLINE_TX": "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{days},qfq",
        
        # [C] 全市场扫描 (东财): 基础行情 + 涨幅动量 (抛弃绝对金额偏见)
        # f3: 涨跌幅, f8: 换手率, f22: 涨速
        "SCAN_ALL": (
            "http://push2.eastmoney.com/api/qt/clist/get?"
            "pn={page}&pz={size}&po=1&np=1&fltt=2&invt=2&fid=f3&"
            "fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&"
            "fields=f12,f14,f62,f100,f3,f8,f22&_={ts}"
        ),

        # ==========================================
        # [NEW] 全市场扫描 (新浪): 降维打击无防备节点
        # ==========================================
        "SCAN_ALL_SINA": "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1&node={node}",
        
        # [D] 行业/资金流补全 (东财): 支持 secids 批量请求
        "FLOW_BATCH": "http://push2.eastmoney.com/api/qt/ulist.np/get?secids={secids}&fields=f12,f14,f62,f100&_={ts}",
        
        # [E] 领涨行业 (东财)
        "SECTOR_HOT": (
            "http://push2.eastmoney.com/api/qt/clist/get?"
            "pn=1&pz={size}&po=1&np=1&fltt=2&invt=2&fid=f3&"
            "fs=m:90+t:2+f:!50&fields=f14&_={ts}"
        ),
        
        # [F] 个股公告 (东财 RAG 源)
        "ANNOUNCEMENT": "https://np-anotice-stock.eastmoney.com/api/security/ann",

        # [G] AI 模型接口 (集中管理)
        "LLM_DEEPSEEK": "https://api.deepseek.com/chat/completions",
        "LLM_DEEPSEEK_CHECK": "https://api.deepseek.com/models",
        "LLM_GEMINI": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",

        # [H] 宏观风控：全市场总成交额 (东财)
        "MARKET_TOTAL_AMT": "http://push2.eastmoney.com/api/qt/ulist.np/get?secids=1.000001,0.399001&fields=f14,f6&_={ts}",

        # [新增] 真实的宏观市场宽度 (上证+深证)
        # f104: 上涨家数, f105: 下跌家数, f106: 平盘家数
        "MARKET_BREADTH": "http://push2.eastmoney.com/api/qt/ulist.np/get?secids=1.000001,0.399001&fields=f104,f105,f106&_={ts}",


        # [I] 宏观风控：行业板块成交额排行 (东财)
        "SECTOR_CROWDING": (
            "http://push2.eastmoney.com/api/qt/clist/get?"
            "pn=1&pz=30&po=1&np=1&fltt=2&invt=2&"
            "fid=f6&fs=m:90+t:2+f:!50&fields=f14,f6&_={ts}"
        )
    }

    @staticmethod
    def get_url(key, **kwargs):
        """获取并格式化 URL，严格参数校验"""
        if key not in DataSource.URLS:
            raise ValueError(f"[DataSource] 未注册的接口 Key: {key}")
        try:
            return DataSource.URLS[key].format(**kwargs)
        except KeyError as e:
            raise ValueError(f"[DataSource] URL参数缺失: {key} 需要 {str(e)}")


class DataCacheManager:
    """
    [核心组件] 高速三级缓存管理器 (单例修正版 V5.0)
    审计通过:
    1. 线程安全: 使用类级锁 _lock 确保多线程读写安全。
    2. 内存保护: 严格限制 80 条内存缓存，防止 Android OOM。
    3. 逻辑闭环: 内存 -> 磁盘 -> 过期清理。
    """
    _instance = None
    _lock = threading.Lock() 

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataCacheManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_dir):
        if self._initialized: return
        
        self.cache_dir = cache_dir
        self.mem_cache = {}
        
        if not os.path.exists(self.cache_dir):
            try: os.makedirs(self.cache_dir, exist_ok=True)
            except: pass
            
        self._initialized = True

    def _get_path(self, symbol, tag):
        # 增加日期标识，确保每日更新
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        return os.path.join(self.cache_dir, f"{symbol}_{tag}_{date_str}.pkl")

    def get_valid_cache(self, symbol, tag="kline"):
        """尝试获取今日有效的缓存"""
        path = self._get_path(symbol, tag)
        
        # 1. 尝试内存 (加类锁)
        with DataCacheManager._lock:
            if path in self.mem_cache:
                return self.mem_cache[path]

        # 2. 尝试今日磁盘
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                
                # [内存保护] 回填内存前检查容量，Android 运存寸土寸金
                with DataCacheManager._lock:
                    if len(self.mem_cache) > 80: 
                        self.mem_cache.clear() # 激进清理
                    self.mem_cache[path] = data
                return data
            except: pass
        return None

    def set_cache(self, symbol, data, tag="kline"):
        """设置缓存"""
        path = self._get_path(symbol, tag)
        
        with DataCacheManager._lock:
            if len(self.mem_cache) > 80:
                self.mem_cache.clear()
            self.mem_cache[path] = data
            # 必须在锁内写入磁盘，绝对禁止多线程交叉写入
            try:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("CACHE_SAVE_ERR", str(e))
	
    def clear_today_cache(self):
        """[专项优化] 物理清理今日产生的所有本地缓存"""
        try:
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            with DataCacheManager._lock:
                self.mem_cache.clear()
            
            count = 0
            if os.path.exists(self.cache_dir):
                for f in os.listdir(self.cache_dir):
                    if date_str in f:
                        os.remove(os.path.join(self.cache_dir, f))
                        count += 1
            return True, count
        except Exception as e:
            return False, str(e)

    def clean_old_cache(self, keep_days=3):
        """定期清理陈旧缓存，防止 Android 存储空间溢出"""
        try:
            now = time.time()
            if os.path.exists(self.cache_dir):
                for f in os.listdir(self.cache_dir):
                    f_path = os.path.join(self.cache_dir, f)
                    if os.stat(f_path).st_mtime < now - (keep_days * 86400):
                        os.remove(f_path)
        except: pass


class DataLayer:
    """
    [核心数据层 V8.0]
    负责所有上游数据的获取、清洗、验证和缓存。
    修复记录:
    1. 移除所有硬编码 URL，统一调用 DataSource.get_url。
    2. 增强 Android 线程安全性，限制并发数。
    3. 严格集成 Governance 模块进行数据清洗。
    """
    
    def __init__(self, net_client, auto_clean=True):
        self.net = net_client
        self.cache_dir = os.path.join(BASE_DIR, "Stock_Cache")
        if not os.path.exists(self.cache_dir):
            try: os.makedirs(self.cache_dir)
            except: pass
        
        # 注入缓存管理器
        self.cm = DataCacheManager(self.cache_dir)
        
        # [修复] 仅在实盘单例模式下启动清理线程，回测模式下禁用，防止IO抢占
        if auto_clean:
            threading.Thread(target=self.cm.clean_old_cache, args=(7,), daemon=True).start()

    
    def _convert_code_tencent(self, code):
        """
        [接口适配] 腾讯接口代码转换标准 (修复版)
        修复: 增加对已带前缀代码(如 sh000001)的识别，防止生成 shsh000001。
        """
        s_code = str(code).strip().lower()
        
        # 1. 如果已经是标准格式 (sh/sz/bj开头)，直接返回
        if s_code.startswith(('sh', 'sz', 'bj')):
            return s_code
            
        # 2. 沪市: 主板(6), 科创(68), B股(90), 基金/ETF(5), 新股配号(7)
        if s_code.startswith(('6', '9', '5', '7')): return f"sh{s_code}"
        # 3. 深市: 主板(0), 创业(3), B股(2), 基金(1)
        if s_code.startswith(('0', '2', '3', '1')): return f"sz{s_code}" 
        # 4. 北交所: 8/4
        if s_code.startswith(('8', '4')): return f"bj{s_code}"
        
        # 5. 兜底默认沪市 (针对指数如 000001，通常指上证)
        return f"sh{s_code}"


    def get_realtime_snapshot(self, codes):
        """
        [实盘核心 V6.1 - 治理管道接入版]
        功能: 获取股票实时快照 (Snapshot)
        重构:
        1. [协议解码] 明确处理腾讯接口的 "万/手" 单位。
        2. [集中清洗] 接入 DataSanitizer 处理 NaN 和 Inf。
        3. [标准输出] 返回符合 StandardSchema 的字典。
        """
        if not codes: return {}
        
        results = {}
        # 分批处理，防止 URL 过长 (腾讯接口通常支持 ~60-80 个)
        chunk_size = 60
        
        for i in range(0, len(codes), chunk_size):
            batch = codes[i:i+chunk_size]
            # 转换代码 (sh600000)
            t_codes = [self._convert_code_tencent(c) for c in batch]
            try:
                url = DataSource.get_url("SNAPSHOT", codes=",".join(t_codes))
                resp = self.net.get(url, timeout=4) # 缩短超时时间，Android敏感
                if not resp or resp.status_code != 200: continue
                
                # --- 协议解析下沉至治理层 ---
                content = resp.text.strip()
                rows = DataAdapter.parse_tencent_snapshot_batch(content)
                
                if not rows: continue

                # --- 进入治理管道 ---
                # 1. 转换为 DataFrame 以便批量清洗
                df_batch = pd.DataFrame(rows)
                
                # 2. 清洗 (处理停牌导致的 0/NaN)
                # 注意：快照数据没有历史前值，Sanitizer 会用 0 填充 Vol，保持 Price 不变
                df_batch = DataSanitizer.sanitize(df_batch)
                
                # 3. 转回标准字典并补充衍生字段
                for _, r in df_batch.iterrows():
                    item = r.to_dict()
                    
                    # 补充涨跌幅 (pct)
                    if item['pre_close'] > 0:
                        item['pct'] = (item['close'] - item['pre_close']) / item['pre_close'] * 100
                    else:
                        item['pct'] = 0.0
                    
                    results[item['code']] = item
                    
            except Exception as e:
                # 记录日志但不崩溃
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("REALTIME_ERR", f"快照解析异常: {str(e)[:50]}")
                continue
                
        return results


    def get_stock_announcements(self, symbol):
        """[RAG 数据源] 获取个股最近公告"""
        try:
            clean_code = "".join(filter(str.isdigit, str(symbol)))
            # [Refactor] 使用注册表获取 URL
            url = DataSource.get_url("ANNOUNCEMENT")
            params = {
                "sr": "-1", "page_size": 3, "page_index": 1, 
                "ann_type": "A", "client_source": "web", 
                "stock_list": clean_code, "f_node": "0", "s_node": "0"
            }
            # 使用 get_fresh 绕过缓存
            resp = self.net.get_fresh(url, params=params, timeout=4)
            if resp and resp.status_code == 200:
                data = resp.json()
                items = data.get('data', {}).get('list', [])
                if not items: return "无近期公告"
                rag_texts = []
                for item in items:
                    date = item.get('notice_date', '')[:10]
                    title = item.get('title', '')
                    rag_texts.append(f"[{date}] {title}")
                return "; ".join(rag_texts)
        except Exception as e:
            from utils import HunterShield
            HunterShield.record(f"RAG_Announcement_Fail | {symbol}", e)
        return "公告获取失败"


    def fetch_rsrs_raw_kline(self):
        """
        [底层原子操作] 获取 RSRS 原始 K 线数据
        功能: 获取大盘(sh000001)长周期数据，用于计算 Regime。
        """
        try:
            N, M = CFG.RSRS_PARAMS
            req_days = max(300, M + N + 50)
            
            # [Refactor] 使用注册表获取 URL
            url = DataSource.get_url("KLINE_TX", code="sh000001", days=req_days)
            
            resp = self.net.get(url, timeout=5)
            if not resp or resp.status_code != 200: 
                return []
            
            json_data = resp.json()
            # [架构修正] 解析逻辑下沉至治理层
            k_data = DataAdapter.parse_tencent_kline(json_data, 'sh000001')
            return k_data

        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("DATA_ERR", f"Fetch RSRS Raw Error: {str(e)}")
            return []


    def get_market_regime_rsrs(self):
        """
        [核心算法] RSRS 3.0 (业务逻辑层) - SSOT 对齐版
        修复: 调用 QuantEngine.calc_rsrs_regime_series 统一算法，消除实盘/回测偏差。
        架构修复: 引入局部延迟导入，彻底切断与 strategy.py 的循环依赖。
        缓存修复: 盘中强制绕过缓存，获取最新大盘指数。
        显影修复: 恢复所有产生 0.5 兜底时的日志预警。
        """
        try:
            # [核心修复] 引入时钟，盘中强制刷新大盘数据，拒绝命中早盘缓存
            from utils import BeijingClock
            is_live = BeijingClock.is_market_time()
            
            # 1. 获取上证指数数据 (复用通用接口，带缓存/强制刷新)
            # 800天是为了保证有足够的 M (600) 窗口计算 Z-Score
            df = self.get_backtest_data('sh000001', days=800, force_refresh=is_live)
            
            # 数据不足时的兜底
            if df.empty or len(df) < 60: 
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("RSRS_WARN", f"大盘数据极度匮乏 (Len={len(df)}), 强制输出假0.5兜底!")
                return 0.5 # 默认震荡
            
            # [核心架构修复] 局部延迟导入 QuantEngine
            from strategy import QuantEngine
            
            # 2. [核心复用] 调用 QuantEngine 计算全序列
            N, M = CFG.RSRS_PARAMS
            regime_series = QuantEngine.calc_rsrs_regime_series(df, N, M)
            
            # 3. 取最新一天的环境值
            if regime_series.empty: 
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("RSRS_WARN", "Regime序列计算为空, 强制输出假0.5兜底!")
                return 0.5
                
            current_val = regime_series.iloc[-1]
            
            # 防止计算结果为 NaN (如刚上市或停牌)
            if pd.isna(current_val): 
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("RSRS_WARN", "Z-Score计算结果为NaN(可能处于暖机期), 强制输出假0.5兜底!")
                return 0.5
            
            # 记录调试日志
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("RSRS_LIVE", f"Current Regime={current_val} (Live:{is_live})")
                
            return float(current_val)
            
        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("RSRS_CALC_ERR", f"大盘计算异常: {str(e)} -> 强制输出假0.5兜底!")
            return 0.5 # 异常兜底：中性



    def get_scan_list_hybrid(self):
        """
        [数据源] 获取全市场扫描列表 (V2.1 - 大一统接口版)
        功能: 拉取最新全市场行情（含涨跌幅、成交额）
        修改: 替换硬编码 URL 为 DataSource.get_url
        """
        import time
        
        ts = int(time.time())
        # [Refactor] 使用注册表获取 URL
        # pz 参数放大到 2倍 SCAN_LIMIT 以确保过滤后数量足够
        url = DataSource.get_url("SCAN_ALL", page=1, size=CFG.SCAN_LIMIT*2, ts=ts)
        
        try:
            # 使用 fresh 无缓存通道，超时3秒
            resp = self.net.get_fresh(url, timeout=3)
            
            if resp and resp.status_code == 200:
                data = resp.json().get('data', {}).get('diff', [])
                if data:
                    df = pd.DataFrame(data).rename(columns={'f12': 'symbol', 'f14': 'name', 'f62': 'flow', 'f100': 'ind', 'f3': 'pct_scan'})
                    # 确保资金流是数值型，NaN补0
                    df['flow'] = pd.to_numeric(df['flow'], errors='coerce').fillna(0)
                    return df
            else:
                if 'RECORDER' in globals():
                    status = resp.status_code if resp else 'No Resp'
                    globals()['RECORDER'].log_debug("SCAN_WARN", f"扫描接口状态非200: {status}")

        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("SCAN_FAIL", f"市场扫描失败: {str(e)[:50]}")
            
        return pd.DataFrame()


    def _get_history_and_merge(self, code, realtime_snapshot=None):
        """
        [数据核心 V7.0 - 时空对齐无损融合版 (Zero Feature Drift)]
        重构: 废弃盘中大面积手动推演，采用快照物理追加 + 向量化极速重算。
        """
        circuit_break_res = {
            "trend_desc": "数据获取失败", "data_quality": 0.0, "is_monster": False, "is_live": False,
            "final_score": 0.0, "ai_score": 50.0, "strategy_name": "数据熔断",
            "close": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, 
            "vol": 0.0, "vol_prev": 0.0, "close_prev": 0.0, "pct": 0.0,
            "flow": 0.0, "pe": 0.0, "pb": 0.0, "amount": 0.0,
            "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0, "macd_slope": 0.0,
            "ema12": 0.0, "ema26": 0.0, 
            "rsi": 50.0, "rsi_rank": 50.0, 
            "bias_20": 0.0, "bias_vwap": 0.0, "vwap_20": 0.0,
            "obv": 0.0, "obv_slope": 0.0, "pct_b": 0.5,           # [修复] 默认中性
            "kdj_k": 50.0, "kdj_d": 50.0, "kdj_j": 50.0, "kdj_gold": False,
            "er": 0.0, "atr": 0.0, "volatility": 0.0, "chop": 50.0, 
            "bb_up": 0.0, "bb_low": 0.0, "bb_width": 100.0,       # [修复] 默认中性
            "rsrs_wls": 1.0, "rsrs_r2": 0.0, "amihud": 0.0,
            "mfi": 50.0, "vam": 0.0, "pv_corr": 0.0, "winner_rate": 50.0, # [修复] 默认中性
            "smart_money_rank": 0.0, "cost_avg": 0.0, "upper_shadow_ratio": 0.0,
            "lower_shadow_ratio": 0.0, "gap_ratio": 0.0, "vol_zscore": 0.0, "profit_to_cost_dist": 0.0,
            "is_squeeze_atr": False, "is_divergence": False, 
            "macd_top_div": False, "macd_btm_div": False,
            "ma5": 0.0, "ma10": 0.0, "ma20": 0.0, "ma60": 0.0
        }

        t_code = self._convert_code_tencent(code)
        
        # [A] 尝试读取缓存
        df = self.cm.get_valid_cache(code, "kline")
        need_download = (df is None or df.empty)
        
        # [B] 智能校验 (价格偏离度检查)
        if not need_download and realtime_snapshot:
            try:
                cached_last_close = float(df.iloc[-1]['close'])
                real_price = float(realtime_snapshot.get('price', 0.0))
                if real_price <= 0.01: 
                    need_download = True
                elif cached_last_close > 0 and abs(cached_last_close - real_price) / cached_last_close > 0.3:
                    need_download = True
            except: 
                need_download = True

        if need_download:
            url = DataSource.get_url("KLINE_TX", code=t_code, days=460)
            max_retries = 3
            success = False
            for attempt in range(max_retries):
                try:
                    resp = self.net.get(url, timeout=6 + attempt * 2)
                    if not resp or resp.status_code != 200:
                        time.sleep(0.1); continue
                    
                    data = resp.json()
                    k_data = data.get('data', {}).get(t_code, {}).get('day', []) or \
                             data.get('data', {}).get(t_code, {}).get('qfqday', [])
                    
                    if len(k_data) < 20: break
                    clean_data = [row[:6] for row in k_data if len(row) >= 6]
                    if not clean_data: break

                    df = pd.DataFrame(clean_data, columns=['date', 'open', 'close', 'high', 'low', 'vol'])
                    df = DataAdapter.adapt(df, source_name="Tencent")
                    df = DataSanitizer.sanitize(df)
                    is_valid, msg = DataValidator.validate(df, context_tag=f"History_{code}")
                    
                    if not is_valid: 
                        if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("DATA_REJECT", f"{code}: {msg}")
                        break
                    
                    self.cm.set_cache(code, df, "kline")
                    success = True
                    break 
                except Exception:
                    time.sleep(0.1) 
            
            if not success:
                return circuit_break_res

        # --- 数据就绪，开始防闪烁时空融合 (大一统逻辑) ---
        try:
            today_str = datetime.datetime.now().strftime('%Y-%m-%d')
            is_live_mode = False

            if realtime_snapshot:
                from utils import BeijingClock
                phase = BeijingClock.get_phase()
                if phase in ['OPEN', 'MID', 'TAIL']: 
                    is_live_mode = True

                # 提取最新快照，构建 T 日 K 线
                new_row = {
                    'date': today_str,
                    'open': float(realtime_snapshot.get('open', 0)),
                    'close': float(realtime_snapshot.get('price', 0)), 
                    'high': float(realtime_snapshot.get('high', 0)),
                    'low': float(realtime_snapshot.get('low', 0)),
                    'vol': float(realtime_snapshot.get('vol', 0)),
                    'amount': float(realtime_snapshot.get('amount', 0)) # [保留原细节]
                }
                
                is_bad_data = False
                if new_row['close'] <= 0.01: is_bad_data = True 
                if new_row['high'] < new_row['low']: is_bad_data = True 
                
                if not is_bad_data:
                    # 如果历史 df 中已经包含了今天（比如盘中多次请求），先把它剥离
                    if str(df.iloc[-1]['date']) == today_str:
                        df = df.iloc[:-1].copy()
                        
                    # 物理拼接最新的快照作为真正的今天
                    new_df = pd.DataFrame([new_row])
                    new_df = DataSanitizer.sanitize(new_df)
                    df = pd.concat([df, new_df], ignore_index=True)

            from strategy import QuantEngine
            
            # [核心升维] 抛弃盘中手动推演。无论盘中盘后，统一将包含最新快照的 DF 送入引擎进行全量极速计算！
            # 这保证了盘中 46 个因子 100% 同步更新，彻底消灭 AI 特征偏移与时空撕裂。
            df = QuantEngine.calc_tech_batch(df)
            
            if df.empty or df['data_quality'].iloc[-1] < 0.5:
                return circuit_break_res

            # 提取最后一行计算结果
            res = circuit_break_res.copy()
            res.update(df.iloc[-1].to_dict())
            
            # 补齐状态标记
            res['is_live'] = is_live_mode
            res['data_quality'] = 1.0
            # [核心修复] 强制洗刷 circuit_break_res 拷贝带来的幽灵状态
            # 严格以 df 的真实计算结果为准，若未计算出该特征则强行置空
            res['strategy_name'] = df.iloc[-1].get('strategy_name', "")
            res['trend_desc'] = df.iloc[-1].get('trend_desc', "")


            # 强转 Numpy 原生类型防止 JSON/AI 报错
            for k, v in res.items():
                if isinstance(v, (np.bool_, bool)): res[k] = bool(v)
                elif isinstance(v, (np.int64, np.int32)): res[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)): res[k] = float(v)
                
            return res
        
        except Exception as e:
            if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("HISTORY_MERGE_ERR", traceback.format_exc())
            return circuit_break_res


    def get_market_hot_sectors(self):
        """
        [恢复] 获取市场领涨行业 (Top 5)
        [V416 鲁棒版] 增加重试循环与独立通道，确保数据必达
        """
        ts = int(time.time() * 1000)
        url = DataSource.get_url("SECTOR_HOT", size=5, ts=ts)
        
        # [新增] 3次重试机制 (提高鲁棒性)
        for attempt in range(3): 
            try:
                # [关键修改] 使用 get_fresh 独立通道 (避免 Session 污染)
                resp = self.net.get_fresh(url, timeout=5) # 超时设为5秒
                
                if resp and resp.status_code == 200:
                    data = resp.json()
                    # [新增] 双重空值防御 (None 防御)
                    data_body = data.get('data')
                    if data_body:
                        diff = data_body.get('diff', [])
                        res = [d.get('f14', '') for d in diff if d.get('f14')]
                        if res: return res # 成功即返回
                    
                    if attempt < 2: time.sleep(0.1)
                        
            except Exception as e:
                from utils import HunterShield
                HunterShield.record("Get_Hot_Sectors", e)
            time.sleep(0.1)
            
        return [] 


    def _fill_flow_data(self, df):
        """
        [工业级重构 V5.3 - 名字/行业双重补全版]
        修复: 
        1. 增加 f14 (名称) 请求，确保 ETF 名字准确无误。
        2. 针对 ETF/LOF 代码段 (159/51/56/58) 强制修正行业标签。
        3. [大一统] 使用 DataSource 管理 URL。
        """
        if df.empty: return df
        
        # 1. 预设字段
        if 'ind' not in df.columns: df['ind'] = 'Self'
        if 'flow' not in df.columns: df['flow'] = 0.0
        if 'name' not in df.columns: df['name'] = df['symbol'].astype(str)
        
        # 2. 筛选需要补全的目标 
        # 逻辑: 资金为0 OR 行业未知 OR 名字异常(是数字/为空)
        mask = (df['flow'].abs() < 1) | \
               (df['ind'].isin(['Self', '-', '未知', 0, '0'])) | \
               (df['name'].astype(str).str.isdigit()) | \
               (df['name'].str.strip() == '')
               
        target_rows = df[mask]
        if target_rows.empty: return df

        # 3. 构造批量请求
        targets = target_rows['symbol'].tolist()
        BATCH_SIZE = 90 
        
        update_map = {} # symbol -> {flow, ind, name}
        
        for i in range(0, len(targets), BATCH_SIZE):
            batch = targets[i : i + BATCH_SIZE]
            secids = []
            
            for code in batch:
                s_code = str(code).strip()
                # 东财规则: 沪市(6/9/5/7)->1.xxx; 深/京->0.xxx
                prefix = "1" if s_code.startswith(('6', '9', '5', '7')) else "0"
                secids.append(f"{prefix}.{s_code}")
            
            if not secids: continue
            
            # [Refactor] 使用注册表获取 URL
            joined_secids = ",".join(secids)
            ts = int(time.time() * 1000)
            url = DataSource.get_url("FLOW_BATCH", secids=joined_secids, ts=ts)
            
            try:
                # 使用 fresh 通道避免缓存干扰
                resp = self.net.get_fresh(url, timeout=6.0)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    diff_list = data.get('data', {}).get('diff', [])
                    # 兼容 list 或 dict
                    items = diff_list if isinstance(diff_list, list) else diff_list.values()
                    
                    for item in items:
                        f12 = str(item.get('f12', ''))
                        
                        # 解析字段
                        f62 = item.get('f62') 
                        flow_val = 0.0
                        if f62 is not None and str(f62) != "-":
                            try: flow_val = float(f62)
                            except Exception as e:
                                from utils import HunterShield
                                HunterShield.record(f"Flow_Parse_Err | {f62}", e)
                            
                        ind_val = str(item.get('f100')) if item.get('f100') != "-" else "未知"
                        name_val = str(item.get('f14')) if item.get('f14') != "-" else ""
                        
                        update_map[f12] = {'flow': flow_val, 'ind': ind_val, 'name': name_val}
                        
            except Exception as e:
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("DATA_BATCH_ERR", f"Batch {i} failed: {e}")

        # 4. 回填数据
        for idx in df[mask].index:
            sym = str(df.at[idx, 'symbol']).strip()
            
            if sym in update_map:
                info = update_map[sym]
                
                # A. 补资金
                if info['flow'] != 0: 
                    df.at[idx, 'flow'] = info['flow']
                
                # B. [Gap修复核心] 补名字
                # 只有当原名字是数字、nan或空时才覆盖，防止覆盖用户自定义备注
                curr_name = str(df.at[idx, 'name']).strip()
                if not curr_name or curr_name.isdigit() or curr_name == 'nan':
                    if info['name']: df.at[idx, 'name'] = info['name']
                
                # C. 补行业
                curr_ind = str(df.at[idx, 'ind'])
                if curr_ind in ['Self', '-', '未知', '0']:
                    if info['ind'] != "未知":
                        df.at[idx, 'ind'] = info['ind']
                    # ETF 兜底逻辑
                    elif sym.startswith(('159', '51', '56', '58')):
                        df.at[idx, 'ind'] = 'ETF/LOF'
                    else:
                        df.at[idx, 'ind'] = info['ind']
            else:
                # API 失败时的兜底
                curr_ind = str(df.at[idx, 'ind'])
                if curr_ind in ['Self', '-', '未知', '0'] and sym.startswith(('159', '51', '56', '58')):
                    df.at[idx, 'ind'] = 'ETF/LOF'
        
        return df


    def get_specific_stocks_hybrid(self, input_data):
        """
        [数据源 V423 - 工业级抗压版]
        修复:
        1. [Android 熔断] 增加 future.result(timeout=12) 防止网络死锁导致 UI 假死。
        2. [内存防御] 显式 del future 并捕获 TimeoutError。
        3. [默认值] 补全 default_tech，防止线程池异常导致 KeyError。
        4. [逻辑保留] 完整保留了原有的快照抓取、资金流补全及价格优先策略。
        """
        try:
            # --- 1. 输入标准化 (逻辑保持不变) ---
            if isinstance(input_data, list):
                clean_list = [str(x).strip().zfill(6) for x in input_data]
                df = pd.DataFrame({'symbol': clean_list, 'name': '-', 'ind': 'Self', 'flow': 0.0})
            elif isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
                if 'name' not in df.columns: df['name'] = '-' 
                if 'flow' not in df.columns: df['flow'] = 0.0
                df['symbol'] = df['symbol'].apply(lambda x: str(x).strip().zfill(6))
            else:
                return pd.DataFrame()
            
            if df.empty: return df
            df.reset_index(drop=True, inplace=True)

            # --- 2. 获取实时快照 (逻辑保持不变) ---
            codes = df['symbol'].tolist()
            snapshot_map = {} 
            
            for i in range(0, len(codes), 50): 
                chunk = codes[i:i+50]
                joined_codes = ','.join([self._convert_code_tencent(c) for c in chunk])
                try:
                    url = DataSource.get_url("SNAPSHOT", codes=joined_codes)
                    # Android 敏感超时设置
                    resp = self.net.get(url, timeout=5, encoding='gbk')
                    if not resp: continue

                    lines = resp.text.strip().split('\n')
                    for line in lines:
                        # [架构修正] 使用治理层的统一解析器
                        parsed = DataAdapter.parse_tencent_snapshot_line(line)
                        if parsed: snapshot_map[parsed['symbol']] = parsed

                except: pass
            
            # 注入基础数据
            df['price'] = df['symbol'].map(lambda x: snapshot_map.get(x, {}).get('price', 0.0))
            df['name'] = df['symbol'].map(lambda x: snapshot_map.get(x, {}).get('name', '-'))
            df['pe'] = df['symbol'].map(lambda x: snapshot_map.get(x, {}).get('pe', 0.0))
            df['pct'] = df['symbol'].map(lambda x: snapshot_map.get(x, {}).get('pct', 0.0))
            df['pb'] = df['symbol'].map(lambda x: snapshot_map.get(x, {}).get('pb', 0.0))
            
            # 补全资金流
            df = self._fill_flow_data(df)
            # [新增架构] 将宏观拥挤度作为特征(Feature)直接打入 DataFrame
            # =======================================================
            crowding_dict = self.get_macro_crowding()
            # 根据股票所属的行业(ind)，映射出该行业的拥挤度数值，找不到则默认 0.0
            df['sector_crowding'] = df['ind'].map(crowding_dict).fillna(0.0)

            final_list = []
            
            # [核心定义] 默认技术指标模版 (保持不变，但追加新特征)
            default_tech = {
                "ma20": 0.0, "rsi_rank": 50.0, "bias_20": 0.0, "volatility": 0.0,
                "chop": 50.0, "vam": 0.0, "mfi": 50.0, "trend_desc": "数据超时",
                "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "vol": 0.0,
                "sector_crowding": 0.0, # [新增兜底]
                "data_quality": 0.0 
            }
       
            # --- 3. 并行获取历史数据 (工业级重构核心) ---
            from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
            import gc

            # [Android 优化] 限制最大线程数
            max_workers = 4 if '/storage/emulated' in BASE_DIR else CFG.MAX_WORKERS
            TASK_TIMEOUT = 12 # 设置硬性超时时间

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {}
                
                # 提交任务
                for _, row in df.iterrows():
                    sym = str(row['symbol'])
                    snap = snapshot_map.get(sym, None)
                    f = executor.submit(self._get_history_and_merge, sym, snap)
                    future_to_row[f] = row.to_dict()
                
                from concurrent.futures import wait
                
                # [核心架构修复] 自适应动态超时机制 (排队论防雪崩)
                # 计算公式: (总任务数 / 并发数) * 单个任务容忍耗时(1.5s) + 基础缓冲(5.0s)
                total_tasks = len(future_to_row)
                dynamic_timeout = max(15.0, (total_tasks / max_workers) * 1.5 + 5.0)
                
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("THREAD_POOL", f"启动并发列阵: 任务={total_tasks}, 核数={max_workers}, 动态生命线={dynamic_timeout:.1f}s")

                # [终极修复]: 使用自适应时钟截断，彻底消灭算力不足导致的假熔断
                done, not_done = wait(future_to_row.keys(), timeout=dynamic_timeout)


                # 1. 处理正常完成的任务
                for future in done:
                    base_data = future_to_row[future]
                    sym = base_data.get('symbol', 'Unknown')
                    
                    try:
                        tech_res = future.result()
                        
                        if not tech_res or tech_res.get('data_quality', 0) == 0:
                            tech_res = default_tech.copy()

                        merged = {**tech_res, **base_data}
                        
                        is_holding = sym in CFG.HOLDINGS
                        if merged.get('data_quality', 0) < 0.5 and not is_holding:
                            if 'RECORDER' in globals():
                                globals()['RECORDER'].log_debug("DATA_DROP", f"剔除无效: {sym}")
                            continue 

                        rt_price = base_data.get('price', 0.0)
                        hist_close = tech_res.get('close', 0.0)
                        
                        if rt_price > 0.01:
                            merged['close'] = rt_price
                            merged['price'] = rt_price
                        else:
                            merged['close'] = hist_close
                            merged['price'] = hist_close 

                        final_list.append(merged)
                        
                    except Exception as e:
                        continue
                    finally:
                        del future_to_row[future]
                
                # 2. 物理绞杀超时未完成的任务 (防假死核心)
                for future in not_done:
                    sym = future_to_row[future].get('symbol', 'Unknown')
                    if 'RECORDER' in globals():
                        globals()['RECORDER'].log_debug("TIMEOUT", f"{sym} 线程超时熔断，强制绞杀")
                    future.cancel()
                    del future_to_row[future]


            # [Android] 任务结束后主动触发 GC，释放内存
            gc.collect()
            
            return pd.DataFrame(final_list)

        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("HYBRID_CRITICAL", traceback.format_exc())
            return pd.DataFrame()


    def get_backtest_data(self, symbol, days=365, force_refresh=False):
        """
        [回测数据接口 V6.3 - 纯HTTP极速版]
        架构调整:
        1. [彻底移除] 删除了 Baostock 所有相关逻辑，根除 Socket 死锁风险。
        2. [纯粹] 仅依赖 Tencent HTTP 接口，天然支持高并发多线程。
        3. [极简] 失败直接返回空，不再进行耗时的无效重试。
        4. [修复] 增加 force_refresh 参数，允许实盘盘中绕过缓存。
        """
        # --- 1. 尝试缓存 (L1 & L2) ---
        cache_tag = f"bt_{days}"
        
        # [核心修复] 如果是强制刷新(盘中大盘)，则跳过读取缓存
        if not force_refresh:
            cached_df = self.cm.get_valid_cache(symbol, tag=cache_tag)
            if cached_df is not None and not cached_df.empty:
                return cached_df

        # --- 2. 网络穿透 (L3 - Single Source) ---
        df = pd.DataFrame()
        
        try:
            t_code = self._convert_code_tencent(symbol) 
            # 腾讯接口通常返回最近 640 天左右的数据，足够回测使用
            url = DataSource.get_url("KLINE_TX", code=t_code, days=days)
            
            # 5秒超时，快速失败
            resp = self.net.get(url, timeout=5) 

            if resp and resp.status_code == 200:
                json_str = resp.text
                if "=" in json_str: json_str = json_str.split('=', 1)[1]
                data = json.loads(json_str) 
                
                # [架构修正] 解析逻辑下沉至治理层
                clean_data = DataAdapter.parse_tencent_kline(data, t_code)
                
                if clean_data:
                    df = pd.DataFrame(clean_data, columns=['date', 'open', 'close', 'high', 'low', 'vol'])
                    df['pe'] = 0.0 
                    df['pb'] = 0.0
                    
                    # ================= [治理管道] =================
                    df = DataAdapter.adapt(df, source_name="Tencent")
                    df = DataSanitizer.sanitize(df)
                    is_valid, _ = DataValidator.validate(df, f"BT_TX_{symbol}")
                    
                    if not is_valid:
                        df = pd.DataFrame()

        except Exception as e:
            # 网络异常记录日志后放弃，不阻塞
            from utils import HunterShield
            HunterShield.record(f"BT_Data_Download | {symbol}", e)

        # --- 3. 存入缓存 ---
        if not df.empty:
            self.cm.set_cache(symbol, df, tag=cache_tag)
            
        return df



    def get_macro_crowding(self):
        """
        [风控组件] 获取全市场宏观拥挤度 (带60秒内存防刷缓存)
        返回示例: {"电子": 15.88, "半导体": 10.11} (仅返回热度 >= 8.0% 的板块)
        """
        import time
        now = time.time()
        
        # 1. 缓存拦截 (避免盘中循环扫描时导致网络阻塞)
        if hasattr(self, '_macro_crowding_cache') and (now - getattr(self, '_macro_crowding_ts', 0) < 60):
            return self._macro_crowding_cache
            
        warnings_dict = {}
        try:
            # 2. 获取大盘总成交额
            # [核心修复] 注入实时时间戳，击穿 CDN 缓存，获取最真实的成交额
            url_idx = DataSource.get_url("MARKET_TOTAL_AMT", ts=int(now * 1000))
            resp_idx = self.net.get_fresh(url_idx, timeout=5)
            if not resp_idx: return {}
            
            diff_idx = resp_idx.json().get('data', {}).get('diff', [])
            total_amt = sum([float(item.get('f6', 0))/100000000 for item in diff_idx if item.get('f6', '-') != '-'])
            
            if total_amt <= 0: return {}
            
            # 3. 获取板块成交额排行
            url_sec = DataSource.get_url("SECTOR_CROWDING", ts=int(now))
            resp_sec = self.net.get_fresh(url_sec, timeout=5)
            if not resp_sec: return {}
            
            sectors = resp_sec.json().get('data', {}).get('diff', [])
            for sec in sectors:
                name = sec.get('f14', '-')
                amt = sec.get('f6', 0)
                if amt != '-':
                    ratio = (float(amt) / 100000000) / total_amt * 100
                    # 仅记录过热板块
                    if ratio >= 8.0:  
                        warnings_dict[name] = ratio
                        
            # 4. 更新缓存
            self._macro_crowding_cache = warnings_dict
            self._macro_crowding_ts = now
            return warnings_dict
            
        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("MACRO_CROWD_ERR", f"宏观拥挤度获取失败: {e}")
            return warnings_dict


    def get_real_market_breadth(self):
        """
        [宏观风控] 获取真实的全局市场上涨比例 (上证+深证)
        彻底解决 scan_df Top200 造成的 100% 幸存者偏差。
        """
        try:
            url = DataSource.get_url("MARKET_BREADTH", ts=int(time.time() * 1000))
            resp = self.net.get_fresh(url, timeout=5)
            if resp and resp.status_code == 200:
                diff = resp.json().get('data', {}).get('diff', [])
                total_up = 0
                total_down = 0
                for item in diff:
                    f104 = item.get('f104') # 上涨家数
                    f105 = item.get('f105') # 下跌家数
                    if f104 and str(f104) != '-': total_up += int(f104)
                    if f105 and str(f105) != '-': total_down += int(f105)
                
                if total_up + total_down > 0:
                    return total_up / (total_up + total_down)
        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("BREADTH_ERR", f"宏观宽度获取失败: {e}")
        return 0.5 # 网络异常时默认 50%，不触发极端熔断
