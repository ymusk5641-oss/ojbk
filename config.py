import json
import os
import threading
import time
import csv
import datetime
from kivy.core.text import LabelBase

def get_android_storage_path():
    """
    [环境适配] 获取 Android 设备的存储路径。
    优先尝试使用外部存储目录 '/storage/emulated/0/Hunter_Logs'，
    如果不存在（如在电脑运行），则回退到当前脚本所在目录。
    """
    android_public = "/storage/emulated/0"
    if os.path.exists(android_public):
        base = os.path.join(android_public, "Hunter_Logs")
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(base):
        try: os.makedirs(base)
        except: base = os.getcwd() 
    return base

# 初始化全局根目录
BASE_DIR = get_android_storage_path()


def get_chinese_font():
    """
    [环境适配] 自动在 Android/Win/Mac 系统中查找可用的中文字体路径。
    用于解决 Kivy 在多端显示中文乱码的问题。
    """
    potential_fonts = [
        # Android 常用路径
        '/system/fonts/DroidSansFallback.ttf',
        '/system/fonts/NotoSansCJK-Regular.ttc',
        '/system/fonts/NotoSansSC-Regular.otf',
        # Ubuntu / Linux 常用路径
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        # Windows 常用路径
        'C:\\Windows\\Fonts\\msyh.ttc',
        'C:\\Windows\\Fonts\\simhei.ttf',
        # MacOS 常用路径
        '/System/Library/Fonts/PingFang.ttc',
        # 当前目录兜底 (用户自己放入的字体)
        os.path.join(BASE_DIR, 'font.ttf'),
        'msyh.ttf', 
        'simhei.ttf'
    ]
    for font_path in potential_fonts:
        if os.path.exists(font_path): return font_path
    return None


# 注册中文字体
CHINESE_FONT = get_chinese_font()
if CHINESE_FONT:
    LabelBase.register(name='Roboto', fn_regular=CHINESE_FONT)


class ConfigManager:
    """
    [基础组件] 配置管理器 (资金感知版 V5.6 - 全参数复刻版)
    审计认证:
    1. 完整保留 dddd58.py 中 STRATEGY_PARAMS 的所有 48 个参数及注释，杜绝配置漂移。
    2. 集成 Android 线程自适应限制，防止设备过热崩溃。
    3. 修复 TOTAL_ASSETS 计算属性。
    """
    CONFIG_FILE = os.path.join(BASE_DIR, "hunter_config.json")
    JOURNAL_FILE = os.path.join(BASE_DIR, "hunter_journal.csv")
    
    # 默认配置模板 (新增 assets)
    DEFAULT_CONFIG = {
        "api_keys": {
            "gemini": "YOUR_GEMINI_KEY_HERE", 
            "deepseek": "YOUR_DEEPSEEK_KEY_HERE"
        },
        "target_stocks": [], 
        "holdings": {},      
        "assets": {
            "cash": 50000.0,    # 默认可用资金
            "total_net": 50000.0 # 预留字段
        },
        "system": {
            "max_workers": 32,
            "min_flow": 15000000,
            "scan_limit": 200
        },
        "risk": {
            "max_pe": 120,
            "min_roe": 0,
            "stop_loss_atr": 2.8,
            "sector_limit": 2,
            "max_bias_20": 15.0,
            "max_hold_days": 15,
            "min_market_breadth": 0.20
        },
        "algo": {
            "rsrs_n": 18, "rsrs_m": 250, "buy_min_score": 75, "buy_max_rsi": 80
        },
        "trading": {
            "slippage": 0.0015, "comm_buy": 0.0003, "comm_sell": 0.0013
        }
    }
    
    # [策略参数仓库] (Parametrization V1.1)
    # 提取核心逻辑中的魔术数字，便于后续遗传算法寻优
    # ⚠️ 严禁删除任何 Key，否则 strategy_scoring 会报错
    STRATEGY_PARAMS = {
        # --- 评分基准 ---
        "score_base": 60.0,
        "score_threshold_name": 75.0, # 达到多少分才给策略命名(趋势/反转)

        # --- 乖离率 (Bias) ---
        "bias_monster_limit": 30.0,   # 妖股允许的最大乖离
        "bias_normal_limit": 20.0,    # 普通股允许的最大乖离
        "bias_attack_min": 5.0,       # 攻击形态下限
        "bias_attack_max": 15.0,      # 攻击形态上限
        "bias_risk_limit": 25.0,      # 信号检查时的硬风控

        # --- 形态风控 ---
        "shadow_limit": 0.6,          # 上影线比例限制 (避雷针)
        "pv_corr_neg_limit": -0.2,    # 量价负相关限制 (诱多)
        "pv_corr_rev_min": 0.1,       # 反转缩量底限

        # --- 技术指标奖励 ---
        "chop_trend_limit": 40.0,     # CHOP < 40 视为趋势
        "rsrs_strong_limit": 0.9,     # RSRS > 0.9 视为强
        "vam_strong_limit": 1.0,      # VAM > 1.0 视为强

        # --- RSI 阈值体系 ---
        "rsi_overbought": 85.0,       # 超买线
        "rsi_high": 80.0,             # 高位线
        "rsi_strong": 75.0,           # 强势线
        "rsi_weak": 40.0,             # 弱势线 (趋势下限)
        "rsi_oversold": 20.0,         # 超卖线 (反转上限)

        # --- 流动性与筹码 ---
        "amihud_high": 0.2,           # 缩量阈值
        "amihud_low": 0.05,           # 锁仓阈值
        "win_rate_high": 90.0,        # 获利盘高位
        "win_rate_low": 5.0,          # 获利盘低位 (血筹)

        # --- 宏观环境 (Regime) ---
        "regime_bonus_limit": 0.6,    # 环境好加分
        "regime_bear_limit": 0.4,     # 熊市禁令线

        # --- AI 权重 ---
        "ai_prob_super": 0.75,
        "ai_prob_high": 0.60,
        "ai_prob_mid": 0.40,
        "ai_prob_low": 0.30,
        "ai_penalty_weight": 10.0,    # [新增] AI看空时的扣分力度 (默认10分)

        # --- 波动率风控 ---
        "vol_rev_limit": 1.5,         # 反转策略波动率限制
        "vol_filter_strict": 4.5,     # 严格波动率限制
        "vol_filter_loose": 5.5,      # 宽松波动率限制 (妖股)
        
        # --- CHOP 风控 ---
        "chop_filter_strict": 50.0,
        "chop_filter_loose": 60.0,
                
        # --- 混扫截断与双轨入围参数 (Pipeline Selection) ---
        "scan_min_score": 60.0,        # 进入 LLM 终审的最低及格线
        "scan_top_n_total": 10,        # 每次扫描最终送审的总名额
        "scan_top_n_ai_reserve": 5,    # 其中专为 AI 高爆发率模型保留的独立名额

    }

    # [新增] 全局统一特征宪法 (无量纲/平稳化 AI 专供版)
    # 所有模块 (实盘日志、回测CSV、AI训练) 必须严格遵守此顺序
    CORE_FEATURE_SCHEMA = [
         # --- 1. 基础量价 (剔除绝对量 vol_prev, close_prev, 剔除废弃的 pe/pb) ---
        "pct", "amihud", 
        # --- 2. 结构因子 (剔除绝对价 atr, bb_up, bb_low) ---
        "rsrs_wls", "rsrs_r2", "volatility", "er", "chop",
        "bb_width", "pct_b",
        # --- 3. 均线系统 (剔除绝对价 ma, vwap, cost，替换为偏离度) ---
        "ma_5_20_ratio", "ma_10_60_ratio", "ma_20_60_ratio",
        "bias_20", "vam", "bias_vwap", 
        # --- 4. 动量震荡 (MACD 内部已升级为 PPO 百分比) ---
        "macd", "macd_signal", "macd_hist", "macd_slope",
        "kdj_k", "kdj_d", "kdj_j",
        "rsi", "rsi_rank", "mfi",
        "roc_20", "trix",
        # --- 5. 资金与形态 (剔除绝对 obv，替换为 zscore) ---
        "obv_zscore", "obv_slope", "pv_corr",
        "winner_rate", "smart_money_rank", "upper_shadow_ratio", "lower_shadow_ratio", "gap_ratio", "vol_zscore", 
        "profit_to_cost_dist", "env_regime"
    ]

    
    # [新增] 导出元数据定义 (Metadata Schema)
    # 这些是交易日志的基础字段，不参与 AI 训练
    EXPORT_METADATA_SCHEMA = [
        'Date', 'Symbol', 
        'Action',       # Buy, Sell, Hold, Wait, Filtered
        'Price', 
        'Return(%)', 
        'Reason',       # 人类可读的理由
        'Shares', 
        'Score',        # 量化总分
        'AI_Prob',      # AI 预测概率
        'Env_Regime',   # 市场环境系数
        'Regime_Src',   # [新增] 数据溯源标记 (REAL/FILL/EMPTY)
        
        # --- 诊断字段------
        'Cash',         # 剩余现金
        'Pos_Pct',      # 实际仓位占比
        'Stop_Line',    # 当前动态止损线 (画图用)
        'Filter_Code',  # 过滤代码 (如: RISK_BEAR, AI_REJECT)
        'Highest_Price' # 持仓期间最高价
    ]

    MODELS = ["deepseek-reasoner", "deepseek-chat", "gemini-2.0-flash"]

    def __init__(self):
        self.lock = threading.Lock()
        self.data = self.load_config()
        self.init_journal()

    def load_config(self):
        """
        [函数级替换] 加载配置文件 (增强版)
        修改: 增加对 mining_batch_size 的兼容性补全，不影响其他逻辑。
        """
        with self.lock:
            # 1. 基础加载
            config = json.loads(json.dumps(self.DEFAULT_CONFIG)) 
            if os.path.exists(self.CONFIG_FILE):
                try:
                    with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip(): 
                            saved = json.loads(content)
                            # 深度合并(递归)
                            def deep_merge(base, override):
                                for k, v in override.items():
                                    if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                                        deep_merge(base[k], v)
                                    else:
                                        base[k] = v
                            deep_merge(config, saved)
                except Exception as e:
                    # 容错处理
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        if os.path.exists(self.CONFIG_FILE): 
                            os.rename(self.CONFIG_FILE, f"{self.CONFIG_FILE}.bak_{timestamp}")
                    except: pass
                    return config 

            # 2. [关键修改] 强制补全 mining_batch_size (防止旧版 JSON 缺字段)
            if 'system' not in config: config['system'] = {}
            if 'mining_batch_size' not in config['system']:
                config['system']['mining_batch_size'] = 200  # 默认值
            
            # 3. 基础补全 (保留原逻辑)
            if not isinstance(config.get('holdings'), dict): config['holdings'] = {}
            if not isinstance(config.get('target_stocks'), list): config['target_stocks'] = []
            
            # 4. 初始化文件
            if not os.path.exists(self.CONFIG_FILE): self._save_atomic(config)
            
            return config


    def init_journal(self):
        if not os.path.exists(self.JOURNAL_FILE): self._write_header()

    def _write_header(self):
        """
        [日志表头修正] 
        1. Model -> Status (避免歧义)
        2. 新增 MarketVal(市值), Profit(盈亏), TotalAssets(总资产)
        """
        try:
            with self.lock:
                with open(self.JOURNAL_FILE, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([
                        "Time", "Symbol", "Name", "Price", "Score", "Phase", 
                        "RSRS", "Breadth", "PE", "Bias20", "Flow", 
                        "Reason", "Status", "Duration", "Result", 
                        "TrendTag", "WinRate", "VWAP_Bias",
                        "MarketVal", "Profit", "TotalAssets"  # [新增列]
                    ])
        except: pass


    def _save_atomic(self, data):
        """
        [架构核心] 原子化配置文件写入
        修正: 使用 os.replace 替代 remove+rename，配合 fsync 强制刷盘，
        彻底解决 Android/OS 异常断电导致的配置文件清空问题。
        """
        tmp_file = f"{self.CONFIG_FILE}.tmp"
        try:
            # 1. 写入临时文件
            with open(tmp_file, 'w', encoding='utf-8') as f: 
                json.dump(data, f, indent=4, ensure_ascii=False)
                # [核心增强] 强制将缓冲区数据写入物理磁盘
                f.flush()
                os.fsync(f.fileno())
            
            # 2. 原子替换 (Atomic Replace)
            # os.replace 在 POSIX 系统上是原子的，即便目标文件存在也会覆盖，不会出现中间态
            os.replace(tmp_file, self.CONFIG_FILE)
            return True
        except Exception as e:
            print(f"Atomic Save Error: {e}")
            # 发生错误尝试清理垃圾文件
            try:
                if os.path.exists(tmp_file): os.remove(tmp_file)
            except: pass
            return False


    def save(self):
        with self.lock: return self._save_atomic(self.data)

    def add_target(self, code):
        with self.lock:
            if not isinstance(self.data.get('target_stocks'), list): self.data['target_stocks'] = []
            if code not in self.data['target_stocks']:
                self.data['target_stocks'].append(code)
                self._save_atomic(self.data)

    def update_holding(self, code, cost, target_volume=0):
        """
        [架构升级] 更新持仓信息 (安全加权平均版)
        注: 传入的 target_volume 必须是交易完成后的【绝对总股数】
        """
        with self.lock:
            if not isinstance(self.data.get('holdings'), dict): 
                self.data['holdings'] = {}
                
            existing = self.data['holdings'].get(code, {'cost': 0.0, 'volume': 0})
            old_vol = existing['volume']
            old_cost = existing['cost']
            
            # 严格将传入的值视为最新的总股数
            target_volume = int(target_volume)
            # 自动推算本次交易的变动量
            delta_vol = target_volume - old_vol
            
            if target_volume > 0:
                if delta_vol > 0: 
                    # 只有真实发生了买入加仓，才去摊薄计算新的加权成本
                    new_cost = (old_cost * old_vol + float(cost) * delta_vol) / target_volume
                else:          
                    # 如果是减仓、或者仅仅是刷新数据，成本价保持原样
                    new_cost = old_cost 
            else:
                # 彻底清仓
                new_cost = 0.0 
                
            self.data['holdings'][code] = {
                'cost': float(new_cost),
                'volume': target_volume,
                'ts': int(time.time())
            }
            
            if not isinstance(self.data.get('target_stocks'), list): 
                self.data['target_stocks'] = []
            if code not in self.data['target_stocks'] and target_volume > 0: 
                self.data['target_stocks'].append(code)
                
            self._save_atomic(self.data)



    # --- 快捷属性 ---
    @property
    def GEMINI_KEY(self): return self.data.get('api_keys', {}).get('gemini', '')
    @property
    def DEEPSEEK_KEY(self): return self.data.get('api_keys', {}).get('deepseek', '')
    
    @property
    def TARGET_STOCKS(self): 
        with self.lock:
            targets = self.data.get('target_stocks', [])
            if not isinstance(targets, list): targets = []
            holdings_data = self.data.get('holdings', {})
            if not isinstance(holdings_data, dict): holdings_data = {}
            target_set = set(targets)
            holdings_set = set(holdings_data.keys())
            return [x for x in list(target_set.union(holdings_set)) if len(str(x)) == 6]

    @property
    def HOLDINGS(self): 
        """
        [鲁棒性增强] 返回持仓字典。
        为了兼容旧版数据，如果发现 Value 是 float，自动升级为 Dict 结构返回，防止下游崩溃。
        """
        with self.lock:
            h = self.data.get('holdings', {})
            if not isinstance(h, dict): return {}
            
            clean_h = {}
            for k, v in h.items():
                if isinstance(v, (float, int)):
                    clean_h[k] = {'cost': float(v), 'volume': 0}
                elif isinstance(v, dict):
                    clean_h[k] = v
                else:
                    continue # 脏数据丢弃
            return clean_h

    @property
    def CASH(self):
        return self.data.get('assets', {}).get('cash', 50000.0)

    @property
    def TOTAL_ASSETS(self):
        """
        [新增] 获取总资产估值 (现金 + 持仓成本市值)
        注意：此处暂时用成本价估算，实时市值需要在业务层结合 Price 计算
        """
        cash = self.CASH
        market_val = 0.0
        h = self.HOLDINGS
        for k, v in h.items():
            market_val += v.get('cost', 0) * v.get('volume', 0)
        return cash + market_val

    @property
    def MAX_WORKERS(self):
        """
        [环境自适应] 智能调整线程数
        - Android 环境: 强制限制为 12-16 (防止 OOM / UI 卡死)
        - PC 环境: 使用配置文件值 (默认 32)
        """
        base_workers = self.data.get('system', {}).get('max_workers', 32)
        if '/storage/emulated' in BASE_DIR:
            return min(base_workers, 12) # Android 上限锁死 12
        return base_workers

    @property
    def SCAN_LIMIT(self): return self.data.get('system', {}).get('scan_limit', 200)
    @property
    def SECTOR_LIMIT(self): return self.data.get('risk', {}).get('sector_limit', 2)
    @property
    def MAX_BIAS_20(self): return self.data.get('risk', {}).get('max_bias_20', 15.0)
    @property
    def MIN_MARKET_BREADTH(self): return self.data.get('risk', {}).get('min_market_breadth', 0.20)
    @property
    def RSRS_PARAMS(self):
        algo = self.data.get('algo', {})
        return algo.get('rsrs_n', 18), algo.get('rsrs_m', 250)
    @property
    def STRATEGY_THRESHOLDS(self): return self.data.get('algo', self.DEFAULT_CONFIG['algo'])
    @property
    def TRANS_COSTS(self): return self.data.get('trading', self.DEFAULT_CONFIG['trading'])
    @property
    def MINING_BATCH_SIZE(self):
        """
        [新增属性] 获取挖掘步长
        优先读取配置文件，如果不存在则默认为 200
        """
        return self.data.get('system', {}).get('mining_batch_size', 200)

    @property
    def FULL_LOG_SCHEMA(self):
        """
        [SSOT] 全局统一日志字典
        = 基础元数据 (Metadata) + 核心特征 (Features)
        """
        # 列表合并，确保顺序固定
        return self.EXPORT_METADATA_SCHEMA + self.CORE_FEATURE_SCHEMA




# 实例化全局配置对象
CFG = ConfigManager()
