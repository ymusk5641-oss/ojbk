import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import threading
import os
import joblib
import json
import time
from config import CFG
from utils import RECORDER, BASE_DIR, BeijingClock


# [新增] 接入数据治理层 (Data Governance Layer)
from governance import DataValidator, DataSanitizer


def safe_compute(context_tag):
    """
    工业级异常捕获装饰器
    作用：防止单个函数的计算错误导致整个 App 闪退，并记录详细日志。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 尝试获取 symbol 用于日志
                sym = "Unknown"
                if len(args) > 0 and hasattr(args[0], 'iloc'):
                     # 尝试从 DataFrame args[0] 中获取 symbol
                     try: sym = args[0].iloc[0]['symbol']
                     except: pass
                
                # 记录详细堆栈
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_exception(f"{context_tag}|{sym}", e)
                
                # 熔断保护：如果是 DataFrame 操作，返回原数据防止 NoneType 报错
                if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                    return args[0]
                return None
        return wrapper
    return decorator


class SignalRegistry:
    """
    [信号宪法] 全局统一信号定义 (Single Source of Truth)
    注意：Value 值保持中文，确保 UI 显示和 AI 理解无缝兼容。
    """
    # --- 1. 逻辑关键字 (Logic Keys) ---
    # 核心！用于 if "关键字" in tag 判断。
    # 无论 UI 显示变成 "🔥妖股通行证" 还是 "妖股(高波)"，只要包含 "妖股" 二字，逻辑就不会断。
    KEY_TREND = "趋势"
    KEY_REV = "反转"
    KEY_MONSTER = "妖股"
    
    KEY_EXIT_PROTECT = "保本"
    KEY_EXIT_LOCK = "锁利"
    KEY_EXIT_TIME = "时间"
    KEY_EXIT_MA20 = "破位MA20"
    KEY_EXIT_RSI = "RSI极值"

    # --- 2. 策略名称 (Strategy Names) ---
    STRAT_TREND = "趋势追击"   # 包含 KEY_TREND
    STRAT_REV = "超跌反转"     # 包含 KEY_REV
    STRAT_ALERT = "起爆预警"
    STRAT_WATCH = "观察"
    STRAT_RISK = "☠️风控拦截"
    STRAT_BAD_DATA = "❌数据污染"
    STRAT_BIAS_FAIL = "乖离过大"
    STRAT_NEEDLE = "☠️避雷针诱多"
    STRAT_FUSE = "市场熔断"
    DESC_INVALID_DATA = "数据无效"
    STRAT_SECTOR_LIMIT = "🚫板块限额"
    # --- 3. 趋势/形态标签 (Tags) ---
    TREND_BULL = "多头"
    TREND_BEAR = "空头"
    TREND_NEW_HIGH = "新高"
    TREND_BREAK_ATR = "🚀突破ATR"
    TREND_PURE = "📈趋势纯净"
    TREND_RSRS_STRONG = "RSRS强"
    TREND_SLOW_BULL = "稳健慢牛"
    
    VOL_HEAVY = "放量"
    VOL_SHRINK_UP = "缩量涨"
    VOL_EXTREME_SHRINK = "⚡极致缩量"
    VOL_LOCK = "🔒锁仓"
    
    PTN_GOLD_CROSS_M = "M金叉"
    PTN_DIV_TOP = "⚠️顶背离"
    PTN_DIV_BTM = "💎底背离"
    PTN_BIAS_ATTACK = "Bias攻击区"
    PTN_MACD_IGNITE = "MACD起爆"
    PTN_LOW_VOL_START = "低波启动"
    
    # --- 4. 风控/负面标签 (Risk Tags) ---
    RISK_MONSTER_PASS = "🔥妖股通行证" # 包含 KEY_MONSTER
    RISK_LURE = "缩量诱多"
    RISK_HIGH_LOCK = "高位锁仓"
    RISK_OVERHEAT = "RSI过热"
    RISK_OVERSOLD = "极度超跌"
    RISK_BLOOD = "血筹"
    RISK_SHRINK_LURE = "⚠️高位缩量诱多" 
    RISK_HUGE_NEEDLE = "💀天量避雷针"
    
    # --- 5. 卖出信号 (Exit Signals) ---
    EXIT_TAG_MONSTER = "🔥妖股跟踪"
    EXIT_TAG_STOP = "止损"
    EXIT_TAG_PROTECT = "🛡️保本"       # 包含 KEY_EXIT_PROTECT
    EXIT_TAG_LOCK_LV1 = "🔒锁利(Lv1)" # 包含 KEY_EXIT_LOCK
    EXIT_TAG_LOCK_LV2 = "🔒锁利(Lv2)" # 包含 KEY_EXIT_LOCK
    EXIT_MSG_MA20 = "📉破位MA20"      # 包含 KEY_EXIT_MA20
    EXIT_MSG_RSI = "RSI极值"          # 包含 KEY_EXIT_RSI
    EXIT_MSG_TIME = "时间止损"        # 包含 KEY_EXIT_TIME
    
    # --- 6. AI 标签 ---
    AI_SUPER = "AI极强"
    AI_LONG = "AI看多"
    AI_KILL = "⛔AI斩杀"

    # [新增] --- 7. 拥挤度与资金博弈标签 (Crowding & Flow Tags) ---
    CROWD_LAG = "拥挤滞涨"
    CROWD_TRAMPLE = "拥挤踩踏"
    CROWD_MICRO = "微观拥挤(爆量)"           # [新增] 对应爆量滞涨
    SMART_MONEY_FLEE = "主力撤退(缩量高乖离)" # [新增] 对应聪明钱代理

    # [新增] --- 8. 审计动作 (Audit Actions) ---
    ACT_TAKE_PROFIT = "止盈离场"
    ACT_SWAP = "建议换股"
    ACT_TREND_BREAK = "趋势破坏"
    ACT_HIGH_SELL = "高抛止盈"
    ACT_CRITICAL = "⚠️ 关键决策"
    ACT_STOP_LOSS = "止损卖出"
    ACT_SIGNAL_PROFIT = "止盈信号"
    ACT_FORCE_REDUCE = "强制减仓"
    ACT_HOLD = "HOLD"



class TechLib:
    """
    [底层算力核心] Numpy 向量化指标库 (Vectorization Kernel)
    修复记录:
    1. 修正 sma 方法返回长度不足的问题 (补齐前导 NaN)。
    """
    
    @staticmethod
    def sliding_window_view(x, window_size):
        """[加速核心] 创建零拷贝的滑动窗口视图"""
        x = np.asarray(x)
        shape = (x.shape[0] - window_size + 1, window_size)
        strides = (x.strides[0], x.strides[0])
        return as_strided(x, shape=shape, strides=strides)

    @staticmethod
    def sma(x, n):
        """
        简单移动平均 (Simple Moving Average) - 长度对齐修正版
        修正: 使用 pad 补齐前 n-1 个数据，确保输出长度 == 输入长度。
        """
        if len(x) < n: return np.full(len(x), np.nan)
        
        # 1. 计算累加和
        cs = np.cumsum(np.insert(x, 0, 0))
        
        # 2. 计算区间均值 (长度会缩短 n-1)
        sma_valid = (cs[n:] - cs[:-n]) / n
        
        # 3. [关键修复] 补齐前导 NaN
        pad = np.full(n - 1, np.nan)
        
        return np.concatenate((pad, sma_valid))

    @staticmethod
    def ema(x, span):
        """
        指数移动平均 (Exponential Moving Average) - [性能极速版]
        优化: 彻底剥离 Pandas 实例化，使用纯 Numpy 矩阵预分配 + 原生循环。
        经测试在移动端 Pydroid3 下，短序列数组速度提升 50 倍，完美消除 GC 卡顿。
        """
        x_arr = np.asarray(x, dtype=np.float64)
        n = len(x_arr)
        if n == 0:
            return x_arr
        
        alpha = 2.0 / (span + 1.0)
        ema_arr = np.empty_like(x_arr)
        ema_arr[0] = x_arr[0]
        
        # 纯 Numpy 数组内部寻址，避免多余的内存碎片
        for i in range(1, n):
            ema_arr[i] = ema_arr[i-1] + alpha * (x_arr[i] - ema_arr[i-1])
            
        return ema_arr


    @staticmethod
    def rolling_max(x, n):
        # [修复] 增加 min_periods=1 防止输出前导 NaN，彻底消除 MACD 背离计算时的 RuntimeWarning
        return pd.Series(x).rolling(n, min_periods=1).max().values

    @staticmethod
    def rolling_min(x, n):
        # [修复] 增加 min_periods=1 防止输出前导 NaN，彻底消除 MACD 背离计算时的 RuntimeWarning
        return pd.Series(x).rolling(n, min_periods=1).min().values

    
    @staticmethod
    def rolling_sum(x, n):
        return pd.Series(x).rolling(n).sum().values

    @staticmethod
    def rolling_std(x, n):
        return pd.Series(x).rolling(n).std().values

    @staticmethod
    def rsi(close, n=14):
        """
        [算法修复 V420] 相对强弱指标 (RSI) - 停牌/一字板防御版
        修复: 当股票无波动(avg_up=0 且 avg_down=0)时，强制 RSI=50，防止误判为超跌(0)。
        """
        # 1. 输入清洗
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 计算差分
        delta = np.diff(close, prepend=close[0])
        
        # 3. 分离涨跌
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # 4. 指数移动平均
        alpha = 1.0 / n
        avg_up = pd.Series(up).ewm(alpha=alpha, adjust=False).mean().values
        avg_down = pd.Series(down).ewm(alpha=alpha, adjust=False).mean().values
        
        # 5. [核心修复] 总波动检测
        total_movement = avg_up + avg_down
        
        # 计算 RS，加 1e-9 防止除零
        rs = np.divide(avg_up, avg_down + 1e-9)
        rsi_val = 100 - (100 / (1 + rs))
        
        # [Defense] 当总波动极小时，强制归为中性 (50.0)
        # 解决长期停牌或一字板导致的 RSI=0 (误报超跌) 问题
        rsi_val = np.where(total_movement < 1e-9, 50.0, rsi_val)
        
        return np.nan_to_num(rsi_val, nan=50.0)

    @staticmethod
    def rolling_corr(x, y, window=10):
        """[极致提速] 纯 Numpy 零拷贝滑动窗口相关系数"""
        n = len(x)
        if n < window:
            return np.zeros(n)
        
        # 利用已有的滑动窗口基建
        x_view = TechLib.sliding_window_view(x, window)
        y_view = TechLib.sliding_window_view(y, window)
        
        x_mean = np.mean(x_view, axis=1, keepdims=True)
        y_mean = np.mean(y_view, axis=1, keepdims=True)
        
        x_sub = x_view - x_mean
        y_sub = y_view - y_mean
        
        cov = np.sum(x_sub * y_sub, axis=1)
        var_x = np.sum(x_sub ** 2, axis=1)
        var_y = np.sum(y_sub ** 2, axis=1)
        
        denom = np.sqrt(var_x * var_y)
        # 防除零保护
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom!=0)
        corr = np.clip(corr, -1.0, 1.0) # 新增钳位保护
        # 补齐前导窗口
        return np.concatenate((np.zeros(window - 1), corr))


    @staticmethod
    def kdj(close, high, low, n=9, m1=3, m2=3):
        low_n = pd.Series(low).rolling(n).min().values
        high_n = pd.Series(high).rolling(n).max().values
        
        # [核心修复] 停牌/僵尸股零波动防御。若区间高低点重叠，强行归于中性 RSV=50.0，防止跌入 0.0 超跌黑洞
        diff = high_n - low_n
        rsv = np.where(diff < 1e-5, 50.0, (close - low_n) / (diff + 1e-9) * 100)
        
        rsv = np.nan_to_num(rsv, nan=50.0)
        k = pd.Series(rsv).ewm(com=m1-1, adjust=False).mean().values
        d = pd.Series(k).ewm(com=m2-1, adjust=False).mean().values
        j = 3 * k - 2 * d
        return k, d, j


    @staticmethod
    def atr(high, low, close, n=14):
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return pd.Series(tr).ewm(alpha=1.0/n, adjust=False).mean().values

    @staticmethod
    def chop(high, low, close, n=14):
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        sum_tr = pd.Series(tr).rolling(n).sum().values
        max_h = pd.Series(high).rolling(n).max().values
        min_l = pd.Series(low).rolling(n).min().values
        numerator = sum_tr / (max_h - min_l + 1e-9)
        
        # [修复] 使用 np.clip 限制最小为 1.0，最大为 n，防止极度横盘导致对数溢出，并允许 NaN 穿透
        numerator = np.clip(numerator, 1.0, n)
        
        # 屏蔽无效期计算产生的 RuntimeWarning (NaN 参与对数运算)
        with np.errstate(invalid='ignore', divide='ignore'):
            chop = 100 * np.log10(numerator) / np.log10(n)
            
        # 最终将所有无效期或计算失败的值，归于绝对中立 (50.0)
        return np.nan_to_num(chop, nan=50.0)



class FactorRegistry:
    """
    [数据治理核心] 因子注册表 (Data Constitution)
    功能：集中定义所有因子的元数据、校验规则和默认值填充策略。
    架构升级 V2.2: 全字段接管，收回所有中间态字段的定义权。
    """
    # 1. 基础物理量 (必须 > 0，允许微小浮点误差)
    PHYSICAL = {
        'close': 'positive', 'open': 'positive', 'high': 'positive', 'low': 'positive',
        'vol': 'positive', 'amount': 'positive',
        'ma5': 'positive', 'ma10': 'positive', 'ma20': 'positive', 'ma60': 'positive',
        'vol_prev': 'positive', 'close_prev': 'positive'
    }
    
    # 2. 震荡指标 (通常在 0-100 或 -100-100)
    OSCILLATOR = {
        'rsi_rank': {'min': 0, 'max': 100},
        'mfi': {'min': 0, 'max': 100},
        'chop': {'min': 0, 'max': 100},
        'winner_rate': {'min': 0, 'max': 100},
        'kdj_j': {'min': -100, 'max': 200},
        'wr': {'min': 0, 'max': 100}
    }
    
    # 3. 策略依赖清单 (入场/离场必需字段)
    REQUIRED_FOR_ENTRY = [
        'close', 'open', 'high', 'ma5', 'vol', 'vol_prev', 'close_prev', 'pct', 
        'bias_20', 'rsi_rank', 'volatility', 'chop', 'pv_corr', 'rsrs_wls'
    ]

    # 4. [新增] 字段默认值宪法 (Field Constitution)
    # 定义当上游数据缺失时，系统认可的唯一安全填充值
    FIELD_DEFAULTS = {
        # --- 0. 基础元数据 (补齐防御黑洞) ---
        'symbol': '000000', 'name': '-',
        'pe': 0.0, 'pb': 0.0,

        # --- 1. 基础物理量 ---
        'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0,
        'vol': 0.0, 'amount': 0.0, 
        'vol_prev': 0.0, 'close_prev': 0.0,
        'data_quality': 1.0, # 默认质量为优
        
        # --- 2. 数值型指标 (默认 0.0) ---
        'ma5': 0.0, 'ma10': 0.0, 'ma20': 0.0, 'ma60': 0.0,
        'ma_5_20_ratio': 0.0, 'ma_10_60_ratio': 0.0, 'ma_20_60_ratio': 0.0, # 新增相对均线
        'bias_20': 0.0, 'bias_vwap': 0.0, 'vwap_20': 0.0, 'cost_avg': 0.0,
        'volatility': 0.0, 'flow': 0.0, 'pct': 0.0, 
        'pv_corr': 0.0, 'vam': 0.0, 'upper_shadow_ratio': 0.0,
        'obv': 0.0, 'obv_slope': 0.0, 'obv_zscore': 0.0, # 新增 obv_zscore
        'amihud': 0.0, 'er': 0.0, 
        'bb_up': 0.0, 'bb_low': 0.0, 'bb_width': 100.0, 'pct_b': 0.5,
        'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0, 'macd_slope': 0.0,
        'roc_20': 0.0, 'trix': 0.0,
        'atr': 0.0,
        'smart_money_rank': 50.0, # [修复] 适配0-100量纲，默认值改为中性50.0防误杀
        'sector_crowding': 0.0,
        'lower_shadow_ratio': 0.0,
        'gap_ratio': 0.0,
        'vol_zscore': 0.0,
        'profit_to_cost_dist': 0.0,
        'env_regime': 0.5, # 宏观环境默认中性
        # --- 3. 策略中间态与结果 ---
        'trend_score': 0.0, 'reversal_score': 0.0, 'final_score': 0.0,
        'ai_score': 50.0, 'llm_score': -1.0,
        
        # --- 4. 文本描述 ---
        'trend_desc': "", 'strategy_name': "", 
        '_trend_desc_tech': "", 
        
        # --- 5. 震荡型指标 (默认 50.0 中性) ---
        'rsi': 50.0, 'rsi_rank': 50.0, 
        'mfi': 50.0, 'chop': 50.0, 
        'kdj_k': 50.0, 'kdj_d': 50.0, 'kdj_j': 50.0,
        'winner_rate': 50.0,
        
        # --- 6. 乘数型 (默认 1.0) ---
        'rsrs_wls': 1.0, 'rsrs_r2': 0.0,
        
        # --- 7. 布尔型/标记 (默认 False) ---
        'is_monster': False, 
        'macd_gold': False, 'kdj_gold': False, 
        'macd_btm_div': False, 'macd_top_div': False, 
        'is_divergence': False, 'is_squeeze_atr': False,
        '_is_trash': False
    }
    
    @staticmethod
    def validate_row(row, schema_type='entry'):
        """[哨兵机制] 单行数据强校验与解包"""
        required = FactorRegistry.REQUIRED_FOR_ENTRY if schema_type == 'entry' else []
        clean_data = {}
        missing = []
        
        for key in required:
            val = row.get(key)
            if val is None:
                missing.append(f"{key}(Missing)")
                continue
            if isinstance(val, (float, np.floating, int, np.integer)):
                if np.isnan(val) or np.isinf(val):
                    missing.append(f"{key}(NaN/Inf)")
                    continue
            
            # 物理量检查
            if key in FactorRegistry.PHYSICAL and val <= 0.0001:
                if key != 'vol': 
                    missing.append(f"{key}(<=0)")
                    continue
            
            clean_data[key] = val
            
        if missing: return False, f"数据熔断: {missing}"
        return True, clean_data

    @staticmethod
    def enforce_std_schema(df, context_tag="Unknown"):
        """
        [强制标准架构对齐 V2.2]
        职责：
        1. 确保所有注册在 FIELD_DEFAULTS 中的列都存在。
        2. 对缺失列或脏数据列，强制使用宪法定义的默认值填充。
        3. 确保数值类型正确 (float/bool/str)，拒绝 Object 类型混入。
        """
        if df.empty: return df
        
        # 1. 遍历宪法定义的每个字段
        for col, default_val in FactorRegistry.FIELD_DEFAULTS.items():
            # A. 如果列不存在，创建并填充默认值 (最为关键的一步)
            if col not in df.columns:
                df[col] = default_val
            
            # B. 针对数值型进行清洗 (处理 NaN/Inf)
            if isinstance(default_val, (int, float)):
                if df[col].dtype == 'object':
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df[col] = df[col].fillna(default_val)
                
                if np.isinf(df[col]).any():
                    df[col] = df[col].replace([np.inf, -np.inf], default_val)

            # C. 针对布尔型进行清洗
            elif isinstance(default_val, bool):
                df[col] = df[col].fillna(default_val).astype(bool)

            # D. [新增] 针对字符串进行清洗 (防止 NaN 混入文本列)
            elif isinstance(default_val, str):
                df[col] = df[col].fillna(default_val).astype(str)
                # 处理可能存在的 "nan" 字符串
                mask_nan = df[col].str.lower() == 'nan'
                if mask_nan.any():
                    df.loc[mask_nan, col] = default_val
                
        return df


class TechCalculator:
    """
    [新独立组件] 技术指标计算引擎 (Stateless)
    职责: 接收原始 OHLCV 数据，返回包含所有技术因子的 DataFrame。
    特点: 纯函数式设计，无副作用，专门负责“脏活累活”。
    """

    @staticmethod
    def execute_calculation_pipeline(df):
        """[对外接口] 执行全量指标计算流水线"""
        # 1. 提取 Numpy 数组 (加速)
        raw, err = TechCalculator._extract_raw_numpy(df)
        if err: return df, err # 返回错误但不崩溃
        
        target_len = len(df)
        
        # 2. 模块化计算 (流水线)
        # 注意：这里我们复用 QuantEngine 中已有的静态算法方法，或者将其逻辑搬运过来。
        # 为了避免重复代码，建议将 QuantEngine 中的 _calc_xxx 逻辑迁移至此，
        # 但鉴于 QuantEngine 中还有 _calc_amihud_illiquidity 等静态方法可能被其他地方引用，
        # 我们这里采用“逻辑搬运 + 引用静态工具”的混合模式。
        
        res, err = TechCalculator._calc_basic_and_rsrs(raw, target_len)
        if err: return df, err
        
        res = TechCalculator._calc_ma_and_bollinger(raw, res)
        res = TechCalculator._calc_trend_and_momentum(raw, res)
        res = TechCalculator._calc_oscillators(raw, res)
        res = TechCalculator._calc_others_and_features(raw, res)
        
        # 3. 维度对齐检查
        for k, v in res.items():
            if len(v) != target_len:
                return df, f"维度崩坏: {k}({len(v)}!={target_len})"

        # 4. 合并结果
        try:
            # 仅保留非因子列，避免重复
            cols_to_update = list(res.keys())
            df_clean = df.drop(columns=[c for c in cols_to_update if c in df.columns], errors='ignore')
            
            tech_df = pd.DataFrame(res, index=df.index)
            df_final = pd.concat([df_clean, tech_df], axis=1)
            
            return df_final, None
            
        except Exception as e:
            return df, f"合并失败: {str(e)}"

    @staticmethod
    def _extract_raw_numpy(df):
        """[内部原子] 提取原始 Numpy 数组 (修复 NaN 黑洞与状态污染版)"""
        try:
            # [核心修复1] 采用 Series 级别的 ffill/bfill 提取 values，绝不修改原始 df 
            # (彻底杜绝多个列表复用同一 df 缓存时导致的状态污染和回测不一致)
            close = df['close'].ffill().bfill().values.astype(np.float64)
            
            high = df['high'].ffill().bfill().values.astype(np.float64) if 'high' in df.columns else close
            low = df['low'].ffill().bfill().values.astype(np.float64) if 'low' in df.columns else close
            open_p = df['open'].ffill().bfill().values.astype(np.float64) if 'open' in df.columns else close
            
            # [核心修复2] A 股停牌物理隔离：价格可以延续，但成交量(vol)在缺失/停牌时必须强制为 0.0，绝不能继承昨天！
            vol = df['vol'].fillna(0.0).values.astype(np.float64) if 'vol' in df.columns else np.zeros_like(close)
            
            # 基础防线
            if np.isnan(close).any(): return None, "Close包含无法修复的NaN"
            
            return {
                'close': close, 'high': high,
                'low': low, 'vol': vol,
                'open': open_p
            }, None
        except Exception as e:
            return None, f"提取Numpy失败: {str(e)}"


    @staticmethod
    def _calc_basic_and_rsrs(raw, target_len):
        """[内部原子] 基础变动与 RSRS"""
        res = {}
        close, high, low, vol = raw['close'], raw['high'], raw['low'], raw['vol']

        # 基础变动
        res['pct'] = np.concatenate(([0.0], np.diff(close) / (close[:-1] + 1e-9) * 100))
        res['vol_prev'] = np.roll(vol, 1); res['vol_prev'][0] = vol[0] 
        res['close_prev'] = np.roll(close, 1); res['close_prev'][0] = close[0]

        # Amihud (直接调用 QuantEngine 的静态方法，保持逻辑一致)
        res['amihud'] = QuantEngine._calc_amihud_illiquidity(pd.Series(close), pd.Series(vol)).values 
        
        # RSRS (调用 QuantEngine 静态方法)
        params = getattr(CFG, 'STRATEGY_PARAMS', {})
        rsrs_df = QuantEngine._calc_weighted_rsrs(pd.Series(high), pd.Series(low), pd.Series(vol), N=params.get('rsrs_window', 18))
        
        if len(rsrs_df) == target_len:
            res['rsrs_wls'] = rsrs_df['beta'].values
            res['rsrs_r2'] = rsrs_df['r2'].values
        else:
            return None, f"RSRS长度错位 ({len(rsrs_df)}!={target_len})"
        return res, None

    @staticmethod
    def _calc_ma_and_bollinger(raw, res):
        """[内部原子] 均线与布林带"""
        close = raw['close']
        res['ma5'] = TechLib.sma(close, 5) 
        res['ma10'] = TechLib.sma(close, 10) 
        res['ma20'] = TechLib.sma(close, 20) 
        res['ma60'] = TechLib.sma(close, 60) 

        # [源头修复] 消除 MA 的前导 NaN 幻觉，对齐到当前价格
        # [Bug 1 修复] 先记录真实有效的 MA 掩码，用于后续拦截冷启动的虚假 Ratio
        valid_ma20 = ~np.isnan(res['ma20'])
        valid_ma60 = ~np.isnan(res['ma60'])

        res['ma5'] = np.where(np.isnan(res['ma5']), close, res['ma5'])
        res['ma10'] = np.where(np.isnan(res['ma10']), close, res['ma10'])
        res['ma20'] = np.where(np.isnan(res['ma20']), close, res['ma20'])
        res['ma60'] = np.where(np.isnan(res['ma60']), close, res['ma60'])

        std20 = TechLib.rolling_std(close, 20)
        res['bb_up'] = res['ma20'] + 2 * std20
        res['bb_low'] = res['ma20'] - 2 * std20

        # [连带修复] 防御 bb_up/low 产生的 NaN，使其退化为中心轴 ma20
        res['bb_up'] = np.where(np.isnan(res['bb_up']), res['ma20'], res['bb_up'])
        res['bb_low'] = np.where(np.isnan(res['bb_low']), res['ma20'], res['bb_low'])
        
        denom = res['ma20'] + 1e-9
        res['bb_width'] = np.divide((res['bb_up'] - res['bb_low']), denom, out=np.zeros_like(res['ma20']), where=denom!=0) * 100
        
        # 👇【新增修复】防御冷启动导致的 bb_width 塌缩为 0
        res['bb_width'] = np.where(np.isnan(std20), 100.0, res['bb_width'])
        
        bb_diff = res['bb_up'] - res['bb_low']
        res['pct_b'] = np.where(
            bb_diff < 1e-5, 
            0.5, 
            (close - res['bb_low']) / (bb_diff + 1e-9)
        )
  
        # [新增特征] 剥离绝对价格，计算供 AI 学习的均线偏离度 (无量纲)
        # [Bug 1 修复] 只有当长均线真实走出来时，才计算 Ratio，否则置为 0.0 中性，彻底消除 AI 特征漂移
        res['ma_5_20_ratio'] = np.where(valid_ma20, (res['ma5'] - res['ma20']) / (res['ma20'] + 1e-9) * 100, 0.0)
        res['ma_10_60_ratio'] = np.where(valid_ma60, (res['ma10'] - res['ma60']) / (res['ma60'] + 1e-9) * 100, 0.0)
        res['ma_20_60_ratio'] = np.where(valid_ma60, (res['ma20'] - res['ma60']) / (res['ma60'] + 1e-9) * 100, 0.0)

        return res





    @staticmethod
    def _calc_trend_and_momentum(raw, res):
        """[内部原子] 趋势与动量 (含 ROC20 与 TRIX)"""
        close, high, low = raw['close'], raw['high'], raw['low']
        
        # MACD (升级为无量纲 PPO: Percentage Price Oscillator)
        ema12 = TechLib.ema(close, 12); ema26 = TechLib.ema(close, 26)
        # 核心修改：除以 ema26 转化为百分比，彻底消除高价股与低价股的量纲差异
        res['macd'] = (ema12 - ema26) / (ema26 + 1e-9) * 100
        res['macd_signal'] = TechLib.ema(res['macd'], 9)
        res['macd_hist'] = (res['macd'] - res['macd_signal']) * 2
        res['macd_slope'] = np.concatenate(([0.0], np.diff(res['macd_hist'])))

        # 金叉
        prev_diff = np.roll(res['macd'], 1); prev_diff[0] = 0
        prev_dea = np.roll(res['macd_signal'], 1); prev_dea[0] = 0
        # [修复] 修改为 <=，防止指标完全粘合时丢失金叉信号
        res['macd_gold'] = (prev_diff <= prev_dea) & (res['macd'] > res['macd_signal'])
        res['macd_gold'][0] = False


        # 背离 (MACD Div)
        w = 20
        roll_high_p = TechLib.rolling_max(high, w); roll_low_p = TechLib.rolling_min(low, w)
        roll_high_m = TechLib.rolling_max(res['macd'], w); roll_low_m = TechLib.rolling_min(res['macd'], w)
        
        res['macd_top_div'] = (close >= roll_high_p * 0.99) & (res['macd'] < roll_high_m * 0.85) & (res['macd'] > 0)
        res['macd_btm_div'] = (close <= roll_low_p * 1.01) & (res['macd'] > roll_low_m * 0.85) & (res['macd'] < 0)

        # ATR & Chop
        res['atr'] = TechLib.atr(high, low, close, 14)
        res['volatility'] = (res['atr'] / (close + 1e-9)) * 100
        res['is_squeeze_atr'] = res['atr'] < (TechLib.sma(res['atr'], 20) * 0.70)
        
         # ER
        er_n = 10
        if len(close) <= er_n:
            res['er'] = np.zeros(len(close))
        else:
            change = np.abs(close[er_n:] - close[:-er_n])
            change = np.concatenate((np.zeros(er_n), change))
            path = TechLib.rolling_sum(np.abs(np.diff(close, prepend=close[0])), er_n)
            
            # [修复] 强制清理初始期 rolling_sum 产生的 NaN 以及除零造成的 inf 幽灵
            er_arr = np.divide(change, path + 1e-9)
            res['er'] = np.nan_to_num(er_arr, nan=0.0, posinf=0.0, neginf=0.0)

        
        res['chop'] = TechLib.chop(high, low, close, 14)

        # ========================================================
        # [新增工业级动量因子] ROC20 与 TRIX
        # ========================================================
        # 1. ROC20 (20日变动率 - 捕获中期绝对动能)
        close_20_ago = np.roll(close, 20)
        close_20_ago[:20] = close[0] # 前20天用首日价格兜底
        res['roc_20'] = (close - close_20_ago) / (close_20_ago + 1e-9) * 100

        # 2. TRIX (三重指数平滑 - 过滤主力洗盘毛刺)
        trix_ema1 = TechLib.ema(close, 12)
        trix_ema2 = TechLib.ema(trix_ema1, 12)
        trix_ema3 = TechLib.ema(trix_ema2, 12)
        # TRIX = (EMA3今日 - EMA3昨日) / EMA3昨日 * 100
        trix_ema3_prev = np.roll(trix_ema3, 1)
        trix_ema3_prev[0] = trix_ema3[0]
        # [逻辑修复] 分母增加 np.abs()，防止极其微小的负数毛刺与 1e-9 抵消导致除零/Inf 崩溃
        res['trix'] = (trix_ema3 - trix_ema3_prev) / (np.abs(trix_ema3_prev) + 1e-9) * 100


        return res


    @staticmethod
    def _calc_oscillators(raw, res):
        """
        [内部原子 V2.1 - RSI 坍缩防御版]
        修复: 引入 60日 RSI 标准差检测。若历史无波动(停牌/一字板)，强行将 Rank 归于中性(50)，
        防止复牌后的微小扰动导致 Rank 瞬间飙升至 100 触发过热误判。
        """
        close, high, low, vol = raw['close'], raw['high'], raw['low'], raw['vol']
        
        # 1. RSI 与 防御性 Rank
        res['rsi'] = TechLib.rsi(close, 14)
        rsi_series = pd.Series(res['rsi'])
        
        rsi_std = rsi_series.rolling(60, min_periods=20).std()
        raw_rank = rsi_series.rolling(60, min_periods=20).rank(pct=True).values * 100
        
        # 当波动极小(std < 1.5)时，剥夺其相对排名的资格，强制设为 50.0
        safe_rank = np.where(rsi_std < 1.5, 50.0, raw_rank)
        res['rsi_rank'] = np.nan_to_num(safe_rank, nan=50.0)
        
         # 2. MFI 与 真实资金流 (Flow) 暴露 (作为全系统的 Single Source of Truth)
        tp = (high + low + close) / 3.0
        flow = tp * vol
        diff_tp = np.diff(tp, prepend=tp[0])
        
        # [无损修复] 补齐断层，将真实的带有方向的资金动能写回 DataFrame 供后续加分逻辑读取
        # 物理含义: (价格变化百分比) * (真实的成交金额)
        res['flow'] = np.divide(diff_tp, tp + 1e-9) * flow
        
        pos_flow = np.where(diff_tp > 0, flow, 0)
        neg_flow = np.where(diff_tp < 0, flow, 0)
        
        pos_sum = TechLib.rolling_sum(pos_flow, 14)
        neg_sum = TechLib.rolling_sum(np.abs(neg_flow), 14)
        pos_sum = np.nan_to_num(pos_sum, nan=0.0)
        neg_sum = np.nan_to_num(neg_sum, nan=0.0)
        mfi_ratio = pos_sum / (neg_sum + 1e-9)
        mfi_raw = 100 - (100 / (1 + mfi_ratio))
        
        # [核心修复] 防御停牌/僵尸股的“零资金流”黑洞。总流极小时强行锚定中性50.0，防止跌入0.0超卖陷阱
        total_flow = pos_sum + neg_sum
        res['mfi'] = np.where(total_flow < 1e-5, 50.0, mfi_raw)

        # 3. KDJ
        res['kdj_k'], res['kdj_d'], res['kdj_j'] = TechLib.kdj(close, high, low)
        prev_k = np.roll(res['kdj_k'], 1); prev_k[0] = 50
        prev_d = np.roll(res['kdj_d'], 1); prev_d[0] = 50
        # [修复] 允许从指标完全粘合的状态 (如停牌复牌或极度缩量引起的 K==D==50) 直接起爆金叉
        res['kdj_gold'] = (res['kdj_k'] > res['kdj_d']) & (prev_k <= prev_d)
        res['kdj_gold'][0] = False
        
        return res


    @staticmethod
    def _calc_others_and_features(raw, res):
        """[内部原子] 其他特征 (集成工业级 Alpha)"""
        # [Gemini Fix: 补充提取 low，供上下影线共同使用]
        close, high, vol, open_p, low = raw['close'], raw['high'], raw['vol'], raw['open'], raw['low']
        
        # OBV
        res['obv'] = np.cumsum(np.sign(np.diff(close, prepend=close[0])) * vol)
        obv_ma = TechLib.sma(np.abs(res['obv']), 20) + 1e-9
        obv_diff = res['obv'][5:] - res['obv'][:-5]
        res['obv_slope'] = np.concatenate((np.zeros(5), obv_diff)) / obv_ma * 10
        
        # [新增特征] OBV Z-Score (将无限增长的非平稳序列转化为 20 日局部平稳动能)
        obv_s = pd.Series(res['obv'])
        obv_ma20 = obv_s.rolling(20, min_periods=1).mean()
        obv_std20 = obv_s.rolling(20, min_periods=1).std().fillna(1.0)
        res['obv_zscore'] = ((obv_s - obv_ma20) / np.where(obv_std20 < 1e-9, 1.0, obv_std20)).values


        # Bias & VAM
        res['bias_20'] = (close - res['ma20']) / (res['ma20'] + 1e-9) * 100
        res['vam'] = (close - res['ma20']) / (res['atr'] + 1e-9)
        
        # Correlation & Shadow
        res['pv_corr'] = TechLib.rolling_corr(close, vol, 10)
        # 修复后代码
        raw_upper = (high - np.maximum(open_p, close)) / ((high - low) + 1e-9)
        res['upper_shadow_ratio'] = np.clip(raw_upper, 0.0, 1.0)

        raw_lower = (np.minimum(open_p, close) - low) / ((high - low) + 1e-9)
        res['lower_shadow_ratio'] = np.clip(raw_lower, 0.0, 1.0)

        # 2. 跳空缺口动量 (Gap Ratio)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0] # 首日无跳空，兜底
        res['gap_ratio'] = (open_p - prev_close) / (prev_close + 1e-9) * 100
        
        # 3. 成交量异动 (Volume Z-Score -> 换手率平替)
        vol_s = pd.Series(vol)
        vol_ma = vol_s.rolling(20, min_periods=1).mean()
        vol_std = vol_s.rolling(20, min_periods=1).std().fillna(1.0)
        vol_std = np.where(vol_std < 1e-9, 1.0, vol_std) # 防除零
        res['vol_zscore'] = ((vol_s - vol_ma) / vol_std).values

        return res




class AIModelServer:
    """
    [独立状态机 V2.1 - 工业级参数持久化推理版]
    职责: 隔离 QuantEngine 状态，保障 Android 多线程安全。
    新增: 启动时加载历史 MAD 参数 (mad_params)，并在实盘调用网关时严格传入，
         彻底解决单股预测无法去极值的理论盲区。
    """
    _model = None
    _features = []
    _mad_params = {} # [新增] 用于存储从 JSON 读取的历史特征尺子
    _threshold = 0.5
    _is_loaded = False
    _lock = threading.Lock()

    @classmethod
    def load(cls, force_reload=False):
        with cls._lock:
            if not force_reload and cls._is_loaded: return
            try:
                import joblib, json, os
                from utils import BASE_DIR, RECORDER
                model_path = os.path.join(BASE_DIR, "hunter_rf_model.pkl")
                feat_path = os.path.join(BASE_DIR, "hunter_features.json")
                
                if os.path.exists(model_path) and os.path.exists(feat_path):
                    cls._model = joblib.load(model_path)
                    with open(feat_path, 'r', encoding='utf-8') as f:
                        schema = json.load(f)
                        cls._features = schema.get("features", [])
                        cls._threshold = schema.get("threshold", 0.5)
                        # [核心新增] 加载历史特征尺子
                        cls._mad_params = schema.get("mad_params", {})
                        
                    cls._is_loaded = True
                    RECORDER.log_debug("AI_INIT", f"✅ 独立服务接管模型 | 阈值: {cls._threshold:.2f} | 尺子: {len(cls._mad_params)}维")
                else:
                    RECORDER.log_debug("AI_INIT", "⚠️ 缺失模型文件，引擎将以纯量化模式运行")
            except Exception as e:
                from utils import RECORDER
                RECORDER.log_debug("AI_LOAD_ERR", str(e))

    @classmethod
    def predict_batch(cls, df_valid):
        """
        高并发批量预测接口 (严格使用历史尺子)
        [深度逻辑修复] 修复了与 predict_single 的三大不对称漏洞：
        1. 强制抹除 PE/PB，防止实盘/回测特征污染。
        2. 安全特征对齐，防止旧模型字典导致 KeyError 闪退。
        3. 引入向量化自适应校准，确保批量输出的分数具备绝对策略基准。
        """
        if not cls._is_loaded or not cls._features: return None
        try:
            from governance import DataSanitizer
            # 1. 拦截数据，强制清洗并运用历史 MAD 极值
            df_clean = DataSanitizer.clean_machine_learning_features(
                df_valid, cls._features, mad_params=cls._mad_params
            ).copy()
            
            # 2. [核心防线] 特征严格对齐与基本面静默
            for feat in cls._features:
                if feat not in df_clean.columns:
                    df_clean[feat] = 0.0
                # 必须与 predict_single 保持绝对一致，抹除不可靠的基本面
                if feat in ['pe', 'pb']:
                    df_clean[feat] = 0.0
            
            # 安全提取矩阵
            X = df_clean[cls._features].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            # 3. 推理获取原始概率
            probs = cls._model.predict_proba(X)[:, 1]
            
            # 4. [算力防线] 向量化阈值校准
            th = cls._threshold
            if th <= 0.0 or th >= 1.0:
                return np.full_like(probs, 0.5) * 100.0
            
            # 使用 numpy 矩阵运算，实现与单行预测完全一致的非线性映射
            calibrated = np.where(
                probs >= th,
                0.5 + 0.5 * (probs - th) / (1.0 - th),
                0.5 * probs / th
            )
            # [统一接口] 强制放大 100 倍
            return calibrated*100.0
            
        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("AI_BATCH_ERR", str(e))
            return None

    @classmethod
    def predict_single(cls, df_row):
        """带自适应校准的单行预测 (严格使用历史尺子，解决盲区)"""
        if not cls._is_loaded or not cls._features: return 50.0
        try:
            df_temp = pd.DataFrame([df_row])
            from governance import DataSanitizer
            # [核心改动] 哪怕只有 1 行数据，只要传了 _mad_params，网关就会执行强制截断！
            df_clean = DataSanitizer.clean_machine_learning_features(
                df_temp, cls._features, mad_params=cls._mad_params
            ).copy()

            clean_row_dict = df_clean.iloc[0].to_dict()
            input_vector = []
            for feat in cls._features:
                val = 0.0 if feat in ['pe', 'pb'] else clean_row_dict.get(feat, 0.0)
                if pd.isna(val) or np.isinf(val): val = 0.0
                input_vector.append(float(val))

            probs = cls._model.predict_proba(np.array([input_vector]))
            raw_prob = float(probs[0][1])
            
            th = cls._threshold
            if raw_prob >= th:
                calibrated = 0.5 if th >= 1.0 else 0.5 + 0.5 * (raw_prob - th) / (1.0 - th)
            else:
                calibrated = 0.5 if th <= 0.0 else 0.5 * raw_prob / th

            # [统一接口] 强制放大 100 倍
            return calibrated * 100.0
        except Exception:
            return 50.0



class QuantEngine:
    """
    [量化策略核心 V10.0]
    觉醒内核：集成机器学习推理与多线程安全锁
    """

    # --- 拥挤度缓存 (保留原有资产) ---
    _crowding_cache = None
    _crowding_ts = 0
    _CROWDING_TTL = 60 
    _crowding_lock = threading.Lock()  # <--- [核心修复] 补齐缺失的线程锁！


    @staticmethod
    def _ensure_features_exist(df):

        """
        [Schema 卫士 - 审计熔断版]
        功能：
        1. 检查 Config 定义的所有特征是否存在。
        2. [报警] 如果缺失，记录详细日志并打印红色警告。
        3. [熔断] 将 data_quality 降级为 0，强制策略层“拒收”此数据，防止错误交易。
        4. [补全] 填充 0.0 防止程序 Crash。
        """
        if df.empty: return df
        
        # 1. 获取宪法定义的标准
        required_cols = CFG.CORE_FEATURE_SCHEMA
        current_cols = set(df.columns)
        
        # 2. 扫描缺失字段
        missing_cols = [c for c in required_cols if c not in current_cols]
        
        if missing_cols:
            # --- A. 触发报警 (Observability) ---
            # 获取当前处理的股票代码（用于日志定位）
            sample_sym = df.iloc[0]['symbol'] if 'symbol' in df.columns else 'Unknown'
            
            error_msg = (
                f"⚠️ [CRITICAL SCHEMA MISSING] 标的:{sample_sym} | "
                f"缺失 {len(missing_cols)} 个核心因子! \n"
                f"   >>> 丢失列表: {missing_cols[:5]}... (共{len(missing_cols)}个)"
            )
            
            # 写入后台日志
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("SCHEMA_LOSS", error_msg)
            
            # 控制台显红报警 (开发模式可见)
            print(f"\033[91m{error_msg}\033[0m")

            # --- B. 执行熔断 (Circuit Breaker) ---
            # 强制标记数据质量为 0 (垃圾)，策略层的 check_entry_signal 会自动过滤掉 data_quality < 0.5 的数据
            # 这样即使填了 0，也不会因为误判而买入
            df['data_quality'] = 0.0
            df['strategy_name'] = SignalRegistry.STRAT_BAD_DATA

            # --- C. 安全填充 (Crash Prevention) ---
            # 虽然标记了垃圾，但为了防止后续代码引用报错，依然需要物理填充
            for c in missing_cols:
                df[c] = FactorRegistry.FIELD_DEFAULTS.get(c, 0.0)

        return df

    @staticmethod
    def validate_data(df, min_len=20):
        """[数据完整性熔断校验]"""
        if df is None or df.empty: 
            return False, "数据为空"
        if len(df) < min_len: 
            return False, f"长度不足({len(df)}<{min_len})"
        if 'close' in df.columns and df['close'].sum() == 0: 
            return False, "收盘价全为0"
        return True, "OK"

    @staticmethod
    def map_regime_score(z_score):
        """
        [环境映射标准] 将连续的 RSRS Z-Score 映射为离散的策略系数
        """
        try:
            z = float(z_score)
            if np.isnan(z): return 0.5 
        except:
            return 0.5
            
        # [核心修复] 提高牛熊确认门槛，防止长熊市中的均线钝化导致的“假突破”
        if z >= 1.0: return 1.0   # [极强] 绝对多头主升浪 (门槛提高)
        if z >= 0.4: return 0.8   # [偏强] 震荡向上 (原为 0.0，现必须有显著向上动量)
        if z >= -0.5: return 0.5  # [震荡] 垃圾时间，控制仓位
        return 0.2                # [熊市] 破位单边下跌，收缩防守



    @staticmethod
    def calc_rsrs_regime_series(df, N=18, M=600):
        """
        [SSOT] RSRS 市场环境计算标准算法 (工业级防钝化版)
        逻辑: OLS RSRS -> Z-Score -> MA 绝对趋势压制 -> 离散系数映射
        """
        # 1. 基础数据校验
        if df.empty or len(df) < N: 
            return pd.Series(0.5, index=df.index) 
        
        # 2. 向量化提取
        high = pd.to_numeric(df['high'], errors='coerce').fillna(0.0)
        low = pd.to_numeric(df['low'], errors='coerce').fillna(0.0)
        # [新增] 提取收盘价用于绝对趋势防守
        close = pd.to_numeric(df['close'], errors='coerce').fillna(0.0)
        
        cov = high.rolling(window=N).cov(low)
        var = low.rolling(window=N).var()
        corr = high.rolling(window=N).corr(low).fillna(0.0)
        
        beta = np.where(var < 1e-8, 1.0, cov / (var + 1e-9))
        r2 = corr ** 2
        rsrs = beta * r2

        # 3. Z-Score 标准化
        rsrs_mean = rsrs.rolling(window=M, min_periods=250).mean()
        rsrs_std = rsrs.rolling(window=M, min_periods=250).std()
        z_score = (rsrs - rsrs_mean) / (rsrs_std + 1e-9)
        
        # ========================================================
        # 4. [终极风控补丁] 引入 MA 绝对趋势过滤，彻底封杀 Z-Score 钝化陷阱！
        # ========================================================
        ma20 = close.rolling(window=20, min_periods=1).mean()
        ma60 = close.rolling(window=60, min_periods=1).mean()
        
        z_adj = z_score.values.copy()
        
        # A: 跌破 20 日均线 (短期空头/弱势)
        # 强行剥夺“偏强”以上的资格，Z-Score 最高只能被钳制在 -0.1 (映射后对应 0.5 震荡)
        mask_under_ma20 = (close < ma20).values
        z_adj = np.where(mask_under_ma20, np.minimum(z_adj, -0.1), z_adj)
        
        # B: 跌破 60 日均线 (中期熊市)
        # 强行剥夺所有多头资格，Z-Score 最高只能被钳制在 -0.6 (映射后必定落入 0.2 熊市)
        mask_under_ma60 = (close < ma60).values
        z_adj = np.where(mask_under_ma60, np.minimum(z_adj, -0.6), z_adj)

        # 5. 映射为离散策略系数
        regime_val = pd.Series(z_adj, index=df.index).apply(QuantEngine.map_regime_score)
        
        return regime_val



    @staticmethod
    def _calc_amihud_illiquidity(close, volume, window=20):
        """
        [修复版] Amihud 非流动性因子
        修复：移除致命的 .clip(upper=1.0) 硬截断，降级量纲乘数，将去极值任务交还给系统统一的 DataSanitizer 处理。
        """
        try:
            pct_abs = close.pct_change().abs()
            amount = close * volume + 1e-9
            daily_illiq = pct_abs / amount
            
            # [核心修复] 移除 clip，改为 1e8 使得常规股票的值落在平稳区间，保留微盘股和小票的区分度
            daily_illiq_scaled = daily_illiq * 1e8
            
            illiq_factor = daily_illiq_scaled.rolling(window, min_periods=1).mean()
            return illiq_factor.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        except Exception as e:
            return pd.Series(0.0, index=close.index)


    @staticmethod
    def _calc_weighted_rsrs(high, low, vol, N=18):
        """
        [算法核心 V2.2] 加权 RSRS (含 R2 计算)
        升级: 修复 Pandas 2.0+ fillna(method='ffill') 弃用警告
        """
        try:
            length = len(high)
            if length < N:
                 return pd.DataFrame({'beta': 1.0, 'r2': 0.0}, index=high.index)

            beta_arr = np.full(length, 1.0)
            r2_arr = np.full(length, 0.0)
            
            # 转换为 numpy 数组加速
            h_arr = high.values
            l_arr = low.values
            v_arr = vol.values
            
            # 权重预计算 (WLS)
            v_mean = np.nanmean(v_arr)
            if v_mean == 0: v_mean = 1.0
            w_raw = v_arr / (v_mean + 1e-9)
            w_sqrt = np.sqrt(w_raw)
            
            # [修复] 修改循环起点并向右平移切片窗口，解决信号永远滞后1天的问题
            for i in range(N - 1, length):
                # 窗口切片
                win_h = h_arr[i - N + 1 : i + 1]
                win_l = l_arr[i - N + 1 : i + 1]
                win_w = w_sqrt[i - N + 1 : i + 1]
     
                # 构造加权矩阵
                y_prime = win_h * win_w
                x1 = win_l * win_w
                x2 = win_w 
                X_prime = np.column_stack((x1, x2))
                
                try:
                    res = np.linalg.lstsq(X_prime, y_prime, rcond=None)
                    slope = res[0][0]
                    beta_arr[i] = slope
                    
                    # 计算 R2
                    if len(res[1]) > 0:
                        ss_res = res[1][0]
                    else:
                        y_pred = np.dot(X_prime, res[0])
                        ss_res = np.sum((y_prime - y_pred) ** 2)
                    
                    # [修复] 加权总平方和 (Weighted SST) 必须使用原始 Y(win_h) 和原始权重计算
                    w_raw_win = win_w ** 2
                    y_mean_w = np.average(win_h, weights=w_raw_win)
                    ss_tot = np.sum(w_raw_win * (win_h - y_mean_w) ** 2)
                    
                    r2 = 1 - (ss_res / (ss_tot + 1e-9))
                    r2_arr[i] = max(0.0, min(1.0, r2))
                    
                except:
                    beta_arr[i] = 1.0
                    r2_arr[i] = 0.0
            
            # [关键] 保持索引一致，防止赋值时对齐错误
            # [修复] 使用 ffill() 替代 fillna(method='ffill')
            return pd.DataFrame({
                'beta': beta_arr, 
                'r2': r2_arr
            }, index=high.index).ffill()
            
        except Exception:
            return pd.DataFrame({'beta': 1.0, 'r2': 0.0}, index=high.index)


    @staticmethod
    def calc_industrial_indicators(df):
        """
        [工业级核心 V5.4 - 妖股真实性过滤版 + 筹码 Alpha 扩展]
        """
        try:
            # 提取基础列，减少 DataFrame 索引开销
            close = df['close']
            vol = df['vol']
            high = df['high']
            low = df['low']
            
            # --- 1. VWAP (成交量加权平均价) ---
            tp = (high + low + close) / 3.0
            pv = tp * vol
            
            cum_pv = pv.rolling(20, min_periods=1).sum()
            cum_vol = vol.rolling(20, min_periods=1).sum()
            
            # [修复] 如果 20 日累计成交量极低(如长期停牌)，将 VWAP 安全锚定为现价，防止计算爆炸
            safe_vwap = np.where(cum_vol < 1e-5, close, cum_pv / (cum_vol + 1e-9))
            df['vwap_20'] = safe_vwap

            df['bias_vwap'] = (close - df['vwap_20']) / (df['vwap_20'] + 1e-9) * 100


            # --- 2. 筹码获利比例 (Winner Rate) ---
            # [修复] 替换为真实的带量加权 EMA (Volume-Weighted EMA)，反映真实的筹码成本中心
            alpha = 2.0 / (50 + 1)
            ema_pv = (tp * vol).ewm(alpha=alpha, adjust=False).mean()
            ema_vol = vol.ewm(alpha=alpha, adjust=False).mean()
            
            # 👇 核心修复：防零成交量黑洞！当 ema_vol 极度衰减(如停牌)时，将成本强制锚定为现价 close
            df['cost_avg'] = np.where(ema_vol < 1e-5, close, ema_pv / (ema_vol + 1e-9))

            std_cost = tp.rolling(50, min_periods=5).std()
            z_score = (close - df['cost_avg']) / (std_cost + 1e-9)
            z_score = np.clip(z_score, -10.0, 10.0) # 新增这一行，完美防止 exp 溢出
            
            df['winner_rate'] = 100 / (1 + np.exp(-1.5 * z_score))
            df['winner_rate'] = df['winner_rate'].fillna(50.0)

            # ===================================================
            # [Gemini Fix: 新增工业级 Alpha 因子]
            # 现价距平均成本乖离 (Profit to Cost Distance)
            # ===================================================
            df['profit_to_cost_dist'] = (close - df['cost_avg']) / (df['cost_avg'] + 1e-9) * 100

             # --- 3. 聪明钱流向 (升级为连续平滑的 SMI) ---
            safe_vol_prev = df['vol_prev'] if 'vol_prev' in df.columns else vol.shift(1).fillna(0.0)
            
            # A. 价格顺向发力度
            price_strength = (close - df['vwap_20']) / (df['vwap_20'] + 1e-9)
            # B. 资金涌入强度 (用 log1p 防止极限爆量失真)
            vol_surge = np.log1p(vol / (safe_vol_prev + 1e-9))
            # C. 动能叠加与标准化
            raw_sm = price_strength * vol_surge
            sm_mean = raw_sm.rolling(20, min_periods=1).mean()
            sm_std = raw_sm.rolling(20, min_periods=1).std().fillna(1.0)
            sm_z = (raw_sm - sm_mean) / (np.where(sm_std < 1e-9, 1.0, sm_std))
            
            # D. Sigmoid 映射到 0-100 平稳区间
            df['smart_money_rank'] = 100 / (1 + np.exp(-sm_z))
            df['smart_money_rank'] = df['smart_money_rank'].fillna(50.0)
            
            # --- 4. 妖股识别 (Super Trend) ---
            rsi = df.get('rsi', 50)
            ma5 = df.get('ma5', close)
            volatility = df.get('volatility', 0)
            
            # [核心修复] 真实流动性验证：当日量不能低于过去5日均量的10%（排除无量一字板）
            vol_ma5 = vol.rolling(5, min_periods=1).mean()
            has_volume = vol > (vol_ma5 * 0.1)
            
            df['is_monster'] = (rsi > 80) & (close > ma5) & (volatility > 2.0) & has_volume

        except Exception as e:
            # [核心修复：打破静默失败，污染显影]
            if 'RECORDER' in globals():
                sym = df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown'
                globals()['RECORDER'].log_debug("IND_CALC_ERR", f"[{sym}] 工业指标污染: {str(e)}")
            
            # 安全兜底 (防闪退)
            df['vwap_20'] = df['ma20'] if 'ma20' in df.columns else df.get('close', 0.0)
            df['winner_rate'] = 50.0
            df['smart_money_rank'] = 50.0
            df['is_monster'] = False
            # [Gemini Fix: 异常块防漏补齐]
            df['profit_to_cost_dist'] = 0.0 
            
            # 强制打上“数据污染”烙印，移交策略层拦截
            df['data_quality'] = 0.0 
            
        return df


    @staticmethod
    @safe_compute("TECH_BATCH")
    def calc_tech_batch(df):
        """
        [算法核心 V12.0 - 瘦身版]
        升级: 将计算逻辑全权委托给 TechCalculator，QuantEngine 只负责流程控制。
        """
        # [兜底模板] (保持不变，用于熔断时补全列)
        REQUIRED_COLS = [
            'ma5', 'ma10', 'ma20', 'ma60', 'atr', 'volatility', 'chop', 
            'ma_5_20_ratio', 'ma_10_60_ratio', 'ma_20_60_ratio', # 新增
            'rsi', 'rsi_rank', 'mfi', 'kdj_k', 'kdj_d', 'kdj_j', 'kdj_gold', 
            'obv', 'obv_slope', 'obv_zscore', 'bias_20', 'vam', 'pv_corr', 'upper_shadow_ratio', # 新增 obv_zscore
            'macd', 'macd_signal', 'macd_hist', 'bb_up', 'bb_low', 'bb_width', 
            'pct_b', 'er', 'macd_slope', 'macd_top_div', 'macd_btm_div', 
            'macd_gold', 'amihud', 'rsrs_wls', 'rsrs_r2', 'vol_prev', 'close_prev'
        ]


        def _abort_with_error(reason, context_df):
            sym = context_df.iloc[0]['symbol'] if not context_df.empty and 'symbol' in context_df.columns else "Unknown"
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("STRATEGY_KILL", f"[{sym}] 计算熔断: {reason}")
            
            failed_df = context_df.copy()
            failed_df['data_quality'] = 0.0 
            for c in REQUIRED_COLS:
                if c not in failed_df.columns: failed_df[c] = 0.0
            return failed_df

        # --- 1. 治理检查 ---
        is_valid, msg = DataValidator.validate(df, context_tag="Strategy_Entry")
        if not is_valid: return _abort_with_error(f"质检失败: {msg}", df)

        df = DataSanitizer.sanitize(df)
        if len(df) < 20: return _abort_with_error("行数不足20", df)

        # --- 2. 委托计算 (调用新类) ---
        df_tech, err = TechCalculator.execute_calculation_pipeline(df)
        
        if err: return _abort_with_error(err, df)

        # --- 3. 后处理 (业务逻辑) ---
        # 工业指标依赖业务逻辑，依然保留在 QuantEngine 中
        df_final = QuantEngine.calc_industrial_indicators(df_tech)
        
        # [显影修复] 严禁盲目覆盖 data_quality！
        # 只有当指标计算没有触发 0.0 的污染标记时，才认可数据质量
        if 'data_quality' not in df_final.columns or df_final['data_quality'].min() > 0.5:
            df_final['data_quality'] = 1.0
            
        return df_final

    @staticmethod
    def _calc_position_math(score, volatility_pct, market_regime, price, available_cash, total_assets, max_pos_cap_override=None, current_sector_exposure=0.0):
        """
        [资金管理核心 V8.0 - 真实凯利+板块锁死版]
        修复: 引入标准 Fractional Kelly Criterion 替代粗糙的线性映射，防止高波妖股仓位失控。
        新增: 引入板块敞口硬锁(Sector Exposure Limit)，绝不梭哈单一题材。
        """
        # --- 1. 基础熔断 ---
        if score < 60: 
            return 0, "分低(<60)", 0.0, (0.5, 0.5, 0.5, 1)
        if not price or price <= 0.01:
            return 0, "价格异常", 0.0, (1, 0.8, 0, 1)

        # [基线参数还原] 保持 1% 的总资产风险预算，严控回撤
        RISK_BUDGET_PER_TRADE = 0.01 
        STOP_LOSS_MULTIPLIER = 2.0 
        
        # [逻辑重构] 彻底移除虚假底座，兜底 1.0 防止零资产状态下的数学除零错误。
        # 让 Kelly 公式严格基于真实购买力运作，彻底封死小账户的满仓幻觉。
        assets_base = max(total_assets, available_cash, 1.0)

        # --- 2. 动态上限与熊市硬锁 (Regime Logic) ---
        regime_desc = ""
        if max_pos_cap_override is not None:
            # [指令覆盖模式] (回测/强制)
            MAX_POS_CAP = float(max_pos_cap_override)
            regime_desc = "指令覆盖"
        else:
            # [实盘风控模式]
            if market_regime <= 0.2:
                # 熊市必须高分才能博弈，否则空仓
                if score < 80:
                    return 0, "熊市保护(<80分)", 0.0, (0.6, 0.6, 0.6, 1)
                MAX_POS_CAP = 0.10 # 熊市上限锁死 10%
                regime_desc = "熊市博弈"
            elif market_regime <= 0.5: 
                MAX_POS_CAP = 0.20
                regime_desc = "震荡"
            else: 
                MAX_POS_CAP = 0.30
                regime_desc = "进攻"

        # ==================== 2.5 [工业级升级] 板块敞口硬锁 ====================
        # 同一行业板块的持仓金额绝对不能超过总资金的 30%
        SECTOR_MAX_LIMIT = 0.30
        # [修复2] 将 > 0 改为 >= 0，确保板块内的第一只股票也受 30% 上限约束
        if current_sector_exposure >= 0 and max_pos_cap_override is None:
            remaining_sector_space = max(0.0, SECTOR_MAX_LIMIT - current_sector_exposure)
            MAX_POS_CAP = min(MAX_POS_CAP, remaining_sector_space)
            # [性能与滑点优化] 如果剩余空间不足 5%，直接放弃，防止碎片化买入(产生不必要的手续费与滑点)
            if MAX_POS_CAP < 0.05:
                return 0, f"板块限额(仅剩{remaining_sector_space*100:.1f}%)", 0.0, (0.5, 0.5, 0.5, 1)


        # --- 3 & 4. 波动率定仓与凯利准则 ---
        real_vol = max(volatility_pct, 1.0)
        
        # [修复1] 如果是强行覆盖模式(如回测)，直接绕过波动率压制，实现物理满仓
        if max_pos_cap_override is not None:
            theoretical_pos_pct = max_pos_cap_override
            kelly_factor = 1.0
        else:
            vol_adjusted_pos = RISK_BUDGET_PER_TRADE / ((STOP_LOSS_MULTIPLIER * real_vol / 100.0) + 1e-9)
            p = max(0.40, min(0.75, 0.40 + (score - 60) * (0.35 / 40.0)))
            b = 2.5 if market_regime > 0.8 else 2.0
            kelly_f = p - (1.0 - p) / b
            kelly_factor = max(0.0, min(1.0, kelly_f * 2.0))
            theoretical_pos_pct = vol_adjusted_pos * kelly_factor


        # --- 5. 物理约束与对齐 ---
        final_pos_pct = min(theoretical_pos_pct, MAX_POS_CAP)
        
        target_amt = assets_base * final_pos_pct
        executable_amt = min(target_amt, available_cash)
        
        # 向下取整到 100 股
        can_buy_shares = int(executable_amt / (price + 1e-9))
        real_shares = (can_buy_shares // 100) * 100
        
        # --- 6. 小资金一手保障 (Small Account Protection) ---
        if real_shares < 100 and available_cash >= (price * 100):
            if max_pos_cap_override is None and score >= 85 and market_regime > 0.2:
                real_shares = 100
                # [核心修正] 限制最大占比为 1.0 (100%)。
                # 防止由于 assets_base 还原真实小资金后，买入高价股导致的数值溢出 (>100%) 崩坏全局。
                final_pos_pct = min(1.0, (price * 100) / assets_base)
                desc = f"强票保底|Vol:{real_vol:.1f}"
            else:
                return 0, f"风控限制(<1手)|Vol:{real_vol:.1f}", final_pos_pct, (0.5, 0.5, 0.5, 1)
        else:
            desc = f"{regime_desc}|Vol:{real_vol:.1f}%|Kelly:{kelly_factor:.2f}"

        # 颜色输出
        if final_pos_pct > 0.20: risk_color = (1, 0.3, 0.3, 1)
        elif final_pos_pct > 0.10: risk_color = (1, 0.8, 0, 1)
        else: risk_color = (0.5, 1, 0.5, 1)

        return real_shares, desc, final_pos_pct, risk_color

    @staticmethod
    def _analyze_crowding(df_current):
        """
        [拥挤度分析 - 性能优化版 & 闭集修正 & 线程安全修复版]
        """
        now = time.time()
        
        # 1. 第一重检查 (无锁，提升并发性能)
        if QuantEngine._crowding_cache is not None:
            if now - QuantEngine._crowding_ts < QuantEngine._CROWDING_TTL:
                return QuantEngine._crowding_cache

        # 2. 获取锁并进行第二重检查 (Double-Checked Locking)
        with QuantEngine._crowding_lock:
            # 拿到锁后再次确认缓存是否已被其他线程更新
            if QuantEngine._crowding_cache is not None:
                if now - QuantEngine._crowding_ts < QuantEngine._CROWDING_TTL:
                    return QuantEngine._crowding_cache

            SR = SignalRegistry # 引用注册表
            crowding_map = {} 
            try:
                if not os.path.exists(CFG.JOURNAL_FILE): 
                    QuantEngine._crowding_cache = {}
                    return {}
                
                from collections import deque
                parsed_data = []
                
                try:
                    # [架构核心优化] 使用 deque 在物理层截取末尾 5000 行（约涵盖数天的高频扫描记录）
                    # 彻底告别 pd.read_csv 全量加载导致的内存爆炸和 IO 阻塞
                    with open(CFG.JOURNAL_FILE, 'r', encoding='utf-8-sig', errors='replace') as f:
                        tail_lines = list(deque(f, maxlen=5000))
                        
                    if not tail_lines:
                        QuantEngine._crowding_cache = {}
                        return {}
                        
                    # 根据 config 宪法: 0:Time, 1:Symbol, 3:Price
                    for line in tail_lines:
                        parts = line.strip().split(',')
                        if len(parts) > 3 and parts[0] != 'Time':
                            try:
                                parsed_data.append({
                                    'Time': parts[0],
                                    'Symbol': str(parts[1]).zfill(6),
                                    'Price': float(parts[3])
                                })
                            except: pass
                            
                    if not parsed_data:
                        QuantEngine._crowding_cache = {}
                        return {}
                        
                    j_df = pd.DataFrame(parsed_data)
                except Exception as e: 
                    if 'RECORDER' in globals():
                        globals()['RECORDER'].log_debug("CROWD_READ_ERR", str(e))
                    QuantEngine._crowding_cache = {}
                    return {}
                
                j_df['Time'] = pd.to_datetime(j_df['Time'], errors='coerce')
                # 只看最近 72 小时
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=72)
                recent = j_df[j_df['Time'] > cutoff].copy()
                
                if not recent.empty:
                    recent['Date'] = recent['Time'].dt.date
                    stats = recent.groupby('Symbol').agg({'Date': 'nunique', 'Price': ['first', 'last']})
                    stats.columns = ['days_count', 'price_start', 'price_end']
                    
                    for sym, row in stats.iterrows():
                        if row['days_count'] >= 2: 
                            start = float(row['price_start'])
                            end = float(row['price_end'])
                            if start == 0: continue
                            change = (end - start) / start
                            
                            # [修正] 使用 SR 常量替代硬编码
                            if change < 0.01 and change > -0.02:
                                crowding_map[sym] = {'penalty': 10, 'desc': f"|{SR.CROWD_LAG}"}
                            # 踩踏判定
                            elif change <= -0.02:
                                crowding_map[sym] = {'penalty': 20, 'desc': f"|{SR.CROWD_TRAMPLE}"}
            except: pass
            
            # 3. 写入缓存并更新时间戳 (在锁内安全执行)
            QuantEngine._crowding_cache = crowding_map
            QuantEngine._crowding_ts = time.time()
            return crowding_map



    #  统一的涨跌停计算工具
    @staticmethod
    def calc_limit_price_math(price, ratio=0.10):
        """
        [基建] A股涨跌停价格计算器 (遵循四舍五入规则)
        被 BacktestEngine 和实盘风控共同调用。
        """
        try:
            p = float(price)
            # A股规则: 涨跌幅基于昨日收盘价，保留两位小数(四舍五入)
            # 普通股 10%, 科创/创业 20%, ST 5%
            # 这里简化处理，由外部传入 ratio
            
            # 涨停价
            limit_up = round(p * (1 + ratio) * 100) / 100.0
            # 跌停价
            limit_down = round(p * (1 - ratio) * 100) / 100.0
            
            return limit_up, limit_down
        except:
            return price * 1.1, price * 0.9


    @staticmethod
    def calc_ols_beta_r2(highs, lows):
        """
        [数学基建 - 工业级增强版] RSRS 核心 OLS 回归计算
        修正: 
        1. 增加方差检测，防止极度横盘(方差为0)导致的奇异矩阵错误。
        2. 增加 inf/nan 预清洗，确保 lstsq 输入纯净。
        3. 保持原有接口返回 (beta, r2)。
        """
        try:
            # 1. 基础校验 (保留原逻辑)
            if len(highs) != len(lows) or len(highs) < 2: 
                return np.nan, np.nan
            
            # 2. 类型转换与清洗 (增强)
            # 强制转为 float64，避免传入 int 导致计算精度丢失
            highs = np.array(highs, dtype=np.float64)
            lows = np.array(lows, dtype=np.float64)
            
            # 检查无效值 (新增防御)
            if not (np.isfinite(highs).all() and np.isfinite(lows).all()):
                return np.nan, np.nan

            # 3. [新增] 方差检测 (防止 Singular Matrix)
            # 如果最高价或最低价几乎没有波动（方差接近0），回归无意义，直接返回中性值
            if np.var(highs) < 1e-9 or np.var(lows) < 1e-9:
                return 1.0, 0.0

            # 4. 执行线性回归 (保留原逻辑)
            A = np.vstack([lows, np.ones(len(lows))]).T
            # rcond=None 让 numpy 自动处理奇异值
            beta, alpha = np.linalg.lstsq(A, highs, rcond=None)[0]
            
            # 5. 计算 R2 (保留原逻辑)
            y_pred = beta * lows + alpha
            mean_high = np.mean(highs)
            tss = np.sum((highs - mean_high) ** 2)
            rss = np.sum((highs - y_pred) ** 2)
            
            # 防止极度横盘导致的除零 (保留原逻辑并增强)
            if tss <= 1e-9: 
                return 1.0, 0.0 # 修正：原代码返回 nan, nan 可能导致后续计算中断，此处给中性值更稳健
            
            r2 = 1 - (rss / tss)
            
            # 6. 边界钳制 (新增，防止浮点误差导致 R2 略大于1或小于0)
            r2 = max(0.0, min(1.0, r2))
            
            return beta, r2
        except:
            return np.nan, np.nan

    @staticmethod
    def filter_top_candidates(df, params=None):
        """
        [选股截断器 V2.0 - 双轨制自适应补位版]
        功能：分离量化综合分与AI概率分，执行双通道选拔，并具备名额互相兜底功能。
        """
        if df.empty: return df
        if params is None: params = CFG.STRATEGY_PARAMS
        
        # 1. 获取配置参数
        min_score = params.get('scan_min_score', 60.0)
        total_n = params.get('scan_top_n_total', 10)
        ai_reserve = params.get('scan_top_n_ai_reserve', 5)
        quant_reserve = max(0, total_n - ai_reserve)

        # 2. 基础底线过滤 (淘汰不及格、熔断、以及触碰物理红线的标的)
        valid_mask = df['final_score'] > min_score
        if '_is_entry_valid' in df.columns:
            valid_mask = valid_mask & df['_is_entry_valid']
            
        valid_df = df[valid_mask]
        if valid_df.empty: return valid_df


        # 3. 量化传统通道 (提取综合分 Top N)
        pool_quant = valid_df.sort_values('final_score', ascending=False).head(quant_reserve)

        # 4. AI 觉醒通道 (排除已被量化通道选中的，防止名额重叠浪费)
        remain_df = valid_df[~valid_df['symbol'].isin(pool_quant['symbol'])]
        pool_ai = remain_df.sort_values('ai_score', ascending=False).head(ai_reserve)

        # 5. 物理合并
        combined_df = pd.concat([pool_quant, pool_ai]).drop_duplicates(subset=['symbol'])

        # 6. [工业级兜底] 智能补位机制
        # 如果 AI 选出的票不足 ai_reserve (导致总数不满 total_n)，
        # 则将剩余的空缺名额，重新按量化 final_score 降序补齐。
        current_count = len(combined_df)
        if current_count < total_n and len(valid_df) > current_count:
            shortfall = total_n - current_count
            pool_fallback = valid_df[~valid_df['symbol'].isin(combined_df['symbol'])] \
                            .sort_values('final_score', ascending=False).head(shortfall)
            combined_df = pd.concat([combined_df, pool_fallback])

        # 7. 日志显影 (便于观察两股力量的占比)
        if 'RECORDER' in globals():
            q_cnt = len(pool_quant)
            a_cnt = len(pool_ai)
            f_cnt = len(combined_df) - q_cnt - a_cnt
            globals()['RECORDER'].log_debug(
                "SELECTION", 
                f"入围结构 => 总数:{len(combined_df)} | 量化主导:{q_cnt} | AI主导:{a_cnt} | 补位:{f_cnt}"
            )

        return combined_df.copy()




    @staticmethod
    def check_exit_signal_v2(symbol, cost, current_price, highest_price, atr, rsi_rank, risk_cfg, regime_factor, current_score, hold_days=0, ma20=0.0):
        """
        [卖出防守引擎 V5.1 - 零降级强校验版]
        拔除硬编码配置与 ATR 软兜底，风控参数必须由配置中心严格下发。
        """
        # 数据物理错误，直接不处理
        if cost <= 0 or current_price <= 0: 
            return False, "", 0.0

        profit_pct = (current_price - cost) / cost * 100.0
        drawdown_from_high = (highest_price - current_price) / highest_price * 100.0 if highest_price > 0 else 0.0

        # --- 1. 动态 ATR 止损 ---
        # 坚决信任前置治理层计算的 ATR，拔掉 0.03 的保底降级
        base_stop_dist = max(current_price * 0.04, min(atr * 2.5, current_price * 0.08))
        hard_stop_line = cost - base_stop_dist

        # 移动止盈 (Trailing Stop)
        # [核心修复]: 锁定历史最大利润率
        max_profit_pct = (highest_price - cost) / cost * 100.0
        
        if max_profit_pct > 15.0:
            hard_stop_line = highest_price - (atr * 2.0)
        elif max_profit_pct > 8.0:
            hard_stop_line = max(cost * 1.01, highest_price - (atr * 2.5))

        hard_stop_line = round(hard_stop_line, 2)

        # 触发硬止损 (静态判断)
        if current_price < hard_stop_line:
            if profit_pct > 8.0: # 区分出真正的移动止盈
                return True, f"锁利防线触及({hard_stop_line})", hard_stop_line # 触发 KEY_EXIT_LOCK
            elif profit_pct > 0: 
                return True, f"保本防线触及({hard_stop_line})", hard_stop_line # 触发 KEY_EXIT_PROTECT
            return True, f"盘中触价闪崩止损[动态]({hard_stop_line})", hard_stop_line
 
        # --- 1.5 左侧动能衰竭抢跑 (防闪崩，纯靠强校验入参) ---
        # 只要持仓超过1天且盈亏在 -4% 到 3% 之间，一旦情绪动能(rsi_rank)跌破强势区，直接抢跑！
        # 绝生死等跌破 MA20 承受额外亏损。
        if hold_days >= 1 and (-4.0 <= profit_pct <= 3.0):
            # 引入宏观环境豁免：熊市/震荡市严格抢跑，牛市放宽至极度弱势才抢跑
            rsi_threshold = 45.0 if regime_factor <= 0.6 else 35.0
            if rsi_rank < rsi_threshold:
                return True, f"动能衰竭(RSI转弱/Env:{regime_factor:.1f})左侧提前抢跑", hard_stop_line


        # --- 2. MA20 趋势破位 ---
        if current_price < ma20:
            if profit_pct < -3.0 and hold_days > 2: # 跌破 20日线且深套，才认输
                return True, f"📉破位MA20[跳空穿透]", hard_stop_line

        # --- 3. 时间止损 ---
        # 兜底修复：防止本地 JSON 未热更新导致全线罢工
        max_hold = risk_cfg.get('max_hold_days', 15)
        if hold_days >= max_hold and profit_pct < 5.0:
            return True, f"时间止损(滞涨{hold_days}天)", hard_stop_line

        # --- 4. 情绪极值止盈 ---
        if rsi_rank >= 98 and profit_pct > 8.0: 
            return True, f"RSI极值({int(rsi_rank)})止盈", hard_stop_line
            
        if rsi_rank >= 93 and profit_pct > 15.0: 
            return True, f"高位获利了结(RSI {int(rsi_rank)})", hard_stop_line

        # --- 5. 评分转弱斩仓 ---
        if current_score < 55 and hold_days > 3:
            return True, "评分转弱(<55)", hard_stop_line

        return False, "", hard_stop_line


    @staticmethod
    def get_board_limit_ratio(symbol, name=None):
        """
        [工业级核心] A股涨跌幅限制路由
        覆盖: 北交所(30%), 科创/创业(20%), ST(5%), 主板(10%)
        """
        symbol = str(symbol).strip().zfill(6)
        
        # 1. 北交所 (8xx, 4xx) - 优先级最高
        if symbol.startswith(('8', '4')):
            return 0.30
            
        # 2. 科创板 (688), 创业板 (300)
        if symbol.startswith(('688', '300')):
            return 0.20
            
        # 3. ST 股 (需依赖 name 字段，若无则默认非ST)
        if name and ('ST' in name or '退' in name):
            return 0.05
            
        # 4. 主板 (60, 00)
        return 0.10

    @staticmethod
    def check_entry_signal(row_data, final_score, strategy_name, regime_factor):
        """[买入信号判定 V5.0 - AI 觉醒特权版]"""
        # 👇 [新增：绝对物理防线，防止 VIP 越权]
        if final_score <= 0:
            return False, "系统级风控斩杀(一票否决/硬合规)"
        # --- 1. 严格提取核心指标 ---
        close = row_data['close']
        ma5 = row_data['ma5']
        ma10 = row_data['ma10']
        ma20 = row_data['ma20']
        vol_idx = row_data['volatility']
        rsi = row_data['rsi_rank']
        chop = row_data['chop']
        pv_corr = row_data['pv_corr']
        macd_slope = row_data['macd_slope']
        bias_20 = row_data['bias_20']
        smart_money = row_data['smart_money_rank']
        ai_prob = row_data['ai_score']
        
        # ==========================================================
        # [核心重构 1] AI 绝对红线 (大盘与胜率的双重底线)
        # ==========================================================
        # [逻辑闭环] 纯量化兜底值为 50.0。如果是纯量化模式，免除专属 AI 拦截，交由后续凡人通道处理
        if ai_prob != 50.0:
            if regime_factor < 0.5 and ai_prob < 40.0:
                return False, f"弱势环境AI要求提升 ({ai_prob:.2f} < 40.0)"
            if ai_prob < 35.0:  # <--- 降低到 35.0 
                return False, f"AI胜率极低拦截 ({ai_prob:.2f} < 35.0)"
                
        is_ai_vip = (ai_prob != 50.0) and (ai_prob >= 45.0)  # <--- VIP 线降到 45.0, 纯量化模式(50.0)不享受VIP


        if not is_ai_vip:
            # --- 凡人通道：必须老老实实看技术分、均线和低波动 ---
            if final_score < 75: 
                return False, "分低(<75)"
            if vol_idx > 3.85:
                return False, f"近期波动率过大失控 (Vol={vol_idx:.2f})"
            if rsi < 65:
                return False, f"动量不足拒绝弱势反弹 (RSIRank={rsi:.1f})"
            if chop > 60: 
                return False, f"趋势杂乱(Chop{chop:.0f})"
            # [布尔逻辑修复] 将致命的 or 改为 and。
            if pv_corr < -0.1 and smart_money < 40.0 and macd_slope <= 0:
                return False, "缩量假突(无资金流入)"
            if close < ma5:
                return False, "受制MA5(短期势弱)"
            if ma5 < ma10 and final_score < 90:
                return False, "短期均线空头"
        else:
            # --- VIP 通道：放宽波动率容忍度 (捕捉妖股主升浪) ---
            if vol_idx > 7.5: 
                return False, f"VIP防线: 极度波动失控 (Vol={vol_idx:.2f})"

        # --- 3. 一票否决区 (The Kill Switch - 即使 VIP 也要遵守的基本物理法则) ---
        if regime_factor < 0.4 and final_score < 85 and not is_ai_vip: 
            return False, "熊市禁追高(Env<0.4)"

        if rsi > 85.0: 
            return False, f"RSI过热({rsi:.1f})"
            
        # [核心底线：生命线防守] 
        if close < ma20:
            return False, "受制MA20(右侧未确认)"

        # [终极护城河：防高位接盘] 
        if bias_20 > 15.0:
            return False, f"高位乖离偏大(Bias{bias_20:.1f})"

        # 👇 ---------- [新增：对齐 F 区特殊物理形态拦截] ----------
        if "避雷针" in strategy_name:
            return False, "天量避雷针(极端派发)"
        if "乖离过大" in strategy_name:
            return False, "乖离偏大拦截"
        # 👆 --------------------------------------------------------

         # --- 4. 信号分类输出 ---
        if is_ai_vip:
            return True, f"🔥AI 特权看多起爆 ({ai_prob:.2f})"
        
        # [降维修复] 激活沉睡的主升浪捕捉通道
        if final_score >= 85 and smart_money > 60.0 and chop < 45:
            return True, "主升浪共振起爆"
            
        if ma5 > ma10 > ma20 and macd_slope > 0:
            return True, "趋势右侧追击"
            
        return True, "形态修复买入"



    @staticmethod
    def calculate_target_position(score, volatility, regime, price, available_cash, total_assets, force_full_pos=False, current_sector_exposure=0.0):
        """
        [决策大脑 V8.0] 标准化仓位计算接口 (适配 UI 与 回测)
        """
        # 如果 force_full_pos 为 True (通常用于单股回测)，则覆盖最大上限为 99%
        # 但依然受 _calc_position_math 内部的 波动率 约束 (高波依然不会满仓，这是特性)
        max_cap = 0.99 if force_full_pos else None 
        
        shares, desc, pct, color = QuantEngine._calc_position_math(
            score, volatility, regime, price, available_cash, total_assets, 
            max_pos_cap_override=max_cap, 
            current_sector_exposure=current_sector_exposure
        )
        return shares, desc, pct, color


    @staticmethod
    def dynamic_position_sizing(score, volatility_pct, market_regime, price=0.0, available_cash=0.0, total_assets_override=None, current_sector_exposure=0.0):
        """[UI 适配器] 保持接口兼容 (安全计算持仓市值防闪退)"""
        if total_assets_override is not None:
            est_assets = total_assets_override
        else:
            # 修复原有的直接 sum(dict.values()) 可能导致的崩溃
            holdings = CFG.HOLDINGS
            est_assets = available_cash
            for v in holdings.values():
                if isinstance(v, dict): est_assets += v.get('cost', 0) * v.get('volume', 0)
                else: est_assets += float(v)
        
        # UI 模式下保持默认 0.20 的风控限制
        shares, reason, pct, color = QuantEngine._calc_position_math(
            score, volatility_pct, market_regime, price, available_cash, est_assets, 
            max_pos_cap_override=0.20, 
            current_sector_exposure=current_sector_exposure
        )
        
        if shares == 0:
            return f"0股 ({reason})", color
        
        amt = int(shares * price)
        tag = " [博]" if reason == "博弈" else ""
        return f"{shares}股 (¥{amt}){tag}", color


    def load_ai_model(self, force_reload=False):
        """[代理接口 V8.0] 保持向后兼容，实际业务委托给独立服务"""
        AIModelServer.load(force_reload)
        
    @staticmethod
    def predict_ai_score(df_row):
        """[代理接口 V8.0] 底层已统一为百分制，直接返回即可"""
        return AIModelServer.predict_single(df_row)
        
        

    @staticmethod
    def generate_trend_tags(df_row, df_hist=None):
        """[Refactored] 统一形态标签生成器 - 严格索引版"""
        tags = []
        SR = SignalRegistry # 引用
        
        try:
            # 基础数据提取 - 废弃 .get() 兜底，强制依赖 Schema 对齐
            close = df_row['close']
            vol = df_row['vol']
            ma5, ma10, ma20 = df_row['ma5'], df_row['ma10'], df_row['ma20']
            
            # 1. 均线形态
            if ma5 > ma10 > ma20: tags.append(SR.TREND_BULL)
            elif ma5 < ma10 < ma20: tags.append(SR.TREND_BEAR)
            
            # 2. 量能与价格形态
            if df_hist is not None and not df_hist.empty:
                prev_vol = df_hist.iloc[-1]['vol'] if len(df_hist) >= 1 else vol
                if vol > prev_vol * 1.5: tags.append(SR.VOL_HEAVY)
                
                if df_row['pv_corr'] < -0.5 and df_row['pct'] > 0:
                    tags.append(SR.VOL_SHRINK_UP)
                
                if len(df_hist) >= 19:
                    recent_high = max(df_hist['high'].tail(19).max(), df_row['high'])
                    if close >= recent_high * 0.99: tags.append(SR.TREND_NEW_HIGH)
            
            # 3. 指标形态
            if df_row['vam'] > 1.8: tags.append(SR.TREND_BREAK_ATR)
            if df_row['er'] > 0.55: tags.append(SR.TREND_PURE)
            if df_row['bb_width'] < 8.0: tags.append(SR.VOL_EXTREME_SHRINK)

            # 4. 背离与金叉
            if df_row.get('macd_gold', False): 
                tags.append(SR.PTN_GOLD_CROSS_M)
            if df_row['macd_btm_div']: tags.append(SR.PTN_DIV_BTM)

            if df_row['macd_top_div']: tags.append(SR.PTN_DIV_TOP)
            
            if df_row['winner_rate'] > 95: tags.append(SR.VOL_LOCK)
            
            # 兼容 DataLayer 计算的背离
            if df_row['is_divergence']: tags.append(SR.PTN_DIV_TOP)

        except KeyError as ke:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("TAG_ERR", f"Schema 缺失核心字段: {ke}")
        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("TAG_ERR", str(e))
            
        return tags


    @staticmethod
    def _calc_base_signals(df, regime=0.5, pre_calculated=False, params=None):
        """[Refactored] 基础信号计算层 - 闭集标签版"""
        if df.empty: return df
        if params is None: params = CFG.STRATEGY_PARAMS
        p_get = params.get
        SR = SignalRegistry # 引用

        # ==================== A. 治理层接入 ====================
        if 'data_quality' not in df.columns: df['data_quality'] = 1.0
        
        if 'price' in df.columns:
            safe_price = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
            mask_p = safe_price > 0
            if mask_p.any(): df.loc[mask_p, 'close'] = safe_price[mask_p]
        
        if not pre_calculated:
            if 'name' not in df.columns: df['name'] = '-'
            df['name'] = df['name'].astype(str).fillna('-')
            mask_st = df['name'].str.contains("ST|退", na=False, regex=True)
            if mask_st.any():
                df = df[~mask_st].copy()
                if df.empty: return df
            if 'ma20' not in df.columns:
                df = QuantEngine.calc_tech_batch(df)

        df = FactorRegistry.enforce_std_schema(df, context_tag="BaseSignals")
        
        # ===================================================
        # [Gemini Fix: 宏观管道注入 - 严格防穿透版]
        # 如果 df 中自带 'regime_val' (历史回测)，绝对不能被参数覆盖
        # 只有在实盘当日截面扫描时，才使用传入的标量 regime
        # ===================================================
        if 'regime_val' in df.columns:
            df['env_regime'] = df['regime_val']
        else:
            df['env_regime'] = regime
            
        mask_trash = (df['data_quality'] < 0.5) | (df['close'] <= 0.01)

        # ==================== B. 标签生成 (使用常量) ====================
        desc = pd.Series("", index=df.index)
        
        close = df['close']
        
        # [核心防线升级] 彻底消除 MA 幻觉，确保回写 df 且补齐遗漏的 ma60
        # [Bug 2 修复] 提取需要修正的 ma20 掩码，若发生修正，必须同步重算衍生指标
        mask_ma20_bad = df['ma20'] <= 0.01
        
        df['ma5'] = np.where(df['ma5'] > 0.01, df['ma5'], close)
        df['ma10'] = np.where(df['ma10'] > 0.01, df['ma10'], close)
        df['ma20'] = np.where(~mask_ma20_bad, df['ma20'], close)
        df['ma60'] = np.where(df.get('ma60', pd.Series(0.0, index=df.index)) > 0.01, df['ma60'], close)
        
        # [Bug 2 修复] 如果触发了 ma20 的强制修正，同步修正依赖它的 bias 和 ratio，消除脱节带来的幽灵熔断
        if mask_ma20_bad.any():
            df.loc[mask_ma20_bad, 'bias_20'] = 0.0  # ma20 被替换成了 close，乖离必然为 0.0
            df.loc[mask_ma20_bad, 'ma_5_20_ratio'] = (df.loc[mask_ma20_bad, 'ma5'] - df.loc[mask_ma20_bad, 'ma20']) / (df.loc[mask_ma20_bad, 'ma20'] + 1e-9) * 100
        
        # 重新绑定局部变量供下方逻辑使用
        ma5, ma10, ma20 = df['ma5'], df['ma10'], df['ma20']
        
        vol = df['vol']; vol_prev = df['vol_prev']
        pct = df['pct']; flow = df['flow']
        bias = df['bias_20']; chop = df['chop']; rsrs = df['rsrs_wls']; vam = df['vam']
        macd = df['macd']; macd_hist = df['macd_hist']; macd_gold = df['macd_gold']
        is_monster = df['is_monster']; rsi = df['rsi_rank']; amihud = df['amihud']
        winner = df['winner_rate']; volatility = df['volatility']

        # [严格判定] 只有当长短均线全部错开，才授予多空标签
        desc += np.where((ma5 > ma10) & (ma10 > ma20) & (ma20 > 0.01), f"{SR.TREND_BULL}|", "")
        desc += np.where((ma5 < ma10) & (ma10 < ma20) & (ma20 > 0.01), f"{SR.TREND_BEAR}|", "")


        ref_vol = np.where(df['vol_prev'] <= 0, 9e99, df['vol_prev'])
        desc += np.where(vol > ref_vol * 1.5, f"{SR.VOL_HEAVY}|", "")
        
        desc += np.where((df['pv_corr'] < -0.5) & (df['pct'] > 0), f"{SR.VOL_SHRINK_UP}|", "")
        roll_high = df['high'].rolling(20, min_periods=1).max().fillna(df['high'])
        if len(df) > 5:
            desc += np.where(close >= roll_high * 0.99, f"{SR.TREND_NEW_HIGH}|", "")
            
        desc += np.where(macd_gold, f"{SR.PTN_GOLD_CROSS_M}|", "")
        desc += np.where(df['macd_top_div'] | df['is_divergence'], f"{SR.PTN_DIV_TOP}|", "")
        
        desc += np.where(vam > 1.8, f"{SR.TREND_BREAK_ATR}|", "")
        desc += np.where(df['er'] > 0.55, f"{SR.TREND_PURE}|", "")
        desc += np.where(df['bb_width'] < 8.0, f"{SR.VOL_EXTREME_SHRINK}|", "")
        desc += np.where(df['winner_rate'] > 95, f"{SR.VOL_LOCK}|", "")

        # ==================== C. 评分逻辑 ====================
        t_score = pd.Series(p_get('score_base', 50), index=df.index)
        
        mask_bias_ok = (bias > p_get('bias_attack_min', 3)) & (bias < p_get('bias_attack_max', 12))
        t_score += np.where(mask_bias_ok, 10, 0)
        desc += np.where(mask_bias_ok, f"{SR.PTN_BIAS_ATTACK}|", "")
        
        t_score += np.where(chop < p_get('chop_trend_limit', 40), 10, 0)
        
        mask_rsrs = rsrs > p_get('rsrs_strong_limit', 0.9)
        t_score += np.where(mask_rsrs, 10, 0)
        desc += np.where(mask_rsrs, f"{SR.TREND_RSRS_STRONG}|", "")
        
        t_score += np.where(vam > p_get('vam_strong_limit', 1.5), 10, 0)
        
        t_score += np.where(macd > 0, 5, 0)
        mask_mg = (macd > 0) & macd_gold
        t_score += np.where(mask_mg, 5, 0)
        desc += np.where(mask_mg, f"{SR.PTN_MACD_IGNITE}|", "")
        
        t_score += np.where(is_monster, 20, 0)
        desc += np.where(is_monster, f"{SR.RISK_MONSTER_PASS}|", "")

        mask_not_monster = ~is_monster
        rsi_lim = p_get('rsi_overbought', 80)
        mask_over = mask_not_monster & (rsi > rsi_lim)
        mask_lure = mask_over & (amihud > p_get('amihud_high', 0.3))
        mask_lock = mask_over & (amihud < p_get('amihud_low', 0.05)) & (~mask_lure)
        mask_heat = mask_over & (~mask_lure) & (~mask_lock)

        t_score -= np.where(mask_lure, (rsi - rsi_lim) * 3, 0)
        desc += np.where(mask_lure, f"{SR.RISK_LURE}|", "")
        
        t_score += np.where(mask_lock, 5, 0)
        desc += np.where(mask_lock, f"{SR.RISK_HIGH_LOCK}|", "")
        
        t_score -= np.where(mask_heat, (rsi - rsi_lim) * 2, 0)
        desc += np.where(mask_heat, f"{SR.RISK_OVERHEAT}|", "")

        mask_slow = mask_not_monster & (winner > p_get('win_rate_high', 85)) & (rsi < p_get('rsi_high', 75))
        t_score += np.where(mask_slow, 10, 0)
        desc += np.where(mask_slow, f"{SR.TREND_SLOW_BULL}|", "")

        t_score += np.where(flow > 0, 5, 0)
        # [核心修复1] 将标量 regime 替换为向量化 df['env_regime']，阻断回测时的未来函数泄漏
        t_score += np.where(df['env_regime'] > p_get('regime_bonus_limit', 0.6), 5, 0)

        r_score = pd.Series(p_get('score_base', 50), index=df.index)

        mask_low_vol = (volatility < p_get('vol_rev_limit', 1.5)) & (pct > 0)
        r_score += np.where(mask_low_vol, 10, 0)
        desc += np.where(mask_low_vol, f"{SR.PTN_LOW_VOL_START}|", "")

        mask_oversold = rsi < p_get('rsi_oversold', 20)
        oversold_bonus = np.round(25 * df['env_regime']) 
        r_score += np.where(mask_oversold, oversold_bonus, 0)
        desc += np.where(mask_oversold, f"{SR.RISK_OVERSOLD}|", "")
        
        r_score += np.where(df['kdj_gold'], 10, 0)
        
        mask_btm = df['macd_btm_div'] 
        r_score += np.where(mask_btm, 10, 0)
        desc += np.where(mask_btm, f"{SR.PTN_DIV_BTM}|", "")
        
        mask_blood = winner < p_get('win_rate_low', 10)
        r_score += np.where(mask_blood, 15, 0)
        desc += np.where(mask_blood, f"{SR.RISK_BLOOD}|", "")
        # ==================== D. AI 预测 (解耦版) ===================
        if not pre_calculated and not AIModelServer._is_loaded:
            AIModelServer.load()

        if not pre_calculated and AIModelServer._is_loaded:
            try:
                valid_idx = df.index[~mask_trash]
                if not valid_idx.empty:
                    df = FactorRegistry.enforce_std_schema(df, context_tag="AIPrediction")
                    # 直接调用独立服务的高性能批量接口
                    pred_scores = AIModelServer.predict_batch(df.loc[valid_idx])
                    if pred_scores is not None:
                        # 预测结果已放大到0-100
                        df.loc[valid_idx, 'ai_score'] = pred_scores
                        df['ai_score'] = df['ai_score'].round(4)
            except Exception as e: 
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("AI_BATCH_ERR", str(e))

        df['trend_score'] = t_score
        df['reversal_score'] = r_score
        df['_trend_desc_tech'] = desc 
        df['_is_trash'] = mask_trash 
        
        return df
 

    @staticmethod
    def _calc_composite_score(df, breadth_panic=False, target_mode=False, pre_calculated=False, params=None):
        """[Refactored] 决策合成层 - 闭集标签与板块限额版 V10.0"""
        if df.empty: return df
        if params is None: params = CFG.STRATEGY_PARAMS
        p_get = params.get
        SR = SignalRegistry # 引用

        # ==================== A. 治理层强介入 ====================
        df = FactorRegistry.enforce_std_schema(df, context_tag="CompositeScore")

        # ==================== B. 现场恢复 ====================
        mask_tech_desc = df['_trend_desc_tech'] != ""
        # [性能修复] 抽离为独立 Series 进行向量化操作，避免 DataFrame 高度碎片化
        temp_desc = df['trend_desc'].astype(str).copy()
        temp_desc[mask_tech_desc] = df.loc[mask_tech_desc, '_trend_desc_tech']
        mask_trash = df['_is_trash'] | (df['data_quality'] < 0.5)

        # ==================== C. 拥挤度 (宏观与微观融合版) ====================
        crowding_penalty = pd.Series(0.0, index=df.index)

        # --- 1. 普适风控指标 (实盘全模式与历史回测 均有效) ---
        # 微观拥挤 (爆量滞涨)
        vol_surge = df['vol'] > (df['vol_prev'] * 3.0)
        mask_micro_crowd = vol_surge & (df['rsi_rank'] > 75) & (df['pct'] < 3.0) & (df['pct'] > -3.0)
        crowding_penalty += np.where(mask_micro_crowd, 15.0, 0.0)
        temp_desc[mask_micro_crowd] += f"{SR.CROWD_MICRO}|"

        # 主力撤退 (高位缩量+上影线)
        mask_smart_money_flee = (df['bias_vwap'] > 8.0) & (df['vol'] < df['vol_prev']) & (df['upper_shadow_ratio'] > 0.4)
        crowding_penalty += np.where(mask_smart_money_flee, 20.0, 0.0)
        temp_desc[mask_smart_money_flee] += f"{SR.SMART_MONEY_FLEE}|"

        # --- 2. 实时专属指标 (仅实盘混合扫描生效，阻断未来函数) ---
        if not target_mode and not pre_calculated:
            # 内部交易日志拥挤度 (防御系统自身实盘踩踏)
            c_map = QuantEngine._analyze_crowding(df)
            def get_cp(s): return c_map.get(str(s).zfill(6), {}).get('penalty', 0.0)
            def get_cd(s): 
                d = c_map.get(str(s).zfill(6), {}).get('desc', "")
                return d+"|" if d else ""
            
            crowding_penalty += df['symbol'].map(get_cp).fillna(0.0)
            crowding_tags = df['symbol'].map(get_cd).fillna("")
            temp_desc += crowding_tags

            # 外部宏观板块拥挤度 (防高位接盘)
            # 依赖实盘网络接口，回测中受 FactorRegistry 保护默认为 0.0，安全静默
            macro_crowding = df['sector_crowding']
            
            # 极度拥挤
            mask_macro_danger = macro_crowding >= 12.0
            crowding_penalty += np.where(mask_macro_danger, 30.0, 0.0)
            temp_desc[mask_macro_danger] += f"{SR.CROWD_TRAMPLE}(板块热度>12%)|"
            
            # 局部过热
            mask_macro_hot = (macro_crowding >= 8.0) & (macro_crowding < 12.0)
            crowding_penalty += np.where(mask_macro_hot, 10.0, 0.0)
            temp_desc[mask_macro_hot] += f"{SR.CROWD_LAG}(板块过热)|"

        # ==================== D. AI 融合 ====================
        ai_prob = df['ai_score']
        ai_penalty = p_get('ai_penalty_weight', 10.0)
        
        conds = [ai_prob > 75, ai_prob > 60, ai_prob > 40, ai_prob < 30]
        choices = [15, 5, 0, -abs(ai_penalty)]
        ai_bonus = np.select(conds, choices, default=0)
        
        temp_desc[conds[0]] += f"{SR.AI_SUPER}|"
        temp_desc[conds[1]] += f"{SR.AI_LONG}|"

        llm_val = df['llm_score']
        veto_mult = pd.Series(1.0, index=df.index)
        mask_llm_valid = llm_val != -1.0
        
        mask_llm_kill = mask_llm_valid & (llm_val < 40)
        veto_mult[mask_llm_kill] = 0.0
        temp_desc[mask_llm_kill] += f"{SR.AI_KILL}|"

        # ==================== E. 算分 ====================
        t_score = df['trend_score']
        r_score = df['reversal_score']
        tech_final = np.maximum(t_score, r_score)
        
        final = (tech_final + ai_bonus) * veto_mult
        final -= crowding_penalty

        # ==================== F. 硬性风控 ====================
        bias = df['bias_20']
        is_monster = df['is_monster']
        
        limit_bias = np.where(is_monster, p_get('bias_monster_limit', 25), p_get('bias_normal_limit', 15))
        mask_bias_fail = bias > limit_bias
        
        # [性能修复] 同样将策略名称抽离为独立 Series
        temp_strat = df['strategy_name'].astype(str).copy()
        temp_strat[mask_bias_fail] = SR.STRAT_BIAS_FAIL
        
        # --- [新增: 3D 量价交叉惩罚矩阵 (高空衰竭核查器)] ---
        # 提取 X, Y, Z 轴向量状态
        is_high_altitude = (bias > 8.0) | (df['rsi_rank'] > 80.0)
        # 上影线极长 (标准可由配置下发，默认 0.4 即可触发警报)
        has_long_shadow = df['upper_shadow_ratio'] > p_get('shadow_limit', 0.40)
        
        # 燃料/量能偏离度 Z轴
        is_shrinking_vol = df['vol_zscore'] < 0.5
        is_huge_vol = df['vol_zscore'] > 2.5
        
        # 陷阱 A: 高位缩量诱多 (主力跟风意愿匮乏，撤退)
        mask_trap_exhaustion = is_high_altitude & has_long_shadow & is_shrinking_vol
        final = np.where(mask_trap_exhaustion, final - 15.0, final)
        temp_desc[mask_trap_exhaustion] += f"{SR.RISK_SHRINK_LURE}|"
        
        # 陷阱 B: 天量避雷针 (高位爆量且长上影，主力坚决派发)
        mask_trap_distribution = is_high_altitude & has_long_shadow & is_huge_vol
 
        temp_strat[mask_trap_distribution] = SR.STRAT_NEEDLE
        temp_desc[mask_trap_distribution] += f"{SR.RISK_HUGE_NEEDLE}|"
        # -------------------------------------------------------------
        
        if breadth_panic and not target_mode:
            final[:] = 0
            temp_strat[:] = SR.STRAT_FUSE

        # ==================== G. 结果输出 ====================
        final = np.clip(final, 0, None)
        df['final_score'] = np.round(final, 1)

        # 👇 ---------- [终极修复: 物理防线 100% 左移对齐 (置于算分与Rounding之后)] ----------
        ai_prob = df['ai_score']
        # [调整] VIP 线从 52.0 降到 45.0
        is_ai_vip = (ai_prob != 50.0) & (ai_prob >= 45.0)
        curr_score = df['final_score']
        
        # 1. 物理绝对红线 (所有标的众生平等，严格映射 check_entry_signal)
        mask_base_dead = (
            (df['close'] < df['ma20']) |  
            (df['rsi_rank'] > 85.0) |     
            (df['bias_20'] > 15.0) |      
            # [调整] 绝对斩杀线降为 35.0，弱势环境降为 40.0
            ((ai_prob != 50.0) & ((ai_prob < 35.0) | ((df['env_regime'] < 0.5) & (ai_prob < 40.0))))
        )
        
        # 2. 凡人通道红线
        mask_mortal_dead = (~is_ai_vip) & (
            # [调整] 波动率容忍度从 3.85 提升到 6.5 (容忍热点股的正常洗盘)
            (df['volatility'] > 6.5) |   
            (df['rsi_rank'] < 65) |       
            (df['chop'] > 60) |           
            (df['close'] < df['ma5']) |   
            ((df['pv_corr'] < -0.1) & (df['smart_money_rank'] < 40.0) & (df['macd_slope'] <= 0)) | 
            (curr_score < 75) |                
            ((df['ma5'] < df['ma10']) & (curr_score < 90)) | 
            ((df['env_regime'] < 0.4) & (curr_score < 85))   
        )
        
        # 3. VIP 通道红线
        mask_vip_dead = is_ai_vip & (df['volatility'] > 7.5)
        
        mask_all_dead = mask_base_dead | mask_mortal_dead | mask_vip_dead | mask_bias_fail | mask_trap_distribution
        
        # 👇 ---------- [架构级修正: 拦截标志位替代物理归零] ----------
        # 绝不能物理归零 final_score，否则会触发持仓的误杀(current_score < 55)！
        # 新增拦截标志位，供选股器(filter_top_candidates)进行 LLM 截断
        df['_is_entry_valid'] = ~mask_all_dead
        # 👆 ----------------------------------------------------------------------------------

        # 重新基于洗礼后的真实分数计算存活状态
        mask_alive = df['final_score'] > 0

        mask_t_ok = t_score >= p_get('score_threshold_name', 75)
        mask_r_ok = r_score >= p_get('score_threshold_name', 75)
        mask_ai_ok = ai_prob >= 80
        
        s_names = pd.Series("", index=df.index)
        s_names += np.where(mask_alive & mask_t_ok, f"{SR.STRAT_TREND}|", "")
        s_names += np.where(mask_alive & mask_r_ok, f"{SR.STRAT_REV}|", "")
        s_names += np.where(mask_alive & mask_ai_ok, f"{SR.STRAT_ALERT}|", "")
        
        mask_curr_empty = temp_strat == ""
        mask_upd = mask_alive & mask_curr_empty
        temp_strat[mask_upd] = s_names[mask_upd]
        
        mask_obs = mask_alive & (temp_strat == "")
        temp_strat[mask_obs] = SR.STRAT_WATCH
        
        mask_dead = (~mask_alive) & (temp_strat == "") & (~mask_trash)
        temp_strat[mask_dead] = SR.STRAT_RISK
        
        # [新增] 将触碰物理红线的标的强制戴上风控帽子 (UI 显影)
        temp_strat[mask_all_dead] = SR.STRAT_RISK
        
        # [性能修复] 将最终拼接完毕的 Series 一次性赋值回 DataFrame
        df['trend_desc'] = temp_desc.str.rstrip('|')
        df['strategy_name'] = temp_strat.str.rstrip('|')

        if mask_trash.any():
            df.loc[mask_trash, 'final_score'] = -999
            df.loc[mask_trash, 'strategy_name'] = SR.STRAT_BAD_DATA
            mask_empty_desc = mask_trash & (df['trend_desc'] == "")
            df.loc[mask_empty_desc, 'trend_desc'] = SR.DESC_INVALID_DATA
            
        return df




    @staticmethod
    @safe_compute("STRATEGY_MAIN") 
    def strategy_scoring(df, phase, regime, breadth_panic=False, target_mode=False, pre_calculated=False, params=None):
        """
        [Facade] 统一入口 - 带显影剂与终极防线
        """
        if df is None or df.empty: return pd.DataFrame()
        
        # 记录初始状态
        start_len = len(df)
        
        # 1. 计算基础技术面
        df = QuantEngine._calc_base_signals(df, regime=regime, pre_calculated=pre_calculated, params=params)
        
        # [显影剂] 检查阶段1后的完整性
        if len(df) != start_len:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("DATA_LOSS", f"BaseSignals 导致数据丢失: {start_len} -> {len(df)}")
        
        # 2. 计算综合得分 (内部已剥离斩杀逻辑)
        df = QuantEngine._calc_composite_score(
            df, 
            breadth_panic=breadth_panic, 
            target_mode=target_mode, 
            pre_calculated=pre_calculated,
            params=params
        )

        # ==================== 2.5 [工业级升级] 终极板块赢家通吃 ====================
        # 必须放在所有算分逻辑之后(即便是实盘二次融合 AI 分数后)，确保最后这一刀绝不漏网
        if 'ind' in df.columns and len(df) > 1 and df['symbol'].nunique() > 1 and not target_mode:
            df['temp_final'] = df['final_score']
            
            # 必须分数及格且不是垃圾数据才参与排名
            mask_potential = (df['final_score'] >= 60) & (df['data_quality'] >= 0.5) & (df['ind'] != 'Self') & (df['ind'] != '未知') & (df['ind'] != 'ETF/LOF')
            
            if mask_potential.any():
                # 按照行业分组，基于当前最终分数进行降序排名
                df.loc[mask_potential, 'sector_rank'] = df[mask_potential].groupby('ind')['temp_final'].rank(method='first', ascending=False)
                sec_limit = getattr(CFG, 'SECTOR_LIMIT', 1) # 默认每个板块只给 1 个名额
                
                mask_sector_kill = mask_potential & (df['sector_rank'] > sec_limit)
                
                # 👇 ---------- [架构级修正: 持仓免死金牌] ----------
                # 防止健康的持仓股仅仅因为不是本板块当天的第一名，就被强制归零，从而引发 audit_holdings 的 55分斩仓误杀！
                if hasattr(CFG, 'HOLDINGS') and CFG.HOLDINGS:
                    holdings_syms = [str(k).zfill(6) for k in CFG.HOLDINGS.keys()]
                    mask_holding = df['symbol'].astype(str).str.zfill(6).isin(holdings_syms)
                    mask_sector_kill = mask_sector_kill & (~mask_holding)
                # 👆 ------------------------------------------------
                
                # 终极斩杀
                df.loc[mask_sector_kill, 'final_score'] = 0
                df.loc[mask_sector_kill, 'strategy_name'] = SignalRegistry.STRAT_SECTOR_LIMIT
            
            df.drop(columns=['temp_final', 'sector_rank'], inplace=True, errors='ignore')

        # 3. Schema 强制对齐 (SSOT 最后一道防线)
        df = QuantEngine._ensure_features_exist(df)
        return df


    @staticmethod
    def audit_holdings(df, regime_factor):
        """[Refactored] 持仓审计 - 严格索引无黑洞版 + 独立状态机防回撤"""
        alerts = {} 
        holdings = CFG.HOLDINGS 
        now_ts = time.time()
        SR = SignalRegistry # 引用

        # [新增防线] 强制执行 Schema 对齐，确保所有所需字段 100% 存在
        df = FactorRegistry.enforce_std_schema(df, context_tag="AuditHoldings")

        # =======================================================
        # [状态机护城河] 独立于 holdings 建立 highest_prices 字典
        # 避免被 UI 强刷覆盖，并清理已卖出的残留数据防内存泄露
        # =======================================================
        if 'highest_prices' not in CFG.data:
            CFG.data['highest_prices'] = {}
        need_save_cfg = False
        
        active_syms = set(holdings.keys())
        for k in list(CFG.data['highest_prices'].keys()):
            if k not in active_syms:
                del CFG.data['highest_prices'][k]
                need_save_cfg = True

        for _, row in df.iterrows():
            try:
                # 强索引，消除静默字典漏洞
                sym = str(row['symbol']).strip().zfill(6)
                if sym in holdings:
                    h_data = holdings[sym]
                    cost = h_data.get('cost', 0.0) if isinstance(h_data, dict) else float(h_data)
                    ts = h_data.get('ts', 0) if isinstance(h_data, dict) else 0
                    
                    # 统一使用宪法定义的物理基础量 close，兼容 price
                    current = row['close']
                    if current <= 0.01 and 'price' in row:
                         current = row['price']
                    
                    if cost <= 0.01 or current <= 0.01: continue 
                    profit_pct = (current - cost) / cost * 100
                    
                    # [核心拔除黑洞] 严格索取因子，拒绝 .get() 兜底
                    atr = row['atr']
                    ma20 = row['ma20']
                    rsi_rank = row['rsi_rank']
                    
                    # =======================================================
                    # [彻底修复移动止盈] 独立持久化状态机提取与更新
                    # =======================================================
                    recorded_high = CFG.data['highest_prices'].get(sym, current)
                    today_high = row['high'] if row['high'] > 0 else current
                    high_price = max(recorded_high, today_high, current)
                    
                    if high_price > recorded_high:
                        CFG.data['highest_prices'][sym] = float(high_price)
                        need_save_cfg = True
                    final_score = row['final_score']
                    stock_name = row['name']

                    # 剥离周末时间，获取真实交易日天数。保留 if ts > 0 兜底，防止 1970 年纪元穿透导致死循环！
                    held_days = BeijingClock.get_trading_days(ts, now_ts) if ts > 0 else 0
                    
                    should_sell, reason, final_stop = QuantEngine.check_exit_signal_v2(
                        sym, cost, current, high_price, atr, 
                        rsi_rank, CFG.data.get('risk', {}), regime_factor, 
                        final_score, hold_days=held_days, ma20=ma20
                    )
                    
                    # --- 决策生成 ---
                    action = SR.ACT_HOLD # 默认状态
                    reason_extras = []
                    
                    if should_sell:
                        if SR.KEY_EXIT_PROTECT in reason:
                            action = SR.ACT_TAKE_PROFIT
                            reason_extras.append(f"触发保本({final_stop:.2f})")
                        elif SR.KEY_EXIT_LOCK in reason:
                            action = SR.ACT_TAKE_PROFIT
                            reason_extras.append(f"棘轮止盈({final_stop:.2f})")
                        elif SR.KEY_EXIT_TIME in reason:
                            action = SR.ACT_SWAP
                            reason_extras.append(f"资金效率低(滞涨{held_days}天)")
                        elif SR.KEY_EXIT_MA20 in reason:
                            action = SR.ACT_TREND_BREAK
                            reason_extras.append(f"有效跌破MA20")
                        elif SR.KEY_EXIT_RSI in reason:
                            action = SR.ACT_HIGH_SELL
                            reason_extras.append(f"情绪见顶")
                        else:
                            # 止损逻辑 (引入 AI 一票否决)
                            ai_prob = row.get('ai_score', 50.0) 
                            is_strong_buy = (final_score >= 75) and (ai_prob >= 40.0)
                            
                            if is_strong_buy:
                                action = SR.ACT_CRITICAL
                                reason_extras.append(f"触及止损({final_stop:.2f}) 但技术/AI双强，建议减半观察")
                            else:
                                action = SR.ACT_STOP_LOSS
                                reason_extras.append(f"破位({final_stop:.2f})")
                                
                    elif rsi_rank > 95 and profit_pct > 10.0:
                        action = SR.ACT_SIGNAL_PROFIT
                        reason_extras.append(f"🚀RSI极值({rsi_rank:.0f})")
                    elif regime_factor <= 0.2:
                        action = SR.ACT_FORCE_REDUCE
                        reason_extras.append("熊市保护")
                    
                    color = "ff5555" if profit_pct < 0 else "55ff55"
                    profit_str = f"[color={color}]{profit_pct:+.2f}%[/color]"
                    
                    full_msg = f"{stock_name}: {action} [{profit_str}] {' '.join(reason_extras)}"
                    alerts[sym] = full_msg
            except KeyError as ke:
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("AUDIT_ERR", f"持仓审计失败，缺失核心字段: {ke}")
            except Exception as e:
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("AUDIT_ERR", str(e))
                
        # =======================================================
        # 统一在循环结束后保存，避免 I/O 频繁刷盘导致界面卡死
        # =======================================================
        if need_save_cfg:
            CFG.save()
            
        return alerts
