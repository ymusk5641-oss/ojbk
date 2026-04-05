import pandas as pd
import numpy as np
import traceback
import time

class StandardSchema:
    """
    [数据宪法] 全局统一列名与单位定义
    """
    # 核心字段标准
    # close, open, high, low, amount -> float (元)
    # vol -> float (股) [注意：绝对禁止使用手]
    # pct -> float (%)
    REQUIRED_COLS = ['date', 'open', 'high', 'low', 'close', 'vol']
    
    # 允许的浮点误差
    FLOAT_TOLERANCE = 1e-9

class DataAdapter:
    """
    [适配器] 负责将不同来源的异构数据转换为标准态
    核心逻辑复刻自原 DataVerifier，确保逻辑零丢失。
    """
    @staticmethod
    def _safe_to_numeric(df, cols):
        for c in cols:
            if c in df.columns:
                # [核心修复] 强制转为数值，无法转换的变为 NaN (0.0)
                # 这一步是防止 'str' 进入数学计算的关键
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        return df

    @staticmethod
    def adapt(df, source_name="Unknown"):
        """
        统一单位适配 (增强版: 全面数值化 V2.0)
        """
        if df is None or df.empty: return df
        
        # 1. 确保数值类型安全 (扩容核心修复)
        # Baostock 的 peTTM/pbMRQ/pctChg 都是 string，必须在此转为 float
        # 否则后续 .abs() 或数学运算会直接崩盘
        # [CRITICAL FIX] 加入 'pe', 'pb', 'pct'
        target_numeric_cols = [
            'close', 'vol', 'amount', 'open', 'high', 'low', 
            'pe', 'pb', 'pct' 
        ]
        df = DataAdapter._safe_to_numeric(df, target_numeric_cols)

        # -----------------------------------------------------------
        # 逻辑复刻 A: 拥有 "成交额(amount)" (Baostock/东财) -> 物理验证
        # -----------------------------------------------------------
        if 'amount' in df.columns and df['amount'].gt(0).any():
            # 抽取非零行进行抽样验证
            sample = df[df['amount'] > 0].iloc[:10]
            if not sample.empty:
                # 计算隐含乘数: Amount / (Price * Vol)
                # 理论值: 股=1.0, 手=100.0
                try:
                    # 加 1e-9 防止除零
                    implied_multiplier = (sample['amount'] / (sample['close'] * sample['vol'] + 1e-9)).mean()
                    
                    # [原逻辑保持] 允许 20% 的滑点误差 (处理涨跌停/均价差异)
                    if 80 < implied_multiplier < 120:
                        df['vol'] = df['vol'] * 100 # 判定为 [手] -> 转 [股]
                    elif 0.8 < implied_multiplier < 1.2:
                        pass # 判定为 [股] -> 保持不变
                    # [新增] 如果既不是1也不是100，可能是单位异常，但为了稳健暂不处理，依赖后续 Sanitizer
                except: pass
        
        # -----------------------------------------------------------
        # 逻辑复刻 B: 无 "成交额" (腾讯) -> 量级验证 + 来源特判
        # -----------------------------------------------------------
        elif source_name == "Tencent":
            # 腾讯接口默认是 [手]，但需要防止它已经是 [股] (如超大盘股)
            avg_vol = df['vol'].mean()
            # [原逻辑保持] 阈值设定：1亿。如果日均成交量小于1亿，极大概率是手
            if avg_vol < 100000000: 
                df['vol'] = df['vol'] * 100
        
        # -----------------------------------------------------------
        # 逻辑复刻 C: 盲测兜底
        # -----------------------------------------------------------
        else:
            avg_vol = df['vol'].mean()
            # [原逻辑保持] 极小量级判定为手
            if 0 < avg_vol < 500: 
                df['vol'] = df['vol'] * 100

        return df

    # =======================================================
    # 以下为从 DataLayer 下沉的异构协议解析器 (Parser)
    # =======================================================

    @staticmethod
    def _safe_float(value):
        """安全浮点数转换，防止空值崩溃"""
        try: return float(value)
        except: return 0.0

    @staticmethod
    def parse_tencent_snapshot_line(line):
        """
        [解析器] 单行腾讯快照协议解析
        接管原 data.py 的 _safe_parse_snapshot 逻辑，集中化解码。
        """
        try:
            if '="' not in line or '~' not in line: return None
            parts = line.split('="')
            if len(parts) < 2: return None
            data_str = parts[1].strip('" ;')
            p = data_str.split('~')
            
            # 硬校验字段长度
            if len(p) < 40: return None
            
            price = DataAdapter._safe_float(p[3])
            if price <= 0: return None
            
            return {
                'symbol': p[2],
                'name': p[1],
                'price': price,
                'pre_close': DataAdapter._safe_float(p[4]),
                'pct': DataAdapter._safe_float(p[32]),
                'pe': DataAdapter._safe_float(p[39]),
                'pb': DataAdapter._safe_float(p[46]),
                'open': DataAdapter._safe_float(p[5]),
                'high': DataAdapter._safe_float(p[33]),
                'low': DataAdapter._safe_float(p[34]),
                # [协议解码] 腾讯接口特性：手 -> 股, 万 -> 元
                'vol': DataAdapter._safe_float(p[36]) * 100,
                'amount': DataAdapter._safe_float(p[37]) * 10000 if len(p) > 37 and p[37] else 0.0,
                'buy1': DataAdapter._safe_float(p[9]),
                'sell1': DataAdapter._safe_float(p[19]),
                'close': price 
            }
        except Exception:
            return None

    @staticmethod
    def parse_tencent_snapshot_batch(raw_text):
        """[解析器] 批量腾讯快照协议解析"""
        import time
        rows = []
        for line in raw_text.split(';'):
            if not line: continue
            parsed = DataAdapter.parse_tencent_snapshot_line(line)
            if parsed:
                parsed['timestamp'] = time.time()
                rows.append(parsed)
        return rows

    @staticmethod
    def parse_tencent_kline(json_data, t_code):
        """
        [解析器] 腾讯历史 K 线数据解析
        接管原 data.py 的杂乱 JSON 提取逻辑。
        """
        try:
            if not isinstance(json_data, dict): return []
            inner_data = json_data.get('data')
            if not isinstance(inner_data, dict): return []
            
            stock_data = inner_data.get(t_code)
            if not isinstance(stock_data, dict): return []
            
            # 优先取前复权，没有则取不复权
            raw_data = stock_data.get('qfqday', []) or stock_data.get('day', [])
            # 严格截取 OHLCV 等前6个字段
            clean_data = [row[:6] for row in raw_data if len(row) >= 6]
            return clean_data
        except Exception:
            return []


class DataSanitizer:
    """
    [清洗器 V4.0 - 工业级参数持久化版]
    职责: 统一处理空值与异常值。
    新增: 引入特征参数提取 (compute_mad_params) 与外部尺子应用 (mad_params)。
    彻底解决实盘单行预测时无法计算横截面极值的理论盲区。
    """
    @staticmethod
    def sanitize(df):
        if df is None or df.empty: return df
        if 'high' in df.columns and 'low' in df.columns:
            mask_swap = df['high'] < df['low']
            if mask_swap.any(): df.loc[mask_swap, ['high', 'low']] = df.loc[mask_swap, ['low', 'high']].values
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [c for c in price_cols if c in df.columns]
        if existing_price_cols:
            df[existing_price_cols] = df[existing_price_cols].replace(0, np.nan)
            df[existing_price_cols] = df[existing_price_cols].ffill()
            df[existing_price_cols] = df[existing_price_cols].bfill()
        vol_cols = ['vol', 'amount']
        for c in vol_cols:
            if c in df.columns: df[c] = df[c].fillna(0.0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def winsorize_mad(series, n=5.0):
        """传统动态计算 MAD (用于向后兼容和训练时兜底)"""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or pd.isna(mad): return series
        upper_limit = median + n * mad
        lower_limit = median - n * mad
        return series.clip(lower=lower_limit, upper=upper_limit)

    @staticmethod
    def compute_mad_params(df, feature_cols):
        """
        [新增核心] 提取特征尺子 (提取全量数据的 Median 和 MAD)
        用于在训练时固化参数，落盘给实盘使用。
        """
        params = {}
        if df is None or df.empty: return params
        for col in feature_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors='coerce')
                s_valid = s.dropna()
                if s_valid.empty:
                    params[col] = {'median': 0.0, 'mad': 0.0}
                else:
                    med = float(s_valid.median())
                    mad = float((s_valid - med).abs().median())
                    params[col] = {'median': med, 'mad': mad}
        return params

    @staticmethod
    def clean_machine_learning_features(df, feature_cols, mad_params=None):
        """
        [核心重构] 机器学习特征清洗专用流水线
        如果传入了 mad_params(历史尺子)，则严格按历史尺子裁切 (无视数据行数)。
        如果没有传，则走动态计算兜底。
        """
        if df is None or df.empty: return df
        df_clean = df.copy()
        is_single_row = len(df_clean) <= 1
        
        for col in feature_cols:
            if col not in df_clean.columns: continue
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
            
            # [严格模式]: 只要有外部尺子，哪怕只有1行数据，也强制裁剪！
            if mad_params and col in mad_params:
                med = mad_params[col].get('median', 0.0)
                mad = mad_params[col].get('mad', 0.0)
                if mad > 0 and not pd.isna(mad):
                    upper_limit = med + 5.0 * mad
                    lower_limit = med - 5.0 * mad
                    df_clean[col] = df_clean[col].clip(lower=lower_limit, upper=upper_limit)
            # [动态兜底模式]: 没传尺子且不是单行，自己算
            elif not is_single_row:
                df_clean[col] = DataSanitizer.winsorize_mad(df_clean[col], n=5.0)
            
            df_clean[col] = df_clean[col].fillna(0.0)
            
        return df_clean



class DataValidator:
    """
    [质检员] 数据完整性与物理约束校验
    """
    @staticmethod
    def validate(df, context_tag="Unknown"):
        """
        返回: (is_valid, error_msg)
        """
        if df is None or df.empty:
            return False, "数据为空"
            
        # 1. 物理约束检查 (Hard Constraints)
        # High >= Low
        if 'high' in df.columns and 'low' in df.columns:
            # 允许微小的浮点误差
            invalid_hl = df[df['high'] < df['low'] - 1e-9]
            if not invalid_hl.empty:
                return False, f"物理逻辑错误: High < Low ({len(invalid_hl)}行)"
        
        # 2. 核心字段检查
        # 收盘价不能全为 0
        if 'close' in df.columns:
            if (df['close'] <= 0).all():
                 return False, "致命错误: 收盘价全为0"
                 
        # 3. 数据长度检查 (原逻辑保留)
        if len(df) < 5: # 至少要有几行数据
            return False, f"数据长度过短({len(df)})"

        return True, "OK"
