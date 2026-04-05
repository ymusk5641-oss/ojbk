import os
import csv
import datetime
import traceback
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import time
import random
from collections import defaultdict

from utils import BASE_DIR, RECORDER
from config import CFG
# [新增] 引入 DataSource
from data import DataLayer, DataSource 
from strategy import QuantEngine, FactorRegistry

class BacktestEngine:
    """
    [回测引擎 ]
    """
    def __init__(self, net_client):
        self.net = net_client
        # [核心] 初始化数据层 (复用 DataLayer 的清洗和验资逻辑)
        self.dl = DataLayer(net_client, auto_clean=False) 
        self.quant = QuantEngine() 
        
        self.save_dir = os.path.join(BASE_DIR, "Backtest_History")
        if not os.path.exists(self.save_dir):
            try: os.makedirs(self.save_dir)
            except: pass
        
        self.benchmark_cache = {} 
        self.is_running = True # 任务控制标志

    def _get_benchmark_regime(self, days=500):
        """
        [回测专用] 获取大盘 RSRS 历史序列
        修复: 增加对缓存数据长度的校验，防止长周期回测命中短周期预热缓存。
        """
        # 优先读内存缓存，并校验长度是否达标 (按一年250个交易日，0.6的容错率算)
        if 'sh000001' in self.benchmark_cache:
            cached_dict = self.benchmark_cache['sh000001']
            if len(cached_dict) >= (days * 0.6):
                return cached_dict
        try:
            # [核心修复] 动态获取大盘历史数据，增加 M (250天) 的 Z-Score 暖机期

            N, M = CFG.RSRS_PARAMS
            warmup_days = days + M + 50 
            
            # 弃用固定的 fetch_rsrs_raw_kline，改用你成熟的动态回测获取器
            df = self.dl.get_backtest_data('sh000001', days=warmup_days)
            if df is None or df.empty: return {}
            
            # 确保按日期升序，防止 RSRS 计算错位
            df = df.sort_values('date').reset_index(drop=True)
            
            # [核心复用] 调用 QuantEngine 统一计算，保持实盘一致性
            # 注意：返回的已经是映射好的系数 (0.2/0.5/0.8/1.0)
            df['regime_val'] = QuantEngine.calc_rsrs_regime_series(df, N, M)
            
            # 格式化日期 Key (YYYY-MM-DD)
            try:
                df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            except:
                df['date_str'] = df['date'].astype(str)

            # 生成字典 { '2023-01-01': 0.8, ... }
            # dropna 确保剔除初期因暖机不足产生的 NaN
            rsrs_dict = df.set_index('date_str')['regime_val'].dropna().to_dict()
            
            self.benchmark_cache['sh000001'] = rsrs_dict
            return rsrs_dict

        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("BT_RSRS_ERR", str(e))
            return {}

    def _prepare_data_for_strategy(self, df, symbol_str):
        """
        [回测数据清洗 V3.0 - 纯净基建版]
        职责: 只做基础数据对齐、技术指标计算和资金流计算。
        严禁在此处调用 AI 或策略打分逻辑！
        """
        if df.empty: return df
        try:
            # 1. 基础类型转换
            cols = ['open', 'close', 'high', 'low', 'vol']
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

            # [关键对齐] 回测中，当日收盘价(Close)即为"当前价"(Price)
            df['price'] = df['close'] 
            if 'pe' not in df.columns: df['pe'] = 0.0
            
            # 2. 计算纯技术指标 (无状态、无偏见)
            df = QuantEngine.calc_tech_batch(df)

            # 3. 补充元数据
            df['name'] = "Backtest"
            df['symbol'] = str(symbol_str)
            df['pct'] = df['close'].pct_change() * 100
            df['pct'] = df['pct'].fillna(0.0)
            
            # 数据清洗兜底 (统一交由 strategy.py 的治理层处理，剥夺此处越权计算 flow 的逻辑)
            df.fillna(0, inplace=True)

            
            # 只在未定义的情况下赋值为 1.0，保护治理层前置触发的 0.0 熔断标志
            if 'data_quality' not in df.columns:
                df['data_quality'] = 1.0
            
            # [架构优化] 预热截断移交至总装线，保证后续指标不断裂
            return df
            
        except Exception as e:
            import traceback
            if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("BT_PREP_ERR", traceback.format_exc())
            return pd.DataFrame()
            
 
    def _save_report(self, symbol, result_dict, trade_logs):
        """[原有] 报告写入 TXT (保持不变，用于阅读)"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"BT_{symbol}_{timestamp}.txt"
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"=== 回测审计报告: {symbol} ===\n")
                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"回测区间: {result_dict.get('period', 'N/A')}\n")
                f.write(f"总收益率: {result_dict.get('return', 0):.2f}%\n")
                f.write(f"交易次数: {result_dict.get('trades_count', 0)}\n")
                f.write("-" * 40 + "\n")
                f.write(">>> 交易流水:\n")
                if not trade_logs:
                    f.write("无交易记录 (未触发买点)\n")
                else:
                    for line in trade_logs:
                        f.write(line + "\n")
                f.write("-" * 40 + "\n")
                f.write("End of Report\n")
        except Exception as e:
            print(f"Log Save Error: {e}")

    def _save_csv(self, symbol, csv_logs):
        """
        [CSV 保存 - 宪法驱动版 V6.0]
        改进: 
        1. [SSOT] 直接引用 CFG.EXPORT_METADATA_SCHEMA，严禁在函数内硬编码列名。
        2. [自适应] 自动接纳代码中产生的但在 Config 中未定义的临时字段（放在最后）。
        """
        if not csv_logs: return
        
        import csv
        import datetime
        import os

        try:
            # 1. [核心修改] 从 Config 获取优先列顺序 (SSOT)
            priority_cols = CFG.EXPORT_METADATA_SCHEMA
            
            # 2. 收集数据中实际存在的所有 Key (作为实际产出的真理)
            all_keys = set()
            for row in csv_logs:
                all_keys.update(row.keys())
            
            # 3. 动态构建最终表头
            # 逻辑: 宪法定义的列(按顺序) + 宪法没定义但实际产生的列(按字母序)
            # 这样既保证了 Regime_Src 的位置，又不至于丢掉未注册的临时Debug字段
            remaining_keys = sorted(list(all_keys - set(priority_cols)))
            final_headers = [k for k in priority_cols if k in all_keys] + remaining_keys
            
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.save_dir, f"BT_{symbol}_{ts}.csv")
            
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=final_headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(csv_logs)
                
        except Exception as e:
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("BT_CSV_ERR", f"保存CSV失败 {symbol}: {str(e)}")


    def _build_augmented_dataframe(self, symbol, days):
        """
        [回测总装车间 V3.1 - 完美单向数据流版 + 时间戳映射修复]
        分离原则: 严格遵循 [技术面 -> 宏观面 -> AI与策略面 -> 预热截断] 的先后顺序。
        """
        def log_step(msg, level="INFO"): 
            tag = "BT_TRACE" if level=="INFO" else "BT_WARN"
            if 'RECORDER' in globals(): globals()['RECORDER'].log_debug(tag, msg)

        # 1. 获取原始 K 线
        raw_df = self.dl.get_backtest_data(symbol, days=days+200)
        if raw_df is None or raw_df.empty or len(raw_df) < 50: 
            return pd.DataFrame()

        # 2. 基础清洗与技术指标组装
        full_df = self._prepare_data_for_strategy(raw_df, str(symbol))
        if full_df.empty: 
            return full_df
        
        # 3. 注入宏观大盘环境 (Regime)
        regime_map = self._get_benchmark_regime(max(600, days + 200)) 
        is_benchmark_valid = bool(regime_map and len(regime_map) > 50)
        
        if is_benchmark_valid: 
            # [隐形地雷修复] 强制转换为标准 YYYY-MM-DD 字符串再映射
            # 增加 .astype(str).str.strip() 容错纯数字格式 (如 20230101)
            date_strs = pd.to_datetime(full_df['date'].astype(str).str.strip()).dt.strftime('%Y-%m-%d')
            
            # [映射率显影诊断] 记录映射交集与丢失率，防止回测大盘数据静默失效
            mapped_series = date_strs.map(regime_map)
            valid_map_count = mapped_series.notnull().sum()
            total_count = len(date_strs)
            
            map_ratio = (valid_map_count / total_count * 100) if total_count > 0 else 0
            if map_ratio < 95.0:
                log_step(f"   -> ⚠️ 宏观因子映射异常: 成功 {valid_map_count}/{total_count} ({map_ratio:.1f}%)", "WARN")
                if total_count > 0 and len(regime_map) > 0:
                    sample_target = date_strs.iloc[0]
                    sample_source = list(regime_map.keys())[0]
                    log_step(f"   -> 🚨 键值格式比对 | 目标字典: '{sample_target}' vs 基准字典: '{sample_source}'", "WARN")
            else:
                log_step(f"   -> ✅ 宏观因子映射完美: 成功率 {map_ratio:.1f}%")

            # [核心修复] 引入 ffill() 前向填充，解决节假日错位导致的环境突变 0.5 问题
            mapped_series_filled = mapped_series.ffill().fillna(0.5)
            full_df['regime_val'] = mapped_series_filled
            full_df['regime_src'] = mapped_series.apply(lambda x: 'REAL' if pd.notnull(x) else 'FILL/INHERIT')
        else: 
            log_step("⚠️ [警告] 大盘环境数据缺失，将启用中性(0.5)兜底", "WARN")
            full_df['regime_val'] = 0.5
            full_df['regime_src'] = 'EMPTY_MAP' 

        
        # 4. 终极一击：执行全套策略打分 (自动内部调用 AI，绝不重复计算)
        # 将 "MID" 统一为 "TAIL"，确保回测与实盘尾盘分数严格对齐
        # 👇 修复为: 强制开启 target_mode=True 实施时空隔离
        full_df = self.quant.strategy_scoring(full_df, "TAIL", 0.5, target_mode=True)
        # 5. Schema 强制校验 (由数据宪法 FactorRegistry 进行最后一次体检，缺列补默认值)
        full_df = FactorRegistry.enforce_std_schema(full_df, context_tag="BacktestRun")

        # 6. 预热期截断 (必须在所有 Rolling 指标计算完毕后执行)
        if len(full_df) > 120:
            return full_df.iloc[120:].reset_index(drop=True)
        else:
            return pd.DataFrame()



    def run_single_stock(self, symbol, days=365, buy_mode='open', force_full_pos=False):
        """
        [回测核心 - 工业级旗舰版 V8.0 - 全真多态与尾盘冲击惩罚]
        架构升级：引入基于 OHLC 序列的盘中动态止损与跳空穿透惩罚，精准还原真实爆仓感。
        核验修复：引入真实持仓股数(held_shares)与物理涨跌停边界钳制，根除碎股复利与灵异价格。
        时间轴升级：完美剥离 T 日尾盘(tail/close) 与 T+1 日早/中盘(open/mid) 的物理交易时差。
        本次升级：引入尾盘 Ask3-Ask5 抢筹冲击成本，以及跳空下杀的 3 倍恐慌滑点惩罚。
        新增升级：引入每日盯市(Mark-to-Market)机制，精准计算最大回撤(MDD)与夏普比率。
        """
        def log_step(msg, level="INFO"): 
            tag = "BT_TRACE" if level=="INFO" else "BT_WARN"
            if 'RECORDER' in globals(): globals()['RECORDER'].log_debug(tag, msg)

        # [新增] 结果字典扩展 max_drawdown 和 sharpe_ratio 字段
        res_dict = {"symbol": symbol, "return": 0, "trades_count": 0, "trade_log": [], "period": "N/A", "max_drawdown": 0.0, "sharpe_ratio": 0.0}
        trades = []   
        csv_logs = [] 
        
        try:
            log_step(f"{symbol} [Start] 启动高精度OHLC回测...") 
            s_sym = str(symbol)
            limit_ratio = 0.20 if s_sym.startswith(('300', '688')) else (0.30 if s_sym.startswith(('8', '4')) else 0.10)
            
            costs = CFG.TRANS_COSTS
            BASE_SLIPPAGE = costs.get('slippage', 0.0015)
            COMM_BUY = costs.get('comm_buy', 0.0003)
            COMM_SELL = costs.get('comm_sell', 0.0013)
            risk_cfg = CFG.data['risk'] 

            full_df = self._build_augmented_dataframe(symbol, days)
            if full_df.empty: return res_dict

            res_dict['period'] = f"{full_df.iloc[0]['date']}~{full_df.iloc[-1]['date']}"
            
            cash = 100000.0
            initial_cash = cash
            held_shares = 0         
            buy_cost_unit = 0.0     
            highest_price = 0.0 
            current_stop_line = 0.0 
            
            COOLDOWN_DAYS = 3
            cooldown_counter = 0 
            is_holding = False
            entry_idx = -1 
            
            records = full_df.to_dict('records')
            
            # [核心修复：时间对齐] 提前装入第0天的初始净值
            daily_equity = [initial_cash]
            
            # [修复] 将计时器移至特征计算完成后，仅统计撮合循环耗时
            thread_start_time = time.time()
            
            for i in range(len(records) - 1):
                # [修复] 僵尸线程自杀机制必须处于真实的业务循环中，并放宽至 15 秒
                if time.time() - thread_start_time > 15.0:
                    if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("BT_TIMEOUT", f"⚠️ {symbol} 撮合运算超时(>15s)，触发内部熔断自保！")
                    break
                    
                today = records[i]

                tomorrow = records[i+1]
                yesterday_close = today['close_prev'] if today.get('close_prev', 0) > 0 else today['open']
                
                if today['vol'] == 0: continue 
                
                # [核心修复] 物理隔离次日停牌。防止 T+1 报单发生幽灵穿越成交
                can_trade_tomorrow = (tomorrow['vol'] > 0)
                
                if cooldown_counter > 0: cooldown_counter -= 1


                current_hold_val = held_shares * today['close'] if is_holding else 0.0
                current_total_assets = cash + current_hold_val
                current_pos_pct = current_hold_val / current_total_assets if current_total_assets > 0 else 0.0
                
                score = today['final_score']
                ai_score = today['ai_score'] 
                regime = today['regime_val']
                regime_src = today['regime_src'] 
                
                daily_action = "HOLD" if is_holding else "WAIT" 
                daily_reason = ""
                daily_filter = "" 
                daily_exec_price = today['close']
                daily_return = 0.0

                # ===================== [买入撮合] =====================
                if not is_holding:
                    should_buy, signal_reason = QuantEngine.check_entry_signal(today, score, today['strategy_name'], regime)
                    if should_buy:
                        if cooldown_counter > 0:
                            daily_action = "FILTERED"; daily_filter = f"冷却期({cooldown_counter})"; daily_reason = signal_reason 
                        else:
                            vol_idx = today['volatility']
                            slippage = BASE_SLIPPAGE + (vol_idx * 0.0005)

                            if buy_mode == 'close' or buy_mode == 'tail':
                                limit_up_today, _ = QuantEngine.calc_limit_price_math(yesterday_close, limit_ratio)
                                if today['close'] >= limit_up_today - 0.01 and today['high'] == today['low']:
                                    daily_action = "FILTERED"; daily_filter = "今日一字涨停无法买入"; daily_reason = signal_reason
                                else:
                                    tail_rush_penalty = 0.003 + (vol_idx * 0.001) 
                                    slippage += tail_rush_penalty
                                    
                                    target_price = today['close']
                                    raw_exec_price = target_price * (1 + slippage)
                                    exec_price = round(min(raw_exec_price, limit_up_today), 2)
                                    
                                    real_shares, _, _, _ = QuantEngine.calculate_target_position(
                                        score, vol_idx, regime, exec_price, cash, current_total_assets, force_full_pos=force_full_pos
                                    )
                                    
                                    if real_shares >= 100:
                                        trade_amount = real_shares * exec_price
                                        cost_fee = trade_amount * COMM_BUY
                                        total_cost = trade_amount + cost_fee
                                        if cash >= total_cost:
                                            cash -= total_cost 
                                            is_holding = True; entry_idx = i  
                                            held_shares = real_shares 
                                            buy_cost_unit = exec_price; highest_price = exec_price 
                                            current_stop_line = 0.0 
                                            
                                            daily_action = "Buy"; daily_reason = signal_reason; daily_exec_price = exec_price
                                            trades.append(f"[{today['date']}] Buy(Tail) {signal_reason} (Score:{score:.0f}, Env:{regime:.1f}[{regime_src}], 滑点:{(slippage*100):.2f}%)")
                                            
                                            new_hold_val = held_shares * exec_price 
                                            current_pos_pct = new_hold_val / (cash + new_hold_val)
                                        else:
                                            daily_action = "FILTERED"; daily_filter = "资金不足以支付滑点/佣金"; daily_reason = signal_reason
                                    else:
                                        daily_action = "FILTERED"; daily_filter = "资金不足1手"; daily_reason = signal_reason

                            else:
                                vol_idx = today['volatility']
                                limit_up_tomorrow, limit_down_tomorrow = QuantEngine.calc_limit_price_math(today['close'], limit_ratio)
                                if not can_trade_tomorrow:
                                    daily_action = "FILTERED"; daily_filter = "次日停牌报单失效"; daily_reason = signal_reason
                                elif tomorrow['low'] < limit_up_tomorrow - 0.01 and tomorrow['open'] < limit_up_tomorrow * 0.99:
                                    is_breakout = False
                                    if buy_mode == 'open':
                                        target_price = tomorrow['open']
                                        slippage += 0.0025
                                        is_breakout = True
                                    elif buy_mode == 'mid':
                                        target_price = tomorrow['open'] * 1.01
                                        is_breakout = tomorrow['high'] >= target_price

                                    if is_breakout:
                                        target_price = min(max(target_price, limit_down_tomorrow), limit_up_tomorrow)
                                        raw_exec_price = target_price * (1 + slippage)
                                        # [核心修复] 增加 tomorrow['high'] 的钳制，绝不生成无法物理成交的价格
                                        exec_price = round(min(raw_exec_price, limit_up_tomorrow, tomorrow['high']), 2)
                                        
                                        real_shares, _, _, _ = QuantEngine.calculate_target_position(
                                            score, vol_idx, regime, exec_price, cash, current_total_assets, force_full_pos=force_full_pos
                                        )

                                        
                                        if real_shares >= 100:
                                            trade_amount = real_shares * exec_price
                                            cost_fee = trade_amount * COMM_BUY
                                            total_cost = trade_amount + cost_fee
                                            if cash >= total_cost:
                                                cash -= total_cost 
                                                is_holding = True; entry_idx = i + 1  
                                                held_shares = real_shares 
                                                buy_cost_unit = exec_price; highest_price = exec_price 
                                                current_stop_line = 0.0 
                                                
                                                tag = "Buy(Open)" if buy_mode == 'open' else "Buy(Mid)"
                                                daily_action = "Buy"; daily_reason = signal_reason; daily_exec_price = exec_price
                                                trades.append(f"[{tomorrow['date']}] {tag} {signal_reason} (Score:{score:.0f}, Env:{regime:.1f}[{regime_src}])")
                                                
                                                new_hold_val = held_shares * exec_price 
                                                current_pos_pct = new_hold_val / (cash + new_hold_val)
                                            else:
                                                daily_action = "FILTERED"; daily_filter = "资金不足以支付滑点/佣金"; daily_reason = signal_reason
                                        else:
                                            daily_action = "FILTERED"; daily_filter = "资金不足1手"; daily_reason = signal_reason
                                    else:
                                        daily_action = "FILTERED"; daily_filter = "盘中未突破目标价"; daily_reason = signal_reason
                                else:
                                    daily_action = "FILTERED"; daily_filter = "一字涨停无法买入"; daily_reason = signal_reason
                    else:
                        daily_reason = signal_reason 
                        
                # ===================== [卖出撮合] ====================
                if is_holding:
                    held_days = i - entry_idx
                    # 无论次日能否卖出，创新高逻辑不受影响(基于 today 真实数据)
                    if held_days > 0 or (held_days == 0 and buy_mode != 'tail' and buy_mode != 'close'):
                        if today['high'] > highest_price: 
                            highest_price = today['high']
                            
                    # [核心修复] 只有在明天开市交易的情况下，才允许执行卖出判断和撮合
                    if held_days >= 0 and can_trade_tomorrow:
                        atr = today['atr']
                        rsi = today['rsi_rank']
                        ma20 = today['ma20']
                        
                        _, _, temp_stop = QuantEngine.check_exit_signal_v2(
                            symbol, buy_cost_unit, today['close'], highest_price, atr, rsi, risk_cfg, regime, score, hold_days=held_days, ma20=ma20
                        )
                        current_stop_line = temp_stop

                        should_sell, sell_reason, hard_stop_line = QuantEngine.check_exit_signal_v2(
                            symbol, buy_cost_unit, today['close'], highest_price,
                            atr, rsi, risk_cfg, regime, score, hold_days=held_days, ma20=ma20
                        )

                        final_sell_signal = False
                        exec_price = tomorrow['open'] 
                        
                        if should_sell:
                            final_sell_signal = True
                            # [核心修复]: 优先按次日开盘价成交，修正滑点
                            exec_price = tomorrow['open'] 
                            
                            if hard_stop_line > 0 and ("触及" in sell_reason or "保本" in sell_reason or "破位" in sell_reason):
                                if tomorrow['open'] < hard_stop_line:
                                    sell_reason += "[跳空穿透]"
                        else:
                            if hard_stop_line > 0 and tomorrow['low'] <= hard_stop_line:
                                final_sell_signal = True
                                if tomorrow['open'] <= hard_stop_line:
                                    exec_price = tomorrow['open']
                                    sell_reason = "开盘跳空跌破止损[穿透]"
                                else:
                                    exec_price = hard_stop_line
                                    sell_reason = "盘中触价闪崩止损[动态]"

                        if final_sell_signal:
                            _, limit_down_tomorrow = QuantEngine.calc_limit_price_math(today['close'], limit_ratio)
                            
                            if tomorrow['open'] <= limit_down_tomorrow + 0.01 and tomorrow['high'] == tomorrow['low']:
                                final_sell_signal = False
                                daily_filter = "一字跌停锁死(无法卖出)"
                            else:
                                if "跳空穿透" in sell_reason or "开盘跳空" in sell_reason:
                                    actual_slippage = 0.0005 
                                else:
                                    actual_slippage = BASE_SLIPPAGE * 3 if "穿透" in sell_reason else BASE_SLIPPAGE
                                
                                raw_final_price = exec_price * (1 - actual_slippage)
                                
                                # [核心修复] 加入 tomorrow['low'] 作为钳制，绝不允许扣减滑点后跌穿当日物理最低价
                                final_price = round(max(raw_final_price, limit_down_tomorrow, tomorrow['low']), 2)

                                revenue = held_shares * final_price * (1 - COMM_SELL)

                                period_return = (final_price - buy_cost_unit) / (buy_cost_unit + 1e-9)
                                cash += revenue
                                
                                if daily_action == "Buy":
                                    daily_action = "Buy ➔ 次日止损"
                                else:
                                    daily_action = "Sell"
                                    
                                daily_reason = sell_reason
                                daily_exec_price = final_price
                                daily_return = round(period_return * 100, 2)
                                trades.append(f"[{tomorrow['date']}] Sell {sell_reason} (Ret:{period_return*100:.2f}%)")
                                
                                is_holding = False; held_shares = 0; buy_cost_unit = 0; highest_price = 0
                                cooldown_counter = COOLDOWN_DAYS
                                current_stop_line = 0.0
                                current_pos_pct = 0.0

                csv_row = {}
                csv_row['Date'] = tomorrow['date'] 
                csv_row['Symbol'] = symbol
                csv_row['Action'] = daily_action
                csv_row['Price'] = daily_exec_price
                csv_row['Return(%)'] = daily_return
                csv_row['Reason'] = daily_reason
                csv_row['Filter_Code'] = daily_filter    
                csv_row['Cash'] = round(cash, 2)          
                csv_row['Pos_Pct'] = round(current_pos_pct * 100, 1) 
                csv_row['Stop_Line'] = round(current_stop_line, 2)   
                csv_row['Highest_Price'] = round(highest_price, 2)   
                csv_row['Shares'] = held_shares   
                csv_row['Score'] = int(score)
                csv_row['AI_Prob'] = round(float(ai_score), 2)
                csv_row['Env_Regime'] = round(regime, 2)
                csv_row['Regime_Src'] = regime_src
                
                for feat in CFG.CORE_FEATURE_SCHEMA:
                    val = today.get(feat, 0.0)
                    if isinstance(val, bool): val = int(val)
                    try: csv_row[feat] = round(float(val), 4)
                    except: csv_row[feat] = 0.0
                
                csv_logs.append(csv_row)
                
                # ===================== [修正: 严格基于 tomorrow 价格结算] =====================
                # 既然上面的逻辑（cash增减、股数变动）都是基于 tomorrow 发生的事实，
                # 那么计算日终净值时，必须使用 tomorrow['close']！
                current_asset_value = cash + (held_shares * tomorrow['close'] if is_holding else 0.0)
                daily_equity.append(current_asset_value)

            if is_holding:
                last_price = records[-1]['close']
                _, limit_down_last = QuantEngine.calc_limit_price_math(records[-2]['close'], limit_ratio)
                
                raw_final_price = last_price * (1 - BASE_SLIPPAGE)
                final_price = round(max(raw_final_price, limit_down_last), 2)
                
                revenue = held_shares * final_price * (1 - COMM_SELL)
                period_return = (final_price - buy_cost_unit) / (buy_cost_unit + 1e-9)
                cash += revenue
                
                if len(csv_logs) > 0:
                    csv_logs[-1]['Action'] = 'ForceSell'
                    csv_logs[-1]['Price'] = final_price   
                    csv_logs[-1]['Reason'] = '回测结束强制平仓'
                    csv_logs[-1]['Return(%)'] = round(period_return * 100, 2)
                    csv_logs[-1]['Shares'] = 0
                    csv_logs[-1]['Cash'] = round(cash, 2)
                    csv_logs[-1]['Pos_Pct'] = 0.0
                
                trades.append(f"[{records[-1]['date']}] ForceSell 回测结束强制平仓 (Ret:{period_return*100:.2f}%)")
                if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("BT_TRACE", f"⚠️ {symbol} 触发期末强制平仓 -> 回笼资金: ¥{revenue:.2f}")
                
                # [修正：期末结算] 强制卖出后，覆写最后一天（也就是今天）的净值，不产生新天数
                daily_equity[-1] = cash

            # ===================== [风险指标计算引擎] =====================
            max_drawdown = 0.0
            sharpe_ratio = 0.0
            
            if len(daily_equity) > 1:
                equity_arr = np.array(daily_equity)
                daily_returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-9)
                
                running_max = np.maximum.accumulate(equity_arr)
                drawdowns = (equity_arr - running_max) / (running_max + 1e-9)
                max_drawdown = abs(drawdowns.min()) * 100 
                
                risk_free_daily = 0.03 / 250
                std_dev = np.std(daily_returns)
                if std_dev > 1e-9:
                    sharpe_ratio = np.sqrt(250) * (np.mean(daily_returns) - risk_free_daily) / std_dev

            res_dict['return'] = (cash - initial_cash) / initial_cash * 100
            res_dict['trades_count'] = len([t for t in trades if 'Buy' in t]) 
            res_dict['trade_log'] = trades
            
            res_dict['max_drawdown'] = max_drawdown
            res_dict['sharpe_ratio'] = sharpe_ratio
            
            if 'RECORDER' in globals():
                globals()['RECORDER'].log_debug("BT_METRICS", f"🏁 {symbol} 结算 | 收益: {res_dict['return']:.2f}% | 回撤: {max_drawdown:.2f}% | 夏普: {sharpe_ratio:.2f}")
            
            self._save_report(symbol, res_dict, trades)
            self._save_csv(symbol, csv_logs)
            return res_dict

        except KeyError as ke:
            if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("BT_SCHEMA_ERR", f"缺失字段 {ke}")
            return res_dict
        except Exception as e:
            import traceback
            if 'RECORDER' in globals(): globals()['RECORDER'].log_debug("BT_RUN_ERR", traceback.format_exc())
            return res_dict




    def run_portfolio_test(self, stock_list, callback=None, buy_mode='open'):
        """
        [回测入口 - 白盒透视版]
        修改:
        1. [显影] 增加大量 print 控制台输出，解决"黑盒"焦虑。
        2. [防卡] 增加 total_timeout (总任务超时) 和 task_timeout (单任务超时)。
        3. [定位] 明确打印"正在提交"、"正在计算"、"完成"的状态。
        """
        from concurrent.futures import as_completed, TimeoutError
        import time
        def log_step(msg):
            # print(msg) # 如果您以后想看控制台，可以解开这行
            RECORDER.log_debug("BT_TRACE", msg)  # 这行让界面能看到！
            
        # 1. 强制控制台输出，确保你能看到
        log_step(f"\n{'='*40}\n🚀 [启动回测] 目标: {len(stock_list)}只 | 线程: 8\n{'='*40}")
        
        stats = {"total_stocks": len(stock_list), "positive_stocks": 0, "total_return_sum": 0.0, "details": []}
        
        # --- 阶段一：数据预热 (串行) ---
        log_step(">> [Step 1] 开始数据预热...")
        if callback: callback("正在预热数据 (防止死锁)...")
        # 👇 加上这行！串行预热大盘，拯救你的并发网络！
        self._get_benchmark_regime(days=600)
        
        for idx, code in enumerate(stock_list):
            # 简单的进度条
            if idx % 10 == 0: log_step(f"   - 预热进度: {idx}/{len(stock_list)}")
            try:
                self.dl.get_backtest_data(code, days=400)
            except Exception as e:
                log_step(f"   ! 预热失败 {code}: {e}")

        # --- 阶段二：并行计算 (核心) ---
        log_step(f">> [Step 2] 启动并行计算...")
        if callback: callback("🔥 启动并行计算引擎...")
        
        start_time = time.time()
        completed_count = 0
        
        # [核心修复] 环境自适应并发控制 (彻底告别硬编码与 OOM)
        # Android 平台回测因数据量巨大，严格锁死在 3-4 线程；PC 平台交由 CFG 控制
        safe_workers = 4 if '/storage/emulated' in BASE_DIR else getattr(CFG, 'MAX_WORKERS', 16)
        log_step(f"   [基建] 当前环境分配并发核数: {safe_workers}")
        
        # 移除原有的 hardcode max_workers=8
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            # 建立任务映射
            future_to_code = {}
            for code in stock_list:
                f = executor.submit(self.run_single_stock, code, 365, buy_mode, True)
                future_to_code[f] = code

            log_step(f">> [Step 2.1] 所有任务已提交，开始等待结果...")

            
            # 使用 as_completed 获取结果
            for i, future in enumerate(as_completed(future_to_code)):
                code = future_to_code[future]
                elapsed = time.time() - start_time
                
                try:
                    # [严格熔断] 单个股票放宽至 25 秒 (特征计算约5-10s + 撮合约15s)
                    res = future.result(timeout=25)
                    
                    # 结果统计
                    ret = res.get('return', 0)
                    stats["total_return_sum"] += ret
                    if ret > 0: stats["positive_stocks"] += 1
                    
                    # 控制台详细日志 (这是你最需要的)
                    print(f"   ✅ [{i+1}/{len(stock_list)}] {code} 完成 | 收益: {ret:>6.2f}% | 耗时: {elapsed:.1f}s")
                    
                    # UI 更新 (用 try 包裹防止 UI 线程炸裂)
                    if callback:
                        try:
                            msg = f"[{i+1}/{len(stock_list)}] {code}: {ret:+.1f}%"
                            callback(msg)
                        except: pass
                        
                    stats["details"].append(f"✅ {code}: {ret:+.2f}%")

                except TimeoutError:
                    print(f"   ❌ [{i+1}/{len(stock_list)}] {code} 超时熔断 (BLOCKING)!")
                    stats['details'].append(f"⚠️ {code}: 超时")
                except Exception as e:
                    print(f"   ❌ [{i+1}/{len(stock_list)}] {code} 异常: {str(e)}")
                    stats['details'].append(f"❌ {code}: Error")

        log_step(f"\n{'='*40}\n🏁 [回测结束] 总耗时: {time.time() - start_time:.1f}s\n{'='*40}")
        
        completed = max(1, len([d for d in stats['details'] if '超时' not in d and 'Error' not in d]))
        stats["avg_return"] = stats["total_return_sum"] / completed
        stats["win_rate"] = (stats["positive_stocks"] / completed) * 100
        
        return stats

    def _get_all_mined_symbols(self):
        """
        [底层核心组件] 获取全量已挖掘股票黑名单 (含文件自愈功能)
        功能: 扫描所有历史 CSV 文件，建立全局去重索引，并修复损坏文件。
        """
        import os
        import pandas as pd
        import glob
        
        train_dir = os.path.join(BASE_DIR, "Hunter_Train_Data")
        mined_symbols = set()
        
        # =========================================================
        # [核心修复] 读取“已尝试”黑名单，彻底隔离毒药数据卡死流水线
        # =========================================================
        attempt_file = os.path.join(BASE_DIR, "hunter_attempted_mining.txt")
        if os.path.exists(attempt_file):
            try:
                with open(attempt_file, 'r', encoding='utf-8') as f:
                    mined_symbols.update([x.strip() for x in f.read().splitlines() if x.strip()])
            except: pass

        if not os.path.exists(train_dir): return mined_symbols
        
        existing_files = glob.glob(os.path.join(train_dir, "*.csv"))
        
        for f_path in existing_files:
            try:
                # A. 脏数据物理修复 (自愈逻辑)
                with open(f_path, 'r+', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        header_cols = len(lines[0].split(','))
                        last_line = lines[-1]
                        if len(last_line.split(',')) != header_cols:
                            print(f"   🛠️ 自动修复损坏文件: {os.path.basename(f_path)}")
                            f.seek(0)
                            f.truncate(len("".join(lines[:-1])))
                
                # B. 提取已存在的股票代码
                if os.path.getsize(f_path) > 0:
                    try:
                        for chunk in pd.read_csv(f_path, usecols=['symbol'], chunksize=5000):
                            batch_syms = chunk['symbol'].apply(lambda x: str(x).strip().split('.')[0].zfill(6)).tolist()
                            mined_symbols.update(batch_syms)
                    except Exception: continue
                    
            except Exception as e:
                continue
                
        return mined_symbols


    def export_ml_training_data(self, stock_list, days=500, source_tag="manual"):
        """
        [工业级挖掘 V10.0 - 尾盘刺客 MFE/MAE 不对称打标版]
        升级: 彻底抛弃“次日收盘绝对收益”的钝化指标。
        引入 MFE(最大有利波动) 与 MAE(最大不利波动)，教会模型寻找“次日早盘冲高猛、下杀极小”的高盈亏比标的。
        """

        # --- 0. 准备专用日志写入器 ---
        mining_log_path = os.path.join(BASE_DIR, "hunter_mining.log")
        
        def write_mining_log(msg):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            full_msg = f"[{ts}] {msg}"
            try:
                print(full_msg) 
                with open(mining_log_path, "a", encoding="utf-8") as f:
                    f.write(full_msg + "\n")
            except: pass

        if not hasattr(self, 'is_running'): self.is_running = True

        write_mining_log(f"\n{'='*40}")
        write_mining_log(f"🚀 [任务启动] Tag:{source_tag} | 目标总数: {len(stock_list)}")

        # 1. 目录准备
        train_dir = os.path.join(BASE_DIR, "Hunter_Train_Data")
        if not os.path.exists(train_dir): os.makedirs(train_dir)

        # 2. 索引建立
        try:
            mined_symbols = self._get_all_mined_symbols()
        except Exception as e:
            write_mining_log(f"❌ 索引建立失败: {e}")
            mined_symbols = set()
        # 3. 任务过滤
        clean_targets = [str(x).strip().split('.')[0].zfill(6) for x in stock_list]
        todo_list = [x for x in clean_targets if x not in mined_symbols]
        
        if not todo_list:
            write_mining_log("✅ 所有目标均已存在，本轮无工作。")
            return None, 0, 0

        # =======================================================
        # [核心防卡死修复] 只要进队列，立刻打上“已处理”烙印！
        # 这样即使它因为长度不足跳过，下次也不会再来污染排队池
        # =======================================================
        attempt_file = os.path.join(BASE_DIR, "hunter_attempted_mining.txt")
        try:
            with open(attempt_file, 'a', encoding='utf-8') as f:
                for sym in todo_list:
                    f.write(f"{sym}\n")
        except: pass

        # =======================================================
        # [防退化保留] 预加载大盘基准数据，保留 Alpha 基因
        # =======================================================
        write_mining_log("📊 正在获取大盘基准数据 (用于辅助构建 Alpha 基因)...")
        bench_map = {}
        regime_map = {} # [Gemini Fix: 新增宏观环境映射表]
        try:
            bench_df = self.dl.get_backtest_data('sh000001', days=days)
            if not bench_df.empty:
                # [核心修复] 强制统一大盘日期的键格式为 YYYY-MM-DD，防止字典 map() 时因类型不匹配而静默失效
                try:
                    bench_clean_dates = pd.to_datetime(bench_df['date'].astype(str).str.strip()).dt.strftime('%Y-%m-%d')
                except:
                    bench_clean_dates = bench_df['date'].astype(str)

                bench_df['bench_next_pct'] = bench_df['close'].pct_change().shift(-1) * 100
                bench_map = dict(zip(bench_clean_dates, bench_df['bench_next_pct']))
                
                # [新增工业级] 计算大盘未来 3 天累计收益，用于 T+3 Alpha 剥离
                bench_df['bench_ret_3d'] = (bench_df['close'].shift(-3) - bench_df['close']) / (bench_df['close'] + 1e-9) * 100
                bench_map_3d = dict(zip(bench_clean_dates, bench_df['bench_ret_3d']))

                write_mining_log(f"✅ 基准映射构建成功，包含 {len(bench_map)} 个交易日。")

                
                # [Gemini Fix: 获取全市场 RSRS 宏观环境字典]
                regime_map = self._get_benchmark_regime(days + 100)
                write_mining_log(f"✅ 宏观环境(Regime)字典预加载成功。")
            else:
                write_mining_log("⚠️ 大盘数据为空，将降级使用绝对收益打标。")
        except Exception as e:
            write_mining_log(f"⚠️ 基准数据获取失败: {e}，将降级使用绝对收益打标。")

        # 4. 初始化环境
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_tag}_{ts_str}.csv"
        save_path = os.path.join(train_dir, filename)
        
        valid_count = 0
        total_rows = 0
        skip_stats = defaultdict(int)
        
        # 5. 执行挖掘
        for idx, symbol in enumerate(todo_list):
            if not self.is_running:
                write_mining_log(f"🛑 [用户终止] 任务在进度 {idx}/{len(todo_list)} 处停止。")
                break
                
            try:
                # --- A. 获取数据 ---
                df = self.dl.get_backtest_data(symbol, days=days)
                
                if df.empty:
                    write_mining_log(f"⚠️ [跳过] {symbol}: 获取数据失败 (Empty)")
                    skip_stats["数据为空"] += 1
                    continue

                # 强制清洗缓存中的脏数据 (String -> Float)
                dirty_cols = ['pe', 'pb', 'pct', 'close', 'open', 'high', 'low']
                for col in dirty_cols:
                    if col in df.columns and df[col].dtype == 'object':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # --- B. 基础有效性熔断 ---
                if len(df) < 100: 
                    write_mining_log(f"⚠️ [跳过] {symbol}: 数据过短 (Len={len(df)})")
                    skip_stats["长度不足"] += 1
                    continue
               
                # =======================================================
                # [Gemini Fix: 注入宏观环境因子]
                # 核心修复：必须在 calc_tech_batch 之前注入，防止底层算分时取不到真实宏观环境导致特征漂移！
                # =======================================================
                if regime_map:
                    try:
                        date_str = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    except:
                        date_str = df['date'].astype(str)
                    df['env_regime'] = date_str.map(regime_map).fillna(0.5)
                else:
                    df['env_regime'] = 0.5
                    
                # 同时赋予 regime_val，确保底层 QuantEngine 兼容不同的字段名调用
                df['regime_val'] = df['env_regime']

                # --- C. 特征计算 (SSOT：所有指标包含 flow 统一交由底层基建计算) ---
                df['symbol'] = str(symbol).zfill(6)
                df = self.quant.calc_tech_batch(df)

                # =======================================================
                # --- E. 工业级 Label 生成 (三维动态波动率打标 V11.0) ---
                # =======================================================
                # 1. 拓宽视野：物理提取未来 3 天的极值点位 (避开 Pydroid3 的 rolling 前瞻漏洞)
                h1 = df['high'].shift(-1)
                h2 = df['high'].shift(-2)
                h3 = df['high'].shift(-3)
                l1 = df['low'].shift(-1)
                l2 = df['low'].shift(-2)
                l3 = df['low'].shift(-3)
                
                # 引入流动性折扣(Slippage Penalty)，防止被极端“避雷针”骗过，强迫模型寻找厚实的多头
                df['max_high_3d'] = pd.DataFrame({'h1': h1, 'h2': h2, 'h3': h3}).max(axis=1) * 0.95 + df['close'] * 0.05
                df['min_low_3d'] = pd.DataFrame({'l1': l1, 'l2': l2, 'l3': l3}).min(axis=1)
                df['next_close_3d'] = df['close'].shift(-3)
                
                # 计算 T+3 MFE 和 MAE
                df['mfe_3d_pct'] = (df['max_high_3d'] - df['close']) / (df['close'] + 1e-9) * 100
                df['mae_3d_pct'] = (df['min_low_3d'] - df['close']) / (df['close'] + 1e-9) * 100
                df['ret_3d_pct'] = (df['next_close_3d'] - df['close']) / (df['close'] + 1e-9) * 100

                # 2. 映射大盘表现与超额剥离 (格式强制对齐修复版)
                # 强制将个股的日期格式 (如 2025/3/10) 统一转化为标准 YYYY-MM-DD，防静默失效
                try:
                    clean_dates = pd.to_datetime(df['date'].astype(str).str.strip()).dt.strftime('%Y-%m-%d')
                except:
                    clean_dates = df['date'].astype(str)
                
                df['bench_next_pct'] = clean_dates.map(bench_map).fillna(0.0)
                try:
                    df['bench_3d_pct'] = clean_dates.map(bench_map_3d).fillna(0.0)
                except NameError:
                    df['bench_3d_pct'] = df['bench_next_pct'] * 3  # 兜底降级

                # =======================================================
                # [安全护城河] 停牌断层剔除 (防止跨月复牌导致的未来函数污染)
                # =======================================================
                try:
                    df['date_obj'] = pd.to_datetime(df['date'])
                    date_diff = (df['date_obj'].shift(-3) - df['date_obj']).dt.days
                    # T+3 跨度正常最多 5-7 天，放宽到 15 天防止长假
                    mask_valid_dates = date_diff <= 15  
                except:
                    mask_valid_dates = True
                
                # 3. 动态波动率自适应标尺 (Volatility-Adjusted Thresholds)
                # 极其核心：牛市振幅大，熊市振幅小。使用真实的个体波动率(ATR)作为杠杆
                df['atr_pct'] = (df['atr'] / (df['close'] + 1e-9) * 100).clip(lower=2.0)
                
                # 4. 生成连续 Alpha 分值 (供下一代回归/排序模型使用)
                excess_ret = np.maximum(0, df['ret_3d_pct'] - df['bench_3d_pct'])
                mfe_adj = df['mfe_3d_pct'] / df['atr_pct']
                mae_adj = df['mae_3d_pct'].abs() / df['atr_pct']
                df['label_value'] = (mfe_adj / (mae_adj + 0.1)) * np.log1p(excess_ret)
                
                # 5. 生成高容错硬标签 (平滑过渡：兼容当前二分类模型)
                # 条件 A: 未来 3 天冲高必须大于该股日常波幅的 0.8 倍
                cond_a = df['mfe_3d_pct'] > (df['atr_pct'] * 0.8)
                # 条件 B: 未来 3 天下杀不能超过该股日常波幅的 0.6 倍 (给了缩量十字星充足的容错)
                cond_b = df['mae_3d_pct'] > -(df['atr_pct'] * 0.6)
                # 条件 C: 盈亏比依然要优秀
                cond_c = df['mfe_3d_pct'] > (df['mae_3d_pct'].abs() * 1.2)
                # 条件 D: T+3 累计绝对跑赢同期大盘
                cond_d = df['ret_3d_pct'] > df['bench_3d_pct']

                # 完美融合打标！
                df['label_class'] = (cond_a & cond_b & cond_c & cond_d & mask_valid_dates).astype(int)

                # --- F. 清洗与截断 ---
                df = df.iloc[60:].copy() # 去掉 Warmup
                
                # 关键因子清洗 (使用 T+3 的锚定点确保时序对齐)
                df.dropna(subset=['next_close_3d', 'ma20', 'label_value'], inplace=True)

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(0, inplace=True) # 兜底填充
                
                if df.empty: 
                    skip_stats["清洗后为空"] += 1
                    continue

                # --- G. 落盘 (白名单模式 - 架构对齐) ---
                feat_cols = CFG.CORE_FEATURE_SCHEMA
                
                # 强制补齐，防止报错
                for col in feat_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                
                # 确保全数值类型
                for c in feat_cols: 
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                    
                # 将连续分值作为新增列持久化保存
                final_cols = ['date', 'symbol'] + feat_cols + ['label_class', 'label_value']
                df = df.loc[:, ~df.columns.duplicated()]
                
                # 写入 CSV
                write_header = (valid_count == 0)
                with open(save_path, 'a', newline='', encoding='utf-8') as f:
                    df[final_cols].to_csv(f, header=write_header, index=False, float_format='%.4f')
                    f.flush()
                    os.fsync(f.fileno())

                valid_count += 1
                total_rows += len(df)
                write_mining_log(f"✅ [成功] {symbol}: 产出 {len(df)} 行样本 (MFE模式)")
                
                # --- H. 内存优化 ---
                del df 
                if idx % 20 == 0:
                    gc.collect() 
                    time.sleep(0.01) 
                    if hasattr(self, 'ui_callback') and self.ui_callback:
                        progress = (idx + 1) / len(todo_list) * 100
                        self.ui_callback(f"挖掘进度: {progress:.0f}% (成功:{valid_count})")

            except Exception as e:
                write_mining_log(f"❌ [异常] {symbol}: {str(e)}")
                skip_stats[f"异常"] += 1
                continue

        # 结束汇报
        write_mining_log(f"\n{'='*40}")
        write_mining_log(f"📊 [挖掘统计报告] 成功: {valid_count} | 总行数: {total_rows}")
        if skip_stats:
            write_mining_log("📉 [跳过原因明细]:")
            for reason, count in skip_stats.items():
                write_mining_log(f"   - {reason}: {count} 只")
        write_mining_log(f"📁 文件路径: {os.path.basename(save_path)}")
        write_mining_log(f"{'='*40}\n")
        
        if valid_count > 0:
            return save_path, valid_count, total_rows
        else:
            return save_path, 0, 0


    def _get_market_codes_cached(self):
        """
        [数据源降维打击] 彻底抛弃东方财富，换用【新浪财经】无防备节点
        [格式穿透修复] 完美解析新浪标准 JSON 与 Unicode 中文解码，精准拦截 ST 股
        """
        import json
        import time
        import os
        import random
        import re
        
        cache_path = os.path.join(BASE_DIR, "market_codes_cache.json")
        CACHE_TTL = 7 * 24 * 3600  # 缓存有效期 7 天
        
        # 1. 尝试读取有效缓存
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if time.time() - data.get('ts', 0) < CACHE_TTL:
                        codes = data.get('codes', [])
                        if codes and len(codes) > 1000: 
                            if 'RECORDER' in globals():
                                globals()['RECORDER'].log_debug("MINE_CACHE", f"命中本地缓存，跳过网络扫描 (共 {len(codes)} 只)。")
                            return codes
            except: pass
            
        if 'RECORDER' in globals():
            globals()['RECORDER'].log_debug("MINE_CACHE", "开始从【新浪财经】雷达拉取全市场A股列表...")
            
        codes_set = set()
        # 新浪行情节点：hs_a(沪深A股), kcb(科创板), bjs(北交所)
        nodes = ['hs_a', 'kcb', 'bjs']

        for node in nodes:
            page = 1
            max_pages = 60 # 每个板块最多扫描 60 页
            
            while page <= max_pages:
                try:
                    url = DataSource.get_url("SCAN_ALL_SINA", page=page, node=node)
                    # 复用 network.py 的能力
                    resp = self.net.get(url, timeout=10)
                    
                    if not resp or resp.status_code != 200:
                        time.sleep(1.0)
                        continue
                        
                    text = resp.text.strip()
                    
                    # 翻页结束标志
                    if text in ("null", "[]", "") or len(text) < 10:
                        break
                        
                    # 【核心修复】原生 JSON 解析 (自动解码 \u5b89 等 Unicode 中文)
                    try:
                        items = resp.json()
                        if isinstance(items, list):
                            for item in items:
                                code = str(item.get('code', ''))
                                name = str(item.get('name', ''))
                                
                                # 有了 json() 的自动解码，这里的 ST 和 退 才能精准匹配
                                if 'ST' in name or '退' in name: continue
                                if len(code) == 6 and code != '000000':
                                    codes_set.add(code)
                                    
                    except json.JSONDecodeError:
                        # 极端防线：如果新浪突然返回非标准 JSON，用宽容正则暴力只抓代码
                        fallback_codes = re.findall(r'["\']?code["\']?\s*:\s*["\']?(\d{6})["\']?', text)
                        if not fallback_codes:
                            break
                        for code in fallback_codes:
                            if len(code) == 6 and code != '000000':
                                codes_set.add(code)
                            
                except Exception as e:
                    time.sleep(2.0)
                    
                page += 1
                time.sleep(random.uniform(0.3, 0.8)) 

        codes = list(codes_set)
        
        # 3. 落盘缓存
        if len(codes) > 1000:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'ts': time.time(), 'codes': codes}, f)
                if 'RECORDER' in globals():
                    globals()['RECORDER'].log_debug("MINE_CACHE", f"✅ 【新浪节点】全市场列表缓存创建成功 (共 {len(codes)} 只)！")
            except: pass
            
        return codes




    def mine_broad_market(self):
        """
        [诊断修复版 V4.1 - DataSource 缓存驱动版]
        修正:
        1. 引入 _get_market_codes_cached 缓存机制，消灭无意义的分页网络请求。
        2. O(N) 级别内存差集运算，瞬间锁定目标。
        """
        import time
        import os
        
        # --- 0. 准备日志写入器 ---
        mining_log_path = os.path.join(BASE_DIR, "hunter_mining.log")
        
        def log_diag(msg):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            full_msg = f"[{ts}] [诊断] {msg}"
            try:
                with open(mining_log_path, "a", encoding="utf-8") as f:
                    f.write(full_msg + "\n")
            except: pass

        # 1. 读取配置
        BATCH_LIMIT = getattr(CFG, 'MINING_BATCH_SIZE', 200)
        if BATCH_LIMIT <= 0: BATCH_LIMIT = 200
        
        log_diag(f"\n{'='*40}")
        log_diag(f"🚀 启动增量挖掘 | 目标步长: {BATCH_LIMIT} | 策略: 本地缓存差集锁定")
        
        # 2. 黑名单构建 (维持原逻辑)
        try: CFG.data = CFG.load_config()
        except: pass
        
        cfg_targets = {str(x).strip().split('.')[0].zfill(6) for x in CFG.TARGET_STOCKS}
        cfg_holdings = {str(x).strip().split('.')[0].zfill(6) for x in CFG.HOLDINGS.keys()}
        try: mined_set = self._get_all_mined_symbols()
        except: mined_set = set()
        
        global_blacklist = cfg_targets | cfg_holdings | mined_set
        log_diag(f"🛡️ 全局黑名单(去重后): {len(global_blacklist)} 只")

        # 3. 获取全市场代码 (高速缓存架构)
        if not self.is_running: return None, 0, "用户提前终止"
        
        all_market_codes = self._get_market_codes_cached()
        if not all_market_codes:
            log_diag("❌ 无法获取全市场股票列表，请检查网络。")
            return None, 0, "获取全市场列表失败"
            
        log_diag(f"📦 成功加载全市场 {len(all_market_codes)} 只正常标的。")

        # 4. 内存差集运算与严格去重提取
        final_tasks = []
        final_set = set() # 引入 Set 防止东方财富分页时产生的重复票
        skipped_count = 0
        
        for code in all_market_codes:
            if code not in global_blacklist and code not in final_set:
                final_tasks.append(code)
                final_set.add(code)
                if len(final_tasks) >= BATCH_LIMIT:
                    break
            else:
                skipped_count += 1
        
        log_diag(f"📝 内存筛选完成: 锁定 {len(final_tasks)} 只新标的 (安全跳过了 {skipped_count} 只老面孔)")
        
        if not final_tasks:
            return None, 0, "当前数据库已囊括全A股所有标的，暂无新股可挖。"
            
        log_diag(f"🚀 任务移交 -> export_ml_training_data")

        # 5. 执行挖掘 (调用修复后的 export 方法)
        return self.export_ml_training_data(
            final_tasks, days=500, source_tag=f"market_cached_mining"
        )


