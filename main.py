# -*- coding: utf-8 -*-
import traceback
import threading
import os
import time 
import pandas as pd
import numpy as np
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.clock import Clock, mainthread

from utils import RECORDER
from config import CFG, BASE_DIR, CHINESE_FONT 
from network import NetworkClient
from data import DataLayer,DataSource
from backtest import BacktestEngine
from ui import MainScreen

class SystemSelfCheck:
    """
    [启动自检 - 核心融合版 V4.3 - 显影剂版]
    架构升级：直接实例化实战(DataLayer)与回测(BacktestEngine)模块进行测试。
    """
    def __init__(self, logger_callback=None):
        self.net = NetworkClient()
        self.logger = logger_callback 
        self.report = []
        
        self.flag_infra = False
        self.flag_real = False
        self.flag_bt = False
        
        self.dl = DataLayer(self.net) 
        self.bt = BacktestEngine(self.net)

    def _log(self, name, status, msg=""):
        """统一日志输出"""
        sign = "✅ [OK]" if status else "❌ [ERR]"
        full_msg = f"{sign} {name}: {msg}"
        color = "00ff00" if status else "ff5555"
        self.report.append(full_msg)
        RECORDER.log_debug("SELF_CHECK", f"[{name}] Status:{status} | {msg}")
        if self.logger: self.logger(full_msg, color)
        print(full_msg)

    def check_infrastructure(self):
        """[基建] 验证存储与配置"""
        name_io = "存储权限 (I/O)"
        io_ok = False
        try:
            test_file = os.path.join(BASE_DIR, ".permission_test")
            with open(test_file, 'w') as f: f.write("ok")
            os.remove(test_file)
            self._log(name_io, True, f"路径可写: {BASE_DIR}")
            io_ok = True
        except Exception as e:
            self._log(name_io, False, f"无写入权限: {e}")

        name_cfg = "环境依赖 (Env)"
        env_ok = False
        if CHINESE_FONT and os.path.exists(CHINESE_FONT):
            try:
                cfg = CFG.load_config()
                self._log(name_cfg, True, "字体/配置加载正常")
                env_ok = True
            except Exception as e:
                self._log(name_cfg, False, f"配置崩溃: {e}")
        else:
            self._log(name_cfg, False, "中文字体缺失")
            
        if io_ok and env_ok: self.flag_infra = True

    def check_quote_integration(self):
        """[行情] 验证：实时报价"""
        name = "实时行情与补全"
        try:
            dummy_df = pd.DataFrame({'symbol': ['600519']})
            res_df = self.dl.get_specific_stocks_hybrid(dummy_df)
            
            if res_df is not None and not res_df.empty:
                row = res_df.iloc[0]
                price = row.get('price', 0)
                ind = row.get('ind', '未知')
                if price > 0 and ind != 'Self' and ind != '未知':
                    self._log(name, True, f"全链路打通 (价:{price} 业:{ind})")
                    self.flag_real = True
                else:
                    self._log(name, False, f"数据缺损 (价:{price} 业:{ind})")
            else:
                self._log(name, False, "返回数据为空")
        except Exception as e:
            self._log(name, False, f"接口崩溃: {e}")

    def check_scan_integration(self):
        """[扫描] 验证：选股雷达"""
        name = "全市场扫描雷达"
        try:
            df = self.dl.get_scan_list_hybrid()
            if df is not None and not df.empty:
                self._log(name, True, f"雷达响应正常 (扫描到 {len(df)} 只)")
            else:
                self._log(name, False, "扫描列表为空")
        except Exception as e:
            self._log(name, False, f"接口崩溃: {e}")

    def check_hot_sectors(self):
        """[热点] 验证：领涨板块"""
        name = "热点风口通道"
        try:
            sectors = self.dl.get_market_hot_sectors()
            if sectors:
                self._log(name, True, f"通道畅通 (Top1: {sectors[0]})")
            else:
                self._log(name, False, "未获取到热点")
        except Exception as e:
            self._log(name, False, f"接口异常: {e}")

    def check_algo_data_source(self):
        """
        [自检核心] 算法数据源联通性检查 (带显影剂)
        修复: 不再只报 -1.0，而是尝试捕获并打印具体的熔断原因。
        """
        # 选取一只极其稳定的蓝筹股作为测试标的 (浦发银行)
        test_code = "600000"
        t_name = "核心算法与数据管道"
        
        try:
            # 1. 强制拉取数据 (走 DataLayer -> Strategy 的完整严苛流程)
            # 我们构造一个单行 DataFrame 模拟 UI 的调用方式
            init_df = pd.DataFrame({'symbol': [test_code]})
            
            # 调用混合接口，它会去下载行情 -> 清洗 -> 计算指标(calc_tech_batch)
            res_df = self.dl.get_specific_stocks_hybrid(init_df)
            
            # 2. 检查容器层级失败
            if res_df is None or res_df.empty:
                self._log(t_name, False, "DataLayer 返回空 (网络中断或接口拒绝)")
                return False

            # 3. 提取首行数据
            data = res_df.iloc[0]
            
            # 4. 显影剂：深入检查字段
            # 如果触发了 strategy.py 的严苛熔断，data_quality 会是 0.0
            quality = data.get('data_quality', 0.0)
            
            if quality < 0.5:
                # 尝试推断原因 (这也是我们刚写的逻辑)
                err_reasons = []
                if data.get('close', 0) <= 0.01: err_reasons.append("收盘价缺失(0)")
                
                # 检查是否是策略层熔断特征: RSRS=1.0 且 MACD=0 (默认值)
                rsrs_default = (data.get('rsrs_wls', 0) == 1.0)
                macd_default = (data.get('macd', 0) == 0.0)
                
                if rsrs_default and macd_default:
                    err_reasons.append("策略层计算熔断(长度对齐失败/NaN)")
                
                reason_str = "|".join(err_reasons) if err_reasons else "未知质量问题(Quality=0)"
                self._log(t_name, False, f"被熔断拦截: {reason_str}")
                return False

            # 5. 检查核心指标值是否正常 (防止 0/NaN 逃逸)
            rsrs_val = data.get('rsrs_wls', 0)
            atr_val = data.get('atr', 0)
            
            # RSRS 通常在 0.5 ~ 1.5 之间， ATR 通常 > 0
            if rsrs_val == 1.0 and data.get('rsrs_r2', 0) == 0:
                 self._log(t_name, True, f"⚠️ 警告: RSRS 为默认值 (1.0), 计算可能未收敛")
            
            if atr_val == 0 or np.isnan(atr_val):
                self._log(t_name, False, f"ATR 计算失效 (Val={atr_val})")
                return False

            # 6. 通过
            self._log(t_name, True, f"RSRS:{rsrs_val:.3f} | ATR:{atr_val:.2f} (计算链路完整)")
            return True

        except Exception as e:
            # 捕获 Python 级别的报错 (比如 import 错误，变量名错误)
            err_msg = str(e)
            traceback.print_exc() # 在控制台打印堆栈，方便截图
            self._log(t_name, False, f"自检崩溃: {err_msg[:50]}...")
            return False


    def check_rag_system(self):
        """[风控] 验证：公告 RAG"""
        name = "舆情风控系统"
        try:
            msg = self.dl.get_stock_announcements('600519')
            if msg and "失败" not in msg:
                self._log(name, True, "公告接口在线")
            else:
                self._log(name, False, f"接口不可用: {msg}")
        except Exception as e:
            self._log(name, False, f"风控异常: {e}")

    def check_backtest_simulation(self):
        """
        [回测] 验证：回测计算逻辑 + 报告写入
        """
        name = "回测引擎演练 (Sim)"
        if self.logger: self.logger("   >>> [DEBUG] 正在启动回测演练 (茅台10日极速版)...", "cccccc")
        
        try:
            # 显影剂: 打印开始信号
            print(">>> [DEBUG] STARTING BACKTEST SIMULATION")
            
            res = self.bt.run_single_stock("600519", days=10, force_full_pos=True)
            
            # 显影剂: 打印结束信号
            print(f">>> [DEBUG] FINISHED BACKTEST SIMULATION. Return: {res.get('return')}")

            if "error" in res:
                self._log(name, False, f"引擎内部错误: {res['error']}")
                return

            save_dir = self.bt.save_dir
            has_files = False
            if os.path.exists(save_dir):
                now = time.time()
                for f in os.listdir(save_dir):
                    if os.stat(os.path.join(save_dir, f)).st_mtime > now - 10:
                        has_files = True; break
            
            if has_files:
                self._log(name, True, f"回测闭环成功 (Ret: {res.get('return',0):.2f}%)")
                self.flag_bt = True
            else:
                self._log(name, False, "逻辑跑通但文件写入失败")

        except Exception as e:
            self._log(name, False, f"回测模块崩溃: {e}")
            print(traceback.format_exc())

    def check_ai_brain(self):
        """[大脑] 验证：LLM"""
        name = "AI 模型大脑"
        if not (CFG.DEEPSEEK_KEY or CFG.GEMINI_KEY):
            self._log(name, True, "⚠️ 未配置 AI Key (仅运行量化模式)")
            return
        
        if CFG.DEEPSEEK_KEY:
            try:
                headers = {"Authorization": f"Bearer {CFG.DEEPSEEK_KEY}"}
                # [Refactor] 使用 DataSource 获取自检 URL
                url = DataSource.get_url("LLM_DEEPSEEK_CHECK")
                
                # 注意: 这里保持使用 session.get 以便传入自定义 headers (Auth)
                resp = self.net.session.get(url, headers=headers, timeout=5, verify=False)
                
                if resp.status_code == 200: self._log(name, True, "DeepSeek 在线")
                else: self._log(name, False, f"DeepSeek 异常 Code:{resp.status_code}")
            except: self._log(name, False, "AI 网络不通")
        elif CFG.GEMINI_KEY:
             self._log(name, True, "Gemini Key 已配置")


    def run_diagnostics(self):
        if self.logger: self.logger("\n[自检模式 V5.0] 正在通过【实战接口】进行分级验证...", "ffff00")
        
        self.check_infrastructure()
        if self.flag_infra:
            self.check_quote_integration()
            self.check_scan_integration()
            self.check_hot_sectors()
            self.check_algo_data_source()
            self.check_rag_system()
            self.check_backtest_simulation() 
            self.check_ai_brain()
        else:
            self._log("核心阻断", False, "基建检查未通过")
        
        return {
            "infra": self.flag_infra,
            "real": self.flag_real,
            "bt": self.flag_bt,
            "report": self.report
        }

class AlphaHunterApp(App):
    def build(self):
        sm = ScreenManager(transition=SlideTransition())
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm

    def on_start(self):
        Clock.schedule_once(lambda dt: self.start_background_check(), 3.0)

    def start_background_check(self):
        threading.Thread(target=self._run_check_logic, daemon=True).start()

    def _run_check_logic(self):
        @mainthread
        def ui_logger(msg, color=None):
            self.main_screen.console_view.update_text(msg, color)

        ui_logger("\n=== 🔒 安全启动 | 正在验证核心接口... ===", "ffff00")
        checker = SystemSelfCheck(logger_callback=ui_logger)
        status_result = checker.run_diagnostics() 
        Clock.schedule_once(lambda dt: self.main_screen.update_lock_state(status_result), 0.1)

if __name__ == '__main__':
    try:
        AlphaHunterApp().run()
    except Exception as e:
        print(traceback.format_exc())
        RECORDER.log_debug("CRASH", traceback.format_exc())
