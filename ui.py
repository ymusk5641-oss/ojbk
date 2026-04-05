import threading
import datetime
import math
import re
import time
import csv
import pandas as pd
import numpy as np
import os
from collections import deque

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.spinner import Spinner
from kivy.clock import Clock, mainthread
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle, RoundedRectangle

from utils import RECORDER, BeijingClock
from config import CFG,CHINESE_FONT
from data import DataLayer
from strategy import QuantEngine
from backtest import BacktestEngine
from ai import AITuningLab, AIEngine, ModelTrainer
from network import NetworkClient

# [核心修复 1] 全局 UI 线程配置锁
# 防止在界面点击保存时，恰好与后台扫描线程发生字典迭代冲突 (RuntimeError: dictionary changed size)
UI_CFG_LOCK = threading.Lock()


# --- 1. UI 风格定义 (Skin) ---
# 金融终端配色表 (Dark Mode)
THEME = {
    "bg": (0.08, 0.08, 0.10, 1),       # 深邃黑背景
    "card": (0.15, 0.15, 0.18, 1),     # 卡片背景
    "text": (0.9, 0.9, 0.9, 1),        # 主文本色
    "text_dim": (0.6, 0.6, 0.6, 1),    # 次文本色
    "accent": (0.2, 0.6, 1.0, 1),      # 科技蓝 (用于强调)
    "up": (1.0, 0.3, 0.3, 1),          # 涨/卖出 (红)
    "down": (0.0, 0.8, 0.4, 1),        # 跌/买入 (绿)
    "gold": (1.0, 0.75, 0.0, 1),       # 警告/重要 (金)
    "btn_scan": (0.1, 0.7, 0.5, 1),    # 扫描按钮色
    "btn_snip": (0.6, 0.2, 0.8, 1),    # 狙击按钮色
    "disabled": (0.3, 0.3, 0.3, 1)     # 禁用灰
}

class ModernButton(Button):
    """[自定义组件] 圆角扁平按钮"""
    def __init__(self, bg_color=THEME["accent"], **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_down = ''
        self.background_color = (0,0,0,0)
        self.bg_color_rgb = bg_color
        self.radius = [dp(10)]
        Clock.schedule_once(lambda dt: self._update_canvas(), 0)
        
    def on_size(self, *args): self._update_canvas()
    def on_pos(self, *args): self._update_canvas()
    def on_press(self): self._update_canvas(is_down=True)
    def on_release(self): self._update_canvas(is_down=False)

    def _update_canvas(self, is_down=False):
        if not self.canvas: return
        self.canvas.before.clear()
        with self.canvas.before:
            r, g, b, a = self.bg_color_rgb
            if self.disabled: Color(*THEME["disabled"])
            elif is_down: Color(r*0.8, g*0.8, b*0.8, a)
            else: Color(r, g, b, a)
            RoundedRectangle(pos=self.pos, size=self.size, radius=self.radius)

class RobustTextInput(TextInput):
    """[UI 组件] 强健型输入框 (Fix Android Focus)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.write_tab = False 
        self.multiline = True

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if not touch.grab_current or touch.grab_current is self:
                Clock.schedule_once(lambda dt: self._force_refocus(), 0.1)
        return super().on_touch_up(touch)

    def _force_refocus(self):
        self.focus = False
        self.focus = True

class OptimizedConsole(ScrollView):
    """[UI 组件] 显存优化版控制台"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (1, 1)
        self.do_scroll_x = False
        self.bar_width = dp(10)
        self.effect_cls = 'ScrollEffect'
        self.layout = GridLayout(cols=1, spacing=dp(2), size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.add_widget(self.layout)
        self.update_text("[color=ffff00]A股猎手 V5.0 (DataSource架构) 就绪[/color]")
        self.MAX_WIDGETS = 150 

    @mainthread
    def update_text(self, msg, color=None):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        color_hex = color if color else "ffffff"
        msg_clean = str(msg).replace("[bold red]", "").replace("[/]", "")
        if "[color=" in msg_clean:
            full_text = f"[color=888888][{timestamp}][/color] {msg_clean}"
        else:
            full_text = f"[color=888888][{timestamp}][/color] [color={color_hex}]{msg_clean}[/color]"
        
        new_label = Label(
            text=full_text, markup=True, size_hint_y=None, font_name='Roboto',
            font_size='13sp', halign='left', valign='top', padding=(dp(5), dp(2))
        )
        new_label.bind(width=lambda *x: new_label.setter('text_size')(new_label, (new_label.width, None)))
        new_label.bind(texture_size=lambda *x: new_label.setter('height')(new_label, new_label.texture_size[1]))
        
        self.layout.add_widget(new_label)
        if len(self.layout.children) > self.MAX_WIDGETS:
            try: self.layout.remove_widget(self.layout.children[-1])
            except: pass
        Clock.schedule_once(lambda dt: self.scroll_to_bottom(), 0.1)

    def scroll_to_bottom(self): self.scroll_y = 0


class BacktestPopup(Popup):
    """
    [UI 组件] 回测实验室 (增强版: 支持持仓独立回测)
    核心功能:
    1. 批量回测时，显示 "X 自选 + Y 持仓" 的构成。
    2. 支持输入框代码临时加入回测。
    3. [新增] 支持一键回测纯持仓股。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "回测与挖掘实验室 (Lab)"
        self.size_hint = (0.95, 0.95)
        # 保持原有的引擎初始化
        self.engine = BacktestEngine(NetworkClient())
        self.ai_lab = AITuningLab(self.engine.net)
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # --- [新增] 0. 挖掘参数设置区 (置顶) ---
        mining_layout = BoxLayout(size_hint_y=None, height=dp(40), spacing=10)
        
        lbl_batch = Label(text="挖掘步长(Batch):", size_hint_x=None, width=dp(110), font_name='Roboto')
        
        # 读取配置，默认为 200
        curr_batch = getattr(CFG, 'MINING_BATCH_SIZE', 200)
        self.txt_batch = TextInput(
            text=str(curr_batch), multiline=False, font_name='Roboto', 
            size_hint_x=None, width=dp(80),
            background_color=(0.3, 0.3, 0.35, 1), foreground_color=(0, 1, 1, 1), 
            input_filter='int'
        )
        # 绑定事件：修改后自动保存
        self.txt_batch.bind(focus=self._save_batch_config)
        
        mining_layout.add_widget(lbl_batch)
        mining_layout.add_widget(self.txt_batch)
        
        # --- 1. 原有控制区 (完整保留) ---
        ctrl_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=dp(90), spacing=5)
        
        # 第一排: 输入框 + 模式
        row1 = BoxLayout(spacing=5)
        self.txt_code = RobustTextInput(hint_text="代码(60xxxx)", multiline=False, font_name='Roboto', size_hint_x=0.4)
        
        # [规范化修复] 彻底告别 close 命名，全局统一使用 Tail 代表尾盘模式
        self.mode_spinner = Spinner(
            text='尾盘确认 (Tail)',
            values=('尾盘确认 (Tail)', '早盘激进 (Open)', '盘中均价 (Mid)'),
            size_hint_x=0.6,
            background_color=THEME['btn_scan'],
            font_name='Roboto'
        )
        row1.add_widget(self.txt_code)
        row1.add_widget(self.mode_spinner)
        
        # 第二排: 功能按钮群 (一个都不能少)
        row2 = BoxLayout(spacing=5)
        
        btn_run = ModernButton(text="单股", bg_color=THEME['accent'], font_name='Roboto', size_hint_x=0.15)
        btn_run.bind(on_press=self.run_single_backtest)
        
        btn_batch = ModernButton(text="自选", bg_color=(0.5, 0.2, 0.6, 1), font_name='Roboto', size_hint_x=0.15)
        btn_batch.bind(on_press=self.run_batch_backtest)

        btn_hold_bt = ModernButton(text="持仓", bg_color=(0.8, 0.4, 0.0, 1), font_name='Roboto', size_hint_x=0.15)
        btn_hold_bt.bind(on_press=self.run_holdings_backtest)
        
        btn_ai = ModernButton(text="AI验", bg_color=THEME['gold'], font_name='Roboto', size_hint_x=0.15)
        btn_ai.bind(on_press=self.run_ai_test)

        # 挖掘与训练按钮 - [修复4：优化按钮文案消除歧义]
        btn_mine_own = ModernButton(text="挖持/自选", bg_color=(0.2, 0.5, 0.5, 1), font_name='Roboto', size_hint_x=0.15)
        btn_mine_own.bind(on_press=lambda x: self.run_ml_mining(mode='holdings'))
        
        btn_mine_mkt = ModernButton(text="挖全场", bg_color=(0.2, 0.3, 0.5, 1), font_name='Roboto', size_hint_x=0.15)
        btn_mine_mkt.bind(on_press=lambda x: self.run_ml_mining(mode='market'))
        
        btn_train = ModernButton(text="训练", bg_color=(0.8, 0.2, 0.2, 1), font_name='Roboto', size_hint_x=0.1)
        btn_train.bind(on_press=self.run_ml_training)

        row2.add_widget(btn_run); row2.add_widget(btn_batch); row2.add_widget(btn_hold_bt)
        row2.add_widget(btn_ai); row2.add_widget(btn_mine_own); row2.add_widget(btn_mine_mkt); row2.add_widget(btn_train)
        
        ctrl_layout.add_widget(row1)
        ctrl_layout.add_widget(row2)
        
        # --- 2. 结果显示区 (保留) ---
        self.scroll = ScrollView()
        self.result_lbl = Label(
            text="[b]回测实验室就绪[/b]\n请在上方输入代码或点击按钮开始...\n[挖掘设置] 步长已生效，修改后自动保存。",
            font_name='Roboto', font_size='13sp',
            size_hint_y=None, halign='left', valign='top', markup=True
        )
        self.result_lbl.bind(width=lambda *x: self.result_lbl.setter('text_size')(self.result_lbl, (self.result_lbl.width, None)))
        self.result_lbl.bind(texture_size=lambda *x: self.result_lbl.setter('height')(self.result_lbl, self.result_lbl.texture_size[1]))
        self.scroll.add_widget(self.result_lbl)
        
        # --- 3. 关闭按钮 ---
        btn_close = Button(text="关闭", size_hint_y=None, height=dp(30), font_name='Roboto')
        btn_close.bind(on_press=self.dismiss)
        
        # 组装
        layout.add_widget(mining_layout) 
        layout.add_widget(ctrl_layout)
        layout.add_widget(self.scroll)
        layout.add_widget(btn_close)
        self.content = layout


    def _save_batch_config(self, instance, value=None):
        """[新增] 自动保存挖掘步长"""
        try:
            raw = self.txt_batch.text.strip()
            if raw and raw.isdigit():
                val = int(raw)
                if 'system' not in CFG.data: CFG.data['system'] = {}
                # 只有当数值变化时才保存，减少IO
                if CFG.data['system'].get('mining_batch_size') != val:
                    CFG.data['system']['mining_batch_size'] = val
                    CFG.save()
                    # 不弹窗打扰，只在日志区小声BB一句
                    if hasattr(self, 'result_lbl'):
                        self.result_lbl.text += f"\n⚙️ 步长已存: {val}"
        except: pass

    def _get_mode_key(self):
        """[规范化修复] 统一解析为标准模式指令，默认兜底设为 tail"""
        text = self.mode_spinner.text
        if 'Open' in text: return 'open'
        if 'Mid' in text: return 'mid'
        if 'Tail' in text: return 'tail'
        return 'tail' # 默认死锁为尾盘

    def run_single_backtest(self, instance):
        code = self.txt_code.text.strip()
        if len(code) != 6: self.result_lbl.text = "请输入6位代码！"; return
        mode = self._get_mode_key()
        self.result_lbl.text = f"正在回测 {code} (模式: {mode})...\n"
        threading.Thread(target=self._thread_single, args=(code, mode), daemon=True).start()
        
    def run_batch_backtest(self, instance):
        """
        [逻辑升级] 批量回测 + 统计显示
        """
        try:
            import threading
            from config import UI_CFG_LOCK
            with UI_CFG_LOCK:
                CFG.data = CFG.load_config()
        except: pass
        
        # 1. 获取要去重的完整列表 (Union)
        targets = list(CFG.TARGET_STOCKS) 
        
        # 2. 智能合并输入框
        manual_code = self.txt_code.text.strip()
        if len(manual_code) == 6:
            if manual_code not in targets:
                targets.insert(0, manual_code)
        
        if not targets: 
            self.result_lbl.text = "⚠️ 列表为空！\n请在[自选管理]添加，或在输入框填入代码。"
            return
            
        # 3. 计算统计数据 (自选 vs 持仓)
        holdings_set = set(CFG.HOLDINGS.keys())
        n_holdings = sum(1 for t in targets if t in holdings_set)
        n_others = len(targets) - n_holdings
        
        mode = self._get_mode_key()
        msg = f"正在回测 {len(targets)} 只股票 (含 {n_others} 只纯自选 + {n_holdings} 只持仓)...\n"
        if len(manual_code) == 6:
             msg += f"(包含输入框代码: {manual_code})\n"
        self.result_lbl.text = msg
        
        threading.Thread(target=self._thread_batch, args=(targets, mode), daemon=True).start()

    def run_holdings_backtest(self, instance):
        """[新增] 仅回测持仓股"""
        try:
            from config import UI_CFG_LOCK
            with UI_CFG_LOCK:
                CFG.data = CFG.load_config()
        except: pass
        
        # 直接读取 HOLDINGS 字典的 Key，不混入自选
        holdings = list(CFG.HOLDINGS.keys())
        
        if not holdings:
            self.result_lbl.text = "⚠️ 当前无持仓！\n请在[持仓管理]中添加。"
            return
            
        mode = self._get_mode_key()
        self.result_lbl.text = f"正在回测 {len(holdings)} 只持仓股...\n"
        threading.Thread(target=self._thread_batch, args=(holdings, mode), daemon=True).start()

    def run_ai_test(self, instance):
        code = self.txt_code.text.strip()
        if len(code) != 6: self.result_lbl.text = "请输入6位代码！"; return
        self.result_lbl.text = f"正在唤醒 DeepSeek 对 {code} 进行历史审计...\n"
        threading.Thread(target=self._thread_ai, args=(code,), daemon=True).start()

    def _thread_single(self, code, mode):
        # [核心修正] 调用时开启 force_full_pos=True
        # 这样单股回测时就会使用 100% 资金，而不是被锁死在 20%
        res = self.engine.run_single_stock(code, buy_mode=mode, force_full_pos=True)
        Clock.schedule_once(lambda dt: self._update_ui_single(res), 0)


    def _thread_batch(self, codes, mode):
        """
        [UI线程桥接 - 增强版]
        修改: 
        1. 将日志更新改为"追加模式"，解决进度被覆盖导致看起来像卡死的问题。
        2. 增加自动滚动到底部。
        """
        def progress_cb(msg):
            def _update_ui(dt):
                # 1. 获取当前时间
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                
                # 2. [关键修改] 改为追加文本，而不是替换 (setattr)
                # 读取旧文本，如果太长则截断，防止手机内存溢出
                current_text = self.result_lbl.text
                if len(current_text) > 8000: 
                    current_text = "...(早期日志已清理)...\n" + current_text[-6000:]
                
                # 拼接新日志
                self.result_lbl.text = f"{current_text}\n[{ts}] {msg}"
                
                # 3. [关键修改] 强制滚动到底部，确保你能看到最新一行
                if hasattr(self, 'scroll'):
                    self.scroll.scroll_y = 0 
            
            # 发送到 UI 主线程执行
            Clock.schedule_once(_update_ui, 0)
            
        # 启动回测引擎 (这里的 callback 现在会追加日志了)
        stats = self.engine.run_portfolio_test(codes, callback=progress_cb, buy_mode=mode)
        
        # 结束后显示统计结果
        Clock.schedule_once(lambda dt: self._update_ui_batch(stats), 0)

    def _thread_ai(self, code):
        def ui_cb(msg, append=True):
            def main_update():
                if not append: self.result_lbl.text = msg
                else: self.result_lbl.text += msg
            Clock.schedule_once(lambda dt: main_update(), 0)
        self.ai_lab.benchmark_prompt(code, model_name="deepseek-chat", ui_callback=ui_cb)

    def _update_ui_single(self, res):
        if "error" in res: self.result_lbl.text = f"回测失败: {res['error']}"; return
        color = "ff5555" if res['return'] < 0 else "00ff00"
        report = (f"[b]对象:[/b] {res['symbol']} ({res['period']})\n"
                  f"[b]收益:[/b] [color={color}]{res['return']:.2f}%[/color]\n"
                  f"[b]次数:[/b] {res['trades_count']}\n{'-'*30}\n")
        report += "\n".join(res['trade_log']) if res['trade_log'] else "无交易"
        self.result_lbl.text = report

    def _update_ui_batch(self, stats):
        avg = stats['avg_return']
        c = "ff5555" if avg < 0 else "00ff00"
        rpt = (f"=== 组合战报 ===\n股票: {stats['total_stocks']} 只\n胜率: {stats['win_rate']:.1f}%\n"
               f"平均: [color={c}]{avg:.2f}%[/color]\n{'-'*30}\n")
        
        # [安全修复] 防御 Timeout 或 Error 导致的 float 转换崩溃
        def _sort_key(x):
            try: return float(x.split(':')[1].split('%')[0])
            except: return -999.0
            
        rpt += "\n".join(sorted(stats['details'], key=_sort_key, reverse=True))
        self.result_lbl.text = rpt


    def run_ml_mining(self, mode='holdings'):
        """
        [交互逻辑] 启动挖掘 (补全缺失方法)
        功能:
        1. 自动保存步长参数。
        2. 重置引擎 is_running 标志位 (确保能启动)。
        3. 启动后台线程。
        """
        # 1. 保存参数
        self._save_batch_config(None)
        
        # 2. 重置引擎运行标志 (关键！否则停止一次后就再也起不来了)
        if not hasattr(self.engine, 'is_running'):
            setattr(self.engine, 'is_running', True)
        self.engine.is_running = True
        
        # 3. 启动线程
        if mode == 'holdings':
            try:
                from config import UI_CFG_LOCK
                with UI_CFG_LOCK:
                    CFG.data = CFG.load_config()
            except: pass
            
            # [修复1：强制去除非数字并补齐6位，防止含有后缀或掉0导致遗漏持仓]
            targets = list(CFG.TARGET_STOCKS) + list(CFG.HOLDINGS.keys())
            raw_list = [re.sub(r'\D', '', str(x)).zfill(6) for x in set(targets)]
            final_list = [x for x in raw_list if len(x) == 6 and x != "000000"]
            
            if not final_list: 
                self.result_lbl.text = "⚠️ 名单为空"; return
            
            self.result_lbl.text = "🚀 正在挖掘 [自选/持仓] 数据..."
            threading.Thread(target=self._thread_ml_mining, args=(final_list, "holdings"), daemon=True).start()
            
        elif mode == 'market':
            batch = getattr(CFG, 'MINING_BATCH_SIZE', 200)
            self.result_lbl.text = f"🚀 正在全市场增量挖掘 (本轮 {batch} 只)..."
            threading.Thread(target=self._thread_mine_broad, daemon=True).start()

    # [配套线程方法]
    def _thread_ml_mining(self, stock_list, tag):
        try:
            path, valid, rows = self.engine.export_ml_training_data(stock_list, days=500, source_tag=tag)
            def ui():
                if path: self.result_lbl.text += f"\n✅ {tag}数据已存:\n{os.path.basename(path)}\n样本:{rows}"
                else: self.result_lbl.text += "\n⚠️ 未生成有效数据"
            Clock.schedule_once(lambda dt: ui(), 0)
        except Exception as e: 
            # [修复2：捕获异常并传回UI主线程，防止前台界面假死挂起]
            def ui_err():
                self.result_lbl.text += f"\n❌ 挖掘失败: {str(e)}"
            Clock.schedule_once(lambda dt: ui_err(), 0)

    def _thread_mine_broad(self):
        """
        [架构修正 V7.0 - UI 表现层]
        职责: 接收 Engine 的三元组数据，负责解包并渲染成人类可读的文案。
        """
        try:
            # 1. 解包数据 (架构层已对齐为 3 个返回值)
            res_path, count, info = self.engine.mine_broad_market()
            
            def ui_update():
                if res_path:
                    # 成功分支: info 是 int (total_rows)
                    # UI 负责决定怎么“吹”这个结果
                    msg = (f"\n✅ [广撒网完成]\n"
                           f"📁 文件: {os.path.basename(res_path)}\n"
                           f"📊 战果: 新增 {count} 只标的 / {info} 行数据\n"
                           f"🚀 状态: 已存入训练库，可直接点击 [⚔️训练]")
                    self.result_lbl.text += msg
                else:
                    # 失败分支: info 是 str (error_message)
                    # UI 负责渲染错误警告
                    self.result_lbl.text += f"\n❌ [广撒网中断] 原因: {info}"
            
            Clock.schedule_once(lambda dt: ui_update(), 0)
            
        except Exception as e:
            # [修复3：改为追加模式，防止抛出异常时清空之前回测大厅里的所有日志记录]
            Clock.schedule_once(lambda dt: setattr(self.result_lbl, 'text', self.result_lbl.text + f"\n💥 系统异常: {str(e)}"), 0)


    def run_ml_training(self, instance):
        """[交互] 启动 AI 训练线程"""
        self.result_lbl.text = "🚀 正在初始化神经网络实验室...\n准备开始训练 Random Forest 模型..."
        threading.Thread(target=self._thread_ml_train, daemon=True).start()

    def _thread_ml_train(self):
        """[线程] 执行训练 -> 线程内热加载 -> UI通知"""
        trainer = ModelTrainer()
        
        # 定义回调函数
        def update_ui_log(msg):
            def _append():
                if len(self.result_lbl.text) > 3000:
                    self.result_lbl.text = "...(旧日志已清理)...\n" + self.result_lbl.text[-1500:]
                self.result_lbl.text += f"\n{msg}"
            Clock.schedule_once(lambda dt: _append(), 0)

        # 1. 执行训练 (耗时操作)
        success = trainer.run_training_task(update_ui_log)
        
        # 2. 热重载 (Hot-Reload) - [核心修复: 在线程内加载，不要去主线程卡UI]
        if success:
            try:
                update_ui_log("⚡ 正在热加载新模型到内存...")
                # 在子线程执行 I/O 加载，这是安全的，因为 strategy.py 有 _model_lock 锁
                # 实例化临时对象调用类方法即可
                QuantEngine().load_ai_model(force_reload=True)
                
                # 加载完后再去通知 UI
                def _notify_finish():
                    self.result_lbl.text += "\n\n✨ [系统通知] 新模型热加载完毕！\n现在回测/实盘均已生效。"
                Clock.schedule_once(lambda dt: _notify_finish(), 0)
                
            except Exception as e:
                update_ui_log(f"❌ 热加载失败: {e}")



class LogManagerPopup(Popup):
    """
    [UI 组件] 日志管理弹窗 (完整分页版)
    修复: 
    1. 完美还原“上一页/下一页”翻页功能，而非简陋的显示最后100行。
    2. 增加字体状态诊断，防止中文乱码。
    3. 支持多种日志文件切换。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # [诊断] 检查字体是否加载成功
        font_status = "✅正常" if CHINESE_FONT else "❌缺失(请放入font.ttf)"
        self.title = f"日志管理系统 (字体: {font_status})"
        self.size_hint = (0.95, 0.95)
        
        # 分页数据结构
        self.all_lines = []        
        self.total_pages = 0       
        self.current_page = 0      
        self.PAGE_SIZE = 100       
        
        layout = BoxLayout(orientation='vertical', padding=5, spacing=5)
        
        # --- 1. 顶部筛选栏 ---
        filter_box = BoxLayout(size_hint_y=0.08, spacing=5)
        
        self.log_type = Spinner(
            text='运行总结', 
            values=('运行总结', '交易日记(Journal)', '决策流水(Factors)', 'AI审计链', 'Debug日志'),
            size_hint_x=0.5,
            font_name='Roboto',
            background_color=(0.2, 0.2, 0.2, 1)
        )
        self.log_type.bind(text=self.on_log_type_change)
        
        btn_refresh = Button(text="读取/刷新", size_hint_x=0.3, background_color=(0,0.7,0,1), font_name='Roboto')
        btn_refresh.bind(on_press=self.query_logs)
        
        filter_box.add_widget(self.log_type)
        filter_box.add_widget(btn_refresh)
        
        # --- 2. 分页控制栏 (原版功能还原) ---
        page_ctrl_box = BoxLayout(size_hint_y=0.08, spacing=10)
        
        self.btn_prev = Button(text="< 上一页", font_name='Roboto', disabled=True)
        self.btn_prev.bind(on_press=lambda x: self.change_page(-1))
        
        self.lbl_page_info = Label(text="第 - / - 页", font_name='Roboto', size_hint_x=0.4)
        
        self.btn_next = Button(text="下一页 >", font_name='Roboto', disabled=True)
        self.btn_next.bind(on_press=lambda x: self.change_page(1))
        
        page_ctrl_box.add_widget(self.btn_prev)
        page_ctrl_box.add_widget(self.lbl_page_info)
        page_ctrl_box.add_widget(self.btn_next)

        # --- 3. 滚动显示区 ---
        self.scroll_view = ScrollView(
            size_hint=(1, 1),
            do_scroll_x=False, 
            do_scroll_y=True,
            bar_width=dp(10)
        )
        
        self.content_label = Label(
            text="请点击刷新读取日志...",
            font_size='13sp',
            font_name='Roboto',
            color=(0.8, 1, 0.8, 1), # 亮绿色保护眼睛
            size_hint_y=None,
            halign='left',
            valign='top',
            padding=(dp(10), dp(10)),
            markup=True 
        )
        
        # 动态高度绑定
        self.content_label.bind(width=lambda *x: self.content_label.setter('text_size')(self.content_label, (self.content_label.width, None)))
        self.content_label.bind(texture_size=lambda *x: self.content_label.setter('height')(self.content_label, self.content_label.texture_size[1]))
        
        self.scroll_view.add_widget(self.content_label)
        
        # --- 4. 底部关闭按钮 ---
        btn_close = Button(text="关闭窗口", size_hint_y=0.08, font_name='Roboto')
        btn_close.bind(on_press=self.dismiss)
        
        layout.add_widget(filter_box)
        layout.add_widget(page_ctrl_box)
        layout.add_widget(self.scroll_view)
        layout.add_widget(btn_close)
        self.content = layout

    def _get_target_file(self):
        t = self.log_type.text
        if t == '运行总结': return RECORDER.summary_file
        if t == 'Debug日志': return RECORDER.debug_file
        if t == 'AI审计链': return RECORDER.trace_file 
        if t == '决策流水(Factors)': return RECORDER.factor_file
        if t == '交易日记(Journal)': return CFG.JOURNAL_FILE 
        return None

    def on_log_type_change(self, instance, value):
        self.content_label.text = "点击刷新以读取..."
        self.all_lines = []
        self.update_page_controls()

    def query_logs(self, instance):
        target_file = self._get_target_file()
        self.content_label.text = "正在读取..."
        
        if not target_file or not os.path.exists(target_file):
            self.content_label.text = "❌ 文件不存在，请先运行策略。"
            return
        try:
            # [核心修复] 使用 deque 只读取文件末尾的 3000 行，防 OOM 内存爆炸
            with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
                self.all_lines = list(deque(f, maxlen=3000))
            
            total_lines = len(self.all_lines)
            if total_lines == 0:
                self.content_label.text = "⚠️ 文件为空"
                self.all_lines = []
                self.update_page_controls()
                return

            self.total_pages = math.ceil(total_lines / self.PAGE_SIZE)
            self.current_page = max(0, self.total_pages - 1) # 默认跳到最后一页
            self.render_current_page()

        except Exception as e:
            self.content_label.text = f"❌ 读取失败: {e}"

    def change_page(self, delta):
        new_page = self.current_page + delta
        if 0 <= new_page < self.total_pages:
            self.current_page = new_page
            self.render_current_page()

    def render_current_page(self):
        start_idx = self.current_page * self.PAGE_SIZE
        end_idx = min(start_idx + self.PAGE_SIZE, len(self.all_lines))
        display_lines = self.all_lines[start_idx:end_idx]
        
        warn_msg = ""
        if not CHINESE_FONT:
            warn_msg = "[color=ff0000][严重警告] 系统中文字体缺失，日志可能无法显示！\n请下载 msyh.ttf 改名为 font.ttf 放入 Hunter_Logs 文件夹。\n\n[/color]"

        header = f"=== 文件: {self.log_type.text} | 行数: {start_idx+1}-{end_idx} (共 {len(self.all_lines)}) ===\n\n"
        
        # 简单清洗非法字符，防止 Markup 报错
        clean_lines = []
        for line in display_lines:
            clean_lines.append(line.replace('[', '【').replace(']', '】')) # 替换掉可能干扰 Kivy 颜色标签的方括号

        full_text = warn_msg + header + "".join(clean_lines)
        self.content_label.text = full_text
        self.update_page_controls()
        self.scroll_view.scroll_y = 1 # 滚回顶部

    def update_page_controls(self):
        if not self.all_lines:
            self.lbl_page_info.text = "第 - / - 页"
            self.btn_prev.disabled = True
            self.btn_next.disabled = True
            return

        self.lbl_page_info.text = f"第 {self.current_page + 1} / {self.total_pages} 页"
        self.btn_prev.disabled = (self.current_page == 0)
        self.btn_next.disabled = (self.current_page >= self.total_pages - 1)


class HelpPopup(Popup):
    """
    [UI 组件] 用户手册弹窗 (完整文案版)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "系统使用说明书"
        self.size_hint = (0.9, 0.85)
        
        # 原版内置说明书文本全面升级 V5.0 工业级实战版
        manual_text = """
【A股猎手 (Alpha Hunter) 工业级操作手册】

一、 基础配置与故障排除
1. 环境准备：系统强依赖中文字体，请务必在 Hunter_Logs 目录下放入 font.ttf（推荐微软雅黑），否则日志将显示乱码。
2. 引擎点火：首次使用请在配置文件 (hunter_config.json) 中填入 DeepSeek 或 Gemini 的 API Key。若无 Key，大模型风控官将下线，系统自动降级为纯量化计算模式。
3. 启动自检：若卡在启动界面，请检查网络连通性或 API Key 额度。若触发“系统基建损坏”，请检查磁盘写入权限。

二、 核心实战功能 (五大引擎)
1. [单股诊断 (Single)]：在上方输入框填入6位代码。此模式无视初级低分过滤，强制执行深度体检，适合盘中对意向股票进行极速排雷与逻辑证伪。
2. [混合扫描 (Scan)]：全市场雷达。调取实时数据扫描 5000+ 标的，过滤出量价异动股并送入 AI 审计。**尾盘绝杀核心按钮，建议在 14:45-14:55 点击。**
3. [自选狙击 (Sniper)]：精细化盯盘。仅对你在配置中添加的“自选股”进行扫描打分，适合早中盘寻找突破买点。
4. [持仓审计 (Audit)]：防守核心。扫描当前持仓股，结合动态 ATR 止损、最高价回撤、RSI 情绪极值与滞涨天数，输出明确的 [HOLD/高抛止盈/止损斩仓] 指令。
5. [回测实验室 (Lab)]：点击 [回测] 进入。支持挖掘历史高盈亏比样本（MFE/MAE），训练本地化机器学习模型（Random Forest + HGB），并支持模型热加载。

三、 仪表盘核心指标解读
1. [状态栏]：
   - 评分(Score)：量化多因子总分（满分100，受拥挤度惩罚可能下降）。>=75分具备操作价值。
   - AI爆发率：本地机器学习模型预测的胜率。>52% 时触发“AI 特权通道”，放宽传统均线约束。
2. [维度与形态]：
   - 筹码 (Winner Rate)：获利盘比例。>95% 视为绝对锁仓，<10% 视为血筹寻底。
   - 乖离率 (Bias20 & VWAP_Bias)：偏离 20 日均线或量价均线的幅度。Bias20 > 15% 属于高危区，系统极大概率拦截。
   - 形态标签：由引擎自动生成的物理画像，如 [多头|放量|🚀突破ATR] 或 负面标签 [缩量诱多]。
3. [资金管理 (Kelly Sizing)]：
   - 系统内置凯利公式与波动率倒数计算仓位。输出如：`200股 (¥36000) [进攻|Vol:2.1%|Kelly:0.45]`。
   - 若建议为 0 股，说明触发硬风控（如波动失控、单板块持仓超 30% 上限、大盘处于熊市冰点等）。
4. [AI 审计官 (LLM Action)]：
   - PASS (做多)：逻辑闭环，主力意图明确，量价健康配合基本面公告。
   - WATCH (观望)：存在逻辑瑕疵（如无明显资金流入、公告平庸），建议等待。
   - REJECT (拒绝)：致命隐患（如高位缩量、顶背离、利好兑现诱多），一票否决。

四、 标准化交易日记 (实战SOP)
▶ 尾盘潜伏流 (T日 14:45)：
   1. 点击 [混合扫描]。观察顶部 RSRS 系数，若提示“市场熔断”，立刻关软件空仓。
   2. 寻找共振标的：AI 审计为 PASS + 评分 >= 75 + 20日乖离 < 12%。
   3. 严格按 [💡 建议] 栏位给出的股数挂单，遵守板块分散原则。
▶ 早盘风控流 (T+1日 09:25-09:40)：
   1. 点击 [持仓审计]。
   2. 若提示 `止损卖出` (破位) 或 `高抛止盈` (情绪极值/保本线触发)，无条件执行。
   3. 若提示 `HOLD`，关闭软件，拒绝盘中情绪干扰，让利润奔跑。

五、 常见问题
Q: 为什么量化评分高达 90 分，AI 却给出 [REJECT]？
A: 量价指标容易被主力画线骗线。AI 首席审计官会通过 RAG 抓取最新公告，若发现高位爆量且配合减持公告，会瞬间识别为“主力派发诱多”并果断拦截。相信风控。
"""

        layout = BoxLayout(orientation='vertical', padding=5, spacing=5)
        
        scroll_view = ScrollView(
            size_hint=(1, 1), 
            do_scroll_x=False, 
            do_scroll_y=True,
            bar_width=dp(10)
        )
        
        self.content_label = Label(
            text=manual_text,
            font_size='14sp',
            font_name='Roboto',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=None,
            halign='left',
            valign='top',
            padding=(dp(10), dp(10))
        )
        
        self.content_label.bind(width=lambda *x: self.content_label.setter('text_size')(self.content_label, (self.content_label.width, None)))
        self.content_label.bind(texture_size=lambda *x: self.content_label.setter('height')(self.content_label, self.content_label.texture_size[1]))
        
        scroll_view.add_widget(self.content_label)
        
        btn_close = Button(text="关闭说明书", size_hint_y=0.1, font_name='Roboto')
        btn_close.bind(on_press=self.dismiss)
        
        layout.add_widget(scroll_view)
        layout.add_widget(btn_close)
        self.content = layout



class AlphaHunterGUIWrapper:
    """
    [业务编排层 - 工业级解耦版 V6.0]
    架构升级:
    1. 彻底实现“标的分析(Stateless)”与“资产估值(Stateful)”的管线隔离。
    2. 解决单股模式下的资产计算失真与 Kelly 仓位错乱 Bug。
    3. 修复大对象驻留导致的移动端内存隐患。
    """
    def __init__(self, ui_console):
        self.ui = ui_console
        self.net = NetworkClient() 
        self.data = DataLayer(self.net)
        self.quant = QuantEngine()
        
        # [审计修复] 异步预热 AI 模型
        threading.Thread(target=self.quant.load_ai_model, daemon=True).start()
        
        self.ai = AIEngine(self.net)

    # [核心修复 2] 移除 @mainthread 装饰器！
    # 让文件 I/O (RECORDER) 留在后台子线程，防止硬盘写入卡死 UI 滑动。
    def log(self, msg, color=None, to_screen=True):
        """双通道日志: to_screen=False 时仅写文件"""
        if to_screen:
            # self.ui.update_text 本身带有 @mainthread，会自动将渲染排队到主线程，绝不越权！
            self.ui.update_text(msg, color)
            
        try:
            RECORDER.log_ui(msg)
        except:
            pass


    def _to_hex(self, color_tuple):
        """[辅助] Kivy Color (r,g,b,a) -> Hex String (rrggbbaa)"""
        try:
            return ''.join([f'{int(c*255):02x}' for c in color_tuple])
        except:
            return "ffffffff"

    def _calc_portfolio_valuation(self):
        """
        [领域二: 独立的资产估值]
        专职独立计算资产，与标的诊断数据流彻底隔离。
        返回: (动态总资产, 现金, 持仓市值, 是否精确计算, 板块暴露度字典)
        """
        available_cash = float(CFG.CASH)
        holdings_map = CFG.HOLDINGS
        
        real_market_val = 0.0
        is_exact = True
        sector_exposure_map = {}
        
        if not holdings_map:
            return available_cash, available_cash, 0.0, True, {}
            
        try:
            # 发起极轻量级的独立请求，仅获取持仓股行情
            h_syms = [str(k).zfill(6) for k in holdings_map.keys()]
            h_df = pd.DataFrame({'symbol': h_syms})
            # 复用基础通道获取持仓股的实时价与板块归属
            quotes_df = self.data.get_specific_stocks_hybrid(h_df)
            
            for h_code, h_data in holdings_map.items():
                clean_h_code = str(h_code).zfill(6)
                vol = h_data.get('volume', 0) if isinstance(h_data, dict) else 0
                cost = h_data.get('cost', 0) if isinstance(h_data, dict) else float(h_data)
                
                h_price = cost
                h_ind = '未知'
                
                if not quotes_df.empty:
                    match_row = quotes_df[quotes_df['symbol'].astype(str).str.zfill(6) == clean_h_code]
                    if not match_row.empty:
                        h_price = match_row.iloc[0]['price']
                        h_ind = match_row.iloc[0].get('ind', '未知')
                    else:
                        is_exact = False
                else:
                    is_exact = False
                    
                real_market_val += h_price * vol
                sector_exposure_map[h_ind] = sector_exposure_map.get(h_ind, 0.0) + (h_price * vol)
                
        except Exception as e:
            RECORDER.log_debug("Valuation_Err", f"独立资产估值回退: {str(e)}")
            is_exact = False
            # 异常兜底：断网等极端情况下使用成本价保守计算
            for h_code, h_data in holdings_map.items():
                vol = h_data.get('volume', 0) if isinstance(h_data, dict) else 0
                cost = h_data.get('cost', 0) if isinstance(h_data, dict) else float(h_data)
                real_market_val += cost * vol
                sector_exposure_map['未知'] = sector_exposure_map.get('未知', 0.0) + (cost * vol)
                
        total_assets_dynamic = available_cash + real_market_val
        
        # 将市值转为实际板块暴露比例
        for k in sector_exposure_map:
            sector_exposure_map[k] = sector_exposure_map[k] / (total_assets_dynamic + 1e-9)
            
        return total_assets_dynamic, available_cash, real_market_val, is_exact, sector_exposure_map

    def run_logic(self, target_mode=False, specific_source=None, extra_code=None):
        """
        [主控调度器 - 逻辑解耦重构]
        职责：按序调度宏观研判、独立估值、标的诊断、聚合决策与终末渲染。
        """
        import csv
        try:
            start_t = time.time(); phase = BeijingClock.get_phase()

            # ==========================================
            # 阶段 1: 模式精准识别与环境预审
            # ==========================================
            if specific_source == 'holdings':
                mode_name = "💼 持仓审计 (Holdings Audit)"; target_mode = True 
            elif specific_source == 'single':
                mode_name = f"🔬 单股诊断 ({extra_code})"; target_mode = True
            elif target_mode:
                mode_name = "🎯 自选狙击 (Sniper)"
            else:
                mode_name = "📡 混合扫描 (Scan)"
                
            self.log(f"=== 系统启动 | {mode_name} | 阶段: {phase} ===", "00ffff")
            
            hot_sectors = self.data.get_market_hot_sectors()
            if hot_sectors: self.log(f"🔥 今日热点主线: {','.join(hot_sectors)}", "ff5555")
            
            self.log("🌍 正在研判大盘环境 (RSRS)...")
            raw_regime = self.data.get_market_regime_rsrs()
            if raw_regime < 0:
                regime = 0.5; self.log("⚠️ RSRS 模型数据获取异常，默认中性", "ff0000")
            else:
                regime = raw_regime
                env_desc = {0.2:"🐻熊市", 0.5:"⚖️震荡", 0.8:"📈偏强", 1.0:"🐂极强"}.get(regime, "未知")
                self.log(f"📊 市场状态: {env_desc} (Coeff={regime})", "00ffff")

            # ==========================================
            # 阶段 2: 独立资产估值 (解耦核心)
            # ==========================================
            total_assets_dynamic, available_cash, real_market_val, is_exact, sector_exposure_map = self._calc_portfolio_valuation()
            
            color_tag = "ffaa00" if is_exact else "aaaaaa"
            self.log(f"💎 [{'实时' if is_exact else '估算'}总资产]: ¥{total_assets_dynamic:,.0f} | 市值: ¥{real_market_val:,.0f} | 现金: ¥{available_cash:,.0f}", color_tag)

            # ==========================================
            # 阶段 3: 纯粹的标的构建与扫描分析 (Stateless)
            # ==========================================
            breadth_panic = False
            ratio = 0.5 

            # [核心修复] 提取真实的宏观宽度计算 (彻底消灭 Top200 幸存者偏差)
            scan_df = self.data.get_scan_list_hybrid() # 保留扫描，供后续送审使用
            ratio = self.data.get_real_market_breadth()
            
            self.log(f"宏观市场宽度: {ratio*100:.1f}% 上涨", "cccccc")
            # 使用 getattr 增加鲁棒性，防配置文件属性缺失
            if ratio < getattr(CFG, 'MIN_MARKET_BREADTH', 0.25): # 当全市场不足 25% 上涨时
                breadth_panic = True
                self.log("💥 [熔断警告] 全场行情极度恶劣，触发一票否决风控！", "ff0000")


            if specific_source == 'single':
                clean_code = re.sub(r'\D', '', str(extra_code)).zfill(6)
                if len(clean_code) != 6: self.log(f"❌ 代码格式错误: {extra_code}", "ff0000"); return
                init_df = pd.DataFrame({'symbol': [clean_code]})
                
            elif specific_source == 'holdings':
                h_list = list(CFG.HOLDINGS.keys())
                if not h_list: self.log("⚠️ 提示：当前无持仓数据", "ffff00"); return
                init_df = pd.DataFrame({'symbol': h_list})
                
            elif target_mode:
                codes = CFG.TARGET_STOCKS if isinstance(CFG.TARGET_STOCKS, list) else []
                combined = list(set(codes + list(CFG.HOLDINGS.keys())))
                init_df = pd.DataFrame({'symbol': combined})
                
            else:
                # 👇 [架构级修复: 混扫防守补丁]
                # 混扫模式必须强行混入持仓股！否则不在热点榜上的弱势持仓股将获取不到K线，彻底失去破位止损审计！
                h_list = list(CFG.HOLDINGS.keys())
                if h_list:
                    h_df = pd.DataFrame({'symbol': h_list})
                    init_df = pd.concat([scan_df, h_df]).drop_duplicates(subset=['symbol'])
                else:
                    init_df = scan_df
                # 👆 --------------------------------------------------

            
            df = self.data.get_specific_stocks_hybrid(init_df)
            if df.empty: self.log("❌ 核心行情数据源响应为空", "ff0000"); return


            # ==========================================
            # 阶段 4: 量化指标计算与 AI 逻辑脑审计
            # ==========================================
            df['llm_score'] = -1.0
            df = self.quant.strategy_scoring(
                df, phase=phase, regime=regime,
                breadth_panic=breadth_panic, target_mode=target_mode,
                pre_calculated=False, params=CFG.STRATEGY_PARAMS 
            )
            
            # 依然保留对命中的标的进行持仓审计（无缝兼容）
            audit_dict = self.quant.audit_holdings(df, regime)

            # ==========================================
            # 阶段 4.5: 双轨制精英截断 (Delegated to QuantEngine)
            # ==========================================
            if target_mode:
                candidates_pre = df.copy()
            else:
                # 干净的 Facade 调用，由策略引擎处理复杂的双轨截断与兜底
                candidates_pre = self.quant.filter_top_candidates(df, params=CFG.STRATEGY_PARAMS)
            
            audited_symbols = [str(x).zfill(6) for x in candidates_pre['symbol'].tolist()]
            audit_map = {} 
            
            if not candidates_pre.empty:
                self.log(f"正在进行 AI 深度审计与逻辑证伪 ({len(candidates_pre)} 只)...", "cccccc")
                
                df['rag_info'] = "无近期公告"
                candidates_pre['rag_info'] = "无近期公告"
                
                # [核心架构修复] 异步并发拉取 RAG 公告，打破 I/O 串行瓶颈
                from concurrent.futures import ThreadPoolExecutor, as_completed
                rag_results = {}
                
                # 限制最多 5 个并发，防止短时间内触发 API 防刷风控
                with ThreadPoolExecutor(max_workers=5) as rag_executor:
                    rag_futures = {
                        rag_executor.submit(self.data.get_stock_announcements, row['symbol']): idx 
                        for idx, row in candidates_pre.iterrows()
                    }
                    
                    for future in as_completed(rag_futures, timeout=15):
                        idx = rag_futures[future]
                        try:
                            rag_results[idx] = future.result()
                        except Exception:
                            rag_results[idx] = "公告获取超时"
                            self.log(f"   ⚠️ 标的 {candidates_pre.loc[idx, 'symbol']} 公告获取超时", "aaaaaa")

                # [核心修复 3] 线程安全的批量回写与降频汇报
                success_count = 0
                for idx, rag_info in rag_results.items():
                    df.loc[idx, 'rag_info'] = rag_info 
                    candidates_pre.loc[idx, 'rag_info'] = rag_info
                    if "超时" not in rag_info and "失败" not in rag_info:
                        success_count += 1
                        
                self.log(f"✅ RAG 公告并发获取完成，成功提取 {success_count} 只标的", "00ff00")
                
                ai_res, _ = self.ai.audit(candidates_pre.to_dict('records'), regime, phase, hot_sectors=hot_sectors)


                audit_map = ai_res.get('audit', {})
                
                score_map = {}
                for sym, info in audit_map.items():
                    clean_k = re.sub(r'\D', '', str(sym)).zfill(6)
                    score_map[clean_k] = info.get('llm_score', 50.0)
                
                for idx, row in df.iterrows():
                    curr_sym = re.sub(r'\D', '', str(row['symbol'])).zfill(6)
                    if curr_sym in score_map:
                        df.at[idx, 'llm_score'] = float(score_map[curr_sym])

                df = self.quant.strategy_scoring(
                    df, phase=phase, regime=regime, 
                    breadth_panic=breadth_panic, target_mode=target_mode, 
                    pre_calculated=True, params=CFG.STRATEGY_PARAMS
                )
                
                if target_mode:
                    candidates_final = df
                else:
                    mask_audited = df['symbol'].isin(audited_symbols)
                    candidates_final = df[mask_audited].sort_values('final_score', ascending=False)
            else:
                candidates_final = pd.DataFrame()
                
                # 👇 ---------- [新增 UX 护城河: 显影沉默的熔断] ----------
                if not df.empty and 'data_quality' in df.columns:
                    trash_count = len(df[df['data_quality'] < 0.5])
                    if trash_count == len(df):
                        self.log("❌ [严重警告] 所有标的 K线数据拉取超时 (腾讯接口被限流)，已被系统防呆机制全部拦截！", "ff0000")
                        self.log("💡 建议: 请等待5分钟，或在 config.json 将 max_workers 降至 8-12 防封 IP。", "ffff00")
                    else:
                        self.log("⚠️ [量化拦截] 本轮扫描无任何标的及格 (全部被得分或物理红线斩杀)。", "ffaa00")
                # 👆 ----------------------------------------------------

            # ==========================================
            # 阶段 5: 聚合计算与 UI 动态渲染 (Aggregation)
            # ==========================================
            self.log("\n[ ====== 策略仪表盘 (Smart View) ====== ]", "00ff00")
            results_buffer = []

            for idx, row in candidates_final.iterrows():
                try:
                    sym = str(row['symbol']).zfill(6)
                    score = row['final_score']
                    
                    is_audited = sym in audited_symbols
                    is_holding_for_log = sym in CFG.HOLDINGS
                    
                    if target_mode or is_audited or is_holding_for_log:
                        try:
                            factor_data = {
                                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'symbol': sym, 'name': row['name'], 'price': row['price'],
                                'final_score': f"{score:.1f}", 'ai_score': f"{row.get('ai_score', 0):.1f}", 
                                'trend_desc': row['trend_desc']
                            }
                            for feat in CFG.CORE_FEATURE_SCHEMA:
                                val = row.get(feat, 0.0)
                                try:
                                    if isinstance(val, bool): val = int(val)
                                    factor_data[feat] = f"{float(val):.4f}"
                                except: factor_data[feat] = "0.0000"
                            RECORDER.log_factors(factor_data)
                        except Exception as e: 
                            # [UI直接显影] 用 self.log 打在屏幕上，颜色设为红色 (ff0000)
                            self.log(f"   ❌ [I/O 警告] 因子写入失败 (CSV文件可能被Excel占用)! 错误: {str(e)}", "ff0000")
                            RECORDER.log_debug("FACTOR_LOG_FAIL", str(e))

                    ai_info = {}
                    sym_normalized = str(sym).zfill(6)
                    for k, v in audit_map.items():
                        k_normalized = str(k).zfill(6)
                        if sym_normalized == k_normalized: ai_info = v; break
                    
                    ai_action = ai_info.get('action', 'WATCH')
                    is_holding = sym in CFG.HOLDINGS
                    stock_ind = row.get('ind', '未知')
                    
                    # 使用独立估值得到的精确资产与板块暴露比例
                    current_exposure = sector_exposure_map.get(stock_ind, 0.0)
                    shares, desc, pct, pos_color_tuple = QuantEngine.calculate_target_position(
                        score, row['volatility'], regime, row['price'], available_cash, total_assets_dynamic,
                        force_full_pos=False, current_sector_exposure=current_exposure
                    )
                    
                    # [架构修正] 无论是持仓还是新标的，统一通过入口风控进行体检，消除UI建议量与拦截标签的矛盾
                    should_buy, signal_reason = QuantEngine.check_entry_signal(row, score, row['strategy_name'], regime)
                    
                    if not should_buy:
                        shares = 0
                        desc = f"量化拦截: {signal_reason}"
                        pos_color_tuple = (0.5, 0.5, 0.5, 1)
                    elif ai_action in ['WATCH', 'REJECT']:
                        shares = 0
                        action_cn = "观望" if ai_action == "WATCH" else "拒绝"
                        desc = f"AI风控: {action_cn} - 强制平调凯利仓位"
                        pos_color_tuple = (1.0, 0.66, 0.0, 1) if ai_action == "WATCH" else (1.0, 0.0, 0.0, 1)
                    else:
                        # 只有非持仓的新标的，才去累加板块敞口（因为持仓的敞口已经在独立估值阶段算过了）
                        if not is_holding and shares > 0 and stock_ind != '未知':
                            added_exposure_pct = (shares * row['price']) / (total_assets_dynamic + 1e-9)
                            sector_exposure_map[stock_ind] = current_exposure + added_exposure_pct

                    
                    pos_color_hex = self._to_hex(pos_color_tuple)
                    pos_txt = f"{shares}股 (¥{int(shares*row['price'])}) [{desc}]" if shares > 0 else f"0股 ({desc})"

                    logs_block = []
                    w_rate = row.get('winner_rate', 50); vwap_bias = row.get('bias_vwap', 0)
                    industrial_info = f"筹码:{w_rate:.0f}%{' [color=ff5555](锁仓)[/color]' if w_rate > 90 else ''} | VWAP乖离:{vwap_bias:.1f}%"
                    
                    if is_holding:
                        h_data = CFG.HOLDINGS[sym]
                        cost = h_data.get('cost', 0) if isinstance(h_data, dict) else float(h_data)
                        vol = h_data.get('volume', 0) if isinstance(h_data, dict) else 0
                        csv_mv = row['price'] * vol
                        csv_pnl = (row['price'] - cost) * vol
                        profit_pct = (row['price'] - cost) / cost * 100 if cost > 0 else 0
                        p_color = "ff5555" if profit_pct < 0 else "55ff55"
                        logs_block.append(f"💼 [持仓] [b]{row['name']}[/b] ({sym}) [color={p_color}]盈亏:{profit_pct:+.2f}%[/color]")
                    else:
                        csv_mv = 0.0; csv_pnl = 0.0
                        logs_block.append(f"🔭 [关注] [b]{row['name']}[/b] ({sym})")
                    
                    if sym in audit_dict:
                        alert_msg = audit_dict[sym]
                        # 👇 [UI修复: 废除暴力 split 导致的关键指令丢失]
                        # 用 ':' 仅剥除前缀股票名，完整保留后续的 动作指令/利润率/原因
                        clean_alert = alert_msg.split(':', 1)[-1].strip() if ':' in alert_msg else alert_msg
                        logs_block.append(f"   ⚠️ [风控信号] {clean_alert}")
                        # 👆 --------------------------------------------------

                    ai_val = row.get('ai_score', 0)
                    
                    logs_block.append(f"   状态: 评分:{score:.0f} | AI爆发率:{ai_val:.1f}% | 策略:{row['strategy_name']}")
                    logs_block.append(f"   维度: [color=cccc00]{industrial_info}[/color]")
                    logs_block.append(f"   形态: [color=dddddd]{row['trend_desc']}[/color] | 20日乖离:{row['bias_20']:.1f}%")
                    logs_block.append(f"   💡 建议: [color={pos_color_hex}]{pos_txt}[/color]")
                    logs_block.append(f"   [color=00ffff][AI审计][/color] {ai_action}: {ai_info.get('reason', 'N/A')}")
                    logs_block.append("[color=333333]" + "-" * 40 + "[/color]")

                    try:
                        with open(CFG.JOURNAL_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                            mark = "HOLD" if is_holding else "WATCH"
                            writer = csv.writer(f)
                            writer.writerow([
                                BeijingClock.now_str(), sym, row['name'], row['price'], f"{score:.1f}", 
                                phase, f"{regime:.2f}", f"{ratio:.2f}", f"{row['pe']:.1f}", f"{row['bias_20']:.1f}", 
                                row.get('flow', 0), ai_info.get('reason', 'N/A'), mark, f"{time.time()-start_t:.1f}s", 
                                ai_action, row['trend_desc'], f"{w_rate:.1f}", f"{vwap_bias:.1f}", 
                                f"{csv_mv:.0f}", f"{csv_pnl:.0f}", f"{total_assets_dynamic:.0f}"
                            ])
                    except Exception as e:
                        # 放弃不可靠的 print，使用系统级框架安全落盘日志
                        RECORDER.log_debug("JOURNAL_ERR", f"日记写入异常: {str(e)}")

                    p_holding = 1 if is_holding else 0
                    action_map = {'PASS': 3, 'WATCH': 2, 'REJECT': 1}
                    p_action = action_map.get(ai_action, 0)
                    p_score = score
                    p_ai_prob = row.get('ai_score', 0)
                    
                    priority = (p_holding, p_action, p_score, p_ai_prob)
                    results_buffer.append({'priority': priority, 'logs': logs_block})
                
                except Exception as e:
                    continue

            # 👇 ---------- [终极架构补丁: 防守盲区扫尾] ----------
            # 补齐盲区：混扫模式下，部分持仓股可能因没进 TopN 或被拦截而未被上方循环渲染。
            # 遍历 audit_dict，将漏掉的持仓报警强行补打到 UI，防止用户错失斩仓/止盈时机！
            processed_symbols = [str(x).zfill(6) for x in candidates_final['symbol'].tolist()] if not candidates_final.empty else []
            for h_sym, alert_msg in audit_dict.items():
                clean_sym = str(h_sym).zfill(6)
                if clean_sym not in processed_symbols:
                    # 只有真正触发了动作的才显影，安静的 HOLD 状态不打扰用户
                    if "HOLD" not in alert_msg:
                        logs_block = [
                            f"🛡️ [持仓防守盲区] [b]{clean_sym}[/b]",
                            f"   {alert_msg}",
                            "[color=333333]" + "-" * 40 + "[/color]"
                        ]
                        # 赋予最高优先级 (0, 0, 999) 强制置顶显示
                        results_buffer.append({'priority': (2, 9, 999, 999), 'logs': logs_block})
            # 👆 --------------------------------------------------

            # ==========================================
            # 阶段 6: 扫尾处理与防视觉蒸发
            # ==========================================

            # [架构修正] 单股诊断模式(single)无需触发持仓断网警报，避免冗余刷屏
            if target_mode and specific_source != 'single':
                processed_holdings = candidates_final['symbol'].astype(str).str.zfill(6).tolist() if not candidates_final.empty else []
                for h_sym, h_data in CFG.HOLDINGS.items():
                    clean_h_sym = str(h_sym).zfill(6)
                    if clean_h_sym not in processed_holdings:
                        logs_block = [
                            f"💼 [持仓] [b]未知(断网/停牌)[/b] ({clean_h_sym}) [color=aaaaaa]盈亏:无法计算[/color]",
                            "   ⚠️ [系统提示] 未拉取到该股最新行情数据，暂时挂起。"
                        ]
                        priority = (1, 0, 0, 0) 
                        results_buffer.append({'priority': priority, 'logs': logs_block})

            results_buffer.sort(key=lambda x: x['priority'], reverse=False)
       
            for item in results_buffer:
                for line in item['logs']: self.log(line)

            self.log(f"✅ 系统作业完成 | 总耗时 {time.time()-start_t:.1f}s", "00ff00")
            
        except Exception as e:
            self.log(f"❌ 系统级故障: {str(e)}", "ff0000")
            RECORDER.log_exception("run_logic_critical", e)



class MainScreen(Screen):
    """
    [主界面 - 单股/扫描/狙击 融合版 V5.0]
    布局优化:
    1. 第一排增加单股代码输入框及单股诊断按钮。
    2. 混合扫描和自选狙击按钮精简文字，移除英文以适配移动端宽度。
    3. 整体风格保持现代扁平化。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "main"
        
        # 根布局
        root = BoxLayout(orientation='vertical', padding=dp(15), spacing=dp(10))
        with root.canvas.before:
            Color(*THEME["bg"])
            Rectangle(pos=self.pos, size=self.size)
        root.bind(pos=self._update_bg, size=self._update_bg)
        self.root_rect = root.canvas.before.children[-1]

        # 1. 顶部状态栏
        header = BoxLayout(size_hint_y=0.08, padding=(dp(10), 0))
        self.status_lbl = Label(
            text=f"ALPHA HUNTER 405.0 | PHASE: {BeijingClock.get_phase()}", 
            font_size='14sp', font_name='Roboto', color=THEME["accent"], bold=True, halign='left', valign='middle'
        )
        self.status_lbl.bind(size=self.status_lbl.setter('text_size'))
        header.add_widget(self.status_lbl)
        root.add_widget(header)
        
        # 2. 中部日志区
        console_container = BoxLayout(size_hint_y=0.72, padding=dp(2))
        with console_container.canvas.before:
            Color(*THEME["card"])
            self.console_bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)])
        console_container.bind(pos=self._update_console_bg, size=self._update_console_bg)
        
        self.console_view = OptimizedConsole()
        console_container.add_widget(self.console_view)
        root.add_widget(console_container)
        
        # 3. 底部操作区
        controls = BoxLayout(orientation='vertical', size_hint_y=0.20, spacing=dp(8))
        
        # --- 第一排: 任务执行 ---
        row1 = BoxLayout(spacing=dp(6))
        
        self.txt_single = RobustTextInput(
            hint_text="代码", multiline=False, font_name='Roboto', 
            size_hint_x=0.20, background_color=(0.15, 0.15, 0.18, 1),
            foreground_color=(1, 1, 1, 1), cursor_color=(0, 1, 0, 1), write_tab=False
        )
        
        self.btn_single = ModernButton(text="单股", font_size='14sp', bold=True, bg_color=THEME["accent"], font_name='Roboto', size_hint_x=0.15)
        self.btn_single.bind(on_press=lambda x: self.start_task(target_mode=True, specific_source='single'))
        
        self.btn_scan = ModernButton(text="混合扫描", font_size='14sp', bold=True, bg_color=THEME["btn_scan"], font_name='Roboto', size_hint_x=0.22)
        self.btn_scan.bind(on_press=lambda x: self.start_task(target_mode=False))
        
        self.btn_sniper = ModernButton(text="自选狙击", font_size='14sp', bold=True, bg_color=THEME["btn_snip"], font_name='Roboto', size_hint_x=0.22)
        self.btn_sniper.bind(on_press=lambda x: self.start_task(target_mode=True))
        
        self.btn_audit = ModernButton(text="持仓审计", font_size='14sp', bold=True, bg_color=(0.8, 0.4, 0.0, 1), font_name='Roboto', size_hint_x=0.21)
        self.btn_audit.bind(on_press=lambda x: self.start_task(target_mode=True, specific_source='holdings'))

        row1.add_widget(self.txt_single); row1.add_widget(self.btn_single)
        row1.add_widget(self.btn_scan); row1.add_widget(self.btn_sniper); row1.add_widget(self.btn_audit)
        
        # --- 第二排: 管理工具 ---
        row2 = BoxLayout(spacing=dp(5)) 
        self.btn_cfg = ModernButton(text="自选", bg_color=(0.25, 0.25, 0.3, 1), font_size='13sp', font_name='Roboto')
        self.btn_cfg.bind(on_press=self.show_config_popup)

        self.btn_hold = ModernButton(text="持仓", bg_color=(0.3, 0.3, 0.4, 1), font_size='13sp', font_name='Roboto')
        self.btn_hold.bind(on_press=self.show_holdings_popup)

        self.btn_bt = ModernButton(text="回测", bg_color=(0.5, 0.2, 0.6, 1), font_size='13sp', font_name='Roboto')
        # [核心修复] 改为调用 self.open_backtest，而不是直接 lambda new
        self.btn_bt.bind(on_press=self.open_backtest)

        self.btn_help = ModernButton(text="说明", bg_color=(0.2, 0.5, 0.6, 1), font_size='13sp', font_name='Roboto')
        self.btn_help.bind(on_press=lambda x: HelpPopup().open())
        
        btn_log = ModernButton(text="日志", bg_color=THEME["gold"], color=(0,0,0,1), font_size='13sp', bold=True, font_name='Roboto')
        btn_log.bind(on_press=lambda x: LogManagerPopup().open())
        
        row2.add_widget(self.btn_cfg); row2.add_widget(self.btn_hold); row2.add_widget(self.btn_bt)
        row2.add_widget(self.btn_help); row2.add_widget(btn_log)
        
        controls.add_widget(row1); controls.add_widget(row2)
        root.add_widget(controls); self.add_widget(root)
        
        # [新增] 缓存回测弹窗实例
        self.popup_backtest = None
        
        self.engine = AlphaHunterGUIWrapper(self.console_view)
        self.toggle_buttons(enabled=False)

    def open_backtest(self, instance):
        """[新增] 单例打开回测窗口，防止重复初始化消耗内存"""
        if self.popup_backtest is None:
            self.console_view.update_text("正在初始化回测实验室...", "cccccc")
            self.popup_backtest = BacktestPopup()
        
        self.popup_backtest.open()

    def _update_bg(self, instance, value):
        """[UI修复] 核心渲染：实时更新根背景矩形的位置和大小"""
        if hasattr(self, 'root_rect'):
            self.root_rect.pos = instance.pos
            self.root_rect.size = instance.size

    def _update_console_bg(self, instance, value):
        """[UI修复] 核心渲染：实时更新日志区圆角背景"""
        if hasattr(self, 'console_bg'):
            self.console_bg.pos = instance.pos
            self.console_bg.size = instance.size


    def update_lock_state(self, status):
        """
        [UI状态机] 根据自检结果进行精细化锁定
        status: {'infra': bool, 'real': bool, 'bt': bool}
        """
        # 1. 默认全开
        self.toggle_buttons(enabled=True)
        
        # 2. 定义按钮组
        btns_real = [self.btn_single, self.btn_scan, self.btn_sniper, self.btn_audit]
        btns_bt = [self.btn_bt]
        # 通用组 (配置/持仓/帮助/日志) 只要基建OK就应该能用
        btns_common = [self.btn_cfg, self.btn_hold, self.btn_help] 
        
        # 3. 逻辑判断
        if not status['infra']:
            # 基建挂了，全锁
            self.toggle_buttons(enabled=False)
            self.console_view.update_text("🛑 严重故障：系统基建损坏，已全功能锁定。", "ff0000")
            return

        # 实盘故障逻辑
        if not status['real']:
            for b in btns_real: 
                b.disabled = True
                b._update_canvas()
            self.txt_single.disabled = True
            self.console_view.update_text("⚠️ 实盘接口故障：扫描与交易功能已锁定。", "ffff00")
        
        # 回测故障逻辑
        if not status['bt']:
            for b in btns_bt: 
                b.disabled = True
                b._update_canvas()
            self.console_view.update_text("⚠️ 回测引擎故障：回测功能已锁定。", "ffff00")
            
        if status['real'] and status['bt']:
            self.console_view.update_text("\n✅ 全系统自检通过，随时待命。", "00ff00")


    def toggle_buttons(self, enabled):
        state = not enabled
        # [修改] 加入 btn_single 到禁用列表
        btns = [
            self.btn_single, self.btn_scan, self.btn_sniper, self.btn_audit, 
            self.btn_cfg, self.btn_hold, self.btn_bt, self.btn_help
        ]
        for btn in btns:
            btn.disabled = state; btn._update_canvas()
        # 输入框也可以选择禁用
        self.txt_single.disabled = state

    def start_task(self, target_mode, specific_source=None):
        self.console_view.update_text("=== SYSTEM STARTED (V405.0) ===", "ffff00")
        
        # [新增] 获取单股代码
        extra_code = None
        if specific_source == 'single':
            extra_code = self.txt_single.text.strip()
            if not extra_code:
                self.console_view.update_text("❌ 请输入股票代码", "ff0000")
                return

        threading.Thread(
            target=self.engine.run_logic, 
            args=(target_mode, specific_source, extra_code), 
            daemon=True
        ).start()

    def show_config_popup(self, instance):
        """
        [界面复原] 系统配置弹窗
        修正: 移除 '挖掘步长' 设置，归还界面空间给用户原有的资金/自选股管理。
        """
        try:
            # 加载最新配置
            try: CFG.data = CFG.load_config()
            except: pass
            
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            
            # --- 提示标签 ---
            lbl_tip = Label(
                text="[自选股列表] 每行一个代码 (如 600519)", 
                size_hint_y=None, height=dp(30), 
                font_name='Roboto', font_size='13sp', color=(0.7, 0.7, 0.7, 1)
            )
            
            # --- 自选股编辑区 (恢复全屏占比) ---
            targets = CFG.TARGET_STOCKS
            if not isinstance(targets, list): targets = []
            
            txt_edit = RobustTextInput(
                text="\n".join(targets), hint_text="在此输入代码...", 
                font_name='Roboto', size_hint_y=1,
                background_color=(0.15, 0.15, 0.18, 1), foreground_color=(1, 1, 1, 1), cursor_color=(0, 1, 0, 1)
            )
            
            # --- 底部按钮 ---
            btn_layout = BoxLayout(size_hint_y=None, height=dp(50), spacing=10)
            btn_cancel = ModernButton(text="取消", bg_color=(0.4, 0.4, 0.4, 1), font_name='Roboto')
            btn_save = ModernButton(text="保存配置", bg_color=THEME["down"], font_name='Roboto')
            btn_layout.add_widget(btn_cancel); btn_layout.add_widget(btn_save)
            
            content.add_widget(lbl_tip)
            content.add_widget(txt_edit)
            content.add_widget(btn_layout)
            
            popup = Popup(title="自选股管理中心", content=content, size_hint=(0.9, 0.85), auto_dismiss=False)
            btn_cancel.bind(on_press=popup.dismiss)
            
            def do_save(btn):
                try:
                    # 仅保存自选股逻辑
                    raw_targets = txt_edit.text
                    codes = re.findall(r'\d{6}', raw_targets if raw_targets else "")
                    unique = sorted(list(set(codes)))
                    
                    # [核心修复 4A] 使用 UI 锁安全接管配置写入
                    with UI_CFG_LOCK:
                        CFG.data['target_stocks'] = unique
                        is_saved = CFG.save()
                        
                    if is_saved:
                        self.console_view.update_text(f"✅ 自选股已更新: {len(unique)}只", "00ff00")
                        popup.dismiss()
                    else: 
                        self.console_view.update_text("❌ 保存失败", "ff0000")
                except Exception as e: 
                    self.console_view.update_text(f"❌ 保存异常: {str(e)}", "ff0000")

            btn_save.bind(on_press=do_save)
            popup.open()
            
        except Exception as e: 
            print(f"❌ 界面错误: {str(e)}")


    def show_holdings_popup(self, instance):
        """[资产管理] (复用原有逻辑)"""
        try:
            content = BoxLayout(orientation='vertical', padding=10, spacing=10)
            cash_layout = BoxLayout(size_hint_y=None, height=dp(40), spacing=10)
            lbl_cash = Label(text="可用资金(¥):", size_hint_x=0.3, font_name='Roboto', font_size='14sp')
            try: CFG.data = CFG.load_config()
            except: pass
            current_cash = CFG.CASH
            self.txt_cash = TextInput(
                text=str(current_cash), multiline=False, font_name='Roboto', size_hint_x=0.7,
                background_color=(0.2, 0.2, 0.25, 1), foreground_color=(0, 1, 1, 1), input_filter='float'
            )
            cash_layout.add_widget(lbl_cash); cash_layout.add_widget(self.txt_cash)
            
            lbl_tip = Label(
                text="[格式] 代码 成本 股数 (如: 600519 1800 100)", 
                size_hint_y=None, height=dp(30), font_name='Roboto', font_size='12sp', color=(0.7, 0.7, 0.7, 1)
            )
            
            holdings = CFG.HOLDINGS
            lines = []
            if holdings:
                for k, v in holdings.items():
                    if isinstance(v, dict): lines.append(f"{k:<8} {v.get('cost',0):<8} {v.get('volume',0)}")
                    else: lines.append(f"{k:<8} {v:<8} 0")
            
            txt_edit = RobustTextInput(
                text="\n".join(lines), hint_text="600519 1800.0 100", 
                font_name='Roboto', size_hint_y=1,
                background_color=(0.15, 0.15, 0.18, 1), foreground_color=(1, 1, 1, 1), cursor_color=(0, 1, 0, 1)
            )
            
            btn_layout = BoxLayout(size_hint_y=None, height=dp(50), spacing=10)
            btn_cancel = ModernButton(text="取消", bg_color=(0.4, 0.4, 0.4, 1), font_name='Roboto')
            btn_save = ModernButton(text="保存", bg_color=THEME["accent"], font_name='Roboto')
            btn_layout.add_widget(btn_cancel); btn_layout.add_widget(btn_save)
            
            content.add_widget(cash_layout); content.add_widget(lbl_tip)
            content.add_widget(txt_edit); content.add_widget(btn_layout)
            
            popup = Popup(title="资产配置中心", content=content, size_hint=(0.95, 0.90), auto_dismiss=False)
            btn_cancel.bind(on_press=popup.dismiss)
            
            def save_all(btn):
                try:
                    raw_cash = self.txt_cash.text.strip()
                    new_h = {}
                    raw_hold = txt_edit.text.strip()
                    if raw_hold:
                        for row in raw_hold.split('\n'):
                            clean = row.replace(',', ' ').strip()
                            parts = clean.split()
                            if len(parts) >= 2:
                                code = parts[0]; cost = float(parts[1])
                                vol = int(parts[2]) if len(parts) >= 3 else 0
                                if len(code) == 6 and code.isdigit():
                                    new_h[code] = {'cost': cost, 'volume': vol, 'ts': int(time.time())}
                    
                    # [核心修复 4B] 使用 UI 锁安全接管复杂结构的深拷贝与写入
                    with UI_CFG_LOCK:
                        if raw_cash: CFG.update_cash(float(raw_cash))
                        CFG.data['holdings'] = new_h
                        if not isinstance(CFG.data.get('target_stocks'), list): CFG.data['target_stocks'] = []
                        for c in new_h.keys():
                            if c not in CFG.data['target_stocks']: CFG.data['target_stocks'].append(c)
                        is_saved = CFG.save()
                        
                    if is_saved:
                        cash_val = float(raw_cash) if raw_cash else 0.0
                        self.console_view.update_text(f"✅ 资产更新: 资金¥{cash_val:.0f} | 持仓{len(new_h)}只", "00ff00")
                        popup.dismiss()
                    else: self.console_view.update_text("❌ 保存失败", "ff0000")
                except Exception as e: self.console_view.update_text(f"❌ 格式错误: {str(e)}", "ff0000")


            btn_save.bind(on_press=save_all)
            popup.open()
        except Exception as e: self.console_view.update_text(f"❌ 错误: {str(e)}", "ff0000")
