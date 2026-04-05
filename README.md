# AlphaHunter Tail v905

> A股量化交易系统 — 实时扫描 · 量化评分 · AI审计 · 回测验证

## 简介

AlphaHunter 是一个面向中国 A 股市场的量化交易系统，集成 Kivy 跨平台 GUI、机器学习模型训练、历史回测和实时行情扫描功能。支持 Android (Pydroid3) 和 Linux 桌面运行。

## 核心功能

- **实时行情扫描** — 从腾讯/东方财富 API 获取全市场行情
- **量化策略引擎** — 41 维因子评分、RSRS 市场状态、ATR 风控
- **AI 审计** — DeepSeek / Gemini LLM 对候选标的进行深度分析
- **回测引擎** — OHLC 模拟交易，含滑点、涨跌停、MFE/MAE 标注
- **ML 训练** — 随机森林 + HGB 投票分类器，冠军/挑战者模型机制
- **数据治理** — MAD 异常值剔除、数据清洗、三级缓存（内存/磁盘/HTTP）

## 系统架构

| 模块 | 职责 |
|---|---|
| `main.py` | Kivy 应用入口，启动自检 |
| `config.py` | 配置管理、字体解析、Android/PC 路径检测 |
| `network.py` | HTTP 客户端（重试、抖动、UA 轮换、连接池） |
| `data.py` | 数据层 — 上游 API 调用、三级缓存 |
| `strategy.py` | 量化引擎、技术指标、因子注册表、信号检查 |
| `backtest.py` | 回测引擎 — OHLC 模拟、滑点、涨跌停 |
| `ai.py` | 模型训练、LLM 审计（DeepSeek/Gemini） |
| `governance.py` | 数据治理 — 清洗、验证、MAD 异常剔除 |
| `ui.py` | Kivy UI — 所有界面、暗色主题 |
| `utils.py` | 日志、北京时间交易时钟、崩溃转储 |

## 安装与运行

### 环境要求

- Python 3.8+
- 依赖：`kivy pandas numpy requests scikit-learn joblib urllib3`

### 安装

```bash
pip install kivy pandas numpy requests scikit-learn joblib urllib3
```

### 启动

```bash
python main.py
```

启动后会自动运行系统自检（网络、数据管道、回测引擎、AI 接口），全部通过后解锁主界面。

## 配置

首次运行会生成 `hunter_config.json`，需要填入你的 API keys：

```json
{
    "api_keys": {
        "gemini": "YOUR_GEMINI_KEY_HERE",
        "deepseek": "YOUR_DEEPSEEK_KEY_HERE"
    },
    "assets": {
        "cash": 50000.0
    }
}
```

> ⚠️ **安全提醒**：API keys 仅存储在本地 `hunter_config.json` 中，请勿提交到版本控制。

## 扫描模式

| 模式 | 说明 |
|---|---|
| **混合扫描** | 全市场扫描 + 持仓防守，双轨筛选（量化 Top5 + AI Top5） |
| **目标扫描** | 对自选股列表进行深度分析 |

## 评分体系

- **基础分** — 技术指标（RSRS、ATR、RSI、VWAP 偏离、筹码集中度等）
- **AI 分** — LLM 对候选标的的基本面、舆情、技术面综合打分
- **入场信号** — 主升浪/趋势右侧/超跌反弹等多策略匹配
- **风控** — 熊市一票否决、RSI 过热拦截、偏离度过高拦截

## 目录结构

```
AlphaHunter_tail_v905/
├── main.py              # 入口
├── config.py            # 配置
├── network.py           # 网络层
├── data.py              # 数据层
├── strategy.py          # 量化引擎
├── backtest.py          # 回测
├── ai.py                # ML 训练 & AI 审计
├── governance.py        # 数据治理
├── ui.py                # Kivy 界面
├── utils.py             # 工具函数
├── hunter_config.json   # 用户配置（含 API keys）
├── Stock_Cache/         # K 线缓存
├── Backtest_History/    # 回测输出
├── Hunter_Logs/         # 日志 & 模型文件
└── hunter_journal.csv   # 扫描历史记录
```

## 清理命令

```bash
# 清缓存
rm -rf Stock_Cache/*.pkl

# 清日志
rm -f hunter_debug.log hunter_summary.txt hunter_factors.csv hunter_journal.csv hunter_mining.log hunter_crash_dump.txt

# 清回测历史
rm -rf Backtest_History/*

# 清训练数据
rm -rf Hunter_Train_Data/*
```

## 注意事项

1. **循环导入**：`strategy.py` 和 `data.py` 互相使用延迟导入，不要在文件顶部交叉引用
2. **模型文件**：`hunter_rf_model.pkl` 和 `hunter_features.json` 必须共存才能启用 ML 推理
3. **Android 限制**：Pydroid3 下线程数限制 12，模型 `n_estimators=150`（PC 为 32 线程 / 250 棵）
4. **回测预热**：需要至少 120+ 天的 K 线数据（前 120 行用于滚动指标计算后被裁剪）
5. **代理问题**：如果系统设置了 HTTP_PROXY，网络层会自动绕过（`trust_env=False`）

## License

本项目仅供学习研究使用，不构成投资建议。股市有风险，入市需谨慎。
