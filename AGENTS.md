# AGENTS.md — AlphaHunter Tail v905

## What This Is

A Chinese A-stock (A股) quantitative trading system with Kivy GUI, ML model training, backtesting, and real-time scanning. Runs on Android (Pydroid3) and Linux desktop.

## Entry Points

- **`main.py`** — Kivy app entry: `AlphaHunterApp().run()`. Runs self-check on start, then locks UI until diagnostics pass.
- **`python main.py`** — how to launch. No virtualenv; uses system Python 3.8+ with `pip install kivy pandas numpy requests scikit-learn joblib urllib3`.

## Architecture (Module Boundaries)

| File | Responsibility |
|---|---|
| `config.py` | Config manager (`CFG` singleton), Chinese font resolver, Android/PC path detection, schema definitions |
| `network.py` | HTTP client with retry, jitter, UA rotation, connection pool (pool_maxsize=50) |
| `data.py` | DataLayer — all upstream API calls (Tencent/Eastmoney/Sina), 3-tier cache (mem/disk/HTTP), feature fetching |
| `strategy.py` | QuantEngine, TechLib (numpy indicators), FactorRegistry (schema/defaults), SignalRegistry, AIModelServer (inference) |
| `backtest.py` | BacktestEngine — OHLC simulation with slippage, limit-up/down, MFE/MAE labeling, ML data mining |
| `ai.py` | ModelTrainer (training pipeline), AIEngine (LLM audit via DeepSeek/Gemini) |
| `governance.py` | Data governance — unit adaptation, sanitization, validation, MAD outlier removal |
| `ui.py` | Kivy UI — all screens, dark theme, thread-safe config access via `UI_CFG_LOCK` |
| `utils.py` | RECORDER (logging), BeijingClock (trading day/phase), HunterShield (crash dump) |

## Key Directories

| Path | Purpose |
|---|---|
| `Stock_Cache/` | Pickled K-line cache (auto-cleaned, daily-tagged) |
| `Backtest_History/` | Backtest output CSVs + TXT reports |
| `Hunter_Train_Data/` | ML training CSVs (mined via `export_ml_training_data`) |
| `.kivy/` | Kivy config + logs (auto-generated) |
| `__pycache__/` | Python bytecode — safe to delete |

## Critical Conventions

### Config & Schema
- `CFG.CORE_FEATURE_SCHEMA` — the canonical 41-feature list. All ML training, logging, and backtest CSVs must use this exact order.
- `CFG.EXPORT_METADATA_SCHEMA` — metadata columns for trade logs.
- `CFG.STRATEGY_PARAMS` — 48 strategy tuning parameters. **Never delete keys** — `strategy_scoring` will crash.
- `FactorRegistry.FIELD_DEFAULTS` — the fallback values for every feature when data is missing. If you add a feature, add a default here.

### Data Flow
1. `NetworkClient` → `DataSource` (URL registry in `data.py`) → `DataAdapter` → `DataSanitizer` → `DataValidator`
2. **Never hardcode URLs** — always use `DataSource.get_url("KEY", **kwargs)`
3. `vol` is always in **shares (股)**, never lots (手). `DataAdapter` auto-converts Tencent's "手" to "股" (×100).

### Threading
- Android: `max_workers` capped at 12 (line 441 of `config.py`). PC uses config value (default 32).
- `DataCacheManager` uses class-level lock — max 80 entries in memory, aggressive clear on overflow.
- `UI_CFG_LOCK` protects config dict during UI interactions.
- `HunterShield` uses a single-consumer queue — never do file I/O in error handlers.

### Environment Detection
- `BASE_DIR` = `/storage/emulated/0/Hunter_Logs` on Android, script directory on PC.
- Check via `'/storage/emulated' in BASE_DIR` — gates thread limits, cache sizes, model tree counts.

## Commands

```bash
# Launch app
python main.py

# Clear cache (safe)
rm -rf Stock_Cache/*.pkl

# Clear all generated logs (safe)
rm -f hunter_debug.log hunter_summary.txt hunter_factors.csv hunter_journal.csv hunter_mining.log hunter_crash_dump.txt

# Clear backtest history
rm -rf Backtest_History/*

# Clear training data
rm -rf Hunter_Train_Data/*
```

## Gotchas

1. **Circular imports**: `strategy.py` imports from `data.py` and vice versa. They use lazy imports inside methods (`from strategy import QuantEngine`). Do not add top-level cross-module imports.
2. **Model files**: `hunter_rf_model.pkl` + `hunter_features.json` must coexist. The JSON contains `features`, `threshold`, and `mad_params` — all required for inference.
3. **API keys**: Stored in `hunter_config.json` under `api_keys.gemini` and `api_keys.deepseek`. The config file has real keys committed — treat as sensitive.
4. **Chinese font**: `CHINESE_FONT` auto-detects across Android/Win/Mac/Linux. If missing, Kivy shows tofu boxes. Test font paths in `config.py:get_chinese_font()`.
5. **Backtest warmup**: `_build_augmented_dataframe` trims first 120 rows after all rolling indicators are computed. Minimum input: ~120+ days.
6. **ML labeling**: Uses T+3 MFE/MAE (not next-day return). Labels account for slippage (5% haircut on max high).
7. **Duplicate URL keys**: `MARKET_TOTAL_AMT` is defined twice in `DataSource.URLS` (lines 64-65). Harmless but messy.
8. **No test framework**: No pytest, no unittest. Verification is done via the app's self-check (`SystemSelfCheck`) or manual backtest runs.
9. **SSL verification disabled**: `verify=False` everywhere. Network uses HTTP (not HTTPS) for most data sources.
10. **Pydroid3 constraints**: Android runs Python 3.8, limited RAM. Model training uses `n_estimators=150` (vs 250 on PC), `n_jobs=4`.

## LLM Integration

- DeepSeek: `POST https://api.deepseek.com/chat/completions` with Bearer auth
- Gemini: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}`
- Timeout: 380s for `deepseek-reasoner`, 30s for others
- AI audit output is JSON-normalized via `_clean_json()` — handles <think> tags, markdown blocks, Chinese punctuation
- `llm_score` is mapped from `sentiment_score` for the strategy engine

## Training Pipeline

1. Mine data: `BacktestEngine.export_ml_training_data()` → writes CSVs to `Hunter_Train_Data/`
2. Train: `ModelTrainer.run_training_task()` → time-series split, MAD sanitization, RF+HGB voting classifier
3. Champion/challenger: new model saved as `_challenger.pkl`, only promoted if it beats old model on OOS precision
4. `hunter_attempted_mining.txt` — blacklist of already-mined symbols (prevents duplicate work)
