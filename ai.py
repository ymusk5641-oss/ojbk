import os
import joblib
import json
import hashlib
import numpy as np
import pandas as pd
import datetime
import threading
import re
import glob

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

from utils import BASE_DIR, RECORDER
from config import CFG
from strategy import QuantEngine
from data import DataLayer,DataSource
from backtest import BacktestEngine

class ModelTrainer:
    """
    [训练核心 V8.1 - 工业级特征参数持久化版]
    升级:
    在训练时提取并固化 MAD 参数 (mad_params)，写入特征 JSON 中，供实盘调用。
    """
    def __init__(self):
        self.train_dir = os.path.join(BASE_DIR, "Hunter_Train_Data")
        self.model_file = os.path.join(BASE_DIR, "hunter_rf_model.pkl")
        self.feature_file = os.path.join(BASE_DIR, "hunter_features.json")
        self.report_file = os.path.join(BASE_DIR, "training_last_report.txt")
        self.history_file = os.path.join(BASE_DIR, "training_history.csv")
        
        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)

    def _get_safe_jobs(self):
        if '/storage/emulated' in BASE_DIR: return 4
        return -1

    def _load_and_merge_data(self, log_func):
        import glob
        try: CFG.data = CFG.load_config()
        except Exception as e: 
            from utils import HunterShield
            HunterShield.record("Trainer_Load_Config", e)
        
        validation_whitelist = set(CFG.TARGET_STOCKS) | set(CFG.HOLDINGS.keys())
        validation_whitelist = {str(x).split('.')[0].zfill(6) for x in validation_whitelist}
        
        files = glob.glob(os.path.join(self.train_dir, "*.csv"))
        if not files: return None, None
        
        log_func(f"📂 扫描到 {len(files)} 个文件 | 🔒 严苛白名单: {len(validation_whitelist)} 只")
        
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'symbol' not in df.columns or 'date' not in df.columns: continue
                file_sym = str(df.iloc[0]['symbol']).split('.')[0].zfill(6)
                
                is_explicit_holding = "holdings" in f
                is_in_whitelist = file_sym in validation_whitelist
                
                if is_explicit_holding or is_in_whitelist: df['_source'] = "holdings"
                else: df['_source'] = "market"
                dfs.append(df)
            except: pass
            
        if not dfs: return None, None
        full_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
        full_df['date'] = pd.to_datetime(full_df['date'], errors='coerce')
        full_df = full_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        full_df.fillna(0, inplace=True)
        return full_df, files

    def _evaluate_model_industrial(self, model, X, y, threshold=0.5):
        try:
            y_probs = model.predict_proba(X)[:, 1]
            y_pred = (y_probs >= threshold).astype(int)
            try: auc = roc_auc_score(y, y_probs)
            except: auc = 0.5
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            cm = confusion_matrix(y, y_pred)
            if cm.shape == (2, 2): tn, fp, fn, tp = cm.ravel()
            else: tn, fp, fn, tp = 0, 0, 0, 0
            return {
                "AUC": float(auc), "Precision": float(prec), "Recall": float(rec),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), 
                "Threshold": float(threshold)
            }
        except Exception:
            return {"AUC": 0.0, "Precision": 0.0, "Recall": 0.0, "TP": 0, "FP": 0, "TN": 0, "FN": 0, "Threshold": 0.5}

    def _find_optimal_threshold(self, model, X_val, y_val, min_precision=0.40):
        try:
            # [防御升级] 防止持仓测试集只有单一类别(全赚或全亏)导致 curve 函数崩溃
            from sklearn.metrics import precision_recall_curve
            y_probs = model.predict_proba(X_val)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
            
            best_th = 0.5; best_f05 = 0.0
            for p, r, t in zip(precisions, recalls, thresholds):
                if p < min_precision: continue
                # F0.5 Score: 偏重胜率(Precision)，兼顾召回(Recall)
                f05 = (1.25 * p * r) / (0.25 * p + r + 1e-9)
                if f05 > best_f05:
                    best_f05 = f05; best_th = t
            return float(best_th)
        except Exception:
            # 如果发生数学计算异常，退回默认的 0.5 中性阈值，保全模型安全落盘
            return 0.5

    def run_training_task(self, ui_callback):
        def log(msg): 
            if ui_callback: ui_callback(msg)

        df, files = self._load_and_merge_data(log)
        if df is None or df.empty:
            log("❌ 训练目录为空！请先点击 [挖全场]")
            return False

        from config import CFG
        from governance import DataSanitizer
        import shutil
        import numpy as np
        import pandas as pd
        
        target_col = "label_class"
        feature_cols = CFG.CORE_FEATURE_SCHEMA
        
        # --- 1. 真实时序切分 (彻底斩断未来函数) ---
   
        mask_holdings = df['_source'] == 'holdings'
        df_market = df[~mask_holdings]
        df_holdings = df[mask_holdings]
        
        if len(df_market) <= 50:
            log(f"❌ 市场训练数据严重不足 ({len(df_market)}条)！")
            return False

        if len(df_holdings) > 10:
            df_train = df_market.copy()
            df_pk = df_holdings.copy()
            pk_source_name = "持仓/自选股 (OOS)"
            log(f"\n🧱 [铁壁防御] 训练集: {len(df_train)} | 考场: {len(df_pk)}")
        else:
            log("\n⚠️ [警告] 无持仓数据！启用严格时序滚动验证 (Walk-Forward)...")
            split_idx = int(len(df_market) * 0.8)
            split_date = df_market.iloc[split_idx]['date'] 

            # 1. 训练集：严格划定在 split_date 及之前的所有股票
            df_train = df_market[df_market['date'] <= split_date].copy()

            # 2. 提取全市场真实的交易日序列 (使用 Pandas 原生操作保安全)
            unique_dates = df_market['date'].drop_duplicates().sort_values().tolist()
            
            # 3. 找到 split_date 在真实交易日序列中的索引
            try:
                date_idx = unique_dates.index(split_date)
                purge_date_idx = date_idx + 10  # 严格物理后移 10 个【交易日】
            except ValueError:
                purge_date_idx = len(unique_dates)

            # 4. 构建绝对隔离的考场测试集
            if purge_date_idx < len(unique_dates):
                purge_date = unique_dates[purge_date_idx]
                df_pk = df_market[df_market['date'] >= purge_date].copy()
                pk_source_name = "市场时序验证集(绝对隔离10个交易日)"
            else:
                # 降级防线：若尾部数据不足10天，至少保证严格推延 1 个交易日，阻断同日截面穿越
                next_date_idx = date_idx + 1
                if next_date_idx < len(unique_dates):
                    purge_date = unique_dates[next_date_idx]
                    df_pk = df_market[df_market['date'] >= purge_date].copy()
                    pk_source_name = "市场时序验证集(短线降级弱隔离)"
                else:
                    df_pk = pd.DataFrame()
            
            # 5. [终极护城河] 拦截考场为空的致命崩溃
            if df_pk.empty:
                log("\n❌ [致命错误] 尾部时间跨度不足，无法切分出考场验证集！请积累更多交易日数据。")
                return False

        # 完美解决冗余：直接提取对齐的日期序列
        train_dates = df_train['date']

        # --- 2. 隔离特征清洗 (杜绝特征泄露) ---
   
        log("🧹 启动工业级数据清洗 (仅在训练集提取MAD标尺，严防泄露)...")
        # 必须只在 df_train 上提取参数！
        extracted_mad_params = DataSanitizer.compute_mad_params(df_train, feature_cols)
        
        # 用训练集的标尺，分别清洗训练集和考场
        df_train_cleaned = DataSanitizer.clean_machine_learning_features(df_train, feature_cols, mad_params=extracted_mad_params)
        df_pk_cleaned = DataSanitizer.clean_machine_learning_features(df_pk, feature_cols, mad_params=extracted_mad_params)

        X_train = df_train_cleaned[feature_cols]
        y_train = df_train_cleaned[target_col].astype(int)
        X_pk = df_pk_cleaned[feature_cols]
        y_pk = df_pk_cleaned[target_col].astype(int)
            
        log("\n🥊 新模型正在修炼 (72W数据量级优化版)...")
        safe_jobs = self._get_safe_jobs() 

        # --- 3. 构建基于真实物理时间的指数衰减权重 ---
 
        max_date = train_dates.max()
        days_diff = (max_date - train_dates).dt.days
        
        # 半衰期 250 个交易日：约一年前的数据权重衰减至一半，最旧兜底 0.3
        half_life = 250
        decay_weights = 0.3 + 0.7 * np.exp(-days_diff / half_life)
        decay_weights = decay_weights.values # 强制转为 Numpy 数组防索引崩溃

        # --- 4. 动态自适应类别平衡 ---
        pos_mask = (y_train == 1).values  
        neg_mask = (y_train == 0).values
        
        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        
        if pos_count > 0:
            imbalance_ratio = neg_count / pos_count
            dynamic_multiplier = imbalance_ratio * 0.9 
            decay_weights[pos_mask] *= dynamic_multiplier
            log(f"⚖️ 触发动态失衡补偿: 自动将正样本权重放大 {dynamic_multiplier:.2f} 倍")
        
        # --- 获取平台安全配置 ---
        is_mobile = '/storage/emulated' in BASE_DIR
        rf_trees = 150 if is_mobile else 250  # 手机端内存有限，限制树的数量；PC端全开
        
        # 1. 随机森林 (RF) - 强制挖掘冷门因子 (100万数据量级优化版)
        rf = RandomForestClassifier(
            n_estimators=rf_trees,   # 动态树量：兼顾群体稳定性与设备内存
            max_depth=15,            # 🚀 [核心调整] 深度放宽到 15，释放 100 万数据的非线性拟合潜力！(原10太浅导致欠拟合)
            min_samples_leaf=30,     # 底层防噪音
            max_features=0.3,        # 每次只看 30% 特征，逼迫模型学习弱 Alpha
            n_jobs=safe_jobs, 
            random_state=42
        )
        
        # 2. 历史梯度提升树 (HGB) - 慢速精细雕刻，极重正则化 (无需修改，完美契合100W数据)
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.01,      
            max_iter=1200,           
            max_depth=8,             
            min_samples_leaf=100,     
            early_stopping=False,    
            l2_regularization=3.0,   
            random_state=42
        )
        
        # 3. 终极融合 - 不等权投票
        challenger = VotingClassifier(
            estimators=[('rf', rf), ('hgb', hgb)], 
            voting='soft',
            weights=[0.35, 0.65]     # 35% RF 防守，65% HGB 进攻
        )

        # --- 4. 模型拟合 (全量数据极速版) ---
        log("⏳ 正在注入物理时间权重进行高维拟合 (100% 最新数据全量喂入)...")
        # 🚀 彻底抛弃校准集剥离，将 100% 的数据（包含最宝贵的近期数据）全部喂给底层模型！
        # 配合下方的 F0.5 动态阈值搜寻，无需依赖耗时且容易引发时序倒置的 Isotonic 校准。
        challenger.fit(X_train, y_train, sample_weight=decay_weights)
        final_model = challenger


         # === 5. 无条件提取并记录本次训练的 Top 因子 ===
        top_features_str = "未提取"
        try:
            from sklearn.inspection import permutation_importance
            
            # [性能优化] 适度扩大样本量以降低方差。如果是在 PC 端，可以全量(不切片)
            sample_size = min(5000, len(X_pk)) # 从 3000 提至 5000
            if sample_size > 0:
                np.random.seed(42)
                idx = np.random.choice(len(X_pk), sample_size, replace=False)
                X_sample = X_pk.iloc[idx]
                y_sample = y_pk.iloc[idx]
                
                log("🔍 正在通过置换算法 (Permutation) 测算核心特征权重...")
                
                # =======================================================
                # 🚀 [核心修复] 强制使用 roc_auc 评估，对冲样本极度不平衡与阈值错位
                # =======================================================
                result = permutation_importance(
                    final_model, X_sample, y_sample, 
                    scoring='roc_auc',  # 👈 必须指定！评价模型排序能力，无视 0.5 默认阈值
                    n_repeats=5,        # 👈 从 3 提升到 5，降低随机置换的方差
                    random_state=42, 
                    n_jobs=safe_jobs
                )
                
                # 组装并排序
                imps = pd.DataFrame({'Feature': feature_cols, 'Imp': result.importances_mean})
                # 过滤掉得分为负数的噪音特征 (打乱后反而变好的特征绝对是噪音)
                imps = imps[imps['Imp'] > 0]
                
                top5 = imps.sort_values(by='Imp', ascending=False).head(5)['Feature'].tolist()
                top_features_str = ' | '.join(top5) if top5 else "未发现显著驱动因子"
                
                log(f"🔑 本次训练核心驱动因子: {top_features_str}")
            else:
                top_features_str = "考场样本不足，跳过提取"

                
        except Exception as e: 
            from utils import HunterShield
            try: HunterShield.record("Top_Feature_Fail", e)
            except: pass
            log(f"⚠️ 特征提取降级: {str(e)[:50]}")


        # =====================================================================
        # [核心漏洞修复] 阈值(Threshold)必须在训练集上寻找，严禁偷窥考场数据 (X_pk)!
        # =====================================================================
        # 1. 在训练集上寻找最佳阈值。为了对冲树模型在训练集上的过拟合(概率偏高)，
        # 我们将寻找及格线时的底线胜率要求 (min_precision) 强行拉高至 0.60。
        best_th = self._find_optimal_threshold(final_model, X_train, y_train, min_precision=0.70)
        
        # 2. 拿着在训练集定下的死规矩(best_th)，去盲测完全未知的考场数据 (X_pk)
        new_metrics = self._evaluate_model_industrial(final_model, X_pk, y_pk, best_th)


        champion = None; old_metrics = None
        if os.path.exists(self.model_file):
            try:
                champion = joblib.load(self.model_file)
                with open(self.feature_file, 'r', encoding='utf-8') as f:
                    old_schema = json.load(f)
                    old_th = old_schema.get("threshold", 0.5)
                    # 1. 提取旧模型当年训练时固化的特征尺子
                    old_mad_params = old_schema.get("mad_params", {})
                
                # 2. [核心修复] 为旧模型专门开辟一个考场
                if old_mad_params:
                    # 使用旧尺子重新清洗原始考场数据 df_pk
                    df_pk_old_cleaned = DataSanitizer.clean_machine_learning_features(
                        df_pk, feature_cols, mad_params=old_mad_params
                    )
                    X_pk_old = df_pk_old_cleaned[feature_cols]
                else:
                    # 兜底：兼容极早期没有保存尺子的旧模型文件
                    X_pk_old = X_pk

                # 3. 让旧模型在自己的尺子体系下进行公平决斗
                old_metrics = self._evaluate_model_industrial(champion, X_pk_old, y_pk, old_th)
            except Exception as e:
                # 显影剂：防止旧模型加载失败被静默掩盖
                log(f"⚠️ [警告] 旧模型评估异常，本次将强制作为首次部署: {e}")
                pass

        log(f"\n⚔️ [决斗] 考题: {pk_source_name}")
        deploy = False; reason = ""
        if old_metrics is None or old_metrics.get('AUC', 0) == 0:
            deploy = True; reason = "首次部署"
            log(f"   🏁 新模型: Prec={new_metrics['Precision']:.1%} | AUC={new_metrics['AUC']:.3f}")
        else:
            delta_prec = new_metrics['Precision'] - old_metrics['Precision']
            log(f"   🛡️ 旧模型: Prec={old_metrics['Precision']:.1%} | AUC={old_metrics['AUC']:.3f}")
            log(f"   🗡️ 新模型: Prec={new_metrics['Precision']:.1%} | AUC={new_metrics['AUC']:.3f}")
            if new_metrics['Precision'] < 0.20: deploy = False; reason = "胜率过低 (<20%)"
            # 🚀 刺客策略胜率优先！要求提升 > 2.0% 且 AUC 提升 > 1.0% 才部署
            elif delta_prec > 0.020 and (new_metrics['AUC'] - old_metrics['AUC']) > 0.010:
                deploy = True; reason = f"胜率显著提升 (+{delta_prec:.1%}), AUC提升 (+{(new_metrics['AUC'] - old_metrics['AUC']):.3f})"
            elif delta_prec > -0.010 and (new_metrics['AUC'] - old_metrics['AUC']) > 0.015: 
                deploy = True; reason = "胜率持平，泛化能力更优"
            else: deploy = False; reason = "未击败旧模型"

        # === 核心重构：无论是否部署，强制保留本次训练资产 ===
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
                
        schema = {
            "features": feature_cols,
            "threshold": float(best_th),
            "mad_params": extracted_mad_params, 
            "version": datetime.datetime.now().strftime("%Y%m%d_%H%M"),
            "metrics": new_metrics,
            "pk_source": pk_source_name
        }

        # [新增机制] 无条件保存“挑战者”副本
        challenger_model_file = self.model_file.replace(".pkl", "_challenger.pkl")
        challenger_feat_file = self.feature_file.replace(".json", "_challenger.json")

        joblib.dump(final_model, challenger_model_file)
        with open(challenger_feat_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
            f.flush(); os.fsync(f.fileno())

        if deploy:
            log(f"\n✅ [DEPLOY] 部署成功! ({reason})")
            shutil.copy2(challenger_model_file, self.model_file)
            shutil.copy2(challenger_feat_file, self.feature_file)
        else:
            log(f"\n🛑 [REJECT] 部署拒绝! ({reason})")
            log(f"💾 (模型资产已无条件备份为: _challenger 版本)")
            
        self._write_report(new_metrics, old_metrics, deploy, reason, pk_source_name)
        
        # === [新增] 将因子持久化追加保存到战报与底座日志中 ===
        try:
            with open(self.report_file, "a", encoding="utf-8") as f:
                f.write(f"\n🔑 [模型核心驱动因子]\n   👉 {top_features_str}\n")
                
            from utils import RECORDER
            RECORDER.log_info("AI_TRAIN", f"新模型训练完毕 | AUC={new_metrics['AUC']:.3f} | 核心因子: {top_features_str}")
        except: pass

        return True

    def _write_report(self, new, old, deployed, reason, source_name):
        import csv
        try:
            with open(self.report_file, "w", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== Hunter AI 演进报告 ({ts}) ===\n")
                f.write(f"决策: {'✅ 部署' if deployed else '🛑 拒绝'} | 原因: {reason}\n")
                f.write(f"来源: {source_name}\n\n")
                f.write(f"🎯 [新模型]\n")
                f.write(f"   Threshold: {new['Threshold']:.2f}\n")
                f.write(f"   Precision: {new['Precision']:.2%}\n")
                f.write(f"   AUC      : {new['AUC']:.3f}\n")
                f.write(f"   TP:{new['TP']} | FP:{new['FP']} | FN:{new['FN']}\n\n")
                if old and old.get('AUC', 0) > 0:
                    f.write(f"🛡️ [旧模型]\n")
                    f.write(f"   Precision: {old['Precision']:.2%}\n")
                    f.write(f"   AUC      : {old['AUC']:.3f}\n")
        except: pass

        try:
            file_exists = os.path.isfile(self.history_file)
            with open(self.history_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Source', 'Deployed', 'Reason', 'New_Prec', 'New_AUC', 'Old_Prec', 'Old_AUC'])
                writer.writerow([
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    source_name, int(deployed), reason,
                    round(new['Precision'], 4), round(new['AUC'], 4),
                    round(old['Precision'], 4) if old else 0.0, 
                    round(old['AUC'], 4) if old else 0.0
                ])
        except Exception: 
            pass 



class AIEngine:
    """
    [AI 智能核心 - 风控官版]
    升级 V402.16: 
    1. 角色升级为“首席风控官”，专注于识别量价背离和诱多。
    2. 强制审计 pv_corr (量价相关性) 和 volatility (波动率)。
    3. 对低波动下的突发巨量进行标记 (SUSPICIOUS)。
    """
    def __init__(self, network_client):
        self.net = network_client 
        self.cache = {}
        self.lock = threading.Lock()
        # 审计时间戳
        self.last_audit_time = 0
        # [新增] 首单日志锁：确保只记录开机后的第一单，不刷屏
        self.first_log_done = False

    # [新增辅助方法] 静默保存第一单日志
    def _save_first_interaction(self, model, prompt, response):
        try:
            log_path = os.path.join(BASE_DIR, "AI_First_Run_Debug.txt")
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== AI First Interaction Analysis [{timestamp}] ===\n")
                f.write(f"Model Used: {model}\n")
                f.write(f"Log File: {log_path}\n\n")
                
                f.write("="*20 + " [INPUT PROMPT] " + "="*20 + "\n")
                f.write(prompt)
                f.write("\n\n")
                
                f.write("="*20 + " [OUTPUT RESPONSE] " + "="*20 + "\n")
                f.write(response)
                f.write("\n" + "="*60 + "\n")
                
            # print(f"✅ [DEBUG] 第一单AI日志已保存至: {log_path}") # 仅控制台可见，不干扰UI
        except Exception as e:
            pass

    
    def _log_trace(self, symbol_list, model, raw_response):
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
            think = match.group(1).strip() if match else "无思维链"
            RECORDER.log_trace(f"[{timestamp}] {model}\n[Think] {think[:500]}...\n{'-'*20}")
        except: pass


    def _dehydrate_news(self, raw_news):
        """
        [模块 1：新闻纯物理脱水]
        只做八股文降噪和压缩，绝不越权进行语义定性，防误杀。
        """
        import re
        if not raw_news or str(raw_news).strip().lower() in ["", "nan", "none"]:
            return "无近期公告"
            
        news_text = str(raw_news)
        
        # 1. 暴力切除 A 股标准公告的无营养表头
        clean_text = re.sub(r'(证券代码|证券简称|公告编号)[:：]\s*[a-zA-Z0-9-]+\s*', '', news_text)
        
        # 2. 切除免责声明等套话
        clean_text = re.sub(r'本公司及董事会全体成员保证信息披露的?内容.*?[，。]', '', clean_text)
        clean_text = re.sub(r'不存在任何虚假记载、误导性陈述或者重大遗漏.*?[，。]', '', clean_text)
        
        # 3. 压缩空白符
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            return "无近期公告"
            
        # 4. 截取前 150 个字符 (公告核心基本都在标题和第一句，防超出 Token 注意力)
        return clean_text[:150]



    def _get_system_prompt(self):
        """
        [模块 3：系统协议 V16.0 - 零幻觉思维链与资产隔离版]
        """
        return (
            "你是一名服务于顶尖量化对冲基金的首席 AI 风控官。\n"
            "你将接收一组资产的特征数据。请抛弃任何主观臆断，严格执行以下【动态决策树】。\n\n"
            
            "## 【终极风控决策树】(按顺序执行，触发即终止)\n\n"
            
            "🔴 节点一：RAG 事实防伪审查 (一票否决)\n"
            "  - [雷区]：若【脱水公告】明确包含“立案/留置/减持/解禁/非标/计提减值”等实质性抽血或合规黑天鹅，无视任何技术面指标，直接 REJECT。\n"
            "  - [诱多陷阱]：若公告为“框架协议/互动易回复”等无约束力利好，且特征层呈现【异常爆量】或【量价背离(派发高危)】，判定为资金借利好出货，直接 REJECT。\n\n"
            
            "🔴 节点二：资产类型隔离校验 (ETF vs 个股)\n"
            "  - [若资产为 ETF]：豁免所有关于“获利盘比例”、“避雷针上影线”的苛刻限制（指数极难被单日操纵）。只看【底层AI裁决】与【大盘宏观状态】。\n"
            "  - [若资产为 个股]：必须严查高位拥挤。若 获利盘>85% 且 Bias20被定性为【极度危险(严重透支)】 且 呈现【异常爆量】，判定为见顶崩塌，直接 REJECT。\n\n"
            
            "🔴 节点三：量价底线斩杀法则\n"
            "  - 若【底层AI裁决】包含“坚决规避”概念（如 AI胜率 < 40%），直接 REJECT。\n"
            "  - 若【宏观状态】处于“极度冰点(防守)”期，除非个股处于绝对的【量价齐升】且【聪明钱】高位沉淀，否则一律从严判定为 WATCH 或 REJECT。\n\n"
            
            "🟢 节点四：真理 Alpha 豁免权 (防错杀机制)\n"
            "  - 若个股未触发上述死刑，且【RSRS微观趋势强度】> 1.1，【聪明钱排行】> 70，且量能【温和】未失控。\n"
            "  - 此时允许豁免部分估值过高或高获利盘的警告，判定为主力强控盘主升浪，果断给予 PASS。\n\n"
            
            "## 【JSON 格式与 CoT 强约束协议】\n"
            "你必须输出包含以下 7 个字段的 JSON 数组。禁止输出其他格式：\n"
            "1. symbol: 6位代码字符串。\n"
            "2. analysis: [思维链约束] 必须严格按照以下格式在一行内输出你的推理过程：\n"
            "   '[公告定性]:xxx; [宏观/资产适配]:xxx; [量价博弈推演]:xxx; [结论]:xxx'\n"
            "3. sentiment_score: 0-100整数 (90=坚决看多，50=中性分歧，10=崩塌派发，只能输出数字)。\n"
            "4. conviction: 0.0-1.0浮点数 (你的逻辑闭环确信度，只能输出数字)。\n"
            "5. risk_factor: 0.0-1.0浮点数 (1.0=安全垫极厚，0.0=即将核按钮，只能输出数字)。\n"
            "6. action: 仅限 ['PASS', 'WATCH', 'REJECT']，只能选一个。\n"
            "7. reason: 15字以内极简归因总结。\n"
        )



    def generate_user_prompt(self, stock_list, regime, phase, hot_sectors=None):
        """
        [模块 2：提示词生成器 V16.0 - 注意力分层与类型隔离版]
        """
        import re
        env_score = "极度冰点(防守)" if regime < 0.3 else ("狂热拥挤(防风险)" if regime > 0.7 else "结构性震荡")
        hot_str = ",".join(hot_sectors) if hot_sectors else "无主线"
        
        base_intro = (
            f"## 【全局环境基准】\n"
            f"> 宏观状态: {env_score} (Regime={regime:.2f})\n"
            f"> 核心题材: [{hot_str}]\n\n"
        )
        
        prompt_body = ""
        for s in stock_list:
            clean_sym = re.sub(r'\D', '', str(s.get('symbol', '000000'))).zfill(6)
            name = str(s.get('name', '')).strip()
            if not name or name == 'nan': name = f"未知标的"
            
            # [Quant 修正] 绝对隔离 ETF 与 个股
            is_etf = clean_sym.startswith(('159', '510', '511', '512', '513', '515', '516', '517', '56', '58')) \
                     or 'ETF' in name.upper() or 'LOF' in name.upper()
            asset_type = "ETF宽基/行业指数" if is_etf else "A股个股"

            raw_rag = s.get('rag_info', '无近期公告')
            clean_rag = self._dehydrate_news(raw_rag)

            # --- 核心语义映射 ---
            pv_corr = float(s.get('pv_corr', 0))
            pv_semantic = "量价齐升(良性)" if pv_corr > 0.4 else ("量价背离(派发高危)" if pv_corr < -0.35 else "量价平稳")
            
            ai_val = float(s.get('ai_score', 50))
            ai_tag = "坚决规避(底座极度看空)" if ai_val < 40 else ("主升起爆" if ai_val > 65 else "平庸观望")

            bias_val = float(s.get('bias_20', 0))
            bias_tag = "极度危险(严重透支)" if bias_val > 15 else ("健康" if bias_val < 5 else "高位偏离")
            
            vol_z = float(s.get('vol_zscore', 0))
            vol_z_tag = "异常爆量(警惕见顶)" if vol_z > 2.0 else ("极度缩量(流动性枯竭)" if vol_z < -1.5 else "温和")

            trend_tag = str(s.get('trend_tag', s.get('trend_desc', '未定义')))

            # [AI 修正] 拆分核心决策区与辅助参考区，引导 LLM 注意力聚焦
            prompt_body += (
                f"---\n"
                f"### 标的: {clean_sym} ({name}) | 资产类型: [{asset_type}]\n"
                f"**【一、 致命排雷层 (决定生死)】**\n"
                f"  - 脱水公告: {clean_rag}\n"
                f"  - 底层AI裁决: 胜率 {ai_val:.1f}% -> [{ai_tag}]\n"
                f"  - 核心量价: 诊断=[{trend_tag}] | 量价关系=[{pv_semantic}] | 爆量Z-Score={vol_z:.2f} [{vol_z_tag}]\n"
                f"  - 泡沫与拥挤: Bias20={bias_val:.2f}% [{bias_tag}] | 获利盘={s.get('winner_rate',50):.1f}% | 避雷针上影线={s.get('upper_shadow_ratio',0):.2%}\n"
                
                f"**【二、 资金与动量层 (判定主升浪)】**\n"
                f"  - 聪明钱排行: {s.get('smart_money_rank',50):.1f}/100 | RSRS微观趋势强度: {s.get('rsrs_wls',1.0):.3f}\n"
                f"  - Amihud冲击成本: {s.get('amihud',0):.5f} | 资金流斜率: {s.get('obv_slope',0):.3f}\n"
                
                f"**【三、 辅助参考层 (仅供辅助，勿做主导)】**\n"
                f"  - 均线/形态: MA5={s.get('ma5',0):.2f}, MA20={s.get('ma20',0):.2f}, MACD={s.get('macd',0):.3f}\n"
                f"  - 震荡/反转: CHOP={s.get('chop',50):.1f}, RSI_Rank={s.get('rsi_rank',50):.0f}, KDJ_J={s.get('kdj_j',50):.1f}\n"
            )

        footer = (
            "\n## 执行指令\n"
            "接收上述信息，严格遵循系统提供的《终极风控决策树》，输出标准 JSON 数组（可使用 markdown 代码块包裹）。\n"
        )
        return base_intro + prompt_body + footer





    def _normalize_item(self, item):
        """
        [数据清洗 V2.0 - 强类型熔断版]
        功能: 将 AI 返回的杂乱数据强制转换为系统可读的标准格式。
        核心修复:
        1. [数值强转] 自动剥离百分号 (e.g., "85%" -> 85) 并限制范围。
        2. [语义映射] 兼容 Buy/Sell 等非标词汇，映射回 PASS/REJECT。
        3. [默认兜底] 任何解析失败均回退到中性值，保证程序不崩。
        """
        import re # 局部引入，确保正则可用
        
        # 1. 准备默认骨架 (中性态度)
        clean = {
            "sentiment_score": 50,
            "conviction": 0.5,
            "risk_factor": 0.5,
            "action": "WATCH",
            "reason": "AI未提供详细理由"
        }
        
        if not isinstance(item, dict): return clean

        # ---------------------------------------------------
        # 2. 提取 Sentiment Score (0-100 int)
        # ---------------------------------------------------
        try:
            raw = str(item.get('sentiment_score', 50)).strip()
            # 提取数字 (支持 "Score: 85", "85%" 等格式)
            nums = re.findall(r'-?\d+', raw)
            if nums:
                val = int(nums[0])
                # 钳位限制在 0-100
                clean['sentiment_score'] = max(0, min(100, val))
        except: 
            pass # 保持默认 50

        # ---------------------------------------------------
        # 3. 提取 Conviction (0.0-1.0 float)
        # ---------------------------------------------------
        try:
            raw = str(item.get('conviction', 0.5)).strip()
            # 优先处理百分数 (e.g., "80%" -> 0.8)
            if '%' in raw:
                nums = re.findall(r'\d+', raw)
                if nums: 
                    clean['conviction'] = float(nums[0]) / 100.0
            else:
                # 处理纯小数 (e.g., "0.8")
                nums = re.findall(r'\d+\.?\d*', raw)
                if nums: 
                    clean['conviction'] = float(nums[0])
            
            # 钳位限制
            clean['conviction'] = max(0.0, min(1.0, clean['conviction']))
        except: 
            pass

        # ---------------------------------------------------
        # 4. 提取 Risk Factor (0.0-1.0 float)
        # 定义: 1.0 = 安全, 0.0 = 极度危险
        # ---------------------------------------------------
        try:
            raw = str(item.get('risk_factor', 0.5)).strip()
            
            # 语义补全: 处理 "High Risk" 这种文字描述
            raw_upper = raw.upper()
            if 'HIGH' in raw_upper or '高' in raw_upper: 
                clean['risk_factor'] = 0.2 # 高危
            elif 'LOW' in raw_upper or '低' in raw_upper: 
                clean['risk_factor'] = 0.9 # 安全
            else:
                # 正常数值提取
                nums = re.findall(r'\d+\.?\d*', raw)
                if nums: 
                    val = float(nums[0])
                    # 容错: 如果 AI 输出了 >1 且 <=100 的整数，视为百分制
                    if 1.0 < val <= 100.0: 
                        val /= 100.0
                    clean['risk_factor'] = max(0.0, min(1.0, val))
        except: 
            pass

        # ---------------------------------------------------
        # 5. 清洗 Action (枚举映射)
        # ---------------------------------------------------
        raw_act = str(item.get('action', 'WATCH')).upper().strip()
        
        # 建立模糊映射表 (处理 AI 的同义词)
        map_rules = {
            'PASS': ['PASS', 'BUY', 'LONG', 'ADD', 'STRONG', '买入', '做多', '推荐'],
            'REJECT': ['REJECT', 'SELL', 'SHORT', 'AVOID', 'WEAK', '卖出', '规避', '拒绝'],
            'WATCH': ['WATCH', 'HOLD', 'OBSERVE', 'WAIT', 'NEUTRAL', '观望', '持有']
        }
        
        found = False
        for std_key, keywords in map_rules.items():
            for kw in keywords:
                # 全字匹配或包含匹配
                if kw == raw_act or (len(raw_act) > 3 and kw in raw_act):
                    clean['action'] = std_key
                    found = True
                    break
            if found: break
        
        # 如果啥都没匹配到，默认 WATCH，防止空值
        if not found: clean['action'] = 'WATCH'
        
        # ---------------------------------------------------
        # 6. 提取 Reason (防止过长)
        # ---------------------------------------------------
        if 'reason' in item:
            reason_txt = str(item['reason']).replace('\n', ' ').strip()
            reason_txt = re.sub(r'^(Reason|理由)[:：]\s*', '', reason_txt, flags=re.IGNORECASE)
            clean['reason'] = reason_txt[:120] 
            
        # 👇 =======================================================
        # [神经桥接修复] 将 AI 吐出的 sentiment_score 映射给底层引擎认识的 llm_score
        # 只有这样，strategy.py 里的 (llm_score < 40) 斩杀线才能真正生效！
        # =======================================================
        clean['llm_score'] = clean['sentiment_score']
            
        return clean
        

    def _clean_json(self, text):
        """
        [架构统一 V1.4 - 工业级清洗版]
        升级记录:
        1. [DeepSeek适配] 强制正则去除 <think>...</think> 标签 (DOTALL模式)。
        2. [暴力提取] 增强型 JSON 提取算法，优先匹配最外层结构，兼容前后废话。
        3. [兼容性] 支持 json/ast/fix 三级解析降级。
        4. [归一化] 统一将 List/Dict/Wrapper 结构转为 {code: data} 标准字典。
        """
        if not text: return None
        
        # 局部引入，防止文件头缺失导致 NameError
        import re
        import json
        import ast
        
        json_str = ""
        data = None
        
        try:
            # ================= 阶段 1: 字符串深度清洗 =================
            # 1. 移除思维链 (DeepSeek R1 核心修复)
            # 必须使用 DOTALL (匹配换行) 和 IGNORECASE
            clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
            
            # 2. 移除 Markdown 标记
            clean_text = re.sub(r'```json|```', '', clean_text, flags=re.IGNORECASE).strip()
            
            # 3. 智能截取 (寻找最外层的 [] 或 {})
            # 逻辑: 找到第一个 [ 或 {，以及最后一个 ] 或 }
            # 优先寻找 [, 因为 Prompt 明确要求返回 List
            idx_list_start = clean_text.find('[')
            idx_obj_start = clean_text.find('{')
            
            start_idx = -1
            end_char = ''
            
            # 判定优先级: 谁在前面取谁，但如果都有，优先取 [ (符合 Schema)
            # 如果两个都存在:
            if idx_list_start != -1 and idx_obj_start != -1:
                if idx_list_start < idx_obj_start:
                    start_idx = idx_list_start; end_char = ']'
                else:
                    start_idx = idx_obj_start; end_char = '}'
            # 只有 [
            elif idx_list_start != -1:
                start_idx = idx_list_start; end_char = ']'
            # 只有 {
            elif idx_obj_start != -1:
                start_idx = idx_obj_start; end_char = '}'
            
            # 截取有效片段
            if start_idx != -1:
                end_idx = clean_text.rfind(end_char)
                if end_idx != -1 and end_idx > start_idx:
                    json_str = clean_text[start_idx : end_idx+1]
            
            # 兜底: 如果截取失败，尝试使用原文本 (防止只有单行 JSON 无废话的情况被误判)
            if not json_str: 
                json_str = clean_text

            # ================= 阶段 2: 反序列化 (三级火箭) =================
            parsed_success = False
            
            # 方案 A: 标准 JSON
            if not parsed_success:
                try:
                    data = json.loads(json_str)
                    parsed_success = True
                except: pass
            
            # 方案 B: Python AST (处理单引号/None/True/False)
            if not parsed_success:
                try:
                    data = ast.literal_eval(json_str)
                    parsed_success = True
                except: pass
                
            # 方案 C: 字符暴力修复 (处理中文符号)
            if not parsed_success:
                try:
                    fixed = json_str.replace("'", '"').replace("，", ",").replace("：", ":")
                    data = json.loads(fixed)
                    parsed_success = True
                except: pass

            if not parsed_success or data is None: 
                # 解析失败，记录日志并返回
                try: RECORDER.log_debug("JSON_FAIL", f"无法解析: {json_str[:50]}...")
                except: pass
                return None

            # ================= 阶段 3: 结构归一化 (Schema Alignment) =================
            # 目标: 将任意结构统一转为 { "600519": { ...data... } }
            normalized = {}
            
            # 辅助函数: 提取 6 位数字代码
            def _extract_code(input_val):
                # [修复] 先强制 zfill，再用正则找 6 位数字，专治大模型发癫
                s = str(input_val).strip().zfill(6)
                match = re.search(r'\d{6}', s)
                return match.group(0) if match else "000000"

            # 情况 A: 列表结构 (标准 Schema) -> [ {...}, {...} ]
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict): continue
                    
                    # 1. 尝试从 Key 中找代码
                    code = "000000"
                    # 优先检查标准字段
                    for key in ['symbol', 'code', 'stock_code', '股票代码', 'f12']:
                        if key in item:
                            code = _extract_code(item[key])
                            if code != "000000": break
                    
                    # 2. 如果没找到，尝试在 Value 中遍历寻找 6 位数字
                    if code == "000000":
                        for v in item.values():
                            code = _extract_code(v)
                            if code != "000000": break
                            
                    normalized[code] = self._normalize_item(item)

            # 情况 B: 字典结构 -> 可能包含包裹 Key，或直接是 {code: data}
            elif isinstance(data, dict):
                # 策略: 遍历所有 Value，如果是 List 则认为是包裹结构，否则认为是直接映射
                is_wrapper = False
                
                # 检查是否是 Wrapper (例如 {"audit_result": [...]})
                for k, v in data.items():
                    if isinstance(v, list):
                        is_wrapper = True
                        # 递归处理列表
                        for sub_item in v:
                            if isinstance(sub_item, dict):
                                # 这里的逻辑与 情况 A 相同
                                code = "000000"
                                for sub_k in ['symbol', 'code', 'stock_code']:
                                    if sub_k in sub_item:
                                        code = _extract_code(sub_item[sub_k])
                                        if code != "000000": break
                                if code == "000000": # 暴力找
                                    for sub_val in sub_item.values():
                                        code = _extract_code(sub_val)
                                        if code != "000000": break
                                        
                                normalized[code] = self._normalize_item(sub_item)
                
                # 如果不是 Wrapper，则是直接映射 { "600519": {...} }
                if not is_wrapper:
                    for k, v in data.items():
                        code = _extract_code(k) # 从 Key 提取代码
                        
                        if isinstance(v, dict):
                            normalized[code] = self._normalize_item(v)
                        else:
                            # 极简模式兼容 { "600519": "PASS" }
                            normalized[code] = {
                                "action": "WATCH", 
                                "reason": str(v),
                                "sentiment_score": 60 # 默认分
                            }

            return normalized

        except Exception as e:
            # 最后的异常兜底
            try: RECORDER.log_debug("JSON_CRASH", f"{str(e)} | Text: {text[:50]}")
            except: pass
            return None

    def _call_llm(self, prompt, model_name, status_callback=None):
        api_key = CFG.DEEPSEEK_KEY if "deepseek" in model_name else CFG.GEMINI_KEY
        if not api_key: return None
        timeout = 380 if "reasoner" in model_name else 30
        
        final_content = None # 用于存储最终结果以便记录日志
        
        try:
            headers = {"Content-Type": "application/json"}
            
            # [Refactor] 使用 DataSource 获取 DeepSeek URL
            if "deepseek" in model_name:
                url = DataSource.get_url("LLM_DEEPSEEK")
                headers["Authorization"] = f"Bearer {api_key}"
                
                # 适度提高温度，让AI去识别模糊的模式
                payload = {
                    "model": model_name, 
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()}, 
                        {"role": "user", "content": prompt}
                    ], 
                    "temperature": 0.4, 
                    "stream": False
                } 
                resp = self.net.post(url, payload, headers, timeout=timeout)
                if resp and resp.status_code == 200:
                    data = resp.json()
                    message = data['choices'][0]['message']
                    content = message.get('content', '')
                    reasoning = message.get('reasoning_content', '')
                    if reasoning: 
                        final_content = f"<think>{reasoning}</think>\n{content}"
                    else:
                        final_content = content
                        
            else: # Gemini
                # [Refactor] 使用 DataSource 获取 Gemini URL (自动填充 model 和 key)
                url = DataSource.get_url("LLM_GEMINI", model=model_name, key=api_key)
                
                payload = {"contents": [{"parts": [{"text": self._get_system_prompt() + "\n" + prompt}]}]}
                resp = self.net.post(url, payload, headers, timeout=timeout)
                if resp and resp.status_code == 200:
                    resp_json = resp.json()
                    if ('candidates' in resp_json and len(resp_json['candidates']) > 0 and
                        'content' in resp_json['candidates'][0] and
                        'parts' in resp_json['candidates'][0]['content'] and
                        len(resp_json['candidates'][0]['content']['parts']) > 0):
                        final_content = resp_json['candidates'][0]['content']['parts'][0]['text']
                    else:
                        final_content = None

            # ================= [新增] 首单日志拦截 (保持原逻辑不丢失) =================
            # 只有当成功获取到内容，且是第一次运行时，才写入文件
            if final_content and not self.first_log_done:
                with self.lock:
                    if not self.first_log_done: # 双重检查锁定(DCL)
                        self._save_first_interaction(model_name, prompt, final_content)
                        self.first_log_done = True 

            return final_content

        except Exception as e: 
            from utils import HunterShield
            HunterShield.record(f"LLM_Call_Fail | {model_name}", e)
        return None


    def audit(self, candidate_data_list, regime_score, phase="MID", status_callback=None, hot_sectors=None):
        """
        [核心审计 - RAG 管道彻底打通版 V7.9]
        架构师修正:
        废除硬编码的简陋 Prompt，全面接入 generate_user_prompt，
        确保【最新公告】和【41维技术因子】能 100% 喂给大模型。
        """
        if not candidate_data_list: return {}, "None"
        if getattr(CFG, 'ENABLE_AI', True) is False: return {}, "AI_Disabled"

        valid_candidates = []
        skipped_results = {}
        
        for row in candidate_data_list:
            # [修复] 强制补齐 6 位字符串
            symbol = str(row.get('symbol', '000000')).strip().zfill(6)
            try:
                if abs(float(row.get('flow', 0))) < 1000:
                    skipped_results[symbol] = {
                        "action": "WATCH", "sentiment_score": 40, "reason": "[系统风控] 资金数据缺失", "risk_factor": 1.0
                    }; continue
                if float(row.get('volatility', 0)) <= 0.001:
                    skipped_results[symbol] = {
                        "action": "REJECT", "sentiment_score": 0, "reason": "[系统风控] 停牌/无波动", "risk_factor": 0.0
                    }; continue
            except: pass
            valid_candidates.append(row)

        if not valid_candidates: return {"audit": skipped_results}, "Skipped All"

        hot_list = hot_sectors if hot_sectors else getattr(CFG, 'HOT_SECTORS', [])

        try:
            r_val = float(regime_score)
        except: r_val = 0.5

        # ========================================================
        # [彻底接通 RAG 管道] 
        # 删除原有的硬编码拼接，直接调用已挂载公告的生成器
        # ========================================================
        user_prompt_body = self.generate_user_prompt(valid_candidates, r_val, phase, hot_sectors=hot_list)

        # 执行调用
        final_audit = skipped_results.copy()
        success_model = "None"
        models = getattr(CFG, 'MODELS', ["deepseek-chat"])
        
        for model in models:
            res = self._call_llm(user_prompt_body, model, status_callback)
            if res:
                parsed = self._clean_json(res)
                if parsed:
                    for k, v in parsed.items(): final_audit[k] = v
                    success_model = model
                    break 
        
        if success_model == "None":
            for row in valid_candidates:
                # [修复] 强制补齐 6 位字符串
                final_audit[str(row['symbol']).strip().zfill(6)] = {
                    "action": "WATCH",
                    "sentiment_score": 50,
                    "llm_score": 50.0,
                    "reason": "AI未响应",
                     "risk_factor": 0.5
                }

        return {"audit": final_audit}, success_model



class AITuningLab:
    """
    [AI 调优实验室]
    """
    def __init__(self, net_client):
        self.net = net_client
        self.data_layer = DataLayer(net_client)
        self.bt_engine = BacktestEngine(net_client)
        self.ai_engine = AIEngine(net_client)
        
    def benchmark_prompt(self, symbol, prompt_ver="v1", model_name="deepseek-chat", ui_callback=None):
        # --- 0. 准备实时存档文件 ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(BASE_DIR, "Backtest_History")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        report_file = os.path.join(save_dir, f"AI_Audit_{symbol}_{timestamp}.txt")
        
        def write_flush(content):
            try:
                with open(report_file, "a", encoding="utf-8") as f:
                    f.write(content + "\n")
                    f.flush(); os.fsync(f.fileno())
            except: pass

        def log(msg, append=True):
            if ui_callback: ui_callback(msg, append)
            else: print(msg)
            clean_msg = re.sub(r'\[color=.*?\]|\[/color\]|\[b\]|\[/b\]', '', msg)
            write_flush(clean_msg)

        # --- 开始业务 ---
        header = f"⚡ [AI验盘] 启动: {symbol} (实战全真模式)\n模型: {model_name}\n文件: {report_file}\n{'='*30}\n"
        log(header, False)
        
        # 1. 获取数据
        raw_df = self.data_layer.get_backtest_data(symbol, days=480)
        if raw_df.empty:
            log("❌ 未获取到数据，请检查网络。"); return

        # 2. 预计算
        df = self.bt_engine._prepare_data_for_strategy(raw_df, str(symbol))
        if df.empty:
            log("❌ 数据预处理失败。"); return
        
        # [核心补全] 注入大盘状态
        log("正在加载同期大盘环境数据 (RSRS)...")
        benchmark_map = self.bt_engine._get_benchmark_regime(600)
        
        try:
            df['date_obj'] = pd.to_datetime(df['date'])
            df['date_str'] = df['date_obj'].dt.strftime('%Y-%m-%d')
        except:
            df['date_str'] = df['date'].astype(str)
            
        df['regime_val'] = df['date_str'].map(benchmark_map).fillna(0.5)

        static_ind = "回测行业"
        try:
            tmp_df = self.data_layer.get_specific_stocks_hybrid([symbol])
            if not tmp_df.empty: static_ind = tmp_df.iloc[0].get('ind', '回测行业')
        except: pass

        # ==============================================================
        # ✅ [核心修复 1: 全量预计算前置] 
        # 彻底消灭 Rolling Bug，把算分逻辑提到循环外面跑一次。
        # 这里传入 0.5 作为中性宏观环境因子，主要为了让底层安全算出所有的指标。
        # ==============================================================
        log(f"正在执行全量特征预计算 (消除切片滚动幻觉)...")
        try:
            df = self.bt_engine.quant.strategy_scoring(
                df, "MID", 0.5, 
                breadth_panic=False, target_mode=True, pre_calculated=False
            )
        except Exception as e:
            log(f"❌ 全量预计算特征失败: {e}")
            return
        
        # 3. 扫描高分样本
        log(f"正在扫描高分样本...")
        all_tasks = []
        
        for i in range(len(df) - 5): 
            row = df.iloc[i]
            regime = row.get('regime_val', 0.5)
            
            # ==============================================================
            # ✅ [核心修复 2: 废除切片重算，改为 O(1) 直接查表]
            # ==============================================================
            score = row.get('final_score', 0)
            if score < 70: continue 
            
            # [同步修复] 过滤掉被物理风控拦截的标的，不浪费 AI 算力
            if not row.get('_is_entry_valid', True): continue
            
            # 兼容底层数据字典字段名（trend_desc 或 trend_tag）
            trend_desc_val = row.get('trend_desc', row.get('trend_tag', '震荡'))
            
            # =======================================================
            # [致命 Bug 修复] 隔夜跳空未来函数与滑点对齐 (保留原版优秀逻辑)
            # =======================================================
            next_day = df.iloc[i+1]
            day_3 = df.iloc[i+1 : i+4]
            
            actual_entry_price = next_day['open']
            if actual_entry_price <= 0.01: continue 
            
            ret_1d = (next_day['close'] - actual_entry_price) / actual_entry_price * 100
            max_pump_3d = (day_3['high'].max() - actual_entry_price) / actual_entry_price * 100
            max_dump_3d = (day_3['low'].min() - actual_entry_price) / actual_entry_price * 100
            
            label = "平"
            # [核心修复] 引入盘中低点穿透判定。优先判断是否在 3 天内触及防守止损线 (如 -5%)，
            # 一旦盘中跌破，即使后续反抽拉高也视为“大面”，对齐实盘爆仓逻辑。
            if ret_1d < -3.0 or max_dump_3d < -5.0: 
                label = "大面"
            elif ret_1d > 2.0 or max_pump_3d > 4.0: 
                label = "大肉"
            # ==============================================================
            # ✅ [核心修复 3: 全维特征无损释放给大模型]
            # 杜绝缺省排雷因子，直接将整行映射给 LLM
            # ==============================================================
            candidate = row.to_dict()
            candidate.update({
                'symbol': symbol, 
                'name': '回测标的', 
                'trend_desc': trend_desc_val,
                'ind': static_ind, 
                'rag_info': "无历史公告"
            })
            
            all_tasks.append({
                'date': row['date'], 'score': score, 'label': label,
                'ret_1d': ret_1d, 'max_3d': max_pump_3d,
                'candidate': candidate, 'regime': regime
            })

        if not all_tasks:
            log("\n⚠️ 无样本: 该股近期评分从未达到 70 分。"); return

        log(f" -> 发现 {len(all_tasks)} 个关键节点，开始流式输出...\n")

        # 4. 并发执行
        def process_batch(batch_tasks):
            batch_results = []
            # [性能起飞] 补全 as_completed 真正实现非阻塞兵发，并加上 Timeout 熔断
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self._process_single_task, t) for t in batch_tasks]
                # 谁先完成就先拿谁的结果，彻底告别木桶效应排队
                for future in as_completed(futures):
                    try: 
                        # 45秒级熔断，防止单一 API 卡死拖跨整个系统
                        batch_results.append(future.result(timeout=45))
                    except: pass
            return batch_results


        # 5. 循环处理
        success = 0; fail = 0
        for i in range(0, len(all_tasks), 5):
            batch = all_tasks[i : i + 5]
            results = process_batch(batch)
            results.sort(key=lambda x: x['date'])
            
            for res in results:
                if res['impact'] == 1: success += 1
                elif res['impact'] == -1: fail += 1
                
                msg = (f"> {res['date']} 分:{res['score']:.0f} ({res['label']})\n"
                       f" [{res['decision']}] {res['judgment']}\n"
                       f"   👉 {res['reason']}\n")
                log(msg, True)

        summary = f"\n{'='*30}\n[战报] {symbol}\n✅ 成功: {success} 次\n❌ 失败: {fail} 次\n🏆 净效能: {success - fail}\n"
        log(summary, True)
        log(f"报告已保存至: {report_file}")

    def _process_single_task(self, task):
        """单个任务的处理逻辑 (剥离出来以便并发调用)"""
        # [核心修复] 历史复盘必须以尾盘视角(TAIL)审视，彻底移除硬编码的 MID
        ai_res, _ = self.ai_engine.audit([task['candidate']], task['regime'], "TAIL")

        decision = "WATCH"; reason = "AI未解释"
        
        # [修复] 终极防线：彻底防御 LLM 幻觉与 Pandas 类型幽灵
        if ai_res and 'audit' in ai_res:
            # 强制提取当前任务的目标代码，并格式化为6位字符串
            req_sym = str(task['candidate']['symbol']).strip().zfill(6)
            
            # 精准去字典里捞数据，而不是盲抓第一个
            if req_sym in ai_res['audit']:
                item = ai_res['audit'][req_sym]
                decision = item.get('action', 'WATCH')
                reason = item.get('reason', 'AI未响应')
            else:
                # 记录 LLM 幻觉拒绝采信
                decision = "WATCH"
                reason = f"[系统风控] 目标 {req_sym} 意外丢失"

        judgment = "无功过"; score_impact = 0
        if decision == 'PASS':
            if task['ret_1d'] > 1.0 or task['max_3d'] > 4.0: judgment = "✅吃肉"; score_impact = 1
            elif task['ret_1d'] < -2.0: judgment = "❌吃面"; score_impact = -1
        elif decision == 'REJECT':
            # [严格保留] 原有判别文案 🛡️避雷 / 😭踏空
            if task['ret_1d'] < -2.0: judgment = "🛡️避雷"; score_impact = 1
            elif task['max_3d'] > 5.0: judgment = "😭踏空"; score_impact = -1
        
        return {
            'date': task['date'], 'score': task['score'], 'label': task['label'],
            'decision': decision, 'judgment': judgment, 'reason': reason,
            'impact': score_impact
        }

