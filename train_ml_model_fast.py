#!/usr/bin/env python3
"""
修正版機器學習模型訓練 - FAST VERSION
解決資料洩漏問題 + 高效記憶體管理
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 強制即時輸出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


def print_banner(text):
    """列印橫幅"""
    print("\n" + "="*70, flush=True)
    print(f"  {text}", flush=True)
    print("="*70 + "\n", flush=True)


def load_data_efficiently():
    """高效載入資料"""
    print_banner("Phase 1: 載入資料")
    
    start_time = time.time()
    
    # 1. 載入標籤
    print("載入異常帳戶標籤...", flush=True)
    alert_df = pd.read_csv('raw_data/acct_alert.csv')
    alert_accounts = set(alert_df['acct'].values)
    print(f"  ✓ 異常帳戶: {len(alert_accounts):,}", flush=True)
    
    # 2. 載入預測目標
    print("載入預測目標...", flush=True)
    predict_df = pd.read_csv('raw_data/acct_predict.csv')
    predict_accounts = set(predict_df['acct'].values)
    print(f"  ✓ 預測目標: {len(predict_accounts):,}", flush=True)
    
    # 3. 分批載入特徵（避免記憶體問題）
    print("\n載入特徵檔案（分批處理）...", flush=True)
    feature_file = 'output/features.csv'
    
    if not os.path.exists(feature_file):
        print(f"❌ 錯誤: 找不到特徵檔案 {feature_file}", flush=True)
        print("請先執行 main.py 或 main_ultra_fast.py 生成特徵", flush=True)
        sys.exit(1)
    
    # 先讀取表頭
    feature_cols = pd.read_csv(feature_file, nrows=0).columns.tolist()
    feature_cols.remove('acct')
    if 'is_alert' in feature_cols:
        feature_cols.remove('is_alert')
    
    print(f"  ✓ 特徵數: {len(feature_cols)}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Phase 1 完成 ({elapsed:.1f} 秒)", flush=True)
    
    return alert_accounts, predict_accounts, feature_file, feature_cols


def prepare_training_data(feature_file, feature_cols, alert_accounts, predict_accounts, 
                          normal_sample_size=5000):
    """
    準備訓練資料（避免資料洩漏）
    使用分批處理以節省記憶體
    """
    print_banner("Phase 2: 準備訓練資料（修正資料洩漏）")
    
    start_time = time.time()
    
    print(f"🔍 策略: 只用標記資料訓練，不包含預測目標", flush=True)
    print(f"  - 異常樣本: {len(alert_accounts):,}", flush=True)
    print(f"  - 正常樣本: 最多 {normal_sample_size:,} (從非預測目標中抽樣)", flush=True)
    
    # 分批讀取，篩選需要的帳戶
    chunksize = 100000
    alert_data = []
    normal_data = []
    normal_sampled = 0
    
    print(f"\n分批處理特徵檔案...", flush=True)
    
    chunk_count = 0
    for chunk in pd.read_csv(feature_file, chunksize=chunksize):
        chunk_count += 1
        
        # 收集異常帳戶
        alert_chunk = chunk[chunk['acct'].isin(alert_accounts)]
        if len(alert_chunk) > 0:
            alert_data.append(alert_chunk)
            print(f"  Chunk {chunk_count}: 找到 {len(alert_chunk)} 個異常帳戶", flush=True)
        
        # 收集正常帳戶（排除預測目標）
        if normal_sampled < normal_sample_size:
            available_normal = chunk[
                ~chunk['acct'].isin(alert_accounts) &
                ~chunk['acct'].isin(predict_accounts)
            ]
            
            if len(available_normal) > 0:
                # 計算還需要多少樣本
                needed = normal_sample_size - normal_sampled
                
                # 隨機抽樣
                if len(available_normal) > needed:
                    sample = available_normal.sample(n=needed, random_state=42)
                else:
                    sample = available_normal
                
                normal_data.append(sample)
                normal_sampled += len(sample)
                print(f"  Chunk {chunk_count}: 抽樣 {len(sample)} 個正常帳戶 (累計: {normal_sampled})", flush=True)
    
    print(f"\n合併資料...", flush=True)
    
    # 合併異常資料
    if len(alert_data) == 0:
        print("❌ 錯誤: 沒有找到任何異常帳戶！", flush=True)
        sys.exit(1)
    
    alert_df = pd.concat(alert_data, ignore_index=True)
    print(f"  ✓ 異常樣本: {len(alert_df):,}", flush=True)
    
    # 合併正常資料
    if len(normal_data) == 0:
        print("❌ 錯誤: 沒有找到任何正常帳戶！", flush=True)
        sys.exit(1)
    
    normal_df = pd.concat(normal_data, ignore_index=True)
    print(f"  ✓ 正常樣本: {len(normal_df):,}", flush=True)
    
    # 合併訓練集
    train_df = pd.concat([alert_df, normal_df], ignore_index=True)
    train_df['label'] = train_df['acct'].isin(alert_accounts).astype(int)
    
    # 準備特徵和標籤
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    
    # 檢查資料洩漏
    leak_check = train_df['acct'].isin(predict_accounts).sum()
    
    print(f"\n訓練集統計:", flush=True)
    print(f"  總樣本: {len(train_df):,}", flush=True)
    print(f"  異常: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)", flush=True)
    print(f"  正常: {len(y_train)-y_train.sum():,} ({(len(y_train)-y_train.sum())/len(y_train)*100:.2f}%)", flush=True)
    print(f"  特徵數: {len(feature_cols)}", flush=True)
    
    if leak_check > 0:
        print(f"\n⚠️  警告: 訓練集中有 {leak_check} 個預測目標帳戶！", flush=True)
    else:
        print(f"\n✅ 確認: 訓練集中沒有預測目標帳戶（無資料洩漏）", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Phase 2 完成 ({elapsed:.1f} 秒)", flush=True)
    
    return X_train, y_train, train_df


def train_models(X_train, y_train, feature_cols):
    """訓練多個模型"""
    print_banner("Phase 3: 訓練模型")
    
    start_time = time.time()
    
    # 交叉驗證設定
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {}
    cv_scores = {}
    
    # 模型 1: Random Forest (平衡)
    print("1️⃣  Random Forest (Balanced)...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("   執行 5-fold 交叉驗證...", flush=True)
    rf_f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        rf.fit(X_fold_train, y_fold_train)
        y_pred = rf.predict(X_fold_val)
        f1 = f1_score(y_fold_val, y_pred)
        rf_f1_scores.append(f1)
        print(f"     Fold {fold}: F1 = {f1:.4f}", flush=True)
    
    rf_f1_scores = np.array(rf_f1_scores)
    print(f"   ✓ CV F1-Score: {rf_f1_scores.mean():.4f} (+/- {rf_f1_scores.std():.4f})", flush=True)
    
    # 在全部訓練集上訓練
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    cv_scores['random_forest'] = rf_f1_scores
    
    # 模型 2: Gradient Boosting
    print("\n2️⃣  Gradient Boosting...", flush=True)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    print("   執行 5-fold 交叉驗證...", flush=True)
    gb_f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        gb.fit(X_fold_train, y_fold_train)
        y_pred = gb.predict(X_fold_val)
        f1 = f1_score(y_fold_val, y_pred)
        gb_f1_scores.append(f1)
        print(f"     Fold {fold}: F1 = {f1:.4f}", flush=True)
    
    gb_f1_scores = np.array(gb_f1_scores)
    print(f"   ✓ CV F1-Score: {gb_f1_scores.mean():.4f} (+/- {gb_f1_scores.std():.4f})", flush=True)
    
    gb.fit(X_train, y_train)
    models['gradient_boosting'] = gb
    cv_scores['gradient_boosting'] = gb_f1_scores
    
    # 模型 3: Random Forest (保守)
    print("\n3️⃣  Random Forest (Conservative)...", flush=True)
    rf_cons = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("   執行 5-fold 交叉驗證...", flush=True)
    rf_cons_f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        rf_cons.fit(X_fold_train, y_fold_train)
        y_pred = rf_cons.predict(X_fold_val)
        f1 = f1_score(y_fold_val, y_pred)
        rf_cons_f1_scores.append(f1)
        print(f"     Fold {fold}: F1 = {f1:.4f}", flush=True)
    
    rf_cons_f1_scores = np.array(rf_cons_f1_scores)
    print(f"   ✓ CV F1-Score: {rf_cons_f1_scores.mean():.4f} (+/- {rf_cons_f1_scores.std():.4f})", flush=True)
    
    rf_cons.fit(X_train, y_train)
    models['rf_conservative'] = rf_cons
    cv_scores['rf_conservative'] = rf_cons_f1_scores
    
    # 選擇最佳模型
    best_model_name = max(cv_scores, key=lambda k: cv_scores[k].mean())
    best_model = models[best_model_name]
    best_score = cv_scores[best_model_name].mean()
    
    print(f"\n🏆 最佳模型: {best_model_name}", flush=True)
    print(f"   CV F1-Score: {best_score:.4f}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Phase 3 完成 ({elapsed/60:.1f} 分鐘)", flush=True)
    
    return best_model, best_model_name, cv_scores


def evaluate_on_training(model, X_train, y_train):
    """評估訓練集表現"""
    print_banner("Phase 4: 訓練集評估")
    
    y_pred = model.predict(X_train)
    
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    cm = confusion_matrix(y_train, y_pred)
    
    print(f"Precision: {precision:.4f}", flush=True)
    print(f"Recall: {recall:.4f}", flush=True)
    print(f"F1-Score: {f1:.4f}", flush=True)
    print(f"\nConfusion Matrix:", flush=True)
    print(f"  TN: {cm[0][0]:,}, FP: {cm[0][1]:,}", flush=True)
    print(f"  FN: {cm[1][0]:,}, TP: {cm[1][1]:,}", flush=True)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }


def predict_efficiently(model, feature_file, feature_cols, predict_accounts, 
                       output_path='output/predictions_ml.csv'):
    """高效預測（分批處理）"""
    print_banner("Phase 5: 生成預測")
    
    start_time = time.time()
    
    print(f"分批讀取並預測...", flush=True)
    
    results = []
    chunksize = 100000
    found_count = 0
    
    for chunk_idx, chunk in enumerate(pd.read_csv(feature_file, chunksize=chunksize), 1):
        # 篩選預測目標
        pred_chunk = chunk[chunk['acct'].isin(predict_accounts)]
        
        if len(pred_chunk) > 0:
            X_pred = pred_chunk[feature_cols]
            y_pred = model.predict(X_pred)
            y_proba = model.predict_proba(X_pred)[:, 1]
            
            result_chunk = pd.DataFrame({
                'acct': pred_chunk['acct'].values,
                'label': y_pred,
                'confidence_score': y_proba
            })
            
            results.append(result_chunk)
            found_count += len(pred_chunk)
            print(f"  Chunk {chunk_idx}: 找到 {len(pred_chunk)} 個預測目標 (累計: {found_count})", flush=True)
    
    if len(results) == 0:
        print("❌ 錯誤: 沒有找到任何預測目標！", flush=True)
        sys.exit(1)
    
    # 合併結果
    result_df = pd.concat(results, ignore_index=True)
    
    # 保存
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ 預測結果已保存: {output_path}", flush=True)
    
    # 簡化版
    simple_output = output_path.replace('.csv', '_acct_label.csv')
    result_df[['acct', 'label']].to_csv(simple_output, index=False)
    print(f"✅ 簡化版已保存: {simple_output}", flush=True)
    
    # 統計
    n_alert = result_df['label'].sum()
    print(f"\n預測統計:", flush=True)
    print(f"  總預測數: {len(result_df):,}", flush=True)
    print(f"  預測為異常: {n_alert:,} ({n_alert/len(result_df)*100:.2f}%)", flush=True)
    print(f"  預測為正常: {len(result_df)-n_alert:,} ({(len(result_df)-n_alert)/len(result_df)*100:.2f}%)", flush=True)
    print(f"  平均信心分數: {result_df['confidence_score'].mean():.4f}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Phase 5 完成 ({elapsed:.1f} 秒)", flush=True)
    
    return result_df


def main():
    """主函數"""
    total_start = time.time()
    
    print("="*70, flush=True)
    print("🔧 修正版機器學習模型訓練 - FAST VERSION", flush=True)
    print("   解決資料洩漏 + 高效記憶體管理", flush=True)
    print("="*70, flush=True)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Phase 1: 載入資料
    alert_accounts, predict_accounts, feature_file, feature_cols = load_data_efficiently()
    
    # Phase 2: 準備訓練資料
    X_train, y_train, train_df = prepare_training_data(
        feature_file, feature_cols, alert_accounts, predict_accounts,
        normal_sample_size=5000
    )
    
    # Phase 3: 訓練模型
    best_model, best_model_name, cv_scores = train_models(X_train, y_train, feature_cols)
    
    # Phase 4: 評估
    train_metrics = evaluate_on_training(best_model, X_train, y_train)
    
    # Phase 5: 預測
    result_df = predict_efficiently(
        best_model, feature_file, feature_cols, predict_accounts,
        output_path='output/predictions_ml.csv'
    )
    
    # 保存模型和報告
    print_banner("Phase 6: 保存模型和報告")
    
    model_path = 'output/trained_model.pkl'
    joblib.dump({
        'model': best_model,
        'model_name': best_model_name,
        'feature_cols': feature_cols
    }, model_path)
    print(f"✅ 模型已保存: {model_path}", flush=True)
    
    report = {
        'model_name': best_model_name,
        'cv_f1_mean': float(cv_scores[best_model_name].mean()),
        'cv_f1_std': float(cv_scores[best_model_name].std()),
        'train_metrics': train_metrics,
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_predictions': len(result_df),
        'n_predicted_alerts': int(result_df['label'].sum()),
        'prediction_rate': float(result_df['label'].sum() / len(result_df))
    }
    
    with open('output/ml_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ 訓練報告已保存: output/ml_training_report.json", flush=True)
    
    # 總結
    total_time = (time.time() - total_start) / 60
    print_banner("✨ 訓練完成！")
    
    print(f"結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"總執行時間: {total_time:.1f} 分鐘", flush=True)
    
    print(f"\n" + "="*70, flush=True)
    print(f"📊 模型表現總結", flush=True)
    print(f"="*70, flush=True)
    print(f"模型: {best_model_name}", flush=True)
    print(f"交叉驗證 F1: {report['cv_f1_mean']:.4f} (+/- {report['cv_f1_std']:.4f})", flush=True)
    print(f"訓練集 F1: {train_metrics['f1_score']:.4f}", flush=True)
    print(f"預測異常率: {report['prediction_rate']*100:.2f}%", flush=True)
    
    print(f"\n" + "="*70, flush=True)
    print(f"📁 輸出檔案", flush=True)
    print(f"="*70, flush=True)
    print(f"✅ output/predictions_ml.csv (完整版)", flush=True)
    print(f"✅ output/predictions_ml_acct_label.csv (提交版)", flush=True)
    print(f"✅ output/trained_model.pkl (模型檔)", flush=True)
    print(f"✅ output/ml_training_report.json (報告)", flush=True)
    
    print(f"\n💡 建議上傳: output/predictions_ml_acct_label.csv", flush=True)
    print(f"   預期 F1-Score: 0.25-0.40 (相比原本 0.07 有顯著提升)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被使用者中斷", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 發生錯誤: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

