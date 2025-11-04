"""
Main Script - FAST VERSION
使用優化的向量化特徵提取，預期加速 10-50 倍
"""

import os
import sys
import time
from datetime import datetime

# 強制即時輸出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# 添加 src 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70, flush=True)
print("⚡ FAST VERSION - 使用向量化優化", flush=True)
print("=" * 70, flush=True)

from data_loader import (
    load_alert_accounts,
    load_predict_accounts,
    create_alert_dict,
    load_transactions_in_batches
)
from feature_engineering_ultra_fast import extract_features_ultra_fast
from eda_analysis import run_eda_analysis
from pattern_discovery import run_pattern_discovery
from rule_predictor import build_and_evaluate_predictor, make_predictions_with_explanations


def print_banner(text):
    """列印橫幅"""
    print("\n" + "="*70, flush=True)
    print(f"  {text}", flush=True)
    print("="*70 + "\n", flush=True)


def main():
    """主執行函數"""
    start_time = time.time()
    print_banner("警示帳戶偵測系統 - 特徵探索與規則建立 (FAST)")
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"\n⚡ 使用優化版本，預期加速 10-50 倍", flush=True)
    
    # 確保輸出目錄存在
    os.makedirs('output', exist_ok=True)
    
    # ======================================================================
    # Phase 1: 載入資料
    # ======================================================================
    print_banner("Phase 1: Loading Data")
    
    alert_df = load_alert_accounts('raw_data/acct_alert.csv')
    alert_accounts = set(alert_df['acct'].values)
    alert_dict = create_alert_dict(alert_df)
    
    predict_df = load_predict_accounts('raw_data/acct_predict.csv')
    
    print(f"\nData loaded successfully!", flush=True)
    print(f"  - Alert accounts: {len(alert_accounts):,}", flush=True)
    print(f"  - Accounts to predict: {len(predict_df):,}", flush=True)
    
    # ======================================================================
    # Phase 2: 特徵工程（優化版）
    # ======================================================================
    print_banner("Phase 2: Feature Engineering (OPTIMIZED)")
    
    feature_file = 'output/features.csv'
    
    if os.path.exists(feature_file):
        print(f"Found existing feature file: {feature_file}", flush=True)
        print("Loading features from file...", flush=True)
        import pandas as pd
        features_df = pd.read_csv(feature_file)
        print(f"Loaded {len(features_df):,} accounts with {len(features_df.columns)-1} features", flush=True)
    else:
        print("⚡ Extracting features using VECTORIZED operations...", flush=True)
        print("Expected time: 5-15 minutes (vs 30-45 minutes with old method)", flush=True)
        
        # 使用優化的向量化特徵提取
        transaction_loader = load_transactions_in_batches(
            file_path='raw_data/acct_transaction.csv',
            chunksize=500000,  # 增加 chunk size
            alert_dict=alert_dict,
            use_time_window=True
        )
        
        features_df = extract_features_ultra_fast(
            transaction_loader,
            alert_accounts,
            save_path=feature_file
        )
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 2 completed in {elapsed:.1f} minutes", flush=True)
    
    # ======================================================================
    # Phase 3: 探索性資料分析 (EDA)
    # ======================================================================
    print_banner("Phase 3: Exploratory Data Analysis")
    
    eda_results = run_eda_analysis(features_df, alert_accounts, output_dir='output')
    comparison_df = eda_results['comparison_df']
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 3 completed in {elapsed:.1f} minutes", flush=True)
    
    # ======================================================================
    # Phase 4: 模式發現
    # ======================================================================
    print_banner("Phase 4: Pattern Discovery")
    
    pattern_results = run_pattern_discovery(features_df, alert_accounts, output_dir='output')
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 4 completed in {elapsed:.1f} minutes", flush=True)
    
    # ======================================================================
    # Phase 5: 規則建立與評估
    # ======================================================================
    print_banner("Phase 5: Rule Building and Evaluation")
    
    predictor = build_and_evaluate_predictor(
        features_df,
        alert_accounts,
        comparison_df,
        output_dir='output'
    )
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n✓ Phase 5 completed in {elapsed:.1f} minutes", flush=True)
    
    # ======================================================================
    # Phase 6: 預測
    # ======================================================================
    print_banner("Phase 6: Making Predictions")
    
    make_predictions_with_explanations(
        predictor,
        features_df,
        predict_df,
        output_path='output/predictions.csv'
    )
    
    # ======================================================================
    # 完成
    # ======================================================================
    total_time = (time.time() - start_time) / 60
    print_banner("✓ All Tasks Completed!")
    
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Total execution time: {total_time:.1f} minutes ({total_time/60:.2f} hours)", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓ Predictions saved to: output/predictions.csv", flush=True)
    print("="*70 + "\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

