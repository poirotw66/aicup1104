"""
Main Script
整合所有模組的主執行腳本 - 警示帳戶特徵探索與規則建立
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
print("SYSTEM STARTING...", flush=True)
print("=" * 70, flush=True)

from data_loader import (
    load_alert_accounts,
    load_predict_accounts,
    create_alert_dict,
    load_transactions_in_batches
)
from feature_engineering import extract_features_from_transactions
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
    print_banner("警示帳戶偵測系統 - 特徵探索與規則建立")
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # 確保輸出目錄存在
    os.makedirs('output', exist_ok=True)
    
    # ======================================================================
    # Phase 1: 載入資料
    # ======================================================================
    print_banner("Phase 1: Loading Data")
    
    # 載入警示帳戶
    alert_df = load_alert_accounts('raw_data/acct_alert.csv')
    alert_accounts = set(alert_df['acct'].values)
    alert_dict = create_alert_dict(alert_df)
    
    # 載入預測目標帳戶
    predict_df = load_predict_accounts('raw_data/acct_predict.csv')
    
    print(f"\nData loaded successfully!")
    print(f"  - Alert accounts: {len(alert_accounts):,}")
    print(f"  - Accounts to predict: {len(predict_df):,}")
    
    # ======================================================================
    # Phase 2: 特徵工程
    # ======================================================================
    print_banner("Phase 2: Feature Engineering")
    
    # 檢查是否已有特徵檔案
    feature_file = 'output/features.csv'
    
    if os.path.exists(feature_file):
        print(f"Found existing feature file: {feature_file}")
        print("Loading features from file...")
        import pandas as pd
        features_df = pd.read_csv(feature_file)
        print(f"Loaded {len(features_df):,} accounts with {len(features_df.columns)-1} features")
    else:
        print("Extracting features from transaction data...")
        print("This may take a while (30-45 minutes)...")
        
        # 載入交易資料並提取特徵
        transaction_loader = load_transactions_in_batches(
            file_path='raw_data/acct_transaction.csv',
            chunksize=100000,
            alert_dict=alert_dict,
            use_time_window=True
        )
        
        features_df = extract_features_from_transactions(
            transaction_loader,
            alert_accounts,
            save_path=feature_file
        )
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nPhase 2 completed in {elapsed:.1f} minutes")
    
    # ======================================================================
    # Phase 3: 探索性資料分析 (EDA)
    # ======================================================================
    print_banner("Phase 3: Exploratory Data Analysis")
    
    eda_results = run_eda_analysis(features_df, alert_accounts, output_dir='output')
    comparison_df = eda_results['comparison_df']
    significant_features = eda_results['significant_features']
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nPhase 3 completed in {elapsed:.1f} minutes")
    
    # ======================================================================
    # Phase 4: 模式發現
    # ======================================================================
    print_banner("Phase 4: Pattern Discovery")
    
    pattern_results = run_pattern_discovery(features_df, alert_accounts, output_dir='output')
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nPhase 4 completed in {elapsed:.1f} minutes")
    
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
    print(f"\nPhase 5 completed in {elapsed:.1f} minutes")
    
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
    print_banner("All Tasks Completed!")
    
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    
    print("\n" + "="*70)
    print("Output Files Generated:")
    print("="*70)
    
    output_files = [
        'output/features.csv',
        'output/feature_comparison.csv',
        'output/alert_features.csv',
        'output/normal_features.csv',
        'output/decision_tree.png',
        'output/decision_tree_rules.txt',
        'output/tree_feature_importance.csv',
        'output/discovered_rules.json',
        'output/alert_clusters.csv',
        'output/final_rules.json',
        'output/rule_evaluation.json',
        'output/predictions.csv',
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
            print(f"  ✓ {file_path} ({size_str})")
        else:
            print(f"  ✗ {file_path} (not generated)")
    
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"  Alert accounts analyzed: {len(alert_accounts):,}")
    print(f"  Features extracted: {len(features_df.columns)-1}")
    print(f"  Significant features: {len(significant_features)}")
    print(f"  Rules created: {len(predictor.rules)}")
    print(f"  Predictions made: {len(predict_df):,}")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Review output/feature_comparison.csv for key insights")
    print("  2. Examine output/decision_tree.png to understand decision logic")
    print("  3. Check output/final_rules.json for the rules being applied")
    print("  4. View output/predictions.csv for final predictions")
    print("  5. Analyze output/rule_evaluation.json for model performance")
    
    print("\n✓ All done!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

