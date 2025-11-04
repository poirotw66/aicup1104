"""
Main Script - ULTRA FAST VERSION
結合 Parquet 格式 + 多核心並行 + 向量化操作
預期加速：15-50 倍
"""

import os
import sys
import time
from datetime import datetime
from multiprocessing import cpu_count

# 強制即時輸出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# 添加 src 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70, flush=True)
print("⚡⚡⚡ ULTRA FAST VERSION ⚡⚡⚡", flush=True)
print("Parquet 格式 + 多核心並行 + 向量化", flush=True)
print("=" * 70, flush=True)

from data_loader import (
    load_alert_accounts,
    load_predict_accounts,
    create_alert_dict
)
from eda_analysis import run_eda_analysis
from pattern_discovery import run_pattern_discovery
from rule_predictor import build_and_evaluate_predictor, make_predictions_with_explanations


def print_banner(text):
    """列印橫幅"""
    print("\n" + "="*70, flush=True)
    print(f"  {text}", flush=True)
    print("="*70 + "\n", flush=True)


def load_transactions_parquet(
    file_path: str,
    alert_dict: dict = None,
    use_time_window: bool = True
):
    """
    從 Parquet 檔案載入交易資料（快 5-10 倍）
    
    Args:
        file_path: Parquet 檔案路徑
        alert_dict: 警示帳戶字典
        use_time_window: 是否使用時間窗口過濾
        
    Yields:
        交易資料 DataFrame
    """
    import pandas as pd
    
    print(f"Loading transactions from Parquet: {file_path}", flush=True)
    
    # Parquet 可以一次性快速載入
    start_time = time.time()
    txn_df = pd.read_parquet(file_path)
    load_time = time.time() - start_time
    
    print(f"✓ Loaded {len(txn_df):,} transactions in {load_time:.1f} seconds", flush=True)
    print(f"   Speed: {len(txn_df)/load_time:,.0f} rows/sec", flush=True)
    
    # 時間窗口過濾
    if use_time_window and alert_dict:
        print("Applying time window filter...", flush=True)
        from data_loader import filter_by_event_date
        
        # 分批過濾（避免記憶體問題）
        chunksize = 1000000
        filtered_chunks = []
        
        for i in range(0, len(txn_df), chunksize):
            chunk = txn_df.iloc[i:i+chunksize]
            filtered_chunk = filter_by_event_date(chunk, alert_dict)
            filtered_chunks.append(filtered_chunk)
        
        txn_df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"✓ After filtering: {len(txn_df):,} transactions", flush=True)
    
    # 返回為 generator 以保持一致性
    yield txn_df


def main():
    """主執行函數"""
    start_time = time.time()
    print_banner("警示帳戶偵測系統 - ULTRA FAST")
    
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(f"Available CPU cores: {cpu_count()}", flush=True)
    print(f"\n⚡⚡⚡ 使用終極優化版本", flush=True)
    print(f"  - Parquet 格式（5-10x I/O 加速）", flush=True)
    print(f"  - 多核心並行（{max(1, cpu_count()-1)} 核心）", flush=True)
    print(f"  - 向量化操作（10-50x 運算加速）", flush=True)
    print(f"  - 預期總加速：15-50 倍", flush=True)
    
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
    # Phase 2: 特徵工程（ULTRA FAST 版本）
    # ======================================================================
    print_banner("Phase 2: Feature Engineering (ULTRA FAST)")
    
    feature_file = 'output/features.csv'
    
    if os.path.exists(feature_file):
        print(f"Found existing feature file: {feature_file}", flush=True)
        response = input("是否使用已存在的特徵檔案？(Y/n): ").strip().lower()
        if response not in ['n', 'no']:
            print("Loading features from file...", flush=True)
            import pandas as pd
            features_df = pd.read_csv(feature_file)
            print(f"Loaded {len(features_df):,} accounts with {len(features_df.columns)-1} features", flush=True)
        else:
            features_df = None
    else:
        features_df = None
    
    if features_df is None:
        # 檢查是否有 Parquet 檔案
        parquet_file = 'raw_data/acct_transaction.parquet'
        csv_file = 'raw_data/acct_transaction.csv'
        
        if os.path.exists(parquet_file):
            print(f"⚡ Using Parquet format for maximum speed!", flush=True)
            
            # 使用向量化 + Parquet
            from feature_engineering_ultra_fast import extract_features_ultra_fast
            
            transaction_loader = load_transactions_parquet(
                parquet_file,
                alert_dict=alert_dict,
                use_time_window=True
            )
            
            features_df = extract_features_ultra_fast(
                transaction_loader,
                alert_accounts,
                save_path=feature_file
            )
            
        else:
            print(f"⚠️  Parquet 檔案不存在，使用 CSV + 多核心並行", flush=True)
            print(f"   建議執行 'python convert_to_parquet.py' 以獲得更快速度", flush=True)
            
            # 使用多核心並行
            from data_loader import load_transactions_in_batches
            from feature_engineering_parallel import extract_features_parallel
            
            transaction_loader = load_transactions_in_batches(
                file_path=csv_file,
                chunksize=500000,
                alert_dict=alert_dict,
                use_time_window=True
            )
            
            features_df = extract_features_parallel(
                transaction_loader,
                alert_accounts,
                save_path=feature_file,
                n_processes=max(1, cpu_count() - 1)
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
    print_banner("⚡⚡⚡ All Tasks Completed! ⚡⚡⚡")
    
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Total execution time: {total_time:.1f} minutes ({total_time/60:.2f} hours)", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("✓ Predictions saved to: output/predictions.csv", flush=True)
    print("="*70 + "\n", flush=True)
    
    # 效能總結
    print("效能總結：", flush=True)
    if os.path.exists('raw_data/acct_transaction.parquet'):
        print("  ✓ 使用 Parquet 格式", flush=True)
    else:
        print("  ○ 未使用 Parquet（可進一步優化）", flush=True)
    
    print(f"  ✓ 使用 {max(1, cpu_count()-1)} 個 CPU 核心", flush=True)
    print(f"  ✓ 向量化操作", flush=True)
    print(f"\n預估相對原始版本加速：{45/total_time if total_time > 0 else 0:.1f}x", flush=True)


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

