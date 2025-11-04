"""
Feature Engineering Module - PARALLEL VERSION
使用多核心並行處理（2-4倍加速）
"""

import pandas as pd
import numpy as np
from typing import Set, List
from tqdm import tqdm
import gc
from multiprocessing import Pool, cpu_count
from functools import partial


def process_account_batch(
    accounts: List[str],
    txn_df: pd.DataFrame,
    alert_accounts: Set[str]
) -> pd.DataFrame:
    """
    處理一批帳戶的特徵提取
    
    這個函數會在不同的進程中並行執行
    
    Args:
        accounts: 要處理的帳戶列表
        txn_df: 交易資料
        alert_accounts: 警示帳戶集合
        
    Returns:
        該批帳戶的特徵 DataFrame
    """
    features_list = []
    
    for acct in accounts:
        # 該帳戶的轉出交易
        out_txn = txn_df[txn_df['from_acct'] == acct]
        # 該帳戶的轉入交易
        in_txn = txn_df[txn_df['to_acct'] == acct]
        
        features = {'acct': acct}
        
        # === 基礎統計特徵 ===
        features['out_txn_count'] = len(out_txn)
        features['in_txn_count'] = len(in_txn)
        features['total_txn_count'] = features['out_txn_count'] + features['in_txn_count']
        
        if features['total_txn_count'] == 0:
            # 該帳戶沒有交易，填充預設值
            features.update({f: 0 for f in [
                'out_txn_sum', 'in_txn_sum', 'out_txn_mean', 'in_txn_mean',
                'out_txn_max', 'in_txn_max', 'out_txn_std', 'in_txn_std',
                'active_days', 'night_txn_ratio', 'alert_txn_count'
            ]})
            features_list.append(features)
            continue
        
        # 金額特徵
        features['out_txn_sum'] = out_txn['txn_amt'].sum() if len(out_txn) > 0 else 0
        features['in_txn_sum'] = in_txn['txn_amt'].sum() if len(in_txn) > 0 else 0
        features['out_txn_mean'] = out_txn['txn_amt'].mean() if len(out_txn) > 0 else 0
        features['in_txn_mean'] = in_txn['txn_amt'].mean() if len(in_txn) > 0 else 0
        features['out_txn_max'] = out_txn['txn_amt'].max() if len(out_txn) > 0 else 0
        features['in_txn_max'] = in_txn['txn_amt'].max() if len(in_txn) > 0 else 0
        features['out_txn_std'] = out_txn['txn_amt'].std() if len(out_txn) > 1 else 0
        features['in_txn_std'] = in_txn['txn_amt'].std() if len(in_txn) > 1 else 0
        
        # 時間特徵
        all_dates = set(out_txn['txn_date'].values) | set(in_txn['txn_date'].values)
        features['active_days'] = len(all_dates)
        
        # 簡化的時間特徵
        if 'hour' in out_txn.columns:
            out_hours = out_txn['hour'].values
            in_hours = in_txn['hour'].values
            all_hours = np.concatenate([out_hours, in_hours])
            valid_hours = all_hours[all_hours >= 0]
            
            if len(valid_hours) > 0:
                features['night_txn_ratio'] = ((valid_hours >= 0) & (valid_hours < 6)).sum() / len(valid_hours)
            else:
                features['night_txn_ratio'] = 0
        else:
            features['night_txn_ratio'] = 0
        
        # 與警示帳戶的互動
        features['alert_txn_count'] = (
            out_txn['to_acct'].isin(alert_accounts).sum() +
            in_txn['from_acct'].isin(alert_accounts).sum()
        )
        
        # 網絡特徵
        features['unique_out_partners'] = out_txn['to_acct'].nunique() if len(out_txn) > 0 else 0
        features['unique_in_partners'] = in_txn['from_acct'].nunique() if len(in_txn) > 0 else 0
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def extract_features_parallel(
    transaction_loader,
    alert_accounts: Set[str],
    save_path: str = 'output/features.csv',
    n_processes: int = None
) -> pd.DataFrame:
    """
    使用多核心並行提取特徵
    
    Args:
        transaction_loader: 交易資料載入器
        alert_accounts: 警示帳戶集合
        save_path: 儲存路徑
        n_processes: 進程數（None = 自動偵測）
        
    Returns:
        特徵 DataFrame
    """
    print("="*60)
    print("PARALLEL Feature Extraction")
    print("="*60)
    
    # 自動偵測 CPU 核心數
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # 保留一個核心給系統
    
    print(f"\n⚡ 使用 {n_processes} 個 CPU 核心進行並行處理")
    print(f"預期加速：{n_processes * 0.7:.1f}x ~ {n_processes * 0.9:.1f}x\n")
    
    # 載入所有交易資料
    print("Loading all transaction data...")
    all_chunks = []
    for chunk in tqdm(transaction_loader, desc="Loading chunks"):
        all_chunks.append(chunk)
    
    print(f"Concatenating {len(all_chunks)} chunks...")
    txn_df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    gc.collect()
    
    print(f"Total transactions: {len(txn_df):,}")
    
    # 預處理：解析時間
    if 'hour' not in txn_df.columns:
        print("Parsing transaction hours...")
        txn_df['hour'] = pd.to_datetime(txn_df['txn_time'], format='%H:%M:%S', errors='coerce').dt.hour
        txn_df['hour'] = txn_df['hour'].fillna(-1).astype(int)
    
    # 獲取所有唯一帳戶
    print("Extracting unique accounts...")
    all_accounts = list(set(txn_df['from_acct'].unique()) | set(txn_df['to_acct'].unique()))
    print(f"Total unique accounts: {len(all_accounts):,}")
    
    # 將帳戶分成 N 批，用於並行處理
    batch_size = max(1, len(all_accounts) // (n_processes * 4))  # 每個進程處理多批
    account_batches = [
        all_accounts[i:i+batch_size]
        for i in range(0, len(all_accounts), batch_size)
    ]
    
    print(f"Split into {len(account_batches)} batches (avg {batch_size} accounts/batch)")
    
    # 並行處理
    print(f"\n⚡ Starting parallel processing with {n_processes} processes...")
    
    # 創建部分函數（固定 txn_df 和 alert_accounts 參數）
    process_func = partial(
        process_account_batch,
        txn_df=txn_df,
        alert_accounts=alert_accounts
    )
    
    # 使用進程池並行處理
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, account_batches),
            total=len(account_batches),
            desc="Processing batches"
        ))
    
    # 合併結果
    print("\nMerging results...")
    features_df = pd.concat(results, ignore_index=True)
    
    # 填充缺失值
    features_df = features_df.fillna(0)
    features_df = features_df.replace([np.inf, -np.inf], 0)
    
    print(f"\n✓ Extracted features for {len(features_df):,} accounts")
    print(f"   Features: {len(features_df.columns) - 1}")
    
    # 儲存
    print(f"\nSaving features to {save_path}...")
    features_df.to_csv(save_path, index=False)
    print("✓ Parallel feature extraction completed!")
    
    return features_df


if __name__ == "__main__":
    print("Parallel feature engineering module ready!")
    print(f"Available CPU cores: {cpu_count()}")

