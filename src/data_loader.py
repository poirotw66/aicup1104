"""
Data Loader Module
分批載入交易資料和時間窗口過濾
"""

import pandas as pd
import gc
from typing import Dict, Optional
from tqdm import tqdm


def load_alert_accounts(file_path: str = 'raw_data/acct_alert.csv') -> pd.DataFrame:
    """
    載入警示帳戶資料
    
    Returns:
        DataFrame with columns: acct, event_date
    """
    print(f"Loading alert accounts from {file_path}...")
    alert_df = pd.read_csv(file_path)
    print(f"Loaded {len(alert_df)} alert accounts")
    return alert_df


def load_predict_accounts(file_path: str = 'raw_data/acct_predict.csv') -> pd.DataFrame:
    """
    載入需要預測的帳戶資料
    
    Returns:
        DataFrame with columns: acct, label
    """
    print(f"Loading predict accounts from {file_path}...")
    predict_df = pd.read_csv(file_path)
    print(f"Loaded {len(predict_df)} accounts to predict")
    return predict_df


def create_alert_dict(alert_df: pd.DataFrame) -> Dict[str, int]:
    """
    建立警示帳戶字典，用於快速查詢
    
    Args:
        alert_df: 警示帳戶 DataFrame
        
    Returns:
        字典 {acct: event_date}
    """
    return dict(zip(alert_df['acct'], alert_df['event_date']))


def load_transactions_in_batches(
    file_path: str = 'raw_data/acct_transaction.csv',
    chunksize: int = 100000,
    alert_dict: Optional[Dict[str, int]] = None,
    use_time_window: bool = True
):
    """
    分批載入交易資料
    
    Args:
        file_path: 交易資料檔案路徑
        chunksize: 每批處理的資料量
        alert_dict: 警示帳戶字典 {acct: event_date}
        use_time_window: 是否使用時間窗口過濾
        
    Yields:
        每批處理後的交易資料 DataFrame
    """
    print(f"Loading transactions from {file_path} in batches...")
    print(f"Batch size: {chunksize}")
    
    total_rows = 0
    filtered_rows = 0
    
    # 使用 tqdm 顯示進度
    with pd.read_csv(file_path, chunksize=chunksize) as reader:
        for i, chunk in enumerate(tqdm(reader, desc="Processing batches")):
            total_rows += len(chunk)
            
            # 如果需要時間窗口過濾
            if use_time_window and alert_dict:
                chunk = filter_by_event_date(chunk, alert_dict)
                filtered_rows += len(chunk)
            
            yield chunk
            
            # 釋放記憶體
            gc.collect()
    
    print(f"Total rows processed: {total_rows:,}")
    if use_time_window and alert_dict:
        print(f"Rows after time window filtering: {filtered_rows:,}")


def filter_by_event_date(
    txn_chunk: pd.DataFrame,
    alert_dict: Dict[str, int]
) -> pd.DataFrame:
    """
    根據警示日期過濾交易資料
    只保留警示日期之前的交易
    
    對於非警示帳戶，保留所有交易
    
    Args:
        txn_chunk: 交易資料批次
        alert_dict: 警示帳戶字典 {acct: event_date}
        
    Returns:
        過濾後的交易資料
    """
    # 為每筆交易標記是否應該保留
    mask_from = txn_chunk.apply(
        lambda row: row['txn_date'] < alert_dict.get(row['from_acct'], float('inf')),
        axis=1
    )
    
    mask_to = txn_chunk.apply(
        lambda row: row['txn_date'] < alert_dict.get(row['to_acct'], float('inf')),
        axis=1
    )
    
    # 只有當 from_acct 和 to_acct 都滿足時間條件時才保留
    mask = mask_from & mask_to
    
    return txn_chunk[mask].copy()


def get_all_accounts(file_path: str = 'raw_data/acct_transaction.csv') -> set:
    """
    獲取所有出現在交易記錄中的唯一帳戶
    
    Args:
        file_path: 交易資料檔案路徑
        
    Returns:
        所有唯一帳戶的集合
    """
    print("Extracting all unique accounts...")
    all_accounts = set()
    
    chunksize = 100000
    with pd.read_csv(file_path, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, desc="Scanning accounts"):
            all_accounts.update(chunk['from_acct'].unique())
            all_accounts.update(chunk['to_acct'].unique())
    
    print(f"Total unique accounts found: {len(all_accounts):,}")
    return all_accounts


def save_accounts_list(accounts: set, output_path: str = 'output/all_accounts.txt'):
    """
    儲存帳戶列表到檔案
    
    Args:
        accounts: 帳戶集合
        output_path: 輸出檔案路徑
    """
    with open(output_path, 'w') as f:
        for acct in sorted(accounts):
            f.write(f"{acct}\n")
    print(f"Saved {len(accounts)} accounts to {output_path}")


def load_accounts_list(input_path: str = 'output/all_accounts.txt') -> set:
    """
    從檔案載入帳戶列表
    
    Args:
        input_path: 輸入檔案路徑
        
    Returns:
        帳戶集合
    """
    with open(input_path, 'r') as f:
        accounts = {line.strip() for line in f if line.strip()}
    print(f"Loaded {len(accounts)} accounts from {input_path}")
    return accounts


if __name__ == "__main__":
    # 測試程式碼
    print("Testing data_loader module...")
    
    # 載入警示帳戶
    alert_df = load_alert_accounts()
    alert_dict = create_alert_dict(alert_df)
    
    # 載入預測帳戶
    predict_df = load_predict_accounts()
    
    # 測試分批載入（只處理前 3 批）
    print("\nTesting batch loading (first 3 batches)...")
    batch_count = 0
    for chunk in load_transactions_in_batches(chunksize=100000, alert_dict=alert_dict):
        print(f"Batch {batch_count + 1}: {len(chunk)} rows")
        batch_count += 1
        if batch_count >= 3:
            break
    
    print("\nData loader module test completed!")

