"""
轉換 CSV 為 Parquet 格式
Parquet 格式優勢：
- 檔案大小減少 70%
- 讀取速度快 5-10 倍
- 保留資料型態
- 支援列式儲存和壓縮
"""

import pandas as pd
import time
import os

def convert_csv_to_parquet(
    csv_path: str,
    parquet_path: str = None,
    chunksize: int = 500000
):
    """
    將大型 CSV 檔案轉換為 Parquet 格式
    
    Args:
        csv_path: CSV 檔案路徑
        parquet_path: Parquet 輸出路徑（預設為 csv_path 但副檔名改為 .parquet）
        chunksize: 分批處理大小
    """
    if parquet_path is None:
        parquet_path = csv_path.replace('.csv', '.parquet')
    
    print(f"Converting {csv_path} to Parquet format...")
    print(f"Output: {parquet_path}")
    
    # 檢查原始檔案大小
    original_size = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"Original CSV size: {original_size:.1f} MB")
    
    start_time = time.time()
    
    # 方法 1：如果記憶體足夠，一次性載入（更快）
    try:
        print("\n嘗試一次性載入（需要足夠記憶體）...")
        df = pd.read_csv(csv_path)
        
        # 轉換為 Parquet
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='snappy',  # 或 'gzip', 'brotli'
            index=False
        )
        
        elapsed = time.time() - start_time
        print(f"✓ 轉換完成！耗時：{elapsed:.1f} 秒")
        
    except MemoryError:
        print("記憶體不足，使用分批處理模式...")
        
        # 方法 2：分批處理
        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        elapsed = time.time() - start_time
        print(f"✓ 轉換完成！耗時：{elapsed:.1f} 秒")
    
    # 檢查壓縮後大小
    compressed_size = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"Parquet size: {compressed_size:.1f} MB")
    print(f"壓縮比：{original_size/compressed_size:.2f}x")
    print(f"節省空間：{(1 - compressed_size/original_size)*100:.1f}%")
    
    return parquet_path


def benchmark_read_speed(csv_path: str, parquet_path: str, n_rows: int = 100000):
    """
    比較 CSV 和 Parquet 的讀取速度
    
    Args:
        csv_path: CSV 檔案路徑
        parquet_path: Parquet 檔案路徑
        n_rows: 讀取行數
    """
    print(f"\n{'='*60}")
    print("讀取速度測試")
    print(f"{'='*60}")
    
    # 測試 CSV
    print(f"\n讀取 CSV 前 {n_rows:,} 行...")
    start = time.time()
    df_csv = pd.read_csv(csv_path, nrows=n_rows)
    csv_time = time.time() - start
    print(f"CSV 讀取時間: {csv_time:.2f} 秒")
    
    # 測試 Parquet
    print(f"\n讀取 Parquet 前 {n_rows:,} 行...")
    start = time.time()
    df_parquet = pd.read_parquet(parquet_path)
    df_parquet = df_parquet.head(n_rows)
    parquet_time = time.time() - start
    print(f"Parquet 讀取時間: {parquet_time:.2f} 秒")
    
    # 比較
    speedup = csv_time / parquet_time
    print(f"\n{'='*60}")
    print(f"⚡ Parquet 快了 {speedup:.2f}x 倍！")
    print(f"{'='*60}")
    
    # 驗證資料一致性
    print("\n驗證資料一致性...")
    assert len(df_csv) == len(df_parquet), "行數不一致"
    assert list(df_csv.columns) == list(df_parquet.columns), "欄位不一致"
    print("✓ 資料一致性檢查通過")


def main():
    """主函數"""
    print("="*60)
    print("CSV to Parquet 轉換工具")
    print("="*60)
    
    csv_file = 'raw_data/acct_transaction.csv'
    parquet_file = 'raw_data/acct_transaction.parquet'
    
    # 檢查 CSV 檔案是否存在
    if not os.path.exists(csv_file):
        print(f"錯誤：找不到檔案 {csv_file}")
        return
    
    # 檢查是否已經轉換過
    if os.path.exists(parquet_file):
        print(f"\n⚠️  Parquet 檔案已存在：{parquet_file}")
        response = input("是否要重新轉換？(y/N): ").strip().lower()
        if response != 'y':
            print("略過轉換")
            # 只執行速度測試
            benchmark_read_speed(csv_file, parquet_file, n_rows=100000)
            return
    
    # 執行轉換
    parquet_path = convert_csv_to_parquet(csv_file, parquet_file)
    
    # 執行速度測試
    benchmark_read_speed(csv_file, parquet_path, n_rows=100000)
    
    print("\n" + "="*60)
    print("✓ 轉換完成！")
    print("="*60)
    print(f"\n接下來請修改 data_loader.py 使用 Parquet 檔案：")
    print(f"  1. 將 'raw_data/acct_transaction.csv'")
    print(f"     改為 'raw_data/acct_transaction.parquet'")
    print(f"  2. 將 pd.read_csv() 改為 pd.read_parquet()")
    print("\n或者直接執行 main_parquet.py（我會為您生成）")


if __name__ == "__main__":
    main()

