#!/usr/bin/env python
"""
測試腳本 - 確認環境和資料載入
"""

import sys
import os

print("=" * 70, flush=True)
print("測試腳本開始執行", flush=True)
print("=" * 70, flush=True)

# 添加 src 到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("\n1. 測試匯入模組...", flush=True)
try:
    from data_loader import load_alert_accounts, load_predict_accounts
    print("   ✓ data_loader 匯入成功", flush=True)
except Exception as e:
    print(f"   ✗ data_loader 匯入失敗: {e}", flush=True)
    sys.exit(1)

print("\n2. 測試載入警示帳戶資料...", flush=True)
try:
    alert_df = load_alert_accounts('raw_data/acct_alert.csv')
    print(f"   ✓ 成功載入 {len(alert_df)} 個警示帳戶", flush=True)
except Exception as e:
    print(f"   ✗ 載入失敗: {e}", flush=True)
    sys.exit(1)

print("\n3. 測試載入預測帳戶資料...", flush=True)
try:
    predict_df = load_predict_accounts('raw_data/acct_predict.csv')
    print(f"   ✓ 成功載入 {len(predict_df)} 個預測帳戶", flush=True)
except Exception as e:
    print(f"   ✗ 載入失敗: {e}", flush=True)
    sys.exit(1)

print("\n4. 檢查交易資料檔案...", flush=True)
txn_file = 'raw_data/acct_transaction.csv'
if os.path.exists(txn_file):
    size = os.path.getsize(txn_file)
    size_mb = size / (1024 * 1024)
    print(f"   ✓ 交易檔案存在: {size_mb:.1f} MB", flush=True)
else:
    print(f"   ✗ 交易檔案不存在", flush=True)
    sys.exit(1)

print("\n5. 測試讀取交易資料前 5 行...", flush=True)
try:
    import pandas as pd
    txn_sample = pd.read_csv(txn_file, nrows=5)
    print(f"   ✓ 成功讀取，欄位: {list(txn_sample.columns)}", flush=True)
    print(f"   資料範例:", flush=True)
    print(txn_sample.head(2).to_string(), flush=True)
except Exception as e:
    print(f"   ✗ 讀取失敗: {e}", flush=True)
    sys.exit(1)

print("\n" + "=" * 70, flush=True)
print("✓ 所有測試通過！可以開始執行主程式", flush=True)
print("=" * 70, flush=True)

