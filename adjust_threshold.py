#!/usr/bin/env python3
"""
調整預測閾值以優化 F1-Score
根據 confidence_score 重新生成預測標籤
"""

import pandas as pd
import numpy as np


def adjust_predictions_threshold(input_file, output_file, threshold=0.75):
    """
    根據指定閾值重新生成預測標籤
    
    Args:
        input_file (str): 包含 confidence_score 的預測檔案
        output_file (str): 輸出檔案路徑
        threshold (float): 判定為異常的閾值 (預設 0.75)
    """
    print(f"讀取預測檔案: {input_file}")
    df = pd.read_csv(input_file)
    
    if 'confidence_score' not in df.columns:
        print("❌ 錯誤: 檔案中沒有 confidence_score 欄位")
        return
    
    # 根據新閾值重新生成標籤
    df['label'] = (df['confidence_score'] > threshold).astype(int)
    
    # 統計資訊
    total = len(df)
    anomaly_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    anomaly_rate = anomaly_count / total * 100
    
    print(f"\n{'='*60}")
    print(f"🎯 使用閾值: {threshold}")
    print(f"{'='*60}")
    print(f"總預測數: {total:,}")
    print(f"正常 (0): {normal_count:,} ({normal_count/total*100:.2f}%)")
    print(f"異常 (1): {anomaly_count:,} ({anomaly_rate:.2f}%)")
    
    # 與訓練集比較
    try:
        alert_df = pd.read_csv('raw_data/acct_alert.csv')
        train_anomaly_rate = len(alert_df) / total * 100
        ratio = anomaly_count / len(alert_df)
        print(f"\n訓練集異常率: {train_anomaly_rate:.2f}%")
        print(f"相對訓練集倍數: {ratio:.2f}x")
        
        if 0.8 <= ratio <= 1.5:
            print("✅ 預測異常數在合理範圍內")
        elif ratio > 1.5:
            print("⚠️  預測異常數偏高，可能需要更高的閾值")
        else:
            print("⚠️  預測異常數偏低，可能需要更低的閾值")
    except:
        pass
    
    # 儲存結果
    # 完整版（包含所有欄位）
    df.to_csv(output_file, index=False)
    print(f"\n✅ 完整版已儲存至: {output_file}")
    
    # 簡化版（只有 acct 和 label）
    simple_output = output_file.replace('.csv', '_acct_label.csv')
    df[['acct', 'label']].to_csv(simple_output, index=False)
    print(f"✅ 簡化版已儲存至: {simple_output}")
    
    # 顯示前幾筆
    print(f"\n前5筆資料:")
    print(df[['acct', 'label', 'confidence_score']].head())
    
    return df


def compare_thresholds(input_file):
    """比較不同閾值下的預測結果"""
    print(f"讀取預測檔案: {input_file}")
    df = pd.read_csv(input_file)
    
    if 'confidence_score' not in df.columns:
        print("❌ 錯誤: 檔案中沒有 confidence_score 欄位")
        return
    
    # 讀取訓練集統計
    try:
        alert_df = pd.read_csv('raw_data/acct_alert.csv')
        train_count = len(alert_df)
    except:
        train_count = None
    
    print(f"\n{'='*70}")
    print(f"📊 不同閾值下的預測結果比較")
    print(f"{'='*70}")
    
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    print(f"\n{'閾值':<8} {'異常數':<10} {'異常率':<12} {'相對訓練集':<15} {'建議'}")
    print("-" * 70)
    
    best_threshold = None
    best_score = float('inf')
    
    for thresh in thresholds:
        pred_label = (df['confidence_score'] > thresh).astype(int)
        anomaly_count = pred_label.sum()
        anomaly_rate = anomaly_count / len(df) * 100
        
        suggestion = ""
        if train_count:
            ratio = anomaly_count / train_count
            
            # 找出最接近 1.0-1.3 倍的閾值
            ideal_distance = abs(ratio - 1.15)  # 1.15 是理想倍數
            if ideal_distance < best_score:
                best_score = ideal_distance
                best_threshold = thresh
            
            ratio_str = f"{ratio:.2f}x"
            
            if 0.9 <= ratio <= 1.3:
                suggestion = "✅ 推薦"
            elif 0.7 <= ratio < 0.9:
                suggestion = "⚠️  偏低"
            elif 1.3 < ratio <= 1.8:
                suggestion = "⚠️  稍高"
            elif ratio > 1.8:
                suggestion = "❌ 過高"
            else:
                suggestion = "❌ 過低"
        else:
            ratio_str = "N/A"
        
        print(f"{thresh:<8.2f} {anomaly_count:<10,} {anomaly_rate:<11.2f}% {ratio_str:<15} {suggestion}")
    
    if best_threshold:
        print(f"\n💡 建議使用閾值: {best_threshold}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    import sys
    
    # 預設檔案路徑
    input_file = "output/predictions.csv"
    output_file = "output/predictions_adjusted.csv"
    
    # 先顯示比較
    print("🔍 分析不同閾值的影響...\n")
    compare_thresholds(input_file)
    
    # 根據命令列參數或預設值調整
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    else:
        threshold = 0.80  # 根據分析結果，0.80 是較好的選擇
    
    print(f"\n\n🔧 應用閾值調整...")
    adjust_predictions_threshold(input_file, output_file, threshold)
    
    print(f"\n" + "="*70)
    print("✨ 完成！")
    print("="*70)
    print(f"\n請上傳 'output/predictions_adjusted_acct_label.csv' 到 leaderboard")
    print(f"預期 F1-Score 應該會有顯著提升！")

