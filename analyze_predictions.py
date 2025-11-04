#!/usr/bin/env python3
"""
分析預測結果和建議改善方向
"""

import pandas as pd
import numpy as np


def analyze_predictions():
    """分析預測結果並提供改善建議"""
    
    # 讀取預測結果
    pred_df = pd.read_csv('output/predictions_acct_label.csv')
    
    # 讀取訓練標籤
    alert_df = pd.read_csv('raw_data/acct_alert.csv')
    
    print("=" * 60)
    print("📊 預測結果分析")
    print("=" * 60)
    
    # 1. 基本統計
    print("\n1️⃣  基本統計:")
    print(f"   預測總數: {len(pred_df):,}")
    print(f"   訓練異常數: {len(alert_df):,}")
    print(f"   異常比例(訓練): {len(alert_df)/len(pred_df)*100:.2f}%")
    
    # 2. 預測分布
    print("\n2️⃣  預測標籤分布:")
    label_counts = pred_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(pred_df) * 100
        print(f"   Label {label}: {count:,} ({pct:.2f}%)")
    
    # 3. 預測異常比例
    pred_anomaly_rate = (pred_df['label'] == 1).sum() / len(pred_df)
    train_anomaly_rate = len(alert_df) / len(pred_df)
    
    print("\n3️⃣  異常比例對比:")
    print(f"   訓練集異常率: {train_anomaly_rate*100:.2f}%")
    print(f"   預測異常率: {pred_anomaly_rate*100:.2f}%")
    print(f"   倍數差異: {pred_anomaly_rate/train_anomaly_rate:.2f}x")
    
    if pred_anomaly_rate > train_anomaly_rate * 2:
        print("   ⚠️  警告: 預測異常率過高，可能導致大量誤報！")
    
    # 4. 建議
    print("\n" + "=" * 60)
    print("💡 改善建議")
    print("=" * 60)
    
    suggestions = []
    
    if pred_anomaly_rate > 0.5:
        suggestions.append({
            'priority': '🔴 高',
            'issue': '預測異常率 > 50%',
            'suggestion': '調整決策閾值 (threshold)，提高判定為異常的標準'
        })
    
    if pred_anomaly_rate > train_anomaly_rate * 2:
        suggestions.append({
            'priority': '🔴 高',
            'issue': f'預測異常率是訓練集的 {pred_anomaly_rate/train_anomaly_rate:.1f} 倍',
            'suggestion': '模型過度敏感，建議:\n' +
                         '      - 調整 class_weight 參數\n' +
                         '      - 使用 predict_proba 並設定更高的閾值 (如 0.7-0.8)\n' +
                         '      - 增加模型正則化 (降低 max_depth, 增加 min_samples_leaf)'
        })
    
    if len(suggestions) == 0:
        suggestions.append({
            'priority': '🟡 中',
            'issue': 'F1-Score 仍然偏低',
            'suggestion': '建議檢查:\n' +
                         '      - 特徵工程是否充分\n' +
                         '      - 是否有特徵洩漏\n' +
                         '      - 模型複雜度是否適當'
        })
    
    for i, sugg in enumerate(suggestions, 1):
        print(f"\n{i}. {sugg['priority']} - {sugg['issue']}")
        print(f"   建議: {sugg['suggestion']}")
    
    # 5. 快速修正建議
    print("\n" + "=" * 60)
    print("🔧 快速修正方案")
    print("=" * 60)
    
    optimal_threshold = train_anomaly_rate * 1.5  # 稍微寬鬆一點
    print(f"\n建議設定異常判定閾值為: {optimal_threshold:.3f}")
    print(f"（目前使用預設 0.5，建議改為 0.7-0.8）")
    
    print("\n範例代碼:")
    print("```python")
    print("# 在模型預測時使用機率閾值")
    print("y_proba = model.predict_proba(X)[:, 1]")
    print("threshold = 0.75  # 調整此值")
    print("y_pred = (y_proba > threshold).astype(int)")
    print("```")
    
    # 6. 生成不同閾值下的預測統計（如果有 confidence_score）
    try:
        full_pred = pd.read_csv('output/predictions.csv')
        if 'confidence_score' in full_pred.columns:
            print("\n" + "=" * 60)
            print("📈 不同閾值下的預測分布")
            print("=" * 60)
            
            thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
            print(f"\n{'閾值':<10} {'預測異常數':<15} {'異常率':<10} {'相對訓練集'}")
            print("-" * 60)
            
            for thresh in thresholds:
                pred_count = (full_pred['confidence_score'] > thresh).sum()
                pred_rate = pred_count / len(full_pred) * 100
                ratio = pred_count / len(alert_df)
                
                marker = ""
                if 0.8 <= ratio <= 1.5:
                    marker = " ← 建議範圍"
                elif ratio > 2:
                    marker = " (過高)"
                
                print(f"{thresh:<10.2f} {pred_count:<15,} {pred_rate:<9.2f}% {ratio:<10.2f}x{marker}")
    
    except Exception as e:
        pass
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    analyze_predictions()

