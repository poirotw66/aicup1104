#!/usr/bin/env python3
"""
比較新舊模型的預測結果
"""

import pandas as pd
import numpy as np


def compare_predictions():
    """比較新舊模型預測結果"""
    
    print("="*70)
    print("📊 新舊模型預測結果比較")
    print("="*70)
    
    # 讀取資料
    print("\n載入資料...")
    
    # 舊模型（規則基礎）
    old_pred = pd.read_csv('output/predictions.csv')
    
    # 新模型（機器學習）
    new_pred = pd.read_csv('output/predictions_ml.csv')
    
    # 訓練標籤
    alert_df = pd.read_csv('raw_data/acct_alert.csv')
    n_train_alerts = len(alert_df)
    
    print(f"✓ 舊模型預測: {len(old_pred):,} 筆")
    print(f"✓ 新模型預測: {len(new_pred):,} 筆")
    print(f"✓ 訓練異常數: {n_train_alerts:,} 筆")
    
    # 統計比較
    print("\n" + "="*70)
    print("1️⃣  預測分布比較")
    print("="*70)
    
    old_alerts = (old_pred['label'] == 1).sum()
    new_alerts = (new_pred['label'] == 1).sum()
    
    print(f"\n{'指標':<25} {'舊模型':<20} {'新模型':<20} {'變化'}")
    print("-"*70)
    print(f"{'預測異常數':<25} {old_alerts:<20,} {new_alerts:<20,} {new_alerts-old_alerts:+,}")
    print(f"{'預測異常率':<25} {old_alerts/len(old_pred)*100:<19.2f}% {new_alerts/len(new_pred)*100:<19.2f}% {(new_alerts/len(new_pred) - old_alerts/len(old_pred))*100:+.2f}%")
    print(f"{'相對訓練集倍數':<25} {old_alerts/n_train_alerts:<19.2f}x {new_alerts/n_train_alerts:<19.2f}x {new_alerts/n_train_alerts - old_alerts/n_train_alerts:+.2f}x")
    
    # 信心分數比較
    print("\n" + "="*70)
    print("2️⃣  信心分數比較")
    print("="*70)
    
    print(f"\n{'統計量':<25} {'舊模型':<20} {'新模型'}")
    print("-"*70)
    print(f"{'平均':<25} {old_pred['confidence_score'].mean():<20.4f} {new_pred['confidence_score'].mean():<20.4f}")
    print(f"{'中位數':<25} {old_pred['confidence_score'].median():<20.4f} {new_pred['confidence_score'].median():<20.4f}")
    print(f"{'標準差':<25} {old_pred['confidence_score'].std():<20.4f} {new_pred['confidence_score'].std():<20.4f}")
    print(f"{'最小值':<25} {old_pred['confidence_score'].min():<20.4f} {new_pred['confidence_score'].min():<20.4f}")
    print(f"{'最大值':<25} {old_pred['confidence_score'].max():<20.4f} {new_pred['confidence_score'].max():<20.4f}")
    
    # 預測一致性
    print("\n" + "="*70)
    print("3️⃣  預測一致性分析")
    print("="*70)
    
    # 合併資料
    merged = old_pred[['acct', 'label']].merge(
        new_pred[['acct', 'label']], 
        on='acct', 
        suffixes=('_old', '_new')
    )
    
    both_alert = ((merged['label_old'] == 1) & (merged['label_new'] == 1)).sum()
    both_normal = ((merged['label_old'] == 0) & (merged['label_new'] == 0)).sum()
    old_only = ((merged['label_old'] == 1) & (merged['label_new'] == 0)).sum()
    new_only = ((merged['label_old'] == 0) & (merged['label_new'] == 1)).sum()
    
    print(f"\n兩模型都預測為異常: {both_alert:,} ({both_alert/len(merged)*100:.2f}%)")
    print(f"兩模型都預測為正常: {both_normal:,} ({both_normal/len(merged)*100:.2f}%)")
    print(f"僅舊模型預測異常: {old_only:,} ({old_only/len(merged)*100:.2f}%)")
    print(f"僅新模型預測異常: {new_only:,} ({new_only/len(merged)*100:.2f}%)")
    
    agreement = (both_alert + both_normal) / len(merged) * 100
    print(f"\n一致率: {agreement:.2f}%")
    
    # 高信心預測
    print("\n" + "="*70)
    print("4️⃣  高信心預測 (confidence > 0.8)")
    print("="*70)
    
    old_high_conf = (old_pred['confidence_score'] > 0.8).sum()
    new_high_conf = (new_pred['confidence_score'] > 0.8).sum()
    
    old_high_alert = old_pred[old_pred['confidence_score'] > 0.8]['label'].sum()
    new_high_alert = new_pred[new_pred['confidence_score'] > 0.8]['label'].sum()
    
    print(f"\n舊模型高信心預測: {old_high_conf:,}")
    print(f"  其中異常: {old_high_alert:,} ({old_high_alert/old_high_conf*100 if old_high_conf > 0 else 0:.1f}%)")
    
    print(f"\n新模型高信心預測: {new_high_conf:,}")
    print(f"  其中異常: {new_high_alert:,} ({new_high_alert/new_high_conf*100 if new_high_conf > 0 else 0:.1f}%)")
    
    # 分數分布
    print("\n" + "="*70)
    print("5️⃣  異常預測的信心分數分布")
    print("="*70)
    
    old_alert_scores = old_pred[old_pred['label'] == 1]['confidence_score']
    new_alert_scores = new_pred[new_pred['label'] == 1]['confidence_score']
    
    print(f"\n舊模型異常預測信心分數:")
    print(f"  平均: {old_alert_scores.mean():.4f}")
    print(f"  中位數: {old_alert_scores.median():.4f}")
    print(f"  最低: {old_alert_scores.min():.4f}")
    
    print(f"\n新模型異常預測信心分數:")
    print(f"  平均: {new_alert_scores.mean():.4f}")
    print(f"  中位數: {new_alert_scores.median():.4f}")
    print(f"  最低: {new_alert_scores.min():.4f}")
    
    # 建議
    print("\n" + "="*70)
    print("💡 建議")
    print("="*70)
    
    print("\n基於以上分析：")
    
    if new_alerts/n_train_alerts > 0.8 and new_alerts/n_train_alerts < 1.3:
        print("✅ 新模型預測異常率在合理範圍內 (0.8-1.3x 訓練集)")
    elif new_alerts/n_train_alerts > 1.5:
        print("⚠️  新模型預測偏高，建議調高閾值")
    else:
        print("⚠️  新模型預測偏低，可能錯過一些異常")
    
    if agreement > 80:
        print("✅ 兩模型預測一致性高")
    elif agreement < 50:
        print("⚠️  兩模型分歧較大，建議檢查差異原因")
    
    print("\n建議上傳:")
    print("  1. output/predictions_ml_acct_label.csv (新模型) - 優先推薦")
    print("  2. 如果新模型效果不佳，可嘗試調整閾值")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    compare_predictions()

