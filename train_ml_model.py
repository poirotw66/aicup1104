#!/usr/bin/env python3
"""
修正版機器學習模型訓練
解決資料洩漏問題，使用適當的訓練/測試分離
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """載入資料"""
    print("="*70)
    print("載入資料...")
    print("="*70)
    
    # 載入特徵
    features_df = pd.read_csv('output/features.csv')
    print(f"總特徵數: {len(features_df):,} 帳戶, {len(features_df.columns)-1} 特徵")
    
    # 載入標籤
    alert_df = pd.read_csv('raw_data/acct_alert.csv')
    alert_accounts = set(alert_df['acct'].values)
    print(f"異常帳戶數: {len(alert_accounts):,}")
    
    # 載入預測目標
    predict_df = pd.read_csv('raw_data/acct_predict.csv')
    predict_accounts = set(predict_df['acct'].values)
    print(f"預測目標數: {len(predict_accounts):,}")
    
    return features_df, alert_accounts, predict_accounts


def prepare_training_data(features_df, alert_accounts, predict_accounts, normal_sample_size=5000):
    """
    準備訓練資料（不包含預測目標，避免資料洩漏）
    
    Args:
        features_df: 特徵資料
        alert_accounts: 異常帳戶集合
        predict_accounts: 預測目標集合
        normal_sample_size: 正常樣本數量
    """
    print("\n" + "="*70)
    print("準備訓練資料（修正資料洩漏問題）...")
    print("="*70)
    
    # 1. 分離異常帳戶
    alert_df = features_df[features_df['acct'].isin(alert_accounts)].copy()
    print(f"\n異常樣本: {len(alert_df):,}")
    
    # 2. 找出可用的正常帳戶（排除預測目標）
    available_normal = features_df[
        ~features_df['acct'].isin(alert_accounts) &
        ~features_df['acct'].isin(predict_accounts)
    ].copy()
    print(f"可用正常樣本: {len(available_normal):,}")
    
    # 3. 從正常帳戶中抽樣
    if len(available_normal) > normal_sample_size:
        normal_sample = available_normal.sample(n=normal_sample_size, random_state=42)
    else:
        normal_sample = available_normal
    print(f"抽樣正常樣本: {len(normal_sample):,}")
    
    # 4. 合併訓練集
    train_df = pd.concat([alert_df, normal_sample], ignore_index=True)
    train_df['label'] = train_df['acct'].isin(alert_accounts).astype(int)
    
    # 5. 準備特徵和標籤
    feature_cols = [col for col in train_df.columns if col not in ['acct', 'label', 'is_alert']]
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    
    print(f"\n訓練集統計:")
    print(f"  總樣本: {len(train_df):,}")
    print(f"  異常: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  正常: {len(y_train)-y_train.sum():,} ({(len(y_train)-y_train.sum())/len(y_train)*100:.2f}%)")
    print(f"  特徵數: {len(feature_cols)}")
    
    # 檢查是否有資料洩漏
    leak_check = train_df['acct'].isin(predict_accounts).sum()
    if leak_check > 0:
        print(f"\n⚠️  警告: 訓練集中有 {leak_check} 個預測目標帳戶！")
    else:
        print(f"\n✅ 確認: 訓練集中沒有預測目標帳戶")
    
    return X_train, y_train, train_df, feature_cols


def train_models(X_train, y_train, feature_cols):
    """訓練多個模型並比較"""
    print("\n" + "="*70)
    print("訓練模型...")
    print("="*70)
    
    models = {}
    cv_results = {}
    
    # 設定交叉驗證
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 模型 1: Random Forest
    print("\n1️⃣  Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1')
    print(f"   CV F1-Score: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
    
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    cv_results['random_forest'] = rf_scores
    
    # 模型 2: Gradient Boosting
    print("\n2️⃣  Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    gb_scores = cross_val_score(gb, X_train, y_train, cv=cv, scoring='f1')
    print(f"   CV F1-Score: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
    
    gb.fit(X_train, y_train)
    models['gradient_boosting'] = gb
    cv_results['gradient_boosting'] = gb_scores
    
    # 模型 3: Random Forest (更保守)
    print("\n3️⃣  Random Forest (Conservative)...")
    rf_cons = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    rf_cons_scores = cross_val_score(rf_cons, X_train, y_train, cv=cv, scoring='f1')
    print(f"   CV F1-Score: {rf_cons_scores.mean():.4f} (+/- {rf_cons_scores.std():.4f})")
    
    rf_cons.fit(X_train, y_train)
    models['rf_conservative'] = rf_cons
    cv_results['rf_conservative'] = rf_cons_scores
    
    # 選擇最佳模型
    best_model_name = max(cv_results, key=lambda k: cv_results[k].mean())
    best_model = models[best_model_name]
    best_score = cv_results[best_model_name].mean()
    
    print(f"\n🏆 最佳模型: {best_model_name} (CV F1: {best_score:.4f})")
    
    return best_model, best_model_name, models, cv_results, feature_cols


def evaluate_on_training(model, X_train, y_train):
    """在訓練集上評估（用於檢查過擬合）"""
    print("\n" + "="*70)
    print("訓練集評估...")
    print("="*70)
    
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    
    print(f"\nPrecision: {precision_score(y_train, y_pred):.4f}")
    print(f"Recall: {recall_score(y_train, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_train, y_pred):.4f}")
    
    cm = confusion_matrix(y_train, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:,}, FP: {cm[0][1]:,}")
    print(f"  FN: {cm[1][0]:,}, TP: {cm[1][1]:,}")
    
    return {
        'precision': precision_score(y_train, y_pred),
        'recall': recall_score(y_train, y_pred),
        'f1_score': f1_score(y_train, y_pred),
        'confusion_matrix': cm.tolist()
    }


def predict_and_save(model, features_df, predict_accounts, feature_cols, output_path='output/predictions_ml.csv'):
    """對預測目標進行預測"""
    print("\n" + "="*70)
    print("生成預測...")
    print("="*70)
    
    # 篩選預測目標
    predict_df = features_df[features_df['acct'].isin(predict_accounts)].copy()
    print(f"預測目標數: {len(predict_df):,}")
    
    # 準備特徵
    X_pred = predict_df[feature_cols]
    
    # 預測
    y_pred = model.predict(X_pred)
    y_proba = model.predict_proba(X_pred)[:, 1]
    
    # 保存結果
    result_df = pd.DataFrame({
        'acct': predict_df['acct'].values,
        'label': y_pred,
        'confidence_score': y_proba
    })
    
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ 預測結果已保存: {output_path}")
    
    # 統計
    n_alert = result_df['label'].sum()
    print(f"\n預測統計:")
    print(f"  預測為異常: {n_alert:,} ({n_alert/len(result_df)*100:.2f}%)")
    print(f"  預測為正常: {len(result_df)-n_alert:,} ({(len(result_df)-n_alert)/len(result_df)*100:.2f}%)")
    print(f"  平均信心分數: {result_df['confidence_score'].mean():.4f}")
    
    # 也生成簡化版
    simple_output = output_path.replace('.csv', '_acct_label.csv')
    result_df[['acct', 'label']].to_csv(simple_output, index=False)
    print(f"✅ 簡化版已保存: {simple_output}")
    
    return result_df


def analyze_feature_importance(model, feature_cols, top_n=20):
    """分析特徵重要性"""
    print("\n" + "="*70)
    print(f"Top {top_n} 重要特徵:")
    print("="*70)
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + importance_df.head(top_n).to_string(index=False))
        
        importance_df.to_csv('output/ml_feature_importance.csv', index=False)
        print(f"\n✅ 特徵重要性已保存: output/ml_feature_importance.csv")
        
        return importance_df
    else:
        print("模型不支援特徵重要性分析")
        return None


def main():
    """主函數"""
    print("\n" + "="*70)
    print("🔧 修正版機器學習模型訓練")
    print("   解決資料洩漏問題")
    print("="*70)
    
    # 1. 載入資料
    features_df, alert_accounts, predict_accounts = load_data()
    
    # 2. 準備訓練資料（不包含預測目標）
    X_train, y_train, train_df, feature_cols = prepare_training_data(
        features_df, alert_accounts, predict_accounts,
        normal_sample_size=5000  # 可調整
    )
    
    # 3. 訓練模型
    best_model, best_model_name, all_models, cv_results, feature_cols = train_models(
        X_train, y_train, feature_cols
    )
    
    # 4. 評估訓練集表現
    train_metrics = evaluate_on_training(best_model, X_train, y_train)
    
    # 5. 分析特徵重要性
    importance_df = analyze_feature_importance(best_model, feature_cols)
    
    # 6. 對預測目標進行預測
    result_df = predict_and_save(
        best_model, features_df, predict_accounts, feature_cols,
        output_path='output/predictions_ml.csv'
    )
    
    # 7. 保存模型
    model_path = 'output/trained_model.pkl'
    joblib.dump({
        'model': best_model,
        'model_name': best_model_name,
        'feature_cols': feature_cols,
        'train_metrics': train_metrics,
        'cv_results': {k: v.tolist() for k, v in cv_results.items()}
    }, model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 8. 生成報告
    report = {
        'model_name': best_model_name,
        'cv_f1_mean': float(cv_results[best_model_name].mean()),
        'cv_f1_std': float(cv_results[best_model_name].std()),
        'train_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                         for k, v in train_metrics.items()},
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_train_alerts': int(y_train.sum()),
        'n_predictions': len(result_df),
        'n_predicted_alerts': int(result_df['label'].sum()),
        'prediction_rate': float(result_df['label'].sum() / len(result_df))
    }
    
    with open('output/ml_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ 訓練報告已保存: output/ml_training_report.json")
    
    print("\n" + "="*70)
    print("✨ 訓練完成！")
    print("="*70)
    print(f"\n建議上傳: output/predictions_ml_acct_label.csv")
    print(f"預期 F1-Score 應該會大幅提升（從 0.07 到 0.25+）")
    
    return best_model, result_df


if __name__ == "__main__":
    best_model, result_df = main()

