"""
Pattern Discovery Module
模式發現 - 使用決策樹、聚類等方法發現警示帳戶的行為模式
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Dict
import json


def train_interpretable_tree(
    features_df: pd.DataFrame,
    alert_accounts: set,
    max_depth: int = 4,
    output_dir: str = 'output'
) -> DecisionTreeClassifier:
    """
    訓練淺層決策樹用於理解決策邏輯
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        max_depth: 決策樹最大深度
        output_dir: 輸出目錄
        
    Returns:
        訓練好的決策樹模型
    """
    print(f"\nTraining interpretable decision tree (max_depth={max_depth})...")
    
    # 準備資料
    features_df['is_alert'] = features_df['acct'].isin(alert_accounts).astype(int)
    
    feature_cols = [col for col in features_df.columns if col not in ['acct', 'is_alert']]
    X = features_df[feature_cols]
    y = features_df['is_alert']
    
    # 訓練決策樹
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=50,
        min_samples_split=100,
        random_state=42,
        class_weight='balanced'
    )
    tree.fit(X, y)
    
    # 計算準確度
    train_score = tree.score(X, y)
    print(f"Training accuracy: {train_score:.4f}")
    
    # 視覺化決策樹
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(tree, feature_names=feature_cols, class_names=['Normal', 'Alert'],
                 filled=True, fontsize=8, rounded=True)
        plt.title('Decision Tree Visualization')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/decision_tree.png', dpi=150, bbox_inches='tight')
        print(f"Decision tree visualization saved to {output_dir}/decision_tree.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate tree visualization: {e}")
    
    # 輸出文字格式的決策規則
    tree_rules = export_text(tree, feature_names=feature_cols)
    with open(f'{output_dir}/decision_tree_rules.txt', 'w') as f:
        f.write(tree_rules)
    print(f"Decision tree rules saved to {output_dir}/decision_tree_rules.txt")
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 15 Important Features from Decision Tree ===")
    print(feature_importance.head(15).to_string())
    
    feature_importance.to_csv(f'{output_dir}/tree_feature_importance.csv', index=False)
    
    return tree


def extract_rules_from_tree(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    min_samples: int = 50
) -> List[Dict]:
    """
    從決策樹中提取規則
    
    Args:
        tree: 決策樹模型
        feature_names: 特徵名稱列表
        min_samples: 最小樣本數要求
        
    Returns:
        規則列表
    """
    tree_structure = tree.tree_
    rules = []
    
    def recurse(node, conditions):
        if tree_structure.n_node_samples[node] < min_samples:
            return
        
        if tree_structure.children_left[node] == tree_structure.children_right[node]:
            # 葉子節點
            class_value = tree_structure.value[node][0]
            if class_value[1] > class_value[0]:  # 預測為警示帳戶
                support = tree_structure.n_node_samples[node]
                confidence = class_value[1] / (class_value[0] + class_value[1])
                
                rules.append({
                    'conditions': conditions.copy(),
                    'support': support,
                    'confidence': confidence,
                    'class': 'alert'
                })
        else:
            # 內部節點
            feature = feature_names[tree_structure.feature[node]]
            threshold = tree_structure.threshold[node]
            
            # 左子樹（<=）
            left_conditions = conditions + [(feature, '<=', threshold)]
            recurse(tree_structure.children_left[node], left_conditions)
            
            # 右子樹（>）
            right_conditions = conditions + [(feature, '>', threshold)]
            recurse(tree_structure.children_right[node], right_conditions)
    
    recurse(0, [])
    
    # 按信心度排序
    rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
    
    print(f"\n=== Extracted {len(rules)} Rules from Decision Tree ===")
    for i, rule in enumerate(rules[:5]):  # 顯示前 5 條
        print(f"\nRule {i+1}:")
        print(f"  Conditions: {rule['conditions']}")
        print(f"  Support: {rule['support']}")
        print(f"  Confidence: {rule['confidence']:.4f}")
    
    return rules


def cluster_alert_accounts(
    features_df: pd.DataFrame,
    alert_accounts: set,
    n_clusters: int = 4,
    output_dir: str = 'output'
) -> Dict:
    """
    對警示帳戶進行聚類分析
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        n_clusters: 聚類數量
        output_dir: 輸出目錄
        
    Returns:
        聚類結果
    """
    print(f"\nPerforming clustering analysis ({n_clusters} clusters)...")
    
    # 只選擇警示帳戶
    alert_df = features_df[features_df['acct'].isin(alert_accounts)].copy()
    
    if len(alert_df) < n_clusters:
        print(f"Warning: Not enough alert accounts for {n_clusters} clusters")
        return {}
    
    feature_cols = [col for col in alert_df.columns if col not in ['acct', 'is_alert']]
    X = alert_df[feature_cols]
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means 聚類
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    alert_df['cluster'] = clusters
    
    # 分析每個聚類的特徵均值
    cluster_profiles = []
    
    print("\n=== Cluster Profiles ===")
    for cluster_id in range(n_clusters):
        cluster_data = alert_df[alert_df['cluster'] == cluster_id]
        cluster_size = len(cluster_data)
        
        print(f"\nCluster {cluster_id} (Size: {cluster_size}, {cluster_size/len(alert_df)*100:.1f}%):")
        
        # 計算該聚類的特徵均值
        profile = cluster_data[feature_cols].mean()
        
        # 找出該聚類最顯著的特徵（與整體警示帳戶均值差異最大）
        overall_mean = alert_df[feature_cols].mean()
        diff = abs(profile - overall_mean) / (overall_mean + 1e-10)
        top_features = diff.nlargest(5)
        
        print("  Top distinctive features:")
        for feature, diff_ratio in top_features.items():
            print(f"    - {feature}: {profile[feature]:.4f} (diff: {diff_ratio:.2f}x)")
        
        cluster_profiles.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'percentage': cluster_size/len(alert_df)*100,
            'top_features': top_features.to_dict()
        })
    
    # 儲存聚類結果
    alert_df[['acct', 'cluster']].to_csv(f'{output_dir}/alert_clusters.csv', index=False)
    print(f"\nCluster assignments saved to {output_dir}/alert_clusters.csv")
    
    return {
        'kmeans': kmeans,
        'scaler': scaler,
        'cluster_profiles': cluster_profiles,
        'cluster_assignments': alert_df[['acct', 'cluster']]
    }


def detect_anomalies(
    features_df: pd.DataFrame,
    alert_accounts: set,
    output_dir: str = 'output'
) -> Dict:
    """
    使用 Isolation Forest 檢測異常
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        output_dir: 輸出目錄
        
    Returns:
        異常檢測結果
    """
    print("\nPerforming anomaly detection...")
    
    feature_cols = [col for col in features_df.columns if col not in ['acct', 'is_alert']]
    X = features_df[feature_cols]
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    
    # 預測異常分數
    anomaly_scores = iso_forest.fit_predict(X)
    anomaly_scores_continuous = iso_forest.score_samples(X)
    
    features_df['anomaly_score'] = anomaly_scores_continuous
    features_df['is_anomaly'] = (anomaly_scores == -1).astype(int)
    features_df['is_alert'] = features_df['acct'].isin(alert_accounts).astype(int)
    
    # 分析警示帳戶的異常分數分布
    alert_anomaly_ratio = features_df[features_df['is_alert'] == 1]['is_anomaly'].mean()
    normal_anomaly_ratio = features_df[features_df['is_alert'] == 0]['is_anomaly'].mean()
    
    print(f"Alert accounts marked as anomalies: {alert_anomaly_ratio:.2%}")
    print(f"Normal accounts marked as anomalies: {normal_anomaly_ratio:.2%}")
    
    # 比較異常分數分布
    alert_scores = features_df[features_df['is_alert'] == 1]['anomaly_score']
    normal_scores = features_df[features_df['is_alert'] == 0]['anomaly_score']
    
    print(f"Alert accounts anomaly score - Mean: {alert_scores.mean():.4f}, Median: {alert_scores.median():.4f}")
    print(f"Normal accounts anomaly score - Mean: {normal_scores.mean():.4f}, Median: {normal_scores.median():.4f}")
    
    # 繪製異常分數分布
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(alert_scores, bins=50, alpha=0.6, label='Alert', color='red')
        plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/anomaly_score_distribution.png', dpi=100)
        print(f"Anomaly score distribution saved to {output_dir}/anomaly_score_distribution.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate anomaly score plot: {e}")
    
    return {
        'iso_forest': iso_forest,
        'alert_anomaly_ratio': alert_anomaly_ratio,
        'normal_anomaly_ratio': normal_anomaly_ratio
    }


def run_pattern_discovery(
    features_df: pd.DataFrame,
    alert_accounts: set,
    output_dir: str = 'output'
) -> Dict:
    """
    執行完整的模式發現流程
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        output_dir: 輸出目錄
        
    Returns:
        模式發現結果
    """
    print("="*60)
    print("Starting Pattern Discovery")
    print("="*60)
    
    # 1. 訓練決策樹
    tree = train_interpretable_tree(features_df, alert_accounts, max_depth=4, output_dir=output_dir)
    
    # 2. 提取規則
    feature_cols = [col for col in features_df.columns if col not in ['acct', 'is_alert']]
    rules = extract_rules_from_tree(tree, feature_cols)
    
    # 3. 聚類分析
    cluster_results = cluster_alert_accounts(features_df, alert_accounts, n_clusters=4, output_dir=output_dir)
    
    # 4. 異常檢測
    anomaly_results = detect_anomalies(features_df, alert_accounts, output_dir=output_dir)
    
    # 儲存規則到 JSON
    rules_to_save = []
    for rule in rules[:10]:  # 保存前 10 條規則
        rules_to_save.append({
            'conditions': [(f, op, float(th)) for f, op, th in rule['conditions']],
            'support': int(rule['support']),
            'confidence': float(rule['confidence'])
        })
    
    with open(f'{output_dir}/discovered_rules.json', 'w') as f:
        json.dump(rules_to_save, f, indent=2)
    print(f"\nDiscovered rules saved to {output_dir}/discovered_rules.json")
    
    print("\n" + "="*60)
    print("Pattern Discovery Completed!")
    print("="*60)
    
    return {
        'tree': tree,
        'rules': rules,
        'cluster_results': cluster_results,
        'anomaly_results': anomaly_results
    }


if __name__ == "__main__":
    print("Pattern discovery module ready!")

