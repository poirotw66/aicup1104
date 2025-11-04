"""
EDA Analysis Module
探索性資料分析 - 比較警示帳戶與正常帳戶的特徵差異
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    計算 Cohen's d 效應量
    
    Args:
        group1: 第一組資料
        group2: 第二組資料
        
    Returns:
        Cohen's d 值
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return 0.0
    
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def compare_features(
    features_df: pd.DataFrame,
    alert_accounts: set,
    output_path: str = 'output/feature_comparison.csv'
) -> pd.DataFrame:
    """
    比較警示帳戶與正常帳戶的特徵差異
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        output_path: 輸出檔案路徑
        
    Returns:
        比較結果 DataFrame
    """
    print("Comparing features between alert and normal accounts...")
    
    # 標記警示帳戶
    features_df['is_alert'] = features_df['acct'].isin(alert_accounts).astype(int)
    
    # 分離警示帳戶和正常帳戶
    alert_df = features_df[features_df['is_alert'] == 1]
    normal_df = features_df[features_df['is_alert'] == 0]
    
    print(f"Alert accounts: {len(alert_df)}")
    print(f"Normal accounts: {len(normal_df)}")
    
    # 獲取所有特徵欄位（排除 acct 和 is_alert）
    feature_cols = [col for col in features_df.columns if col not in ['acct', 'is_alert']]
    
    comparison_results = []
    
    for feature in feature_cols:
        alert_values = alert_df[feature].values
        normal_values = normal_df[feature].values
        
        # 基本統計量
        alert_mean = np.mean(alert_values)
        alert_median = np.median(alert_values)
        alert_std = np.std(alert_values)
        
        normal_mean = np.mean(normal_values)
        normal_median = np.median(normal_values)
        normal_std = np.std(normal_values)
        
        # 差異倍數
        if normal_mean != 0:
            diff_ratio = alert_mean / normal_mean
        else:
            diff_ratio = float('inf') if alert_mean > 0 else 1.0
        
        # 統計檢驗（Mann-Whitney U test，適用於非正態分布）
        try:
            statistic, p_value = mannwhitneyu(alert_values, normal_values, alternative='two-sided')
        except:
            p_value = 1.0
        
        # 效應量
        effect_size = cohen_d(alert_values, normal_values)
        
        # 顯著性標記
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = ''
        
        comparison_results.append({
            'feature': feature,
            'alert_mean': alert_mean,
            'alert_median': alert_median,
            'alert_std': alert_std,
            'normal_mean': normal_mean,
            'normal_median': normal_median,
            'normal_std': normal_std,
            'diff_ratio': diff_ratio,
            'p_value': p_value,
            'effect_size': effect_size,
            'significance': significance
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # 按效應量排序
    comparison_df = comparison_df.sort_values('effect_size', ascending=False, key=abs)
    
    # 儲存結果
    comparison_df.to_csv(output_path, index=False)
    print(f"Feature comparison saved to {output_path}")
    
    # 顯示 Top 15 差異最大的特徵
    print("\n=== Top 15 Features with Largest Differences ===")
    print(comparison_df[['feature', 'alert_mean', 'normal_mean', 'diff_ratio', 'effect_size', 'significance']].head(15).to_string())
    
    return comparison_df


def generate_statistical_summary(
    features_df: pd.DataFrame,
    alert_accounts: set,
    output_dir: str = 'output'
):
    """
    生成統計摘要
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        output_dir: 輸出目錄
    """
    print("\nGenerating statistical summary...")
    
    # 標記警示帳戶
    features_df['is_alert'] = features_df['acct'].isin(alert_accounts).astype(int)
    
    # 分離資料
    alert_df = features_df[features_df['is_alert'] == 1]
    normal_df = features_df[features_df['is_alert'] == 0]
    
    # 儲存警示帳戶和正常帳戶的統計量
    alert_stats = alert_df.describe()
    normal_stats = normal_df.describe()
    
    alert_stats.to_csv(f'{output_dir}/alert_features.csv')
    normal_stats.to_csv(f'{output_dir}/normal_features.csv')
    
    print(f"Alert features statistics saved to {output_dir}/alert_features.csv")
    print(f"Normal features statistics saved to {output_dir}/normal_features.csv")


def identify_significant_features(
    comparison_df: pd.DataFrame,
    p_threshold: float = 0.05,
    effect_size_threshold: float = 0.5
) -> List[str]:
    """
    識別統計顯著的特徵
    
    Args:
        comparison_df: 特徵比較 DataFrame
        p_threshold: p-value 閾值
        effect_size_threshold: 效應量閾值
        
    Returns:
        顯著特徵列表
    """
    significant_features = comparison_df[
        (comparison_df['p_value'] < p_threshold) &
        (abs(comparison_df['effect_size']) > effect_size_threshold)
    ]['feature'].tolist()
    
    print(f"\n=== {len(significant_features)} Significant Features Identified ===")
    print(f"(p-value < {p_threshold} and |effect_size| > {effect_size_threshold})")
    for feature in significant_features[:20]:  # 顯示前 20 個
        print(f"  - {feature}")
    if len(significant_features) > 20:
        print(f"  ... and {len(significant_features) - 20} more")
    
    return significant_features


def plot_feature_distributions(
    features_df: pd.DataFrame,
    alert_accounts: set,
    top_features: List[str],
    output_dir: str = 'output',
    max_plots: int = 10
):
    """
    繪製關鍵特徵的分布圖
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        top_features: 要繪製的特徵列表
        output_dir: 輸出目錄
        max_plots: 最多繪製幾個特徵
    """
    print(f"\nGenerating distribution plots for top {max_plots} features...")
    
    # 標記警示帳戶
    features_df['is_alert'] = features_df['acct'].isin(alert_accounts).astype(int)
    
    # 只繪製前 N 個特徵
    features_to_plot = top_features[:max_plots]
    
    # 設定圖形大小
    n_features = len(features_to_plot)
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 5*n_features))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for idx, feature in enumerate(features_to_plot):
        # 左側：直方圖
        ax1 = axes[idx, 0]
        alert_values = features_df[features_df['is_alert'] == 1][feature]
        normal_values = features_df[features_df['is_alert'] == 0][feature]
        
        # 過濾極端值以便更好的視覺化
        alert_q99 = alert_values.quantile(0.99)
        normal_q99 = normal_values.quantile(0.99)
        max_val = max(alert_q99, normal_q99)
        
        ax1.hist(alert_values[alert_values <= max_val], bins=50, alpha=0.6, label='Alert', color='red')
        ax1.hist(normal_values[normal_values <= max_val], bins=50, alpha=0.6, label='Normal', color='blue')
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution: {feature}')
        ax1.legend()
        
        # 右側：箱型圖
        ax2 = axes[idx, 1]
        data_to_plot = [
            alert_values[alert_values <= max_val],
            normal_values[normal_values <= max_val]
        ]
        ax2.boxplot(data_to_plot, labels=['Alert', 'Normal'])
        ax2.set_ylabel(feature)
        ax2.set_title(f'Box Plot: {feature}')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=100, bbox_inches='tight')
    print(f"Distribution plots saved to {output_dir}/feature_distributions.png")
    plt.close()


def generate_correlation_heatmap(
    features_df: pd.DataFrame,
    alert_accounts: set,
    top_features: List[str],
    output_dir: str = 'output',
    max_features: int = 20
):
    """
    生成相關性熱力圖
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        top_features: 關鍵特徵列表
        output_dir: 輸出目錄
        max_features: 最多包含幾個特徵
    """
    print(f"\nGenerating correlation heatmap...")
    
    # 只選擇警示帳戶
    alert_df = features_df[features_df['acct'].isin(alert_accounts)]
    
    # 選擇 top features
    features_to_plot = top_features[:max_features]
    
    # 計算相關性
    correlation_matrix = alert_df[features_to_plot].corr()
    
    # 繪製熱力圖
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap (Alert Accounts)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=100, bbox_inches='tight')
    print(f"Correlation heatmap saved to {output_dir}/correlation_heatmap.png")
    plt.close()


def run_eda_analysis(
    features_df: pd.DataFrame,
    alert_accounts: set,
    output_dir: str = 'output'
) -> Dict:
    """
    執行完整的 EDA 分析
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        output_dir: 輸出目錄
        
    Returns:
        分析結果字典
    """
    print("="*60)
    print("Starting Exploratory Data Analysis")
    print("="*60)
    
    # 1. 特徵比較
    comparison_df = compare_features(features_df, alert_accounts, f'{output_dir}/feature_comparison.csv')
    
    # 2. 統計摘要
    generate_statistical_summary(features_df, alert_accounts, output_dir)
    
    # 3. 識別顯著特徵
    significant_features = identify_significant_features(comparison_df)
    
    # 4. 繪製分布圖
    try:
        plot_feature_distributions(features_df, alert_accounts, significant_features, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate distribution plots: {e}")
    
    # 5. 相關性分析
    try:
        generate_correlation_heatmap(features_df, alert_accounts, significant_features, output_dir)
    except Exception as e:
        print(f"Warning: Could not generate correlation heatmap: {e}")
    
    print("\n" + "="*60)
    print("EDA Analysis Completed!")
    print("="*60)
    
    return {
        'comparison_df': comparison_df,
        'significant_features': significant_features
    }


if __name__ == "__main__":
    print("EDA analysis module ready!")

