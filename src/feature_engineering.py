"""
Feature Engineering Module
提取約 40 個特徵用於警示帳戶分析
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import gc
from typing import Dict, Set
from tqdm import tqdm


class FeatureExtractor:
    """特徵提取器"""
    
    def __init__(self, alert_accounts: Set[str]):
        """
        初始化特徵提取器
        
        Args:
            alert_accounts: 警示帳戶集合
        """
        self.alert_accounts = alert_accounts
        self.feature_dict = defaultdict(lambda: {
            # 基礎統計特徵
            'out_txn_count': 0,
            'in_txn_count': 0,
            'out_txn_sum': 0.0,
            'in_txn_sum': 0.0,
            'out_txn_amounts': [],
            'in_txn_amounts': [],
            
            # 時間特徵
            'txn_dates': set(),
            'txn_hours': [],
            'first_txn_date': float('inf'),
            'last_txn_date': 0,
            
            # 網絡特徵
            'out_partners': set(),
            'in_partners': set(),
            'self_txn_count': 0,
            'unk_count': 0,
            
            # 與警示帳戶互動
            'alert_out_count': 0,
            'alert_in_count': 0,
            'alert_out_amt': 0.0,
            'alert_in_amt': 0.0,
            
            # 帳戶類型
            'from_acct_types': [],
            'to_acct_types': [],
            
            # 管道類型
            'channel_types': [],
            
            # 交易日期分布
            'daily_txn_counts': defaultdict(int),
        })
    
    def process_chunk(self, chunk: pd.DataFrame):
        """
        處理一批交易資料
        
        Args:
            chunk: 交易資料批次
        """
        for _, row in chunk.iterrows():
            from_acct = row['from_acct']
            to_acct = row['to_acct']
            amt = float(row['txn_amt'])
            date = int(row['txn_date'])
            time_str = row['txn_time']
            
            # 解析時間
            try:
                hour = int(time_str.split(':')[0])
            except:
                hour = -1
            
            # 處理轉出帳戶
            self._update_out_features(from_acct, to_acct, amt, date, hour, row)
            
            # 處理轉入帳戶
            self._update_in_features(to_acct, from_acct, amt, date, hour, row)
    
    def _update_out_features(self, from_acct, to_acct, amt, date, hour, row):
        """更新轉出帳戶特徵"""
        features = self.feature_dict[from_acct]
        
        # 基礎統計
        features['out_txn_count'] += 1
        features['out_txn_sum'] += amt
        features['out_txn_amounts'].append(amt)
        
        # 時間特徵
        features['txn_dates'].add(date)
        features['txn_hours'].append(hour)
        features['first_txn_date'] = min(features['first_txn_date'], date)
        features['last_txn_date'] = max(features['last_txn_date'], date)
        features['daily_txn_counts'][date] += 1
        
        # 網絡特徵
        features['out_partners'].add(to_acct)
        
        # 自我交易
        is_self = row['is_self_txn']
        if is_self == 'Y':
            features['self_txn_count'] += 1
        
        # 未知資訊
        if row['channel_type'] == 'UNK' or is_self == 'UNK':
            features['unk_count'] += 1
        
        # 與警示帳戶互動
        if to_acct in self.alert_accounts:
            features['alert_out_count'] += 1
            features['alert_out_amt'] += amt
        
        # 帳戶類型
        features['from_acct_types'].append(row['from_acct_type'])
        
        # 管道類型
        features['channel_types'].append(row['channel_type'])
    
    def _update_in_features(self, to_acct, from_acct, amt, date, hour, row):
        """更新轉入帳戶特徵"""
        features = self.feature_dict[to_acct]
        
        # 基礎統計
        features['in_txn_count'] += 1
        features['in_txn_sum'] += amt
        features['in_txn_amounts'].append(amt)
        
        # 時間特徵
        features['txn_dates'].add(date)
        features['txn_hours'].append(hour)
        features['first_txn_date'] = min(features['first_txn_date'], date)
        features['last_txn_date'] = max(features['last_txn_date'], date)
        features['daily_txn_counts'][date] += 1
        
        # 網絡特徵
        features['in_partners'].add(from_acct)
        
        # 與警示帳戶互動
        if from_acct in self.alert_accounts:
            features['alert_in_count'] += 1
            features['alert_in_amt'] += amt
        
        # 帳戶類型
        features['to_acct_types'].append(row['to_acct_type'])
    
    def compute_final_features(self) -> pd.DataFrame:
        """
        計算最終特徵
        
        Returns:
            特徵 DataFrame
        """
        print("Computing final features...")
        
        features_list = []
        
        for acct, raw_features in tqdm(self.feature_dict.items(), desc="Computing features"):
            features = {'acct': acct}
            
            # === 1. 基礎統計特徵 (14個) ===
            features['out_txn_count'] = raw_features['out_txn_count']
            features['in_txn_count'] = raw_features['in_txn_count']
            features['total_txn_count'] = features['out_txn_count'] + features['in_txn_count']
            
            features['out_txn_sum'] = raw_features['out_txn_sum']
            features['in_txn_sum'] = raw_features['in_txn_sum']
            features['total_txn_sum'] = features['out_txn_sum'] + features['in_txn_sum']
            
            # 平均金額
            features['out_txn_mean'] = features['out_txn_sum'] / features['out_txn_count'] if features['out_txn_count'] > 0 else 0
            features['in_txn_mean'] = features['in_txn_sum'] / features['in_txn_count'] if features['in_txn_count'] > 0 else 0
            features['total_txn_mean'] = features['total_txn_sum'] / features['total_txn_count'] if features['total_txn_count'] > 0 else 0
            
            # 最大/最小金額
            out_amounts = raw_features['out_txn_amounts']
            in_amounts = raw_features['in_txn_amounts']
            all_amounts = out_amounts + in_amounts
            
            features['out_txn_max'] = max(out_amounts) if out_amounts else 0
            features['in_txn_max'] = max(in_amounts) if in_amounts else 0
            features['out_txn_min'] = min(out_amounts) if out_amounts else 0
            features['in_txn_min'] = min(in_amounts) if in_amounts else 0
            
            # 金額標準差
            features['out_txn_std'] = np.std(out_amounts) if len(out_amounts) > 1 else 0
            features['in_txn_std'] = np.std(in_amounts) if len(in_amounts) > 1 else 0
            
            # === 2. 行為模式特徵 (8個) ===
            # 轉出/轉入比例
            features['out_in_ratio'] = features['out_txn_sum'] / features['in_txn_sum'] if features['in_txn_sum'] > 0 else 0
            features['out_in_count_ratio'] = features['out_txn_count'] / features['in_txn_count'] if features['in_txn_count'] > 0 else 0
            
            # 交易活躍度
            features['active_days'] = len(raw_features['txn_dates'])
            features['txn_lifespan'] = raw_features['last_txn_date'] - raw_features['first_txn_date'] + 1 if raw_features['first_txn_date'] != float('inf') else 0
            
            # 平均每日交易次數
            features['avg_daily_txn'] = features['total_txn_count'] / features['active_days'] if features['active_days'] > 0 else 0
            
            # 單日最大交易次數
            daily_counts = list(raw_features['daily_txn_counts'].values())
            features['max_daily_txn'] = max(daily_counts) if daily_counts else 0
            
            # 交易對手數量
            features['unique_out_partners'] = len(raw_features['out_partners'])
            features['unique_in_partners'] = len(raw_features['in_partners'])
            
            # === 3. 時間特徵 (6個) ===
            hours = raw_features['txn_hours']
            valid_hours = [h for h in hours if h >= 0]
            
            if valid_hours:
                # 凌晨交易比例 (0-6點)
                features['night_txn_ratio'] = sum(1 for h in valid_hours if 0 <= h < 6) / len(valid_hours)
                
                # 工作時間交易比例 (9-18點)
                features['work_hour_txn_ratio'] = sum(1 for h in valid_hours if 9 <= h < 18) / len(valid_hours)
                
                # 交易時間平均值
                features['avg_txn_hour'] = np.mean(valid_hours)
                
                # 交易時間標準差（時間集中度）
                features['txn_hour_std'] = np.std(valid_hours)
            else:
                features['night_txn_ratio'] = 0
                features['work_hour_txn_ratio'] = 0
                features['avg_txn_hour'] = 0
                features['txn_hour_std'] = 0
            
            # 首次和最後交易日期
            features['first_txn_date'] = raw_features['first_txn_date'] if raw_features['first_txn_date'] != float('inf') else 0
            features['last_txn_date'] = raw_features['last_txn_date']
            
            # === 4. 風險指標特徵 (7個) ===
            # 自我交易比例
            features['self_txn_ratio'] = raw_features['self_txn_count'] / features['total_txn_count'] if features['total_txn_count'] > 0 else 0
            
            # 未知資訊比例
            features['unk_ratio'] = raw_features['unk_count'] / features['total_txn_count'] if features['total_txn_count'] > 0 else 0
            
            # 大額交易比例（超過平均值的2倍標準差）
            if all_amounts and features['total_txn_mean'] > 0:
                threshold = features['total_txn_mean'] + 2 * np.std(all_amounts)
                features['large_txn_ratio'] = sum(1 for amt in all_amounts if amt > threshold) / len(all_amounts)
            else:
                features['large_txn_ratio'] = 0
            
            # 與警示帳戶的交易
            features['alert_txn_count'] = raw_features['alert_out_count'] + raw_features['alert_in_count']
            features['alert_txn_amt'] = raw_features['alert_out_amt'] + raw_features['alert_in_amt']
            features['alert_txn_ratio'] = features['alert_txn_count'] / features['total_txn_count'] if features['total_txn_count'] > 0 else 0
            
            # 小額交易比例（小於平均值的一半）
            if all_amounts and features['total_txn_mean'] > 0:
                threshold = features['total_txn_mean'] * 0.5
                features['small_txn_ratio'] = sum(1 for amt in all_amounts if amt < threshold) / len(all_amounts)
            else:
                features['small_txn_ratio'] = 0
            
            # === 5. 網絡特徵 (5個) ===
            # 轉入/轉出對手重疊度
            if features['unique_out_partners'] > 0 and features['unique_in_partners'] > 0:
                overlap = len(raw_features['out_partners'] & raw_features['in_partners'])
                features['partner_overlap_ratio'] = overlap / min(features['unique_out_partners'], features['unique_in_partners'])
            else:
                features['partner_overlap_ratio'] = 0
            
            # 平均每個對手的交易次數
            features['avg_txn_per_out_partner'] = features['out_txn_count'] / features['unique_out_partners'] if features['unique_out_partners'] > 0 else 0
            features['avg_txn_per_in_partner'] = features['in_txn_count'] / features['unique_in_partners'] if features['unique_in_partners'] > 0 else 0
            
            # 交易集中度（單一最大對手的交易比例）
            # 簡化版：用平均交易次數與最大可能的比值
            features['txn_concentration'] = features['avg_txn_per_out_partner'] / features['out_txn_count'] if features['out_txn_count'] > 0 else 0
            
            # 總交易對手數
            features['total_unique_partners'] = features['unique_out_partners'] + features['unique_in_partners']
            
            # === 6. 帳戶類型多樣性 (2個) ===
            from_types = raw_features['from_acct_types']
            to_types = raw_features['to_acct_types']
            
            features['from_acct_type_diversity'] = len(set(from_types)) if from_types else 0
            features['to_acct_type_diversity'] = len(set(to_types)) if to_types else 0
            
            # === 7. 管道類型特徵 (2個) ===
            channel_types = raw_features['channel_types']
            features['channel_type_diversity'] = len(set(channel_types)) if channel_types else 0
            
            # UNK 管道比例
            features['unk_channel_ratio'] = sum(1 for c in channel_types if c == 'UNK') / len(channel_types) if channel_types else 0
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # 填充 NaN 和 Inf
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"Extracted {len(df)} accounts with {len(df.columns)-1} features")
        
        return df


def extract_features_from_transactions(
    transaction_loader,
    alert_accounts: Set[str],
    save_path: str = 'output/features.csv'
) -> pd.DataFrame:
    """
    從交易資料中提取特徵
    
    Args:
        transaction_loader: 交易資料載入器（generator）
        alert_accounts: 警示帳戶集合
        save_path: 儲存路徑
        
    Returns:
        特徵 DataFrame
    """
    print("Starting feature extraction...")
    
    # 初始化特徵提取器
    extractor = FeatureExtractor(alert_accounts)
    
    # 分批處理交易資料
    for chunk in transaction_loader:
        extractor.process_chunk(chunk)
        gc.collect()
    
    # 計算最終特徵
    features_df = extractor.compute_final_features()
    
    # 儲存特徵
    print(f"Saving features to {save_path}...")
    features_df.to_csv(save_path, index=False)
    print("Feature extraction completed!")
    
    return features_df


if __name__ == "__main__":
    print("Testing feature engineering module...")
    
    # 這裡可以添加測試程式碼
    print("Feature engineering module ready!")

