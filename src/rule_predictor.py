"""
Rule Predictor Module
基於規則的預測器 - 建立、評估和應用規則
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from typing import List, Dict, Tuple
import json


class Rule:
    """單一規則類別"""
    
    def __init__(self, feature: str, operator: str, threshold: float, weight: float = 1.0):
        """
        初始化規則
        
        Args:
            feature: 特徵名稱
            operator: 運算符 ('>', '<', '>=', '<=', '==')
            threshold: 閾值
            weight: 權重
        """
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.weight = weight
    
    def apply(self, df: pd.DataFrame) -> np.ndarray:
        """
        應用規則到資料
        
        Args:
            df: 特徵 DataFrame
            
        Returns:
            布林陣列，True 表示滿足規則
        """
        if self.feature not in df.columns:
            return np.zeros(len(df), dtype=bool)
        
        values = df[self.feature].values
        
        if self.operator == '>':
            return values > self.threshold
        elif self.operator == '<':
            return values < self.threshold
        elif self.operator == '>=':
            return values >= self.threshold
        elif self.operator == '<=':
            return values <= self.threshold
        elif self.operator == '==':
            return values == self.threshold
        else:
            return np.zeros(len(df), dtype=bool)
    
    def __str__(self):
        return f"{self.feature} {self.operator} {self.threshold:.4f}"


class RuleBasedPredictor:
    """基於規則的預測器"""
    
    def __init__(self):
        self.rules = []
        self.method = 'voting'  # 'voting', 'scoring', 'any'
        self.threshold = 0.5
    
    def add_rule(self, feature: str, operator: str, threshold: float, weight: float = 1.0):
        """添加規則"""
        rule = Rule(feature, operator, threshold, weight)
        self.rules.append(rule)
        print(f"Added rule: {rule}")
    
    def add_rules_from_comparison(
        self,
        comparison_df: pd.DataFrame,
        n_rules: int = 10,
        min_diff_ratio: float = 1.5,
        min_effect_size: float = 0.5
    ):
        """
        從特徵比較結果中自動生成規則
        
        Args:
            comparison_df: 特徵比較 DataFrame
            n_rules: 要生成的規則數量
            min_diff_ratio: 最小差異倍數
            min_effect_size: 最小效應量
        """
        print(f"\nGenerating rules from feature comparison...")
        
        # 篩選顯著特徵
        significant = comparison_df[
            (comparison_df['p_value'] < 0.05) &
            (abs(comparison_df['effect_size']) > min_effect_size) &
            (comparison_df['diff_ratio'] > min_diff_ratio)
        ].copy()
        
        # 按效應量排序
        significant = significant.sort_values('effect_size', ascending=False, key=abs)
        
        # 生成規則
        for _, row in significant.head(n_rules).iterrows():
            feature = row['feature']
            alert_mean = row['alert_mean']
            normal_mean = row['normal_mean']
            
            # 根據警示帳戶是否高於正常帳戶決定運算符
            if alert_mean > normal_mean:
                operator = '>'
                # 設定閾值為正常帳戶均值 + 1 個標準差
                threshold = normal_mean + row['normal_std']
            else:
                operator = '<'
                threshold = normal_mean - row['normal_std']
            
            # 根據效應量設定權重
            weight = min(abs(row['effect_size']), 3.0)
            
            self.add_rule(feature, operator, threshold, weight)
    
    def predict(self, df: pd.DataFrame, method: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        進行預測
        
        Args:
            df: 特徵 DataFrame
            method: 預測方法 ('voting', 'scoring', 'any')
            
        Returns:
            (predictions, scores) - 預測結果和信心分數
        """
        if method is None:
            method = self.method
        
        if len(self.rules) == 0:
            print("Warning: No rules defined!")
            return np.zeros(len(df)), np.zeros(len(df))
        
        # 應用所有規則
        rule_results = []
        for rule in self.rules:
            result = rule.apply(df)
            rule_results.append(result)
        
        rule_results = np.array(rule_results)
        
        if method == 'voting':
            # 投票法：多數規則觸發則判定為警示
            votes = rule_results.sum(axis=0)
            predictions = (votes >= len(self.rules) * self.threshold).astype(int)
            scores = votes / len(self.rules)
            
        elif method == 'scoring':
            # 分數法：加權分數超過閾值
            weights = np.array([rule.weight for rule in self.rules])
            scores = (rule_results.T @ weights) / weights.sum()
            predictions = (scores >= self.threshold).astype(int)
            
        elif method == 'any':
            # 任一規則觸發即判定為警示
            predictions = (rule_results.sum(axis=0) > 0).astype(int)
            scores = rule_results.sum(axis=0) / len(self.rules)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return predictions, scores
    
    def evaluate(
        self,
        df: pd.DataFrame,
        true_labels: np.ndarray,
        method: str = None
    ) -> Dict:
        """
        評估規則效果
        
        Args:
            df: 特徵 DataFrame
            true_labels: 真實標籤
            method: 預測方法
            
        Returns:
            評估指標字典
        """
        predictions, scores = self.predict(df, method)
        
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'n_predicted_alerts': int(predictions.sum()),
            'n_true_alerts': int(true_labels.sum())
        }
        
        return results
    
    def optimize_threshold(
        self,
        df: pd.DataFrame,
        true_labels: np.ndarray,
        method: str = 'voting'
    ) -> float:
        """
        優化分類閾值以最大化 F1-Score
        
        Args:
            df: 特徵 DataFrame
            true_labels: 真實標籤
            method: 預測方法
            
        Returns:
            最佳閾值
        """
        print(f"\nOptimizing threshold for method '{method}'...")
        
        # 獲取分數
        _, scores = self.predict(df, method)
        
        # 嘗試不同閾值
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            predictions = (scores >= threshold).astype(int)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Best threshold: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
        
        return best_threshold
    
    def explain_prediction(
        self,
        df: pd.DataFrame,
        idx: int
    ) -> Dict:
        """
        解釋單一預測
        
        Args:
            df: 特徵 DataFrame
            idx: 資料索引
            
        Returns:
            解釋字典
        """
        row = df.iloc[idx:idx+1]
        
        triggered_rules = []
        for rule in self.rules:
            if rule.apply(row)[0]:
                triggered_rules.append(str(rule))
        
        prediction, score = self.predict(row)
        
        return {
            'prediction': int(prediction[0]),
            'confidence_score': float(score[0]),
            'triggered_rules': triggered_rules,
            'n_triggered': len(triggered_rules),
            'total_rules': len(self.rules)
        }
    
    def save_rules(self, file_path: str):
        """儲存規則到檔案"""
        rules_data = {
            'method': self.method,
            'threshold': self.threshold,
            'rules': [
                {
                    'feature': rule.feature,
                    'operator': rule.operator,
                    'threshold': float(rule.threshold),
                    'weight': float(rule.weight)
                }
                for rule in self.rules
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        print(f"Rules saved to {file_path}")
    
    def load_rules(self, file_path: str):
        """從檔案載入規則"""
        with open(file_path, 'r') as f:
            rules_data = json.load(f)
        
        self.method = rules_data['method']
        self.threshold = rules_data['threshold']
        self.rules = []
        
        for rule_data in rules_data['rules']:
            self.add_rule(
                rule_data['feature'],
                rule_data['operator'],
                rule_data['threshold'],
                rule_data['weight']
            )
        
        print(f"Loaded {len(self.rules)} rules from {file_path}")


def build_and_evaluate_predictor(
    features_df: pd.DataFrame,
    alert_accounts: set,
    comparison_df: pd.DataFrame,
    output_dir: str = 'output'
) -> RuleBasedPredictor:
    """
    建立和評估規則預測器
    
    Args:
        features_df: 特徵 DataFrame
        alert_accounts: 警示帳戶集合
        comparison_df: 特徵比較 DataFrame
        output_dir: 輸出目錄
        
    Returns:
        訓練好的預測器
    """
    print("="*60)
    print("Building Rule-Based Predictor")
    print("="*60)
    
    # 準備資料
    features_df['is_alert'] = features_df['acct'].isin(alert_accounts).astype(int)
    true_labels = features_df['is_alert'].values
    
    # 建立預測器
    predictor = RuleBasedPredictor()
    predictor.method = 'scoring'
    
    # 從特徵比較中生成規則
    predictor.add_rules_from_comparison(comparison_df, n_rules=15, min_diff_ratio=1.3)
    
    # 優化閾值
    predictor.optimize_threshold(features_df, true_labels, method='scoring')
    
    # 評估效果
    print("\n=== Evaluation Results ===")
    results = predictor.evaluate(features_df, true_labels, method='scoring')
    
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Predicted alerts: {results['n_predicted_alerts']}")
    print(f"True alerts: {results['n_true_alerts']}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {results['confusion_matrix'][0][0]}, FP: {results['confusion_matrix'][0][1]}")
    print(f"  FN: {results['confusion_matrix'][1][0]}, TP: {results['confusion_matrix'][1][1]}")
    
    # 儲存規則
    predictor.save_rules(f'{output_dir}/final_rules.json')
    
    # 儲存評估結果
    with open(f'{output_dir}/rule_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Rule-Based Predictor Ready!")
    print("="*60)
    
    return predictor


def make_predictions_with_explanations(
    predictor: RuleBasedPredictor,
    features_df: pd.DataFrame,
    predict_accounts: pd.DataFrame,
    output_path: str = 'output/predictions.csv'
):
    """
    使用規則預測器進行預測並提供解釋
    
    Args:
        predictor: 規則預測器
        features_df: 特徵 DataFrame
        predict_accounts: 需要預測的帳戶 DataFrame
        output_path: 輸出檔案路徑
    """
    print("\nMaking predictions for target accounts...")
    
    # 篩選出需要預測的帳戶
    predict_features = features_df[features_df['acct'].isin(predict_accounts['acct'])].copy()
    
    if len(predict_features) == 0:
        print("Warning: No matching accounts found in features!")
        return
    
    print(f"Predicting for {len(predict_features)} accounts...")
    
    # 進行預測
    predictions, scores = predictor.predict(predict_features)
    
    # 為每個帳戶生成解釋
    explanations = []
    for idx in range(len(predict_features)):
        explanation = predictor.explain_prediction(predict_features, idx)
        explanations.append(explanation['triggered_rules'])
    
    # 建立結果 DataFrame
    result_df = pd.DataFrame({
        'acct': predict_features['acct'].values,
        'label': predictions,
        'confidence_score': scores,
        'triggered_rules': [';'.join(rules) if rules else '' for rules in explanations]
    })
    
    # 儲存結果
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # 統計摘要
    n_alert = result_df['label'].sum()
    print(f"\n=== Prediction Summary ===")
    print(f"Total accounts predicted: {len(result_df)}")
    print(f"Predicted as alert: {n_alert} ({n_alert/len(result_df)*100:.1f}%)")
    print(f"Predicted as normal: {len(result_df) - n_alert} ({(len(result_df)-n_alert)/len(result_df)*100:.1f}%)")
    print(f"Average confidence score: {result_df['confidence_score'].mean():.4f}")
    
    # 顯示幾個預測為警示的帳戶範例
    alert_examples = result_df[result_df['label'] == 1].head(5)
    if len(alert_examples) > 0:
        print(f"\n=== Sample Alert Predictions ===")
        for _, row in alert_examples.iterrows():
            print(f"\nAccount: {row['acct'][:16]}...")
            print(f"  Confidence: {row['confidence_score']:.4f}")
            print(f"  Triggered rules: {row['triggered_rules']}")


if __name__ == "__main__":
    print("Rule predictor module ready!")

