# 🚨 模型訓練問題診斷報告

## 核心問題：訓練集過大導致嚴重的數據洩漏

### 問題 1: 訓練資料包含預測目標 ❌

**發現：**
```
rule_evaluation.json:
- n_true_alerts: 945  (訓練時標記為異常)
- n_predicted_alerts: 45,086 

但實際上：
- acct_alert.csv 只有 1,004 筆異常帳戶
- 訓練資料竟然有 1,798,484 + 945 = 1,799,429 筆！
```

**問題：模型在訓練時使用了所有特徵資料（包含預測目標帳戶）！**

這導致：
1. ❌ **Data Leakage**: 預測目標帳戶也被用來訓練
2. ❌ **訓練效果極差**: F1-Score 只有 0.0176 (連訓練集都預測不好)
3. ❌ **規則過度嚴格**: Precision 只有 0.97% (99% 都是誤報)

---

## 問題 2: 模型在訓練集上表現就很差

```
Training Performance (from rule_evaluation.json):
- Precision: 0.0106 (1.06%)  ← 太低！
- Recall: 0.5037 (50.37%)    ← 還可以
- F1-Score: 0.0207 (2.07%)   ← 極低！

Confusion Matrix:
          Predicted 0    Predicted 1
Actual 0:  1,753,874      44,610  (2.5% FP rate)
Actual 1:  469            476     (50% recall)
```

**分析：**
- 預測了 45,086 個異常，但只對了 476 個
- 有 44,610 個 False Positives（誤報率極高）
- 這表示規則本身就不準確

---

## 問題 3: 規則生成策略有誤

**當前策略（rule_predictor.py line 114-118）：**
```python
if alert_mean > normal_mean:
    operator = '>'
    threshold = normal_mean + normal_std  # 問題在這！
```

**問題：**
- 閾值設定為 `normal_mean + 1*std` 太寬鬆
- 導致大量正常帳戶被誤判為異常
- 應該使用更嚴格的標準（如 2-3 個標準差）

**證據：**
```
從 feature_comparison.csv:
max_daily_in_txn:
  - Alert mean: 4.45, std: 8.45
  - Normal mean: 0.78, std: 1.25
  - 當前閾值: 0.78 + 1.25 = 2.03
  - 但這樣會涵蓋太多正常帳戶！
```

---

## 問題 4: 特徵閾值設定不合理

看 `final_rules.json` 的規則：
```json
{
  "feature": "avg_daily_txn",
  "operator": ">",
  "threshold": 1.939  // normal_mean + 1*std
}
```

但從資料分布看：
- Normal mean: 1.15, std: 0.79
- Alert mean: 3.30, std: 4.07
- 閾值 1.94 太低了！應該至少 2.5-3.0

---

## 問題 5: 訓練/預測資料分離錯誤

**應該這樣做：**
```
1. 只用 acct_alert.csv (1,004) 作為正樣本
2. 從其他帳戶隨機抽樣作為負樣本（如 5,000-10,000 筆）
3. 在這個小資料集上訓練
4. 然後預測 acct_predict.csv
```

**實際做了什麼：**
```
1. 提取了所有帳戶的特徵（180 萬筆！）
2. 在所有資料上訓練（包含預測目標）
3. 閾值在訓練集上優化
4. 然後在同樣的資料上預測
→ 這是嚴重的資料洩漏！
```

---

## 解決方案

### 🔧 立即修正方案

#### 1. 修正訓練流程（最重要）

**修改 `rule_predictor.py` 的 `build_and_evaluate_predictor`：**

```python
def build_and_evaluate_predictor(
    features_df: pd.DataFrame,
    alert_accounts: set,
    comparison_df: pd.DataFrame,
    output_dir: str = 'output'
) -> RuleBasedPredictor:
    
    # 🔴 關鍵修改：不要在所有資料上訓練！
    # 只用有標籤的資料訓練（alert + random normal sample）
    
    # 1. 分離有標籤和無標籤資料
    alert_df = features_df[features_df['acct'].isin(alert_accounts)]
    unlabeled_df = features_df[~features_df['acct'].isin(alert_accounts)]
    
    # 2. 從無標籤資料中抽樣作為負樣本（不包含預測目標）
    predict_accounts = load_predict_accounts('raw_data/acct_predict.csv')['acct']
    
    # 從非預測目標中抽樣
    available_normal = unlabeled_df[~unlabeled_df['acct'].isin(predict_accounts)]
    normal_sample = available_normal.sample(
        n=min(len(alert_accounts) * 3, len(available_normal)),
        random_state=42
    )
    
    # 3. 構建訓練集
    train_df = pd.concat([alert_df, normal_sample])
    train_df['is_alert'] = train_df['acct'].isin(alert_accounts).astype(int)
    
    # 4. 在訓練集上建立規則...
```

#### 2. 修正閾值設定策略

**修改 `add_rules_from_comparison` 中的閾值計算：**

```python
# 當前（太寬鬆）：
threshold = normal_mean + normal_std

# 改為（更嚴格）：
threshold = normal_mean + 2.0 * normal_std  # 2 個標準差
# 或使用百分位數：
threshold = normal_p95  # 第 95 百分位數
```

#### 3. 增加規則過濾

**只選擇高品質規則：**

```python
# 在 add_rules_from_comparison 中加入：
significant = comparison_df[
    (comparison_df['p_value'] < 0.01) &      # 更嚴格的 p-value
    (abs(comparison_df['effect_size']) > 1.0) &  # 更大的效應量
    (comparison_df['diff_ratio'] > 2.0)      # Alert/Normal 比例要 > 2
].copy()
```

#### 4. 使用更好的模型

**考慮用機器學習模型代替規則：**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 在訓練集上訓練
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)

# 使用交叉驗證
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"CV F1-Score: {scores.mean():.4f}")
```

---

## 預期改善

### 修正前（當前）：
- Training F1: 0.0207
- Test F1: 0.0724
- 大量誤報（Precision 1.06%）

### 修正後（預期）：
- Training F1: 0.3-0.5 (proper training)
- Test F1: 0.25-0.40 (realistic)
- Precision: > 20% (減少誤報)
- Recall: 40-60% (維持檢出率)

---

## 行動計畫

1. ✅ **立即修正訓練流程**（分離訓練/測試資料）
2. ✅ **調整閾值策略**（使用 2-3 個標準差）
3. ✅ **考慮使用機器學習模型**（Random Forest / XGBoost）
4. ✅ **進行適當的交叉驗證**
5. ✅ **重新訓練並評估**

---

## 結論

當前模型的問題是**系統性的**：
1. 資料洩漏（訓練集包含預測目標）
2. 規則生成策略過於寬鬆
3. 沒有適當的訓練/測試分離

**這解釋了為什麼調整閾值沒有幫助** - 問題在模型本身，不在預測閾值！

必須重新設計訓練流程才能得到有效的模型。

