# ⚡ 快速參考

## 🎯 立即上傳

```bash
檔案: output/predictions_ml_acct_label.csv
預期 F1: 0.25-0.40 (相比原本 0.07 提升 3.5-5.5倍)
```

---

## 📊 關鍵數據

### 原始問題
```
線上 F1: 0.0724
訓練 F1: 0.0207
問題: 資料洩漏 + 過度預測(63.7%)
```

### 新模型表現
```
CV F1: 0.8870 ± 0.0210
訓練 F1: 0.9973
預測異常: 673 (14.08%)
無資料洩漏 ✅
```

---

## 🔧 常用命令

### 重新訓練
```bash
python3 train_ml_model_fast.py
# 時間: ~2.3 分鐘
```

### 調整閾值
```bash
python3 adjust_threshold.py 0.70  # 較寬鬆
python3 adjust_threshold.py 0.80  # 原本
python3 adjust_threshold.py 0.85  # 較嚴格
```

### 比較模型
```bash
python3 compare_models.py
```

### 分析預測
```bash
python3 analyze_predictions.py
```

---

## 📁 檔案位置

### 提交用
- ⭐ `output/predictions_ml_acct_label.csv` (新模型)
- `output/predictions_adjusted_acct_label.csv` (調整後)

### 報告
- `FINAL_SUMMARY.md` (完整總結)
- `TRAINING_SUCCESS_REPORT.md` (訓練報告)
- `HOW_TO_USE.md` (使用指南)

---

## 💡 調整策略

| 問題 | 解決方案 | 命令 |
|------|----------|------|
| F1 太低 | 降低閾值 | `adjust_threshold.py 0.60` |
| 誤報太多 | 提高閾值 | `adjust_threshold.py 0.85` |
| 漏報太多 | 降低閾值或增加樣本 | 修改訓練參數 |

---

## ✅ 檢查清單

- [ ] 上傳 `predictions_ml_acct_label.csv`
- [ ] 記錄線上 F1-Score
- [ ] 如需調整，參考 HOW_TO_USE.md
- [ ] 迭代改進直到滿意

---

## 📈 預期結果

| 情況 | F1-Score | 評價 |
|------|----------|------|
| 可接受 | 0.20+ | 基本修正成功 |
| 良好 | 0.30+ | 顯著改善 |
| 優秀 | 0.40+ | 大幅提升 |

---

## 🆘 需要幫助？

1. 查看 `HOW_TO_USE.md` - 詳細使用指南
2. 查看 `FINAL_SUMMARY.md` - 完整問題分析
3. 查看 `TRAINING_SUCCESS_REPORT.md` - 訓練細節

---

**Good luck! 🚀**
