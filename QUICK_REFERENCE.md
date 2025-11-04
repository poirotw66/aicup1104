# 🚀 快速參考卡片

## 一分鐘快速上手

### 第一次使用
```bash
# 最簡單的方式
./quick_start.sh
```

### 追求極速（一次性設定）
```bash
# Step 1: 轉換格式（約 1-2 分鐘）
python convert_to_parquet.py

# Step 2: 使用終極版本（之後每次都很快）
python main_ultra_fast.py
```

---

## 📋 執行方式對比

| 命令 | 時間 | 何時使用 |
|------|------|---------|
| `./quick_start.sh` | - | ⭐ 新手/不確定用哪個 |
| `python main_fast.py` | 5-15 分 | ⭐⭐ 日常使用 |
| `python main_ultra_fast.py` | 2-10 分 | ⚡⚡ 已轉換 Parquet |
| `python main.py` | 30-45 分 | 測試/偵錯 |

---

## 🔍 常用命令

### 檢查進度
```bash
./check_progress.sh
```

### 查看 Log
```bash
tail -f output/execution_fast.log
```

### 查看預測結果
```bash
head -20 output/predictions.csv
```

### 統計預測
```bash
# 預測為警示的帳戶數
awk -F',' 'NR>1 && $2==1 {count++} END {print count}' output/predictions.csv
```

---

## 📂 重要檔案

### 輸入
- `raw_data/acct_alert.csv` - 警示帳戶（1,004個）
- `raw_data/acct_transaction.csv` - 交易資料（703MB）
- `raw_data/acct_predict.csv` - 待預測帳戶（4,780個）

### 輸出
- **`output/predictions.csv`** - 最終預測結果 ⭐⭐⭐
- `output/features.csv` - 特徵資料
- `output/feature_comparison.csv` - 特徵分析
- `output/decision_tree.png` - 決策樹圖

---

## 💡 關鍵問題速查

### Q: GPU 能加速嗎？
**A: ❌ 不能。本專案瓶頸在 I/O 和資料處理，不是數學運算。**

### Q: 多核心 CPU 有用嗎？
**A: ⭐⭐⭐ 有用，2-4倍加速（已在 `main_ultra_fast.py` 實現）。**

### Q: 最有效的優化？
**A: ✅ 向量化操作（10-50x），已在 `main_fast.py` 實現。**

### Q: 如何進一步加速？
**A: 轉換為 Parquet 格式，獲得額外 5-10x I/O 加速。**

---

## 🛠️ 故障排除

### 程式沒有輸出
```bash
# 使用 -u 參數強制即時輸出
python -u main_fast.py
```

### 記憶體不足
```bash
# 減少 chunk size
# 編輯 main_fast.py，將 500000 改為 100000
```

### 想重新提取特徵
```bash
# 刪除舊的特徵檔案
rm output/features.csv

# 重新執行
python main_fast.py
```

---

## 📊 效能速查表

| 優化 | 工具 | 加速 | 難度 |
|------|------|------|------|
| 向量化 | `main_fast.py` | 10-50x | ✅ 已實現 |
| Parquet | `convert_to_parquet.py` | 5-10x | ⭐ 簡單 |
| 多核心 | `main_ultra_fast.py` | 2-4x | ✅ 已實現 |
| 組合 | `main_ultra_fast.py` + Parquet | 15-50x | ⭐⭐ 需設定 |

---

## 📚 詳細文件

- `README.md` - 完整專案說明
- `OPTIMIZATION_GUIDE.md` - 進階優化指南
- `PERFORMANCE_GUIDE.md` - 效能原理
- `SUMMARY.md` - 專案總結

---

## ⚡ 一行命令速查

```bash
# 安裝依賴
pip install -r requirements.txt

# 最簡單執行
./quick_start.sh

# 快速版本
python main_fast.py

# 終極版本（需先轉換）
python convert_to_parquet.py && python main_ultra_fast.py

# 檢查進度
./check_progress.sh

# 查看結果
head -20 output/predictions.csv

# 測試環境
python test_start.py
```

---

## 🎯 記住這三點

1. **日常使用**：`python main_fast.py` 已經很快
2. **追求極速**：先轉 Parquet，再用 `main_ultra_fast.py`
3. **不需要 GPU**：向量化 + Parquet 已是最佳方案

---

**儲存這個檔案以便隨時查閱！** 📌

