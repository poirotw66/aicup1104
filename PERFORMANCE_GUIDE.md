# 效能優化指南

## 📊 效能分析結果

### 原始版本效能瓶頸
1. **I/O 瓶頸**：讀取 703MB CSV 檔案
2. **Python 逐行迴圈**：使用 `iterrows()` 處理資料（極慢）
3. **記憶體管理**：儲存大量 list 和 set

### 預估執行時間
- **原始版本 (main.py)**：30-45 分鐘
- **優化版本 (main_fast.py)**：5-15 分鐘
- **加速比**：約 3-9 倍

---

## ⚡ 優化方案比較

### 1. 向量化操作（已實現）⭐⭐⭐⭐⭐
**效果：10-50 倍加速**

**原理：**
```python
# ❌ 慢速版本（逐行處理）
for _, row in df.iterrows():
    result += row['value']

# ✅ 快速版本（向量化）
result = df['value'].sum()
```

**實作：**
- 使用 `main_fast.py` 和 `feature_engineering_fast.py`
- 用 Pandas 內建的 `groupby()` 和 `agg()` 取代迴圈

**使用方式：**
```bash
python main_fast.py  # 使用優化版本
```

---

### 2. 增加 Chunk Size ⭐⭐⭐
**效果：20-30% 加速**

**說明：**
- 減少 I/O 次數
- 已在 `main_fast.py` 中實現（100K → 500K）

---

### 3. 多核心並行處理 ⭐⭐⭐
**效果：2-4 倍加速（取決於 CPU 核心數）**

**適用場景：**
- 多個獨立任務（如批次預測）
- 特徵提取階段

**實作建議：**
```python
from multiprocessing import Pool

# 並行處理多個帳戶的特徵提取
with Pool(processes=4) as pool:
    results = pool.map(extract_features, account_batches)
```

**注意：**
- 需要足夠的記憶體
- 目前的向量化版本已經很快，多核心效益有限

---

### 4. 使用 Parquet 格式 ⭐⭐⭐⭐
**效果：5-10 倍讀取加速，70% 檔案大小減少**

**轉換 CSV → Parquet：**
```python
import pandas as pd

# 一次性轉換（只需執行一次）
df = pd.read_csv('raw_data/acct_transaction.csv')
df.to_parquet('raw_data/acct_transaction.parquet', engine='pyarrow')

# 之後讀取更快
df = pd.read_parquet('raw_data/acct_transaction.parquet')
```

**優點：**
- 壓縮率高（703MB → 約 200MB）
- 讀取速度快 5-10 倍
- 保留資料型態

---

## ❌ 無效或效果有限的優化

### 1. GPU 加速 ❌
**為什麼無效：**
- 瓶頸在 I/O 和 Python 迴圈，不是數學運算
- GPU 適合：
  - 深度學習訓練
  - 大量矩陣運算
  - 圖像/視頻處理
- 本專案主要是資料處理和統計計算，GPU 無法加速

### 2. 單純提高 CPU 頻率 ⭐
**效果有限：** 約 10-20% 提升
- 因為瓶頸不是 CPU 運算速度
- 主要是 I/O 和記憶體存取

### 3. 增加記憶體 ⭐
**效果有限**
- 除非記憶體不足導致 swap
- 目前的資料量（~1GB）在 8GB+ 記憶體應該足夠

---

## 🚀 推薦的執行方式

### 快速開始（推薦）
```bash
# 使用優化版本
python main_fast.py
```

### 如果需要更快
1. **先轉換為 Parquet**（一次性操作）
```bash
python convert_to_parquet.py
```

2. **修改 data_loader.py 讀取 Parquet**
```python
# 替換 CSV 讀取為 Parquet
df = pd.read_parquet('raw_data/acct_transaction.parquet')
```

---

## 📈 效能提升總結

| 方法 | 加速倍數 | 實作難度 | 成本 |
|------|---------|---------|------|
| 向量化操作（已實現） | 10-50x | 中 | 無 |
| 增加 chunksize（已實現） | 1.2-1.3x | 易 | 無 |
| Parquet 格式 | 5-10x | 易 | 無 |
| 多核心並行 | 2-4x | 中 | 需要多核 CPU |
| 提高 CPU 頻率 | 1.1-1.2x | - | 需要升級硬體 |
| GPU 加速 | **無效** | - | - |

---

## 💡 最佳實踐

### 第一次執行
```bash
# 1. 使用優化版本（預期 5-15 分鐘）
python main_fast.py

# 特徵會被儲存到 output/features.csv
```

### 後續執行
```bash
# 特徵已存在，會直接載入（< 1 分鐘）
python main_fast.py
```

### 如果要重新提取特徵
```bash
# 刪除特徵檔案後重新執行
rm output/features.csv
python main_fast.py
```

---

## 🔧 進階優化（選擇性）

### 1. 使用 Dask 處理超大資料集
如果資料超過記憶體限制：
```bash
pip install dask
```

### 2. 使用 Numba JIT 編譯
加速數值計算：
```python
from numba import jit

@jit(nopython=True)
def fast_calculation(arr):
    return arr.sum()
```

### 3. 資料庫索引
將資料載入 SQLite/PostgreSQL 並建立索引

---

## ⏱️ 執行時間估算

### 使用 `main_fast.py`（優化版本）

| 階段 | 時間 | 說明 |
|------|------|------|
| 載入資料 | < 1 分鐘 | 讀取小檔案 |
| 特徵提取（首次） | 5-15 分鐘 | 向量化處理 |
| 特徵提取（已存在） | < 30 秒 | 直接讀取 CSV |
| EDA 分析 | 1-2 分鐘 | 統計計算 |
| 模式發現 | 2-3 分鐘 | 決策樹、聚類 |
| 規則建立 | < 1 分鐘 | 規則生成 |
| 預測 | < 1 分鐘 | 應用規則 |
| **總計（首次）** | **10-25 分鐘** |  |
| **總計（後續）** | **5-8 分鐘** |  |

---

## 📝 結論

**最有效的優化：**
1. ✅ 使用向量化操作（已實現在 `main_fast.py`）
2. ✅ 增加 chunk size（已實現）
3. 💡 可選：轉換為 Parquet 格式

**無需考慮：**
- ❌ GPU 加速（不適用於此類任務）
- ❌ 升級 CPU 頻率（效益不大）

**直接使用 `python main_fast.py` 即可獲得最佳效能！**

