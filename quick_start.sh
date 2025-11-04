#!/bin/bash
# 快速啟動腳本 - 自動選擇最佳執行方式

echo "========================================"
echo "警示帳戶偵測系統 - 快速啟動"
echo "========================================"
echo ""

# 檢查 Parquet 檔案是否存在
PARQUET_FILE="raw_data/acct_transaction.parquet"
CSV_FILE="raw_data/acct_transaction.csv"
FEATURE_FILE="output/features.csv"

# 功能函數
check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo "✓ $1 ($size)"
        return 0
    else
        echo "✗ $1 (不存在)"
        return 1
    fi
}

echo "檢查資料檔案："
echo "----------------------------------------"
check_file "$CSV_FILE"
PARQUET_EXISTS=$?
check_file "$PARQUET_FILE"
PARQUET_EXISTS=$?
check_file "$FEATURE_FILE"
FEATURE_EXISTS=$?
echo ""

# 決定執行策略
if [ $FEATURE_EXISTS -eq 0 ]; then
    echo "✓ 特徵檔案已存在，將快速載入"
    echo ""
    read -p "是否要重新提取特徵？(y/N): " REEXTRACT
    
    if [ "$REEXTRACT" = "y" ] || [ "$REEXTRACT" = "Y" ]; then
        echo "將重新提取特徵..."
        rm -f "$FEATURE_FILE"
    else
        echo "使用已存在的特徵檔案"
    fi
fi

echo ""
echo "========================================"
echo "選擇執行方式："
echo "========================================"
echo "1. 快速版本 (main_fast.py) - 推薦"
echo "2. 終極版本 (main_ultra_fast.py) - 最快"
echo "3. 先轉換 Parquet，再執行終極版本"
echo "4. 原始版本 (main.py)"
echo ""

read -p "請選擇 (1-4，預設 1): " CHOICE

case "${CHOICE:-1}" in
    1)
        echo ""
        echo "執行快速版本..."
        python -u main_fast.py 2>&1 | tee output/execution.log
        ;;
    2)
        if [ ! -f "$PARQUET_FILE" ]; then
            echo ""
            echo "⚠️  Parquet 檔案不存在"
            echo "建議先執行選項 3 轉換為 Parquet 格式以獲得最佳效能"
            echo ""
            read -p "是否繼續使用 CSV？(Y/n): " CONTINUE
            
            if [ "$CONTINUE" = "n" ] || [ "$CONTINUE" = "N" ]; then
                echo "已取消"
                exit 0
            fi
        fi
        
        echo ""
        echo "執行終極版本..."
        python -u main_ultra_fast.py 2>&1 | tee output/execution.log
        ;;
    3)
        echo ""
        echo "步驟 1: 轉換為 Parquet 格式..."
        python convert_to_parquet.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "步驟 2: 執行終極版本..."
            python -u main_ultra_fast.py 2>&1 | tee output/execution.log
        else
            echo "轉換失敗，請檢查錯誤訊息"
            exit 1
        fi
        ;;
    4)
        echo ""
        echo "執行原始版本..."
        echo "注意：此版本較慢（30-45 分鐘）"
        python -u main.py 2>&1 | tee output/execution.log
        ;;
    *)
        echo "無效的選擇"
        exit 1
        ;;
esac

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 執行完成！"
    echo "========================================"
    echo ""
    echo "輸出檔案："
    ls -lh output/predictions.csv 2>/dev/null && echo "  ✓ output/predictions.csv"
    ls -lh output/features.csv 2>/dev/null && echo "  ✓ output/features.csv"
    ls -lh output/feature_comparison.csv 2>/dev/null && echo "  ✓ output/feature_comparison.csv"
    echo ""
    echo "查看預測結果："
    echo "  head -10 output/predictions.csv"
else
    echo ""
    echo "========================================"
    echo "✗ 執行失敗"
    echo "========================================"
    echo ""
    echo "請查看錯誤訊息或執行："
    echo "  tail -50 output/execution.log"
fi

