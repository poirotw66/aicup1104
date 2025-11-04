#!/bin/bash
# 檢查執行進度

echo "========================================"
echo "檢查程式執行狀態"
echo "========================================"
echo ""

# 檢查程式是否在執行
if pgrep -f "main_fast.py" > /dev/null; then
    echo "✓ 程式正在執行中"
else
    echo "✗ 程式未在執行"
fi

echo ""
echo "最新 log 輸出（最後 30 行）："
echo "----------------------------------------"
if [ -f output/execution_fast.log ]; then
    tail -30 output/execution_fast.log
else
    echo "Log 檔案尚未建立"
fi

echo ""
echo "========================================"
echo "輸出檔案狀態："
echo "========================================"
for file in output/features.csv output/predictions.csv output/feature_comparison.csv; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "✓ $file ($size)"
    else
        echo "✗ $file (尚未產生)"
    fi
done

echo ""
echo "執行時間："
if [ -f output/execution_fast.log ]; then
    start_time=$(grep "Start time:" output/execution_fast.log | head -1)
    echo "$start_time"
    echo "當前時間: $(date '+%Y-%m-%d %H:%M:%S')"
fi

