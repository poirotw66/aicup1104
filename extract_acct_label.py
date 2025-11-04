#!/usr/bin/env python3
"""
Extract acct and label columns from predictions CSV file
從預測結果中提取 acct 和 label 欄位
"""

import pandas as pd
import os


def extract_acct_label(input_file, output_file):
    """
    Extract only acct and label columns from the input CSV file
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    print(f"讀取檔案: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"錯誤: 找不到檔案 {input_file}")
        return
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if required columns exist
    if 'acct' not in df.columns or 'label' not in df.columns:
        print(f"錯誤: 檔案缺少必要欄位 'acct' 或 'label'")
        print(f"現有欄位: {list(df.columns)}")
        return
    
    # Select only acct and label columns
    df_filtered = df[['acct', 'label']]
    
    # Save to output file
    df_filtered.to_csv(output_file, index=False)
    
    print(f"\n✓ 成功輸出 {len(df_filtered)} 筆資料到 {output_file}")
    print(f"✓ 欄位: {list(df_filtered.columns)}")
    print(f"\n前5筆資料預覽:")
    print(df_filtered.head())
    print(f"\nlabel 分布:")
    print(df_filtered['label'].value_counts().sort_index())


if __name__ == "__main__":
    # Default input and output files
    input_file = "output/predictions.csv"
    output_file = "output/predictions_acct_label.csv"
    
    # Extract acct and label columns
    extract_acct_label(input_file, output_file)

