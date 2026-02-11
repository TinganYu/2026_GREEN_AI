#!/bin/bash
# 環境安裝腳本 - 解決依賴順序問題

set -e  # 遇到錯誤就停止

echo "=== 階段 1: 安裝 PyTorch ==="
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 triton==3.5.1

echo "=== 階段 2: 安裝需要 torch 的編譯套件 ==="
# --no-build-isolation: 使用當前環境編譯，避免隔離環境找不到 torch
pip install --no-build-isolation "fast_hadamard_transform @ git+https://github.com/Dao-AILab/fast-hadamard-transform.git@f134af63deb2df17e1171a9ec1ea4a7d8604d5ca"

echo "=== 階段 3: 安裝其餘依賴 ==="
pip install -r requirements_clean.txt

echo "=== 安裝完成 ==="
