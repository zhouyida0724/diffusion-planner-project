#!/bin/bash
# 下载并配置 Bench2Drive 数据集
# 用法: ./scripts/download_bench2drive.sh

set -e

echo "=========================================="
echo "📊 下载 Bench2Drive 数据集"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
B2D_DIR="$DATA_DIR/bench2drive"

# 创建目录
mkdir -p "$B2D_DIR"

# 检查是否已存在
if [ -d "$B2D_DIR/Bench2Drive-Mini" ]; then
    echo "Bench2Drive-Mini 已存在，跳过下载"
    exit 0
fi

# 使用 huggingface-cli 下载
echo "下载 Bench2Drive-Mini (~20GB)..."
echo "如果没有安装 huggingface-cli，先运行: pip install huggingface_hub"

# 方法1: 使用 huggingface-cli (推荐)
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download Thinklab/Bench2Drive-Mini --local-dir "$B2D_DIR/Bench2Drive-Mini"
else
    # 方法2: 使用 Python
    echo "使用 Python 下载..."
    python3 << EOF
from huggingface_hub import snapshot_download
import os

# 设置代理
os.environ['HTTP_PROXY'] = os.environ.get('HTTP_PROXY', 'http://192.168.110.67:7890')
os.environ['HTTPS_PROXY'] = os.environ.get('HTTPS_PROXY', 'http://192.168.110.67:7890')

snapshot_download(
    repo_id="Thinklab/Bench2Drive-Mini",
    local_dir="$B2D_DIR/Bench2Drive-Mini",
    local_dir_use_symlinks=False
)
print("下载完成!")
EOF
fi

echo ""
echo "✅ Bench2Drive 下载完成!"
echo "   位置: $B2D_DIR/Bench2Drive-Mini"
