#!/bin/bash
# 下载并配置 CARLA 仿真器
# 用法: ./scripts/download_carla.sh

set -e

echo "=========================================="
echo "🚗 下载 CARLA 0.9.15"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CARLA_DIR="$PROJECT_DIR/CARLA"

# 检查是否已存在
if [ -d "$CARLA_DIR/CarlaUE4" ]; then
    echo "CARLA 已存在，跳过下载"
    exit 0
fi

# 检查压缩包是否存在 (可能在 data/ 目录)
DATA_DIR="$PROJECT_DIR/data"
CARLA_TAR="$DATA_DIR/CARLA_0.9.15.tar.gz"

if [ ! -f "$CARLA_TAR" ]; then
    # 下载 CARLA
    echo "下载 CARLA 0.9.15 (~20GB)..."
    echo "请等待下载完成..."
    
    mkdir -p "$DATA_DIR"
    
    # 使用代理下载
    PROXY=""
    if [ -n "$HTTP_PROXY" ]; then
        PROXY="-x $HTTP_PROXY"
    fi
    
    wget $PROXY -O "$CARLA_TAR" \
        "https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz"
fi

# 解压
echo "解压 CARLA..."
tar -xvf "$CARLA_TAR" -C "$PROJECT_DIR/"

# 移动到正确位置
if [ -d "$PROJECT_DIR/CARLA_0.9.15" ]; then
    mv "$PROJECT_DIR/CARLA_0.9.15" "$CARLA_DIR"
fi

echo ""
echo "✅ CARLA 下载并配置完成!"
echo "   位置: $CARLA_DIR"
echo ""
echo "启动 CARLA:"
echo "   cd $CARLA_DIR"
echo "   ./CarlaUE4.sh -prefernvidia -nosound -carla-port=2000"
