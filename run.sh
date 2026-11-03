#!/bin/zsh
# Qwen3-TTS 本地工作台启动脚本
set -e
cd "$(dirname "$0")"

PORT="${QWEN_TTS_PORT:-8808}"

if [ ! -d ".venv" ]; then
  echo "未找到 .venv，请先按 README.md 安装依赖"
  exit 1
fi

# 三个模型各约 3.8 GB；缺哪个补哪个（ModelScope 直连）
download() {
  local dir="$1" repo="$2"
  if [ ! -f "models/$dir/model.safetensors" ]; then
    echo "→ 下载 $repo（约 3.8 GB）"
    python3 download_model.py "$repo" "./models/$dir"
  fi
}
download Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
download Qwen3-TTS-12Hz-1.7B-VoiceDesign Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
download Qwen3-TTS-12Hz-1.7B-Base        Qwen/Qwen3-TTS-12Hz-1.7B-Base

echo "→ http://127.0.0.1:${PORT}"
exec .venv/bin/python server.py
