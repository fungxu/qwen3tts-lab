#!/bin/zsh
# 从零搭建运行环境：Python 3.12 venv + 依赖 + 上游 qwen-tts 源码
# 已装过可重复执行，会跳过存在的部分
set -e
cd "$(dirname "$0")"

PY="${PYTHON_BIN:-python3.12}"
UPSTREAM="https://github.com/QwenLM/Qwen3-TTS.git"

# ── 1. 虚拟环境 ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "→ 创建 .venv（Python 3.12）"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  else
    "$PY" -m venv .venv
  fi
fi

PIP=(uv pip install --python .venv/bin/python)
command -v uv >/dev/null 2>&1 || PIP=(.venv/bin/python -m pip install)

# ── 2. 依赖 ────────────────────────────────────────────────
echo "→ 安装依赖"
"$PIP[@]" -r requirements.txt

# ── 3. 上游 qwen-tts 源码 ──────────────────────────────────
# 官方仓库未发布到 PyPI，只能从源码引入
if [ ! -d "_src/qwen_tts" ]; then
  echo "→ 克隆 QwenLM/Qwen3-TTS"
  git clone --depth 1 "$UPSTREAM" _src
fi

# 用 .pth 注册到 site-packages（比 editable 安装更省事，避免构建缓存问题）
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "$PWD/_src" > "$SP/qwen_tts_src.pth"

.venv/bin/python -c "from qwen_tts import Qwen3TTSModel; print('→ qwen-tts 就绪')"
echo "完成。接着跑：./run.sh"
