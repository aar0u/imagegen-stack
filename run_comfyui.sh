#!/usr/bin/env bash
# Linux defaults to GPU (ROCm on the 780M); set COMFYUI_DEVICE=cpu for CPU mode.
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMAGEGEN_DIR="${IMAGEGEN_DIR:-${HOME}/imagegen}"
COMFY_DIR="${COMFY_DIR:-${IMAGEGEN_DIR}/runtime/comfyui}"
COMFY_REPO_URL="${COMFY_REPO_URL:-https://github.com/comfy-org/ComfyUI.git}"
OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then
  DEVICE="${COMFYUI_DEVICE:-gpu}"
else
  DEVICE="${COMFYUI_DEVICE:-cpu}"
fi
ROCM_INDEX="https://rocm.nightlies.amd.com/v2/gfx110X-all/"

case "$DEVICE" in cpu | gpu) ;; *) echo "COMFYUI_DEVICE must be cpu or gpu" >&2; exit 2 ;; esac
[ "$DEVICE" = "cpu" ] || [ "$OS" = "Linux" ] || { echo "COMFYUI_DEVICE=gpu is Linux only" >&2; exit 2; }
VENV=".venv"
[ "$DEVICE" = "gpu" ] && VENV=".venv-rocm"

echo "=================================================="
echo "  Anima-Turbo (ComfyUI 自动化运行脚本)"
echo "  ComfyUI: ${COMFY_DIR} (${DEVICE})"
echo "=================================================="

# 0. 检查 ComfyUI 目录，不存在则自动 clone
if [ ! -d "$COMFY_DIR" ]; then
  echo "📥 检测到未安装 ComfyUI，正在克隆至 ${COMFY_DIR} ..."
  git clone "$COMFY_REPO_URL" "$COMFY_DIR"
fi

REF_ADAPTER_DIR="${COMFY_DIR}/custom_nodes/ComfyUI-Anima-Ref-Adapter"
if [ ! -d "$REF_ADAPTER_DIR" ]; then
  mkdir -p "${COMFY_DIR}/custom_nodes"
  git clone https://github.com/ArcAnima/ComfyUI-Anima-Ref-Adapter.git "$REF_ADAPTER_DIR"
fi

cd "$COMFY_DIR"
export UV_CACHE_DIR="${COMFY_DIR}/.uv_cache"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "${IMAGEGEN_DIR}/logs"
LOG_FILE="${IMAGEGEN_DIR}/logs/comfyui.log"

# 1. 检查模型文件（共享目录 ${IMAGEGEN_DIR}/models，ComfyUI 与 sd.cpp 共用）
echo "[1/3] 检查 Anima 模型文件..."
"${RUNNER_DIR}/fetch_models.sh"

# ComfyUI loads this file from its root and finds the shared models through it.
cat > extra_model_paths.yaml <<EOF
imagegen:
  base_path: ${IMAGEGEN_DIR}/models/
  is_default: true
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  vae: vae/
  loras: loras/
EOF

# 2. 检查 Python 虚拟环境与依赖（维护在 ${COMFY_DIR}/${VENV}）
echo "[2/3] 检查 Python 运行环境与依赖..."
torch_check="import yaml, torch"
[ "$DEVICE" = "gpu" ] && torch_check="import yaml, torch; assert torch.version.hip"
if [ ! -d "$VENV" ] || ! "$VENV/bin/python" -c "$torch_check" 2>/dev/null; then
  echo "📦 配置 Python 3.12 虚拟环境并安装依赖..."
  uv venv --python 3.12 --seed --allow-existing "$VENV"
  pip_install() { uv pip install --python "$VENV/bin/python" "$@"; }
  if [ "$OS" = "Darwin" ]; then
    echo "🍏 检测到 macOS，安装通用 PyTorch (MPS 加速)..."
    pip_install torch torchvision torchaudio
  elif [ "$DEVICE" = "gpu" ]; then
    echo "🐧 检测到 Linux，安装 ROCm 7.13 原生 gfx110X PyTorch (nightly)..."
    # --index outranks PyPI, so torch resolves to the AMD build; torchvision/torchaudio must come from the same index.
    pip_install --index "$ROCM_INDEX" "torch==2.11.0+rocm7.13.0a20260423" torchvision torchaudio
  else
    echo "🐧 检测到 Linux，安装 CPU 版 PyTorch..."
    pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
  pip_install -r requirements.txt
fi

# 3. 平台参数
EXTRA_ARGS=()
if [ "$OS" = "Darwin" ]; then
  echo "[3/3] 配置 macOS (Apple Silicon MPS) 运行环境..."
  EXTRA_ARGS=(--use-split-cross-attention)
elif [ "$DEVICE" = "gpu" ]; then
  echo "[3/3] Linux: GPU (ROCm). If the amdgpu driver hangs again, see README.md."
  # fp16 because the 780M has no bf16 matrix instructions; the other flags are the conservative set of the old 32 s setup.
  EXTRA_ARGS=(--fp16-unet --use-split-cross-attention --cpu-vae)
  if command -v journalctl >/dev/null 2>&1; then
    journalctl -k -f -n 0 --no-pager | grep --line-buffered -E 'MES failed|failed to remove|Failed to evict|GPU reset|ring .* timeout' >> "${IMAGEGEN_DIR}/logs/kernel_comfyui.log" &
    trap 'kill $! 2>/dev/null || true' EXIT
  fi
else
  echo "[3/3] Linux: CPU only"
  # Even a stray torch.cuda call must not create a GPU queue, so hide the device from a ROCm build too.
  export HIP_VISIBLE_DEVICES=-1
  # One thread per physical core: SMT siblings slow bf16 matmul (~5% on a 7840HS).
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)}"
  "$VENV/bin/python" "${RUNNER_DIR}/patch_cpu_bf16.py" "$COMFY_DIR"
  EXTRA_ARGS=(--cpu --bf16-unet --use-pytorch-cross-attention)
fi

echo "--------------------------------------------------"
echo "ComfyUI 运行地址: http://127.0.0.1:8188"
echo "日志文件: ${LOG_FILE}"
echo "--------------------------------------------------"

"$VENV/bin/python" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
