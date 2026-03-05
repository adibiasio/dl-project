curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$PWD/uv-cache"
mkdir -p "$UV_CACHE_DIR"

uv venv
source .venv/bin/activate
uv sync
export KAGGLE_API_TOKEN=XXX
kaggle datasets download parthplc/facebook-hateful-meme-dataset
