#!/bin/bash
# RunPod 초기 설정 및 학습 실행 스크립트
# Usage: bash setup.sh --config config/exaone_2.4b.yaml

CONFIG=$1

# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 의존성 설치
cd /workspace/cafe_recomendation/keyword_selection
uv sync

# 학습 실행
uv run python train.py --config $CONFIG
