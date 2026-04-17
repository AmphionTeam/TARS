#!/bin/bash

wd=$(pwd)

cd scripts/install_xfinder
git clone https://github.com/IAAR-Shanghai/xFinder
pip install tenacity ipywidgets num2words word2number jiwer datasets==3.6.0 accelerate librosa soundfile transformers
cp pyproject.toml xFinder
cp answer_extractor.py xFinder/src/xfinder/modules
cp eval.py xFinder/src/xfinder
pip install -e xFinder

# python -m xfinder.eval --run-example --model-name xFinder-qwen1505 --inference-mode local

cd $wd