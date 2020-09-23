#!/usr/bin/env bash

# glow tts training requires a cython-optimized function, this script builds it so that the imports work correctly afterwards

echo "==========================================================="
echo "Make sure this is called from the right python environment!"
echo "==========================================================="

python monotonic_align/setup.py build_ext --inplace