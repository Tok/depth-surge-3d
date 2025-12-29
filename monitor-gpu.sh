#!/bin/bash

wslGPU=$(ls /dev/dxg)

if [[ -n "$wslGPU" ]]; then
    if [[ -f "/usr/lib/wsl/lib/nvidia-smi" ]]; then
        /usr/lib/wsl/lib/nvidia-smi
    elif [[ -f "rocminfo" ]]; then
        rocminfo
    fi
elif [[ -f "nvidia-smi" ]]; then
    nvidia-smi
fi
