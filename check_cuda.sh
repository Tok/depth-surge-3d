#!/bin/sh

python src/depth_surge_3d/utils/check_cuda.py

ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -encoders | grep nvenc
ffmpeg -hide_banner -decoders | grep cuvid