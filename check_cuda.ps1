python src/depth_surge_3d/utils/check_cuda.py

ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -encoders | Select-String "nvenc"
ffmpeg -hide_banner -decoders | Select-String "cuvid"