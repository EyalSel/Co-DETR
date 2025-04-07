#!/bin/bash

# Run latency profiling for all models
resolutions=(
    "1920 1280"
    "1920 1080"
    "960 640"
    "480 320"
    "240 160"
    "100 100"
    "200 200"
    "300 300"
    "400 400"
    "500 500"
    "600 600"
)

for resolution in "${resolutions[@]}"; do
    # Get width and height from resolution
    width=$(echo $resolution | cut -d' ' -f1)
    height=$(echo $resolution | cut -d' ' -f2)
    for batchsize in 1 2 4 8 16 32 64 128 256; do
        for model in co_detr_swin_T co_detr_swin_S co_detr_swin_B co_detr_swin_L_1000 co_dino_5scale_vit_large; do
            echo "Running: model=$model, batchsize=$batchsize, resolution=${width}x${height}"
            CUDA_VISIBLE_DEVICES=0 python profile_codetr_latency.py --model=$model --batchsize=$batchsize --width=$width --height=$height
            if [ $? -ne 0 ]; then
                echo "Error occurred, but continuing with next configuration..."
            fi
        done
    done
done
