export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export FPS=1
export FPS_MAX_FRAMES=35
export VIDEO_MAX_PIXELS=65536
export AUDIO_MEMMAP=true  # 启用内存映射加载音频
export AUDIO_CHUNK_SIZE=8000  # 按16k采样点分块
export AUDIO_NUM_WORKERS=4  # 专用音频解码进程

torchrun --nproc_per_node=6 /home/Userlist/kongzicheng/omni-critic-r1/ms-swift-main/swift/cli/sft.py \
  --model /home/kongzicheng/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
  --dataset /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/audio/sft_data.jsonl \
            /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/sft_data.jsonl \
            /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/video/sft_data.jsonl \
  --train_type full \
  --output_dir /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/outputs \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 6 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.03 \
  --max_length 5120 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --dataloader_num_workers 8 \
  --torch_dtype bfloat16 \
  --deepspeed /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/configs/ds_zero2.json  2>&1 | tee /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/logs/log_file.txt
