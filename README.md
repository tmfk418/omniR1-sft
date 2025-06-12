# OmniR1-SFT

## 项目概述

这个仓库包含了使用 Swift 框架和自定义数据集进行多模态模型微调的代码。实验由以下几个部分组成：

- **代码仓库**：包含了训练框架 (`ms-swift-main`) 和 OmniCritic-SFT 项目文件。
- 
- **数据集**：包含音频、视频和图像数据集，这些数据集已经上传到 Hugging Face，用于训练模型。

## 步骤 1: 克隆所有分支

首先，克隆整个仓库，并包含所有分支。


git clone --branch main https://github.com/tmfk418/omniR1-sft.git

cd omniR1-sft

git fetch --all

git checkout master1  # 选择用于微调的框架分支

git checkout master   # 选择包含代码和jsonl数据的分支

##步骤 2: 设置虚拟环境

接下来，您需要创建一个虚拟环境，并安装必要的依赖项。

python3 -m venv sft_env

激活虚拟环境：

在 Linux/macOS 上：

source sft_env/bin/activate

pip install -r requirements.txt

##步骤 3: 下载数据集

需要下载的数据集托管在 Hugging Face 上。可以使用 huggingface_hub 库进行下载。

安装 huggingface_hub 库：

pip install huggingface_hub

下载数据集：

from huggingface_hub import hf_hub_download

下载音频、视频和图像数据集

image_dataset = hf_hub_download("TMFK/omnir1-dataset", "local_path")

video_dataset = hf_hub_download("TMFK/omnir1-dataset", "C:\Users\tmfk1\video\video.zip")

audio_dataset = hf_hub_download("TMFK/omnir1-dataset", "Clotho-AQA dataset.zip")

##步骤 4: 更新数据集路径

运行以下脚本来更新音频和视频路径：

更新路径

python configs/change_path.py

确保路径与您下载的文件相匹配。脚本会根据指定的基础路径更新 JSONL 文件中的音频和视频路径。

##步骤5：运行微调代码

bash scripts/sft_multi.sh

注意修改里面的所有路径

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export FPS=1
export FPS_MAX_FRAMES=35
export VIDEO_MAX_PIXELS=65536
export AUDIO_MEMMAP=true  # 启用内存映射加载音频
export AUDIO_CHUNK_SIZE=8000  # 按16k采样点分块
export AUDIO_NUM_WORKERS=4  # 专用音频解码进程

torchrun --nproc_per_node=6 /home/Userlist/kongzicheng/omni-critic-r1/ms-swift-main/swift/cli/sft.py \
  --model /home/kongzicheng/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \（模型路径也辛苦进行修改，直接下载Qwen2.5-Omni-3B和7B即可）
  --dataset /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/audio/sft_data.jsonl \（辛苦修改jsonl路径）
            /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/sft_data.jsonl \（辛苦修改jsonl路径）
            /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/video/sft_data.jsonl \（辛苦修改jsonl路径）
  --train_type full \
  --output_dir /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/outputs \（辛苦修改outputs路径）
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
  --deepspeed /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/configs/ds_zero2.json  2>&1 | tee /home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/logs/log_file.txt（辛苦修改ds_zero2.json及log_file.txt路径）

  现在的问题是爆显存，运行不了也没法调节具体的参数能使训练加快，我现在本地跑视频和音频都没问题，但几个step后会爆显存，视频的还一直爆显存，很抱歉第一次做微调也第一次上传文件和写readme，还请学长们见谅！
