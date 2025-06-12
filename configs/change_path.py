import json
import os

def modify_paths(jsonl_file, base_path="/new/path/to/files"):
    # 获取文件名
    file_name = os.path.basename(jsonl_file)
    
    # 获取文件所在目录
    dir_name = os.path.dirname(jsonl_file)
    
    # 创建新的文件路径，添加 modified_ 前缀
    modified_file_path = os.path.join(dir_name, "modified_" + file_name)

    # 打开并读取原始的 JSONL 文件
    with open(jsonl_file, "r") as infile, open(modified_file_path, "w") as outfile:
        for line in infile:
            data = json.loads(line)
            
            # 修改音频路径
            if "audio" in data:
                audio_path = data["audio"]
                if audio_path.startswith("/home/Userlist/kongzicheng"):
                    new_audio_path = audio_path.replace("/home/Userlist/kongzicheng", base_path)
                    data["audio"] = new_audio_path

            # 修改视频路径
            if "video" in data:
                video_path = data["video"]
                if video_path.startswith("/home/kongzicheng"):
                    new_video_path = video_path.replace("/home/kongzicheng", base_path)
                    data["video"] = new_video_path
            
            # 写回修改后的内容
            outfile.write(json.dumps(data) + "\n")

# 调用函数时直接指定新的基础路径
modify_paths("/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/audio/sft_data.jsonl", base_path="/home")
