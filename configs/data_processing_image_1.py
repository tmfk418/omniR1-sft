import os
import json
from PIL import Image
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


# === 1. 加载 JSON 数据（包含 image_path 和文本信息） ===
with open('/home/Userlist/kongzicheng/omni-critic-r1/selected preference dataset/image/doubao_image_3b_7b_output.json', 'r') as f:
    json_data = json.load(f)

# === 2. 加载 Arrow 图像数据集 ===
arrow_dataset = load_from_disk("/home/Userlist/kongzicheng/omni-critic-r1/rlaif-v-dataset/train")

# === 3. 构建 image_path → row_index 映射，避免一次性解码图像 ===
image_index = {
    path: i for i, path in enumerate(arrow_dataset["image_path"])
}
print(f"✅ Image index built: {len(image_index)} entries")

# === 4. 加载 Qwen 分词器（仅使用 tokenizer） ===
tokenizer = AutoTokenizer.from_pretrained("/home/Userlist/kongzicheng/omni-critic-r1/Qwen2.5-Omni-3B/qwen/Qwen2.5-Omni-3B")
print("✅ Tokenizer loaded.")

# === 5. 构建 Dataset（来自 JSON） ===
dataset = Dataset.from_dict({
    "image_path": [entry["image_path"] for entry in json_data],
    "question": [entry["question"] for entry in json_data],
    "qwen2_5_vl_3b_output": [entry["qwen2_5_vl_3b_output"] for entry in json_data],
    "qwen2_5_vl_7b_output": [entry["qwen2_5_vl_7b_output"] for entry in json_data],
    "score_A": [entry["score_A"] for entry in json_data],
    "score_B": [entry["score_B"] for entry in json_data],
    "better": [entry["better"] for entry in json_data],
    "reasoning": [entry["reasoning"] for entry in json_data],
    "final_verdict": [entry["final_verdict"] for entry in json_data],
    "identical": [entry["identical"] for entry in json_data]
})
print("✅ JSON dataset loaded with", len(dataset), "entries.")

# === 6. 图像加载函数（根据 image_path 从 Arrow 数据中查找） ===
def load_image_by_path(image_path):
    try:
        idx = image_index.get(image_path, None)
        if idx is None:
            raise ValueError("Image path not found")
        image = arrow_dataset[idx]["image"]
        return image.convert("RGB")
    except Exception as e:
        print(f"⚠️ Error loading image: {image_path} | {e}")
        return None

# === 7. 数据预处理函数 ===
def preprocess_data(example):
    image = load_image_by_path(example['image_path'])

    question_encoding = tokenizer(example['question'], truncation=True, padding="max_length", max_length=256)
    answer_3b_encoding = tokenizer(example['qwen2_5_vl_3b_output'], truncation=True, padding="max_length", max_length=1024)
    answer_7b_encoding = tokenizer(example['qwen2_5_vl_7b_output'], truncation=True, padding="max_length", max_length=1024)

    return {
        'image': image,
        'question': question_encoding,
        'answer_3b': answer_3b_encoding,
        'answer_7b': answer_7b_encoding,
        'score_A': example['score_A'],
        'score_B': example['score_B'],
        'better': example['better'],
        'reasoning': example['reasoning'],
        'final_verdict': example['final_verdict'],
        'identical': example['identical']
    }

# === 8. 应用预处理 ===
print("🚀 Starting dataset preprocessing...")
dataset = dataset.map(preprocess_data)
print("✅ Preprocessing complete.")

# === 9. 划分训练集 / 验证集 ===
split = dataset.train_test_split(test_size=0.2)
train_dataset = split['train']
val_dataset = split['test']
print(f"✅ Train set: {len(train_dataset)}, Validation set: {len(val_dataset)}")

# === 10. 保存至磁盘 ===
output_dir = '/home/Userlist/kongzicheng/omni-critic-r1/OmniCritic-SFT/sft_dataset/image/3b_7b'
train_dataset.save_to_disk(os.path.join(output_dir, "train"))
val_dataset.save_to_disk(os.path.join(output_dir, "val"))
print("💾 Datasets saved to:", output_dir)




