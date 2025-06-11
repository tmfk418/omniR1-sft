import json
import os
from pathlib import Path
import shutil

def update_video_field(input_file, output_file=None, backup=True):
    """
    强制统一视频路径字段为 'video_path'，无论原字段是 'video' 还是 'video_path'
    """
    input_path = Path(input_file)
    if output_file is None:
        output_file = input_file
    
    # 备份原文件
    if backup and input_path.exists():
        backup_path = input_path.with_suffix('.bak')
        shutil.copy(input_path, backup_path)
        print(f"✅ 备份已创建: {backup_path}")

    updated_count = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            try:
                data = json.loads(line.strip())
                
                # 关键修改：强制统一字段名
                if 'video' in data or 'video_path' in data:
                    # 获取视频路径（优先取video_path，其次取video）
                    path = data.pop('video_path', None) or data.pop('video', None)
                    if path:
                        data['video_path'] = path
                        updated_count += 1
                
                # 写入更新后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"❌ 无效JSON行: {line.strip()}")
                continue
    
    print(f"\n处理结果:")
    print(f"- 总处理记录: {updated_count}")
    print(f"- 输出文件: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="输入JSONL文件路径")
    parser.add_argument("--output", help="输出文件路径(默认覆盖原文件)", default=None)
    parser.add_argument("--no-backup", help="禁用自动备份", action="store_true")
    args = parser.parse_args()
    
    update_video_field(
        input_file=args.input,
        output_file=args.output,
        backup=not args.no_backup
    )