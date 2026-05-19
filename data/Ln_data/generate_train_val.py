import os
import shutil
import random

# 设置源文件夹和目标文件夹路径
source_folder = '/home/scli/Ln_LigandDiff/data/Ln_data/all_ln_oct_xyz'
base_folder = os.path.dirname(source_folder)
train_folder = os.path.join(base_folder, 'train')
val_folder = os.path.join(base_folder, 'val')

# 获取所有.xyz文件的列表
xyz_files = [f for f in os.listdir(source_folder) if f.endswith('.xyz')]

# 打乱文件列表
random.shuffle(xyz_files)

# 按9:1的比例分配文件到训练集和验证集
split_index = len(xyz_files) * 9 // 10
train_files = xyz_files[:split_index]
val_files = xyz_files[split_index:]

# 创建目标文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 移动文件
for f in train_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(train_folder, f))

for f in val_files:
    shutil.move(os.path.join(source_folder, f), os.path.join(val_folder, f))

print("文件已成功分配。")
