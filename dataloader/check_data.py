import os
import numpy as np


txt_file = "annotation/DFEW/th14_vit_g_16_4/set_1_test.txt"

print(f"正在读取标签文件: {txt_file}")
if not os.path.exists(txt_file):
    print("找不到 txt 标签文件，请检查 annotation 文件夹是否在当前目录下！")
    exit()

with open(txt_file, 'r') as f:
    lines = f.readlines()


first_line = lines[0].strip().split()
feature_path = first_line[0]
frames = first_line[1]
label = first_line[2]

print(f"\n成功读取标签，解析第一行数据:")
print(f" - 预期特征路径: {feature_path}")
print(f" - 预期帧数/参数: {frames}")
print(f" - 预期表情标签: {label}")


print(f"\n正在尝试加载特征文件...")
if os.path.exists(feature_path):
    data = np.load(feature_path)
    print(f"成功加载特征文件！")
    print(f"【关键信息】该特征数据的维度 (Shape) 是: {data.shape}")
else:
    print(f"找不到特征文件: {feature_path}")
    print("路径对齐失败：请确保 'get_features' 文件夹直接放在了当前运行目录下！")