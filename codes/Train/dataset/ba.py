import os
import shutil

# RAF目录和RAF_synthetic目录的路径
raf_dir = 'RAF'
raf_synthetic_dir = 'RAF_synthetic'

# 标签和对应的文件夹名称
labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# 读取list_patition_label.txt文件,获取RAF目录中每个类别的训练图像数量
label_counts = [0] * 7
train_lines = []
test_lines = []
with open('list_patition_label.txt', 'r') as f:
    for line in f:
        filename, label = line.strip().split()
        if filename.startswith('train'):
            label_counts[int(label) - 1] += 1
            train_lines.append(line)
        else:
            test_lines.append(line)

print(len(test_lines))

# 找到图像数量最多的类别(Happiness)的数量
max_count = max(label_counts)

# 读取list_patition_label_s.txt文件,获取RAF_synthetic目录中每个图像文件的标签信息
synthetic_labels = {}
with open('list_patition_label_s.txt', 'r') as f:
    for line in f:
        filename, label = line.strip().split()
        if filename.startswith('train'):
            synthetic_labels[filename] = int(label)

# 复制图像并更新list_patition_label.txt文件
train_count = len(train_lines) + 1
with open('list_patition_label.txt', 'w') as f:
    for line in train_lines:
        f.write(line)
            
    for filename, label in synthetic_labels.items():
        label_index = label - 1
        if label_index == 3 or label_counts[label_index] >= max_count:
            continue
        
        src_path = os.path.join(raf_synthetic_dir, filename)
        dst_file = f'train_{str(train_count).zfill(5)}.jpg'
        dst_path = os.path.join(raf_dir, dst_file)
        
        shutil.copy(src_path, dst_path)
        f.write(f'{dst_file} {label}\n')
        label_counts[label_index] += 1
        train_count += 1
            
    for line in test_lines:
        f.write(line)

print("复制完成,list_patition_label.txt文件已更新。")
