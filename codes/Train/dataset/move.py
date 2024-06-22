import os
import shutil

# 定义标签和对应的类别名称
label_to_class = {
    '1': 'surprise_facial_expression',
    '2': 'fear_facial_expression',
    '3': 'disgust_facial_expression',
    '4': 'happiness_facial_expression',
    '5': 'sadness_facial_expression',
    '6': 'anger_facial_expression',
    '7': 'neutral_facial_expression'
}

# 获取当前目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取RAF_expanded目录路径
raf_expanded_dir = os.path.join(current_dir, 'RAF_expanded')

# 获取cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x目录路径
clip_dir = os.path.join(current_dir, 'cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x')

# 获取已有训练图像的最大编号
max_train_num = 0
for filename in os.listdir(raf_expanded_dir):
    if filename.startswith('train_') and filename.endswith('.jpg'):
        train_num = int(filename[6:11])
        if train_num > max_train_num:
            max_train_num = train_num

# 打开list_patition_label.txt文件并读取内容
with open(os.path.join(current_dir, 'list_patition_label.txt'), 'r') as file:
    lines = file.readlines()

# 找到测试集开始的行号
test_start_index = next(i for i, line in enumerate(lines) if line.startswith('test_'))

# 重新打开list_patition_label.txt文件用于写入
with open(os.path.join(current_dir, 'list_patition_label.txt'), 'w') as file:
    # 写入训练集部分的内容
    file.writelines(lines[:test_start_index])

    # 遍历cifar100_stable_diffusion_scale50_strength0.9_CLIP_optimization_up0.8_batch_5x目录下的每个类别目录
    for label, class_name in label_to_class.items():
        class_dir = os.path.join(clip_dir, class_name)
        
        # 遍历当前类别目录下的图片文件
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                # 提取图片编号
                x, y = map(int, filename[7:-4].split('_expanded_'))
                
                # 生成新的图片编号和文件名
                max_train_num += 1
                new_filename = f'train_{max_train_num:05d}.jpg'
                
                # 移动图片到RAF_expanded目录并重命名
                src_path = os.path.join(class_dir, filename)
                dst_path = os.path.join(raf_expanded_dir, new_filename)
                shutil.move(src_path, dst_path)
                
                # 将新图片信息写入list_patition_label.txt文件
                file.write(f'{new_filename} {label}\n')

    # 写入测试集部分的内容
    file.writelines(lines[test_start_index:])

print("图片移动和重命名完成,list_patition_label.txt文件已更新。")