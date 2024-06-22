import os

# 定义标签和对应的类别名称
label_to_class = {
    '1': 'Surprise',
    '2': 'Fear',
    '3': 'Disgust',
    '4': 'Happiness',
    '5': 'Sadness',
    '6': 'Anger',
    '7': 'Neutral'
}

# 遍历CIFAR_10000目录下的每个子目录
for class_name in os.listdir('CIFAR_10000'):
    # 构建旧目录路径
    old_dir_path = os.path.join('CIFAR_10000', class_name)
    
    # 构建新目录名称
    new_class_name = f'{class_name.lower()}_facial_expression'
    
    # 构建新目录路径
    new_dir_path = os.path.join('CIFAR_10000', new_class_name)
    
    # 重命名目录
    os.rename(old_dir_path, new_dir_path)
    
    # 统计当前类别目录下的图片数量
    num_images = len(os.listdir(new_dir_path))
    
    # 打印类别名称和图片数量
    print(f'{new_class_name}: {num_images} images')