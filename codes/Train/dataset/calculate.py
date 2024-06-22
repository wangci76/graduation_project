import os
from PIL import Image
import matplotlib.pyplot as plt

def count_images_by_individual_kb(directory):
    # 初始化大小范围字典，0KB到12KB每个KB一个条目，以及一个12KB+的条目
    size_ranges = {f"{i}KB": 0 for i in range(13)}
    size_ranges['12KB+'] = 0
    
    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                # 检查文件是否为图片
                with Image.open(os.path.join(root, file)) as img:
                    # 获取文件大小
                    file_size = os.path.getsize(os.path.join(root, file)) / 1024  # 转换为KB
                    file_size_int = int(file_size)

                    # 根据文件大小增加相应的计数
                    if file_size_int <= 12:
                        size_ranges[f"{file_size_int}KB"] += 1
                    else:
                        size_ranges['12KB+'] += 1
            except IOError:
                # 如果文件不是图片，忽略它
                continue

    return size_ranges

def plot_histogram(size_ranges, directory):
    # 准备数据
    labels = list(size_ranges.keys())
    values = list(size_ranges.values())

    # 创建直方图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='blue')
    plt.xlabel('Image Size (KB)')
    plt.ylabel('Number of Images')
    plt.title('Histogram of Image Sizes')
    plt.xticks(rotation=45)

    # 保存图表
    plt.savefig(os.path.join(directory, 'image_size_histogram.png'))
    plt.close()

# 调用函数并打印结果
directory = 'RAF_og'  # 你的图片目录
result = count_images_by_individual_kb(directory)
for size_range, count in result.items():
    print(f"{size_range}: {count}张")

# 绘制并保存直方图
plot_histogram(result, directory)
