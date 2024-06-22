# 标签和对应的文件夹名称
labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# 初始化计数器
label_counts = [0] * 7

# 读取更新后的list_patition_label.txt文件,统计每个类别的图像数量
with open('list_patition_label.txt', 'r') as f:
    for line in f:
        _, label = line.strip().split()
        if _.startswith('train'):
            label_index = int(label) - 1
            label_counts[label_index] += 1

# 打印每个类别的图像数量
for i in range(7):
    print(f"{labels[i]}: {label_counts[i]} images")