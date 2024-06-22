import os

# 定义训练集和测试集的路径
dataset_dir = "/lab/tangb_lab/30011373/zmj/92/dataset"
train_dir = os.path.join(dataset_dir, "RAF_synthetic")
test_dir = os.path.join(dataset_dir, "RAF_synthetic")
label_file = os.path.join(dataset_dir, "list_patition_label_v1.txt")

# 定义要删除的训练集图像数量
num_train_to_delete = 12271

# 读取标签文件
with open(label_file, "r") as f:
    lines = f.readlines()

# 获取训练集和测试集中的所有图像文件名和标签
train_files = []
test_files = []
for line in lines:
    filename, label = line.strip().split()
    if filename.startswith("train_"):
        train_files.append((filename, int(label)))
    else:
        test_files.append((filename, int(label)))

# 删除前 12271 张训练集图像
for i in range(num_train_to_delete):
    os.remove(os.path.join(train_dir, train_files[i][0]))
train_files = train_files[num_train_to_delete:]

# 重新编号训练集和测试集图像
new_train_files = [f"train_{i:05d}.jpg" for i in range(1, len(train_files) + 1)]
new_test_files = [f"test_{i:05d}.jpg" for i in range(1, len(test_files) + 1)]

# 更新训练集图像文件名
for i, (old_filename, label) in enumerate(train_files):
    os.rename(os.path.join(train_dir, old_filename), os.path.join(train_dir, new_train_files[i]))

# 更新测试集图像文件名
for i, (old_filename, label) in enumerate(test_files):
    os.rename(os.path.join(test_dir, old_filename), os.path.join(test_dir, new_test_files[i]))

# 更新标签文件
new_label_file = os.path.join(dataset_dir, "new_list_patition_label_v1.txt")
with open(new_label_file, "w") as f:
    for i, (train_file, label) in enumerate(zip(new_train_files, [t[1] for t in train_files])):
        f.write(f"{train_file} {label}\n")
    for i, (test_file, label) in enumerate(zip(new_test_files, [t[1] for t in test_files])):
        f.write(f"{test_file} {label}\n")

print("操作完成!")
