import torch
import numpy as np
import matplotlib.pyplot as plt
from backbones import get_model
from expression.datasets import RAFDBDataset, AfData
from expression.models import SwinTransFER
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix

dataset = 'a'

# 初始化模型
swin = get_model('swin_t')
net = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True)

# 准备数据集
if dataset == 'r':
    dataset_val = RAFDBDataset(choose="test",
                            data_path="dataset/RAF",
                            label_path="dataset/list_patition_label.txt",
                            img_size=112)
if dataset == 'a':
    dataset_val = AfData(path='/lab/tangb_lab/30011373/zmj/92/dataset/AffectNet_val/validation.csv',
                         directory='/lab/tangb_lab/30011373/zmj/92/dataset/AffectNet_val/imgs/',)

    mapping = {
        0: 6,  # Neutral -> Neutral
        1: 3,  # Happy -> Happiness
        2: 4,  # Sad -> Sadness
        3: 0,  # Surprise -> Surprise
        4: 1,  # Fear -> Fear
        5: 5,  # Anger -> Anger
        6: 2   # Disgust -> Disgust
    }

# 根据模型输出的类别编号，获取对应的类别名称
def get_emotion_label(output_class):
    return mapping[output_class]

test_loader = DataLoader(dataset_val, batch_size=128,
                         shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

# 检查点文件名列表
steps = range(5999, 100000, 6000)  # 从5999开始，每次增加6000
checkpoint_files = [
    f"results/copy2_checkpoint_step_{i}_gpu_0.pt" for i in steps
]

# 评估每个检查点
accuracies = []
mean_accuracies = []

# 在循环外初始化一个列表来存储所有的混淆矩阵
all_conf_matrices = []

# 定义类别名称
class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

for index, checkpoint_file in enumerate(checkpoint_files):
    # 加载模型状态
    dict_checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(dict_checkpoint["state_dict_model"])
    net.cuda()
    net.eval()

    # 评估
    bingo_cnt = 0
    sample_cnt = 0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))

    # 初始化列表来存储预测和真实标签
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, target, p in test_loader:
            if dataset == 'a':
                # 将PyTorch张量转换为Python列表
                class_indices = target.tolist()

                # 根据测试集类别序号，获取对应的模型输出类别序号
                model_output_class_indices = [get_emotion_label(idx) for idx in class_indices]

                # 将映射后的类别序号列表转换为PyTorch张量
                target = torch.tensor(model_output_class_indices)


            img = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            outputs, _ = net(img)
            _, predicts = torch.max(outputs, 1)

            # 收集预测和真实标签
            all_preds.extend(predicts.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            correct = torch.eq(predicts, target)
            bingo_cnt += correct.sum().cpu()
            sample_cnt += outputs.size(0)
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)
    accuracies.append(acc)
    
    mean_acc = np.mean([class_correct[i] / class_total[i] for i in range(7) if class_total[i] != 0])
    mean_acc = np.around(mean_acc, 4)
    mean_accuracies.append(mean_acc)

    # 计算混淆矩阵并存储
    cm = confusion_matrix(all_targets, all_preds)
    all_conf_matrices.append(cm)

# 绘制并保存准确率折线图，并显示每个点的数值
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracies, marker='o', linestyle='-', color='b')
for i, txt in enumerate(accuracies):
    plt.text(steps[i], accuracies[i], f'{txt:.4f}', fontsize=8, ha='right')
plt.title('Model Total Accuracy at Various Checkpoints')
plt.xlabel('Training Steps')
plt.ylabel('Total Accuracy')
plt.grid(True)
plt.xticks(steps, rotation=45)
plt.savefig('results/total_accuracy_plot'+'_'+str(dataset)+'.png')
plt.close()

# 绘制并保存平均准确率折线图，并显示每个点的数值
plt.figure(figsize=(10, 5))
plt.plot(steps, mean_accuracies, marker='o', linestyle='-', color='r')
for i, txt in enumerate(mean_accuracies):
    plt.text(steps[i], mean_accuracies[i], f'{txt:.4f}', fontsize=8, ha='right')
plt.title('Model Mean Accuracy at Various Checkpoints')
plt.xlabel('Training Steps')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.xticks(steps, rotation=45)
plt.savefig('results/mean_accuracy_plot'+'_'+str(dataset)+'.png')
plt.close()

# 绘制每个检查点的混淆矩阵
for step, cm in zip(steps, all_conf_matrices):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix at Step {step}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'results/confusion_matrix_step_{step}'+'_'+str(dataset)+'.png')
    plt.close()

##########################

dataset = 'r'

# 初始化模型
swin = get_model('swin_t')
net = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True)

# 准备数据集
if dataset == 'r':
    dataset_val = RAFDBDataset(choose="test",
                            data_path="dataset/RAF",
                            label_path="dataset/list_patition_label.txt",
                            img_size=112)
if dataset == 'a':
    dataset_val = AfData(path='/lab/tangb_lab/30011373/zmj/92/dataset/AffectNet_val/validation.csv',
                         directory='/lab/tangb_lab/30011373/zmj/92/dataset/AffectNet_val/imgs/',)

    mapping = {
        0: 6,  # Neutral -> Neutral
        1: 3,  # Happy -> Happiness
        2: 4,  # Sad -> Sadness
        3: 0,  # Surprise -> Surprise
        4: 1,  # Fear -> Fear
        5: 5,  # Anger -> Anger
        6: 2   # Disgust -> Disgust
    }

# 根据模型输出的类别编号，获取对应的类别名称
def get_emotion_label(output_class):
    return mapping[output_class]

test_loader = DataLoader(dataset_val, batch_size=128,
                         shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

# 检查点文件名列表
steps = range(5999, 100000, 6000)  # 从5999开始，每次增加6000
checkpoint_files = [
    f"results/copy2_checkpoint_step_{i}_gpu_0.pt" for i in steps
]

# 评估每个检查点
accuracies = []
mean_accuracies = []

# 在循环外初始化一个列表来存储所有的混淆矩阵
all_conf_matrices = []

# 定义类别名称
class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

for index, checkpoint_file in enumerate(checkpoint_files):
    # 加载模型状态
    dict_checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(dict_checkpoint["state_dict_model"])
    net.cuda()
    net.eval()

    # 评估
    bingo_cnt = 0
    sample_cnt = 0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))

    # 初始化列表来存储预测和真实标签
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, target, p in test_loader:
            if dataset == 'a':
                # 将PyTorch张量转换为Python列表
                class_indices = target.tolist()

                # 根据测试集类别序号，获取对应的模型输出类别序号
                model_output_class_indices = [get_emotion_label(idx) for idx in class_indices]

                # 将映射后的类别序号列表转换为PyTorch张量
                target = torch.tensor(model_output_class_indices)


            img = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            outputs, _ = net(img)
            _, predicts = torch.max(outputs, 1)

            # 收集预测和真实标签
            all_preds.extend(predicts.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            correct = torch.eq(predicts, target)
            bingo_cnt += correct.sum().cpu()
            sample_cnt += outputs.size(0)
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)
    accuracies.append(acc)
    
    mean_acc = np.mean([class_correct[i] / class_total[i] for i in range(7) if class_total[i] != 0])
    mean_acc = np.around(mean_acc, 4)
    mean_accuracies.append(mean_acc)

    # 计算混淆矩阵并存储
    cm = confusion_matrix(all_targets, all_preds)
    all_conf_matrices.append(cm)

# 绘制并保存准确率折线图，并显示每个点的数值
plt.figure(figsize=(10, 5))
plt.plot(steps, accuracies, marker='o', linestyle='-', color='b')
for i, txt in enumerate(accuracies):
    plt.text(steps[i], accuracies[i], f'{txt:.4f}', fontsize=8, ha='right')
plt.title('Model Total Accuracy at Various Checkpoints')
plt.xlabel('Training Steps')
plt.ylabel('Total Accuracy')
plt.grid(True)
plt.xticks(steps, rotation=45)
plt.savefig('results/total_accuracy_plot'+'_'+str(dataset)+'.png')
plt.close()

# 绘制并保存平均准确率折线图，并显示每个点的数值
plt.figure(figsize=(10, 5))
plt.plot(steps, mean_accuracies, marker='o', linestyle='-', color='r')
for i, txt in enumerate(mean_accuracies):
    plt.text(steps[i], mean_accuracies[i], f'{txt:.4f}', fontsize=8, ha='right')
plt.title('Model Mean Accuracy at Various Checkpoints')
plt.xlabel('Training Steps')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.xticks(steps, rotation=45)
plt.savefig('results/mean_accuracy_plot'+'_'+str(dataset)+'.png')
plt.close()

# 绘制每个检查点的混淆矩阵
for step, cm in zip(steps, all_conf_matrices):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix at Step {step}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'results/confusion_matrix_step_{step}'+'_'+str(dataset)+'.png')
    plt.close()