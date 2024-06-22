import cv2
import matplotlib.pyplot as plt
import os

def calculate_laplacian_variance(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用拉普拉斯算子
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    return laplacian_var

def plot_laplacian_variances(directory):
    # 存储拉普拉斯方差值
    laplacian_variances = []
    
    # 按顺序处理每张图片
    for i in range(1, 12272):  # 从1到12271
        filename = f"train_{i:05d}.jpg"
        image_path = os.path.join(directory, filename)
        variance = calculate_laplacian_variance(image_path)
        if variance is not None:
            laplacian_variances.append(variance)
        else:
            laplacian_variances.append(0)  # 如果无法读取图像，添加0作为占位符

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(laplacian_variances, label='Laplacian Variance')
    plt.title('Laplacian Variance of Images')
    plt.xlabel('Image Index')
    plt.ylabel('Variance')
    plt.legend()
    
    # 保存图表到同一目录
    plt.savefig(os.path.join(directory, 'laplacian_variance_plot.png'))
    plt.close()  # 关闭图形窗口以释放资源
# 指定图像目录
directory = 'RAF_og'
plot_laplacian_variances(directory)
