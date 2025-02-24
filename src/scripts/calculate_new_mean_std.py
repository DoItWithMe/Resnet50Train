import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count

def calculate_image_stats(image_path):
    """Calculate the sum and sum of squares of the pixel values for an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None, 0
    image = image / 255.0  # Normalize to [0, 1]
    pixel_sum = np.sum(image, axis=(0, 1))
    pixel_sum_squared = np.sum(image ** 2, axis=(0, 1))
    num_pixels = image.shape[0] * image.shape[1]
    return pixel_sum, pixel_sum_squared, num_pixels

def calculate_mean_and_std_parallel(image_paths, process_num):
    """Calculate the mean and standard deviation of images in the given paths using multiprocessing."""
    with Pool(process_num) as pool:
        results = pool.map(calculate_image_stats, image_paths)
    
    # Filter out None results
    results = [result for result in results if result[0] is not None]
    
    total_pixel_sum = np.sum([result[0] for result in results], axis=0)
    total_pixel_sum_squared = np.sum([result[1] for result in results], axis=0)
    total_num_pixels = np.sum([result[2] for result in results])

    mean = total_pixel_sum / total_num_pixels
    std = np.sqrt(total_pixel_sum_squared / total_num_pixels - mean ** 2)

    return mean, std, total_num_pixels

def get_image_paths(folder):
    """Get all image paths in the given folder and its subfolders."""
    image_paths = []
    image_extensions = ['.png', '.jpg', '.jpeg']
    
    for root, dirs, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    
    return image_paths

# ImageNet 数据集的均值和标准差
mu1 = np.array([0.485, 0.456, 0.406])
sigma1 = np.array([0.229, 0.224, 0.225])

# ImageNet 数据集大小 (约 1,281,167 张图片)
n1 = 1281167

# 获取新图片的路径
new_images_folder = '/data/jinzijian/resnet50/assets/vt-imgs-for-test'  # 替换为你的新图片文件夹路径
image_paths = get_image_paths(new_images_folder)

# 计算新增数据集的均值、标准差和总像素数 (使用多进程)
mu2, sigma2, n2 = calculate_mean_and_std_parallel(image_paths, 20)
print("New mean alone: ", mu2)
print("New stad alone: ", sigma2)

# 计算新的均值
mu_new = (n1 * mu1 + n2 * mu2) / (n1 + n2)

# 计算新的标准差
sigma_new = np.sqrt((n1 * (sigma1**2 + (mu1 - mu_new)**2) + n2 * (sigma2**2 + (mu2 - mu_new)**2)) / (n1 + n2))

print("New mean:", mu_new)
print("New standard deviation:", sigma_new)
# =>
# New mean: [0.43922153 0.54558667 0.46546938]
# New standard deviation: [0.33420441 0.26491391 0.28718131]
