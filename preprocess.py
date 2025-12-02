import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置参数 =================
RAW_DATA_DIR = './raw_300w'     # 原始解压路径 (包含 helen, lfpw 等文件夹)
OUTPUT_DIR = './processed_data' # 处理后的保存路径
IMG_SIZE = 224                  # 目标图片大小
PADDING_RATIO = 0.25            # 裁剪时向外扩充的比例 (保留更多下巴和额头)
# ===========================================

def parse_pts(filename):
    """解析 .pts 文件，返回 (68, 2) 的 numpy 数组"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 忽略头3行和最后1行
    points = []
    for line in lines[3:-1]:
        parts = line.strip().split()
        if len(parts) >= 2:
            points.append([float(parts[0]), float(parts[1])])
    return np.array(points)

def process_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 用于记录标签文件： 图片名 x1 y1 x2 y2 ...
    label_file_path = os.path.join(OUTPUT_DIR, 'landmarks_list.txt')
    label_lines = []

    # 遍历所有子文件夹
    sub_dirs = ['helen/trainset', 'lfpw/trainset', 'afw', 'ibug'] 
    # 注意：helen和lfpw分trainset和testset，这里演示取训练集
    
    img_count = 0

    print("开始处理数据...")
    
    # 递归查找所有 .pts 文件
    pts_files = []
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith('.pts'):
                pts_files.append(os.path.join(root, file))

    for pts_path in tqdm(pts_files):
        img_path = pts_path.replace('.pts', '.jpg')
        if not os.path.exists(img_path):
            img_path = pts_path.replace('.pts', '.png') # 有些可能是png
            if not os.path.exists(img_path):
                continue
        
        # 1. 读取图片和关键点
        img = cv2.imread(img_path)
        if img is None: continue
        landmarks = parse_pts(pts_path) # Shape: (68, 2)

        # 2. 计算包围盒 (Bounding Box)
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        w_box = x_max - x_min
        h_box = y_max - y_min

        # 3. 添加 Padding 并裁剪 (防止脸紧贴边缘)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        size = max(w_box, h_box) * (1 + PADDING_RATIO)
        
        x1 = int(cx - size / 2)
        y1 = int(cy - size / 2)
        x2 = int(cx + size / 2)
        y2 = int(cy + size / 2)

        # 边界检查（防止裁剪出负数坐标）
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0: continue

        # 4. 调整关键点坐标 (相对于裁剪后的左上角)
        landmarks[:, 0] -= x1
        landmarks[:, 1] -= y1

        # 5. 缩放 (Resize) 到目标大小 (224x224)
        h_crop, w_crop = cropped_img.shape[:2]
        resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
        
        scale_x = IMG_SIZE / w_crop
        scale_y = IMG_SIZE / h_crop
        
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y

        # 6. 保存处理后的图片和标签
        save_name = f"img_{img_count:05d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), resized_img)
        
        # 展平坐标并转为字符串
        flat_landmarks = landmarks.flatten()
        str_landmarks = ' '.join(map(str, flat_landmarks))
        label_lines.append(f"{save_name} {str_landmarks}\n")
        
        img_count += 1

    # 写入总标签文件
    with open(label_file_path, 'w') as f:
        f.writelines(label_lines)
    
    print(f"处理完成！共生成 {img_count} 张训练样本。")
    print(f"数据保存在: {OUTPUT_DIR}")
    print(f"标签文件在: {label_file_path}")

if __name__ == "__main__":
    process_dataset()