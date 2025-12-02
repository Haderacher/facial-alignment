import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard
import cv2
import numpy as np
import os
import math
import random

# ================= 配置参数 =================
DATA_DIR = './processed_data'
LABEL_FILE = './processed_data/landmarks_list.txt'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
IMG_SIZE = 224
LOG_DIR = 'runs/face_alignment_experiment' # TensorBoard 日志目录
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

EXPERIMENT_TYPE = 'Wing'

if EXPERIMENT_TYPE == 'MSE':
    LOG_DIR = 'runs/experiment_MSE'
    SAVE_NAME = 'best_model_mse.pth'
else:
    LOG_DIR = 'runs/experiment_Wing'
    SAVE_NAME = 'best_model_wing.pth'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定为: {seed}")

# 1. 定义 Wing Loss
class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred, target):
        diff = target - pred
        abs_diff = torch.abs(diff)
        flag = (abs_diff < self.w).float()
        y = flag * (self.w * torch.log(1 + abs_diff / self.epsilon)) + \
            (1 - flag) * (abs_diff - self.C)
        return torch.mean(y)

# 2. 定义 Dataset
class FaceLandmarksDataset(Dataset):
    def __init__(self, image_dir, labels_file):
        self.image_dir = image_dir
        self.landmarks_frame = []
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.landmarks_frame = f.readlines()

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        line = self.landmarks_frame[idx].strip().split()
        img_name = line[0]
        landmarks = np.array([float(x) for x in line[1:]]).astype('float32')
        landmarks = landmarks / IMG_SIZE 
        
        image_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(image_path)
        if image is None:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.zeros(136)
            
        image = image.astype('float32') / 255.0
        image = image.transpose((2, 0, 1)) 
        return torch.from_numpy(image), torch.from_numpy(landmarks)

# 3. 定义模型
def get_model():
    model = models.resnet18(pretrained=True)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 68 * 2)
    return model

# --- 新增辅助函数：在 TensorBoard 中画图 ---
def draw_landmarks_on_batch(images, landmarks_gt, landmarks_pred):
    """
    将一个 batch 的 tensor 图片转换为画了关键点的 numpy 图片
    images: (B, 3, H, W)
    landmarks: (B, 136) 归一化的
    """
    vis_imgs = []
    # 取 Batch 中的前 4 张图展示，节省资源
    limit = min(4, images.shape[0])
    
    for i in range(limit):
        # 1. Tensor -> Numpy (H, W, 3) & 反归一化像素值
        img = images[i].cpu().numpy().transpose((1, 2, 0)).copy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 转回 BGR 用于 cv2 画图
        
        # 2. 解析坐标
        gt = landmarks_gt[i].cpu().numpy().reshape(-1, 2) * IMG_SIZE
        pred = landmarks_pred[i].cpu().numpy().reshape(-1, 2) * IMG_SIZE
        
        # 3. 画点：真实值=绿色，预测值=红色
        for (gx, gy) in gt:
            cv2.circle(img, (int(gx), int(gy)), 2, (0, 255, 0), -1) # Green
        for (px, py) in pred:
            cv2.circle(img, (int(px), int(py)), 2, (0, 0, 255), -1) # Red
            
        # 转回 RGB 供 TensorBoard 显示
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vis_imgs.append(img)
        
    # Stack 成 (4, H, W, 3) -> (4, 3, H, W)
    vis_imgs = np.array(vis_imgs).transpose((0, 3, 1, 2))
    return torch.from_numpy(vis_imgs)

# 4. 训练主循环
def main():
    set_seed(42)
    # 初始化 TensorBoard
    print(f"TensorBoard 日志将保存在: {LOG_DIR}")
    print("----------------------------------------------------------------")
    print(f"训练开始 | 设备: {device} | Batch Size: {BATCH_SIZE} | 学习率: {LEARNING_RATE}")
    print("----------------------------------------------------------------")
    
    dataset = FaceLandmarksDataset(DATA_DIR, LABEL_FILE)
    
    # 划分数据集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = get_model().to(device)
    # criterion = WingLoss() # 或者 nn.MSELoss()
    if EXPERIMENT_TYPE == 'MSE':
        print("当前实验模式: MSE Loss")
        criterion = nn.MSELoss()
    else:
        print("当前实验模式: Wing Loss")
        criterion = WingLoss() # 使用之前定义的 WingLoss 类
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(LOG_DIR)
    best_loss = float('inf')
    global_step = 0
    
    # 定义打印频率：每多少个 Batch 打印一次？
    PRINT_FREQ = 10 

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # === 训练循环 ===
        for i, (images, landmarks) in enumerate(train_loader):
            images = images.to(device)
            landmarks = landmarks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            global_step += 1
            
            # ---【新增】实时打印训练信息 ---
            if (i + 1) % PRINT_FREQ == 0:
                # 计算当前 Epoch 的平均 Loss (截至目前)
                current_avg_loss = running_loss / (i + 1)
                
                print(f"Epoch [{epoch+1}/{EPOCHS}] "
                      f"Step [{i+1}/{len(train_loader)}] "
                      f"Batch Loss: {loss.item():.6f} | "
                      f"Avg Loss: {current_avg_loss:.6f}")
                
                # 写入 TensorBoard
                writer.add_scalar('Loss/Train_Step', loss.item(), global_step)

        avg_train_loss = running_loss / len(train_loader)
        
        # === 验证循环 ===
        model.eval()
        val_loss = 0.0
        vis_images, vis_targets, vis_outputs = None, None, None
        
        # 我们可以加一个简单的进度提示，防止验证集太大导致以为卡死
        # print("正在验证...", end='\r') 
        
        with torch.no_grad():
            for i, (images, landmarks) in enumerate(val_loader):
                images = images.to(device)
                landmarks = landmarks.to(device)
                outputs = model(images)
                loss = criterion(outputs, landmarks)
                val_loss += loss.item()
                
                if i == 0: # 保存第一批用于可视化
                    vis_images, vis_targets, vis_outputs = images, landmarks, outputs
        
        avg_val_loss = val_loss / len(val_loader)
        
        # === Epoch 结束总结 ===
        print("-" * 60)
        print(f"Epoch [{epoch+1}/{EPOCHS}] 完成 Summary:")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val   Loss: {avg_val_loss:.6f}")
        
        # 记录 Epoch 级 Loss 到 TensorBoard
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
        
        # 可视化图片
        if vis_images is not None:
            imgs_with_landmarks = draw_landmarks_on_batch(vis_images, vis_targets, vis_outputs)
            writer.add_images('Predictions vs GT', imgs_with_landmarks, epoch)

        # 保存最优模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ 发现更优模型，已保存 (Loss: {best_loss:.6f})")
        
        print("-" * 60) # 分割线
    
    writer.close()
    print("所有训练任务完成。")
if __name__ == "__main__":
    main()