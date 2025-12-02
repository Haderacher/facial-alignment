import cv2
import numpy as np

# 读取刚才生成的标签文件的第一行
with open('./processed_data/landmarks_list.txt', 'r') as f:
    line = f.readline().strip().split()

img_name = line[0]
coords = np.array([float(x) for x in line[1:]]).reshape(-1, 2)

img = cv2.imread(f'./processed_data/{img_name}')
for (x, y) in coords:
    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

cv2.imshow("Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()