import numpy as np

# 创建一个示例图像
img = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# 设置填充宽度
padding = 1

# 生成随机偏移量
shift_x, shift_y = 2, -1

# 对图像进行填充
padded_img = np.pad(img, ((padding, padding), (padding, padding)), 'constant')

# 获取填充后图像的形状
H, W = padded_img.shape

# 根据随机偏移量裁剪图像
cropped_img = padded_img[padding + shift_x: H + padding + shift_x, padding + shift_y: W + padding + shift_y]

print("原始图像:")
print(img)
print("\n填充后的图像:")
print(padded_img)
print("\n裁剪后的图像:")
print(cropped_img)

