import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据（经纬度和数值）
np.random.seed(42)
x = np.random.uniform(-180, 180, 100)  # 经度
y = np.random.uniform(-90, 90, 100)    # 纬度
values = np.random.randint(10, 500, 100)  # 数值

# 动态缩放圆圈大小（基准大小 + 比例缩放）
base_size = 20  # 基础大小
scaled_sizes = base_size * (values / np.max(values))  # 按比例缩放

# 绘制散点图
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    x, y,
    s=scaled_sizes,    # 圆圈大小映射数值
    c=values,          # 颜色映射数值（可选）
    cmap='viridis',    # 颜色方案
    alpha=0.7,         # 透明度
    edgecolors='black' # 边框颜色
)

# 添加颜色条和图例
plt.colorbar(scatter, label='Value')
plt.xlabel('Lon')
plt.ylabel('Lat')
plt.title('Spatial Data (Radius - value)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
