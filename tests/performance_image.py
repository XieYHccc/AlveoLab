import matplotlib.pyplot as plt
import numpy as np

# 数据准备
labels = ['46143/92282', '57326/114648', '38394/76784', '136022/272040', '28717/57430',
          '26625/53246', '30002/60000', '38268/76532', '39696/79388', '40528/81052',
          '39614/79228', '39167/78330']

time_a = np.array([0.2868, 0.2583, 0.3074, 0.4731, 0.1992, 0.1719, 0.2941, 0.1466, 0.2918, 0.3220, 0.2442, 0.2469])
time_b = np.array([0.2762, 0.2913, 2.3662, 1.0923, 0.3080, 0.2256, 0.3619, 1.8694, 0.2651, 0.2461, 0.3244, 2.6001])
time_c = np.array([0.4813, 0.5852, 0.3355, 1.3857, 0.2579, 0.3071, 0.2226, 0.2801, 0.4865, 0.4429, 0.4488, 0.2882])
time_d = np.array([1.2656, 1.4419, 3.2421, 3.7984, 0.9162, 0.8111, 1.0435, 2.5270, 1.2767, 1.2518, 1.2475, 3.3526])

x = np.arange(len(labels))  # 标签位置
width = 0.15  # 条形图宽度

# 设置深色调颜色
colors = {
    'a': '#1f77b4',  # 深蓝
    'b': '#9467bd',  # 深紫
    'c': '#2ca02c',  # 深绿
    'd': '#d62728'   # 深红
}

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 2 * width, time_a, width, label='Time (a)', color=colors['a'])
ax.bar(x - width, time_b, width, label='Time (b)', color=colors['b'])
ax.bar(x, time_c, width, label='Time (c)', color=colors['c'])
ax.bar(x + width, time_d, width, label='Time (d)', color=colors['d'])


# 添加水平虚线
y_values = np.arange(0, 4, 0.5)  # 每隔0.2画一条线
for y in y_values:
    ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# 添加标签和标题
ax.set_ylabel('Time/s')
ax.set_xlabel('The number of points/triangles')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

# 调整布局
fig.tight_layout()

# 导出为矢量图
plt.savefig("chart_with_grid.pdf")  # 导出为PDF格式
plt.savefig("chart_with_grid.svg")  # 导出为SVG格式
plt.show()