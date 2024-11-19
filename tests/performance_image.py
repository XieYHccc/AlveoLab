import matplotlib.pyplot as plt
import numpy as np

# 数据准备
labels = ['65468/130944', '76566/153092', '73679/147354', '28398/56792', '26127/52206',
          '26710/53420', '44624/89164', '35414/70775', '73844/147684', '87297/174590',
          '83338/166672', '70762/141520']
time_a = np.random.rand(12)
time_b = np.random.rand(12)
time_c = np.random.rand(12)
time_d = np.random.rand(12)
time_e = np.random.rand(12)

x = np.arange(len(labels))  # 标签位置
width = 0.15  # 条形图宽度

# 设置深色调颜色
colors = {
    'a': '#1f77b4',  # 深蓝
    'b': '#9467bd',  # 深紫
    'c': '#2ca02c',  # 深绿
    'd': '#d62728',  # 深红
    'e': '#8c564b'   # 深棕
}

# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 2 * width, time_a, width, label='Time (a)', color=colors['a'])
ax.bar(x - width, time_b, width, label='Time (b)', color=colors['b'])
ax.bar(x, time_c, width, label='Time (c)', color=colors['c'])
ax.bar(x + width, time_d, width, label='Time (d)', color=colors['d'])
ax.bar(x + 2 * width, time_e, width, label='Time (e)', color=colors['e'])

# 添加水平虚线
y_values = np.arange(0, 1.2, 0.2)  # 每隔0.2画一条线
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