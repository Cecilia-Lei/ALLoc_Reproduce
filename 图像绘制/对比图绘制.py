import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei']

# 设置负号显示
plt.rcParams['axes.unicode_minus'] = False

# 定义文件路径和对应的标签、样式（统一为实线）
files = [
    {
        'path': 'E:\\HCI\\lab\\training_log_Proposed_Scheme.xlsx',
        'label': 'ALLoc',
        'color': 'blue',
        'marker': 'o',
        'linestyle': '-'  # 统一为实线
    },
    {
        'path': 'E:\\HCI\\lab\\training_log_AD_CNN.xlsx',
        'label': 'AD_CNN',
        'color': 'red',
        'marker': 's',
        'linestyle': '-'  # 统一为实线
    },
    {
        'path': 'E:\\HCI\\lab\\training_log_CNN.xlsx',
        'label': 'CNN',
        'color': 'green',
        'marker': '^',
        'linestyle': '-'  # 统一为实线
    },
    {
        'path': 'E:\\HCI\\lab\\training_log_MFCNet.xlsx',
        'label': 'MFCNet',
        'color': 'purple',
        'marker': 'd',
        'linestyle': '-'  # 统一为实线
    }
]

# 定义横坐标显示2000的倍数
x_ticks = [0, 2000, 4000, 6000, 8000, 10000]  # 根据数据范围调整

# 绘制每条曲线
for file in files:
    # 读取Excel文件
    df = pd.read_excel(file['path'], sheet_name='Sheet1')

    # 提取数据
    x = df['epoch']
    y = df['avg_distance']

    # 平滑曲线处理 - 使用三次样条插值
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)  # k=3 表示三次样条
    y_smooth = spl(x_smooth)

    # 绘制平滑曲线（添加alpha参数降低不透明度）
    plt.plot(x_smooth, y_smooth,
             color=file['color'],
             linestyle=file['linestyle'],
             linewidth=1.0,
             alpha=0.7,  # 曲线不透明度（0-1之间，值越小越透明）
             label=file['label'])

    # 只在关键节点标记数据点
    for epoch in x_ticks:  # 使用2000倍数的刻度作为关键节点
        closest_idx = np.argmin(np.abs(x - epoch))
        if abs(x[closest_idx] - epoch) < 50:  # 适当放宽误差范围
            plt.plot(x[closest_idx], y[closest_idx],
                     color=file['color'],
                     marker=file['marker'],
                     markersize=6,
                     markeredgewidth=1.5,
                     markeredgecolor='black')

# 设置纵坐标为对数坐标
plt.yscale('log')

# 设置图表标题和坐标轴标签
plt.title('不同模型avg_distance随epochs变化对比图')
plt.xlabel('epochs')
plt.xticks(x_ticks, rotation=45)  # 只显示2000的倍数
plt.ylabel('avg_distance')

# 添加网格线
plt.grid(True, which="both", ls="-", alpha=0.2)

# 添加图例
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()