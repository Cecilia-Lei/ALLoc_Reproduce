import matplotlib.pyplot as plt
import pandas as pd

# 读取Excel文件（请替换为你的实际文件路径）
df = pd.read_excel('E:\\HCI\\lab\\training_log_Proposed_Scheme.xlsx', sheet_name='Sheet1')

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei']

# 设置负号显示
plt.rcParams['axes.unicode_minus'] = False

# 提取数据
x = df['epoch']
y = df['avg_distance']

# 绘制折线图
plt.plot(x, y, marker='o', markersize=3)  # 增加标记点，更清晰显示数据点

# 设置纵坐标为10的次方（对数坐标）
plt.yscale('log')

# 设置图表标题和坐标轴标签
plt.title('avg_distance随epochs变化图')
plt.xlabel('epochs')
plt.xticks(rotation=45)
plt.ylabel('avg_distance')

# 添加网格线，便于读取数值
plt.grid(True, which="both", ls="-", alpha=0.2)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
