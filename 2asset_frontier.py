import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 资产参数设置
mu = [0.1, 0.15]    # 预期收益率
sigma = [0.12, 0.2] # 标准差

# 初始化相关系数
initial_rho = 0.5

# 生成投资组合权重
w = np.linspace(-0.5, 1.5, 100)  # 允许空头头寸

# 计算有效前沿的函数
def calculate_frontier(rho):
    cov = rho * sigma[0] * sigma[1]
    portfolio_returns = w*mu[0] + (1-w)*mu[1]
    portfolio_volatility = np.sqrt(
        (w**2)*(sigma[0]**2) + 
        ((1-w)**2)*(sigma[1]**2) + 
        2*w*(1-w)*cov
    )
    return portfolio_volatility, portfolio_returns

# 创建图表
fig, ax = plt.subplots(figsize=(10,6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# 绘制初始有效前沿
vol, ret = calculate_frontier(initial_rho)
frontier_line, = plt.plot(vol, ret, lw=2, color='b')

# 添加原始资产点
asset1_point = ax.scatter(sigma[0], mu[0], color='red', zorder=5, label='Asset 1')
asset2_point = ax.scatter(sigma[1], mu[1], color='green', zorder=5, label='Asset 2')
plt.legend()

plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Two-Asset Efficient Frontier')

# 添加相关系数滑块
ax_rho = plt.axes([0.2, 0.1, 0.6, 0.03])
rho_slider = Slider(
    ax=ax_rho,
    label='Correlation',
    valmin=-1,
    valmax=1,
    valinit=initial_rho,
)

# 更新函数
def update(val):
    rho = rho_slider.val
    new_vol, new_ret = calculate_frontier(rho)
    frontier_line.set_xdata(new_vol)
    frontier_line.set_ydata(new_ret)
    
    # 自动调整坐标轴范围
    ax.set_xlim(0, np.max(new_vol)*1.05)
    ax.set_ylim(np.min(new_ret)*0.95, np.max(new_ret)*1.05)
    
    fig.canvas.draw_idle()


# 注册更新事件
rho_slider.on_changed(update)

plt.show()