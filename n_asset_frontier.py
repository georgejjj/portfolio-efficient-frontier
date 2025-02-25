import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from scipy.optimize import minimize

# 随机生成20个资产参数
np.random.seed(42)
n_assets = 20
mu = np.random.uniform(0.05, 0.2, n_assets)          # 收益率范围5%-20%
sigma = np.random.uniform(0.1, 0.3, n_assets)        # 波动率范围10%-30%
min_weight = 0.0   # 禁止卖空
max_weight = 1.0   # 最大权重100%

DIRICHLET_ALPHA = 0.3  # 更集中的分布

# 生成随机相关矩阵
corr_mat = np.random.uniform(-0.5, 0.7, (n_assets, n_assets))
corr_mat = (corr_mat + corr_mat.T)/2  # 对称化
np.fill_diagonal(corr_mat, 1)         # 对角线设为1
cov_mat = np.outer(sigma, sigma) * corr_mat  # 转换为协方差矩阵

# 初始化参数
initial_n = 2
n_simulations = 10000  # 蒙特卡洛模拟次数

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)
info_text = ax.text(0.5, 0.02, '', 
                   transform=fig.transFigure,
                   ha='center', va='bottom',
                   fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
asset_scatter = ax.scatter(sigma, mu, c='gray', alpha=0.5, label='All Assets')
portfolio_scatter = ax.scatter([], [], c='blue', alpha=0.3, s=10, label='Portfolios')  # 新增组合散点
ax.set_xlabel('Volatility')
ax.set_ylabel('Return')
ax.set_title('Multi-Asset Efficient Frontier')
ax.legend()

# 新增按钮区域
ax_buttons = plt.axes([0.2, 0.15, 0.6, 0.05], frameon=False)
ax_buttons.set_xticks([])
ax_buttons.set_yticks([])

# 创建控件
ax_slider = plt.axes([0.3, 0.1, 0.4, 0.03])  # 调整滑块位置
n_slider = Slider(
    ax=ax_slider,
    label='Assets Included: ',
    valmin=1,
    valmax=n_assets,
    valinit=initial_n,
    valstep=1
)

# 添加+/-按钮
ax_minus = plt.axes([0.1, 0.1, 0.05, 0.03])
ax_plus = plt.axes([0.8, 0.1, 0.05, 0.03])
btn_minus = Button(ax_minus, '-')
btn_plus = Button(ax_plus, '+')

# 更新权重生成函数

def generate_weights(n, size):
    # 基础Dirichlet分布生成（集中分布）
    weights = np.random.dirichlet(DIRICHLET_ALPHA * np.ones(n), size)
        
    
    # 确保权重和为1（二次归一化防止浮点误差）
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights[:size]


def update(val):
    n = int(n_slider.val)
    
    # 更新散点颜色
    asset_colors = ['red' if i < n else 'gray' for i in range(n_assets)]
    asset_scatter.set_color(asset_colors)
    
    # 生成并更新组合散点
    if n > 1:
        selected = slice(0, n)
        sub_mu = mu[selected]
        sub_cov = cov_mat[selected, :][:, selected]
        
        # 生成随机权重（无卖空）
        weights = generate_weights(n, n_simulations)
        
        # 计算组合收益风险
        portfolio_ret = weights @ sub_mu
        portfolio_vol = np.sqrt(np.diag(weights @ sub_cov @ weights.T))
        
        # 更新散点数据
        portfolio_scatter.set_offsets(np.c_[portfolio_vol, portfolio_ret])
        
        # 计算最小风险组合
        min_vol_idx = np.argmin(portfolio_vol)
        min_vol = portfolio_vol[min_vol_idx]
        min_vol_ret = portfolio_ret[min_vol_idx]

        # 更新信息显示
        info_text.set_text(f'Minimum Volatility Portfolio: σ={min_vol:.4f}, μ={min_vol_ret:.4f}\n'
                          f'Selected Assets: {n} assets (1-{n})')
        

        # 调整坐标轴范围
        ax.set_xlim(0, np.max(sigma)*1.05)
        ax.set_ylim(np.min(mu)*0.9, np.max(mu)*1.1)
    else:
        # 单个资产时清空组合散点
        portfolio_scatter.set_offsets(np.empty((0, 2)))
        info_text.set_text(f'Single Asset: σ={sigma[0]:.4f}, μ={mu[0]:.4f}')

    
    fig.canvas.draw_idle()

# 按钮回调函数
def increment(val):
    current = n_slider.val
    if current < n_assets:
        n_slider.set_val(current + 1)
        update(current + 1)

def decrement(val):
    current = n_slider.val
    if current > 1:
        n_slider.set_val(current - 1)
        update(current - 1)

# 绑定按钮事件
btn_plus.on_clicked(increment)
btn_minus.on_clicked(decrement)
n_slider.on_changed(update)
update(initial_n)  # 初始绘制
plt.show()