import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Model parameters
N = 100  # 粒子个数
L = 10  # 边界大小
v = 0.05  # 粒子速度
eta = 0.1  # 噪声
a = 0.25  #个体半径
r_neighbor = 1.0  # 邻居监视范围

# 初始化粒子位置和角度
theta = 2 * np.pi * np.random.rand(N)
x = L * np.random.rand(N)
y = L * np.random.rand(N)

fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)

def vicsek(x, y, theta, dt):
    new_theta = np.zeros(N)
    x_new = np.zeros(N)
    y_new = np.zeros(N)
    for i in range(N):
        # 计算距离并应用周期性边界条件
        dx = x - x[i]
        dy = y - y[i]
        dx = dx - L * np.round(dx / L)
        dy = dy - L * np.round(dy / L)

        distance = np.sqrt(dx ** 2 + dy ** 2)
        in_radius = distance < r_neighbor

        #求个体i与个体j的相对角度
        if np.all (dx == 0) and np.all (dy == 0) :
            gama = np.pi / 2
            alpha = 0
        elif np.all (dx ==0) and np.all (dy > 0) :
            gama = np.pi / 2
            alpha = 4 * np.arcsin(a /  dy ** 2)
        elif np.all (dx == 0) and np.all (dy < 0) :
            gama = 3 * np.pi / 4
            alpha = 4 * np.arcsin(a /  dy ** 2)
        elif np.all (dy == 0) and np.all (dx > 0) :
            gama = 0
            alpha = 4 * np.arcsin(a / dx ** 2)
        elif np.all (dy == 0) and np.all (dx < 0) :
            gama = np.pi
            alpha = 4 * np.arcsin(a / dx ** 2)
        else :
            gama = np.arccos(dx / np.sqrt(dx ** 2 + dy ** 2))
            gama = np.clip(gama, 0, 2 * np.pi)
            alpha = 4 * np.arcsin(a / np.sqrt(dx ** 2 + dy ** 2))
            alpha = np.clip(alpha, 0, np.pi)
            if np.any(dy < 0):
                gama = 2 * np.pi - gama
            else:
                gama = gama

        #判别是否处危险角
        import random
        i = random.uniform(0, 2 * np.pi)
        def check_valid(i):
            for x in np.arange (0, 2 * np.pi, 0.01):
                if np.any(i >= gama - alpha / 2) or np.any(i <= gama + alpha / 2) :
                    return False
                return true
        while not check_valid(i):
            i = random.uniform(0 , 2 * np.pi)
        avg_angle = i

    #更新速度方向
    new_theta[i] = avg_angle

    # 更新位置
    x_new = (x + v * np.cos(new_theta)) % L
    y_new = (y + v * np.sin(new_theta)) % L

    return x_new, y_new, new_theta

def update(frame):
    global x, y, theta
    x, y, theta = vicsek(x, y, theta, 0.1)
    plt.cla()
    plt.scatter(x, y)
    plt.quiver(x, y, v * np.cos(theta), v * np.sin(theta), color='r')
    plt.xlim(0, L)
    plt.ylim(0, L)

# 创建动画
animation = FuncAnimation(fig, update, frames=200, interval=100)
plt.show()


