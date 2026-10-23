import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def cost(x):
    m = A.shape[0]
    return 0.5/m * np.linalg.norm(A.dot(x) - B, 2)**2

def grad(x):
    m = A.shape[0]
    return 1/m * A.T.dot(A.dot(x) - B)

def check_grad(x):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps

        g[i] = (cost(x1) - cost(x2))/(2*eps)
    g_grad = grad(x)
    if np.linalg.norm(g-g_grad) > 1e-7:
        print("WARNING: CHECK GRADIENT FUNCTION")

def gradient_descent(x_init, learning_rate, iteration):
    x_list = [x_init]
    m = A.shape[0]

    for i in range(iteration):
        x_new = x_list[-1] - learning_rate*grad(x_list[-1])
        if np.linalg.norm(grad(x_new))/m < 0.3: # khi nào dừng gd
            break
        x_list.append(x_new)
    
    return x_list

# dữ liệu
B = np.array([[2,9,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

# vẽ dữ liệu
fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(-10,50), ylim=(-10,50))
plt.plot(A, B, 'ro', label='Data points')

# thêm cột số 1 vào A
ones = np.ones((A.shape[0],1), dtype=np.int8)
A = np.concatenate((ones,A), axis=1)
# tạo bình phương cho A
x_square = np.array([A[:,1]**2]).T 
A = np.concatenate((A,x_square), axis=1)

# -- VẼ PARABOL BẰNG CÔNG THỨC
x_fomular = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(B)
x0_gd = np.linspace(1,46,10000)
y0_fomular = x_fomular[0][0] + x_fomular[1][0] * x0_gd + x_fomular[2][0] * x0_gd**2
plt.plot(x0_gd, y0_fomular, color='green', label='Solution by formula')

# đường khởi tạo ngẫu nhiên
x_init = np.array([[-2.1],[5.1],[-2.1]])
y_init = x_init[0][0] + x_init[1][0]*x0_gd + x_init[2][0]*x0_gd*x0_gd
plt.plot(x0_gd, y_init, color="black", label='Initial value for GD')

# chạy gradient descent
iteration = 70
learning_rate = 0.000001

x_list = gradient_descent(x_init, learning_rate, iteration)

# vẽ x_init (đường màu đen)
for i in range(len(x_list)):
    y0_x_init = x_list[i][0] + x_list[i][1]*x0_gd + x_list[i][2]*x0_gd*x0_gd
    plt.plot(x0_gd, y0_x_init, color='black', alpha=0.3)

line, = ax.plot([], [], color="blue", label='GD Animation')

# vẽ hoạt ảnh (animation)
def update(i):
    y0_gd = x_list[i][0][0] + x_list[i][1][0]*x0_gd + x_list[i][2][0]*x0_gd*x0_gd
    line.set_data(x0_gd, y0_gd)
    return line,

iters = np.arange(1, len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

# chú thích cho biểu đồ
plt.legend(loc="lower right")

plt.title("Gradient Descent Animation")

plt.show()