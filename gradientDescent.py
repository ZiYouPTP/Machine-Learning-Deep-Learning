import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
import matplotlib.animation as animation

#--- Momentum ---
# check convergence
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new))/
                            len(theta_new) < 1e-3

def GD_momentum(theta_init, grad, eta, gamma):
    # Suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break 
        theta.append(theta_new)
        v_old = v_new
    return theta 
    # this variable includes all points in the path
    # if you just want the final answer, 
    # use `return theta[-1]`



#--- Stochastic Gradient Descent --- 
# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w


# --- 1. ĐỊNH NGHĨA CÁC HÀM TOÁN HỌC ---
def cost(x, A, B):
    m = A.shape[0]
    return 0.5/m * np.linalg.norm(A.dot(x) - B, 2)**2

def grad(x, A, B):
    m = A.shape[0]
    return 1/m * A.T.dot(A.dot(x) - B)

def check_grad(x, A, B):
    eps = 1e-4
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps 
        g[i] = (cost(x1, A, B) - cost(x2, A, B)) / (2 * eps)
    
    g_grad = grad(x, A, B)
    if np.linalg.norm(g - g_grad) > 1e-7:
        print("WARNING: CHECK GRADIENT FUNCTION")

def gradient_descent(x_init, learning_rate, iteration, A, B):
    x_list = [x_init]
    m = A.shape[0]

    for i in range(iteration):
        x_new = x_list[-1] - learning_rate * grad(x_list[-1], A, B)
        if np.linalg.norm(grad(x_new, A, B))/m < 0.3: # Điều kiện dừng sớm
            break
        x_list.append(x_new)
    
    return x_list

# --- 2. CHUẨN BỊ DỮ LIỆU ---
A = np.array([[2,9,7,9,11,16,25,23,22,29,29,35,37,40,46]]).T 
B = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]).T 

# Thiết lập figure
fig1 = plt.figure("GD for Linear Regression")
ax = plt.axes(xlim=(-10, 60), ylim=(-1, 20))
plt.title("Gradient Descent Animation")

# Vẽ điểm dữ liệu thực tế
plt.plot(A, B, 'ro', label='Data points')

# --- 3. ĐỐI CHIẾU VỚI SKLEARN ---
lr = linear_model.LinearRegression() 
lr.fit(A, B)
x0_gd = np.linspace(1, 46, 2)
y0_sklearn = lr.intercept_ + lr.coef_[0][0] * x0_gd
plt.plot(x0_gd, y0_sklearn, color="green", label='Solution by sklearn')

# Thêm cột bias (số 1) vào ma trận A để tính GD
ones = np.ones((A.shape[0], 1), dtype=np.int8)
A = np.concatenate((ones, A), axis=1)

# --- 4. CHẠY GRADIENT DESCENT ---
x_init = np.array([[1.], [2.]])
check_grad(x_init, A, B)

iteration = 100
learning_rate = 0.0001
x_list = gradient_descent(x_init, learning_rate, iteration, A, B)

# Vẽ đường khởi tạo (đen đậm)
y_init = x_init[0, 0] + x_init[1, 0] * x0_gd
plt.plot(x0_gd, y_init, color="black", label='Initial value for GD')

# Vẽ lịch sử các đường cập nhật (đen mờ)
for i in range(len(x_list)):
    y0_x_init = x_list[i][0, 0] + x_list[i][1, 0] * x0_gd
    plt.plot(x0_gd, y0_x_init, color='black', alpha=0.1)

# --- 5. ANIMATION ---
line, = ax.plot([], [], color="blue", label='GD Animation')

def update(i):
    y0_gd = x_list[i][0, 0] + x_list[i][1, 0] * x0_gd
    line.set_data(x0_gd, y0_gd)
    return line,

iters = np.arange(1, len(x_list), 1)
line_ani = animation.FuncAnimation(fig1, update, iters, interval=50, blit=True)

# Hiển thị chú thích an toàn, chính xác
plt.legend(loc="lower right")

plt.show()
