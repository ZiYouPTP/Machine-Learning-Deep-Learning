import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
from sklearn import linear_model

#random data
#height
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')

#create model
lr = linear_model.LinearRegression()
#train the model
lr.fit(X,y)

print('w_1 (coef_) = ', lr.coef_[0][0])
print('w_0 (intercept_) = ', lr.intercept_[0])


x0 = np.array([[145, 185]]).T
# 2. Tính y0 tương ứng dựa trên mô hình đã train
y0 = lr.predict(x0) 
# (Tương đương với công thức thủ công: y0 = lr.intercept_ + lr.coef_[0][0] * x0)

# 3. Vẽ biểu đồ
plt.plot(X, y, 'ro')           # Vẽ các điểm dữ liệu thực tế
plt.plot(x0, y0)               # Vẽ đường thẳng dự đoán (fitting line)

# Căn chỉnh hiển thị
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()