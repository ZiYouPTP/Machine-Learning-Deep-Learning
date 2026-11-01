# These are the libraries will be used for this lab.
import numpy as np 
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Check PyTorch version:
print("PyTorch version:", torch.__version__)

# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]
def plotVec(vectors):
    ax = plt.axes()
    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.show()


# ===================
# 1. TYPES AND SHAPE 
# ===================
print("\n--- 1. TYPES AND SHAPE ---")

# Convert a integer list with length 5 to a tensor
ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The dtype of tensor object after converting it to tensor:", ints_to_tensor.dtype)
print("The type of tensor object after converting it to tensor:", ints_to_tensor.type())
print("Python type:", type(ints_to_tensor))

# Chuyển list số thực thành tensor
floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
print("The dtype of tensor object after converting it to tensor:", floats_to_tensor.dtype)
print("The type of tensor object after converting it to tensor:", floats_to_tensor.type())

# Ép kiểu ngay khi khởi tạo
list_floats = [0.0, 1.0, 2.0, 3.0, 4.0]
floats_int_tensor = torch.tensor(list_floats, dtype=torch.int64)
print("The dtype of tensor object is:", floats_int_tensor.dtype)
print("The type of tensor object is:", floats_int_tensor.type())

# Đổi kiểu dữ liệu của tensor đã có bằng .type()
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
print("The type of the new_float_tensor:", new_float_tensor.type())
old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor)
print("The type of the converted new_float_tensor:", new_float_tensor.type())

# Kiểm tra kích thước , số chiều và đổi hình dạng 
print("The size of the new_float_tensor:", new_float_tensor.size())
print("The dimension of the new_float_tensor:", new_float_tensor.ndimension())

twoD_float_tensor = new_float_tensor.view(5, 1)
print("Original Size:", new_float_tensor)
print("Size after view method (5, 1):", twoD_float_tensor)

twoD_float_tensor = new_float_tensor.view(-1, 1)
print("Size after view method (-1, 1):", twoD_float_tensor)

# Chuyển đổi qua lại với NumPy
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)
print("The dtype of new tensor from numpy:", new_tensor.dtype)
print("The type of new tensor from numpy:", new_tensor.type())

back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor:", back_to_numpy)
print("The dtype of numpy array:", back_to_numpy.dtype)

# Lưu ý: back_to_numpy và new_tensor trỏ cùng vùng nhớ với numpy_array
numpy_array[:] = 0
print("The new tensor points to numpy_array:", new_tensor)
print("and back to numpy array points to the tensor:", back_to_numpy)

# Chuyển đổi từ Pandas Series sang Tensor
pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
new_tensor = torch.from_numpy(pandas_series.values.copy()) # nếu không .copy() sẽ bị lỗi do pandas
print("The new tensor from numpy array:", new_tensor)
print("The dtype of new tensor:", new_tensor.dtype)
print("The type of new tensor:", new_tensor.type())

# Lấy giá trị Python thuần bằng .item() và .tolist()
this_tensor = torch.tensor([0, 1, 5, 3])
print("the first item is given by:", this_tensor[0].item(), "the first tensor value is given by:", this_tensor[0])
print("the second item is given by:", this_tensor[1].item(), "the second tensor value is given by:", this_tensor[1])
print("the third item is given by:", this_tensor[2].item(), "the third tensor value is given by:", this_tensor[2])

torch_to_list = this_tensor.tolist()
print('tensor:', this_tensor, "\nlist:", torch_to_list)


# =========================
# 2. INDEXING AND SLICING 
# =========================
print("\n--- 2. INDEXING AND SLICING ---")

index_tensor = torch.tensor([-5, 1, 11, 13, 47])
print("The value on index 0:", index_tensor[0])
print("The value on index 1:", index_tensor[1])
print("The value on index 2:", index_tensor[2])
print("The value on index 3:", index_tensor[3])
print("The value on index 4:", index_tensor[4])

# Thay đổi giá trị phần tử
tensor_sample = torch.tensor([20, 1, 2, 3, 4])
print("Inital value on index 0:", tensor_sample[0])
tensor_sample[0] = 100
print("Modified tensor:", tensor_sample)

print("Inital value on index 4:", tensor_sample[4])
tensor_sample[4] = 0
print("Modified tensor:", tensor_sample)

# Slicing 
subset_tensor_sample = tensor_sample[1:4]
print("Original tensor sample:", tensor_sample)
print("The subset of tensor sample:", subset_tensor_sample)

# Gán giá trị mới cho một đoạn slice
print("Inital value on index 3 and index 4:", tensor_sample[3:5])
tensor_sample[3:5] = torch.tensor([300.0, 400.0])
print("Modified tensor:", tensor_sample)

# Dùng biến list làm chỉ mục
selected_indexes = [3, 4]
subset_tensor_sample = tensor_sample[selected_indexes]
print("The inital tensor_sample:", tensor_sample)
print("The subset of tensor_sample with the values on index 3 and 4:", subset_tensor_sample)

print("The inital tensor_sample:", tensor_sample)
selected_indexes = [1, 3]
tensor_sample[selected_indexes] = 100000
print("Modified tensor with one value:", tensor_sample)


# =====================
# 3. TENSOR FUNCTIONS
# =====================
print("\n--- 3. TENSOR FUNCTIONS ---")

math_tensor = torch.tensor([1.0, -1.0, 1, -1])
print("Tensor example:", math_tensor)
print("The mean of math_tensor:", math_tensor.mean())
print("The standard deviation of math_tensor:", math_tensor.std())

max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
print("Tensor example:", max_min_tensor)
print("Maximum number in the tensor:", max_min_tensor.max())
print("Minimum number in the tensor:", max_min_tensor.min())

pi_tensor = torch.tensor([0, np.pi/2, np.pi])
sin = torch.sin(pi_tensor)
print("The sin result of pi_tensor:", sin)

len_5_tensor = torch.linspace(-2, 2, steps=5)
print("First Try on linspace (5 steps):", len_5_tensor)

len_9_tensor = torch.linspace(-2, 2, steps=9)
print("Second Try on linspace (9 steps):", len_9_tensor)

pi_tensor = torch.linspace(0, 2*np.pi, 100)
sin_result = torch.sin(pi_tensor)
plt.plot(pi_tensor.numpy(), sin_result.numpy())
plt.title("Sin Wave (0 to 2*pi)")
plt.show() # Đóng cửa sổ biểu đồ này để code chạy tiếp xuống dưới


# ======================
# 4. TENSOR OPERATIONS
# ======================
print("\n--- 4. TENSOR OPERATIONS ---")

u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
w = u + v
print("The result tensor u + v:", w)

plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])

# Broadcasting
u = torch.tensor([1, 2, 3, -1])
v = u + 1
print("Addition with constant (u + 1):", v)

# Scalar multiplication
u = torch.tensor([1, 2])
v = 2 * u
print("The result of 2 * u:", v)

# Hadamard product
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v
print("The result of u * v:", w)

# Dot Product
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
print("Dot Product of u, v:", torch.dot(u, v))
