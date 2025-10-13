# 0-1 实现一个神经网络 - XOR 问题
import numpy as np
import src.utils.activation_func.sigmoid as sigmoid

# 这是一个比较简单的问题，我的网络只有输入层、隐藏层、输出层
# 输入层的神经元数量为2（因为XOR问题有2个输入）
# 隐藏层的神经元数量为30（可以根据问题的复杂程度调整）
# 输出层的神经元数量为1（因为XOR问题是一个二分类问题）


# 训练样本
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)

# 初始化神经网络的参数

np.random.seed(42)  # 固定好随机种子，这样每次运行结果都一样，方便调试

# 输入层的维度，由问题的输入决定，我们的XOR问题有2个输入，所以输入层的维度为2
# 举一反三：如果我们的输入是一个图片，图片的像素是28*28，那么输入层的维度可能是28*28=784. 如果还要区分RGB，就要再乘以3
input_size = 2

# 隐藏层的维度，要怎么选型（什么时候用多少层）
# 隐藏层的维度，要根据问题的复杂程度来选择。如果问题比较简单，隐藏层的维度可以小一些。如果问题比较复杂，隐藏层的维度可以大一些。
# 一般来说，隐藏层的维度越大，神经网络的表达能力越强，但是也会增加计算量和过拟合的风险。
hidden_size = 4  # 隐藏层的神经元数量


# 简单任务(线性回归和低维分类)，一层就够了（万能近似定理）
# 复杂任务(图像、语音、自然语言)：需要深度网络（3~10+个隐藏层）
# 大型复杂任务(语音识别、图像分类、自然语言处理)：需要更深的网络（50+个隐藏层,还需要残差链接等技巧防止退化）
hidden_layer_count = (
    1  # 我们这里只用一层，因为我们才刚入门，更复杂的概念我现在还不会，等以后慢慢学习
)

# 输出层的维度，由问题的输出决定，我们的XOR问题有1个输出，所以输出层的维度为1
output_size = 1  # 输出层的神经元数量

# 初始化权重和偏置
W1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重矩阵, 形状为(2, 3)
b1 = np.zeros((1, hidden_size))  # 隐藏层的偏置量为0, 形状为(1, 3)

W2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重矩阵, 形状为(3, 1)
b2 = np.zeros((1, output_size))  # 输出层的偏置量为0, 形状为(1, 1)

# ------ 初始化参数结束 ------


# 定义激活函数
# 引入激活函数，有两个目的：1)引入非线性的能力;2)控制信息传递
# 1）引入非线性的能力：如果不引入非线性函数，神经网络就只能学习线性函数，无法解决复杂的问题。
# 2）控制信息传递：避免信息在神经网络中被压缩或放大，从而影响模型的性能。（梯度爆炸/消失）


# sigmoid 函数: f(x) = 1/(1+e^(-x))
# 优点：1）输出范围在(0,1)之间，这使得它非常适合用于二分类问题。2）函数是单调递增的，这使得它在反向传播中很方便计算梯度。
# 缺点：1）输出不是0均值的，这在某些情况下可能会影响模型的性能。2）当输入非常大或非常小的时候，函数的梯度会非常小，这会导致训练过程中的梯度消失问题。
# 适用场景：1）二分类问题。2）需要将输入映射到(0,1)之间的问题。
activation_function = sigmoid.activation
activation_function_derivative = sigmoid.derivative


# 向前传播
# 计算出每一层的结果，在反向传播的时候要用这些结果去计算梯度，用于优化神经网络中的W1，b1，W2，b2
def forward(X):
    # 隐藏层
    z1 = np.dot(X, W1) + b1
    a1 = activation_function(z1)
    # 输出层
    z2 = np.dot(a1, W2) + b2
    a2 = activation_function(z2)

    return z1, a1, z2, a2  # a2是输出层的输出，可以认为是这次计算的最终结果


# 定义损失函数：衡量a2与真实值y的差距，用于指导梯度下降
# 损失函数的选型，要根据问题的类型来选择。
# 1）回归问题（预测连续值，如房价预测，温度预测等）：常见：均方误差（MSE）和平均绝对误差（MAE）。
# 2）分类问题（预测离散的类别，如图片分类、文本分类）：常见：交叉熵损失函数（Cross-Entropy Loss）和合页损失（Hinge Loss）。
# 3）其他特殊任务（如序列标注、机器翻译等）：根据任务的具体要求选择损失函数。


# 二元交叉熵损失函数
def binary_cross_entropy(y_true, y_pred):
    # 避免log(0)的情况
    epsilon = 1e-15
    # 对y_pred的最大值和最小值做限制，避免出现log(0)或者log(1)的情况
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# 反向传播 + 参数更新
# 梯度下降通过迭代更新参数，使得损失函数最小化。
# 具体来说，梯度下降的过程如下：
# 1）计算损失函数对每个参数的梯度（导数）。
# 2）根据梯度的方向和大小，更新参数。
# 3）重复以上步骤，直到损失函数收敛（即梯度接近于0）。

# 常见的梯度下降：
# 1）批量梯度下降（Batch Gradient Descent）：每次使用所有样本计算梯度，更新参数。
# 2）随机梯度下降（Stochastic Gradient Descent）：每次使用一个样本计算梯度，更新参数，有一个改进方法叫Momentum, 可以加速收敛。
# 3）小批量梯度下降（Mini-batch Gradient Descent）：每次使用一小批量样本（如32个样本）计算梯度，更新参数。


def backward(X, y, z1, a1, z2, a2, learning_rate=0.2):
    global W1, W2, b1, b2
    # 这一步需要计算出输出结果的误差，用这个误差，去指导W1,W2,b1,b2的更新方向
    # 更新的公式: W -> W - 学习率 * d_loss/d_W
    # 更新的公式: b -> b - 学习率 * d_loss/d_b
    # 直接推导 d_loss/d_W 和d_loss/d_b 会非常复杂，所以我们会利用微积分求导的链式法则，一步一步往前推导
    # 先求出d_loss对a2的导数，再对z2求导，再一步一步往前推

    m = X.shape[0]  # 样本数量

    

    # loss = binary_cross_entropy(y, a2)
    # d_loss/d_a2 = (a2 - y) / (a2 * (1 - a2))
    # d_loss/d_z2 = d_loss/d_a2 * d_a2/d_z2 = (a2 - y) / (a2 * (1 - a2)) * a2 * (1 - a2) = (a2 - y) / m
    d_z2 = (a2 - y) / m

    d_W2 = np.dot(a1.T, d_z2)  # 损失对W2的导数
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)  # 损失对b2的导数

    # 隐藏层误差
    d_a1 = np.dot(d_z2, W2.T)  # 损失对a1的导数
    d_z1 = d_a1 * activation_function_derivative(z1)  # 损失对z1的导数
    d_W1 = np.dot(X.T, d_z1)  # 损失对W1的导数
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # 损失对b1的导数

    # 更新参数
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1


# 训练神经网络

epochs = 5000
for i in range(epochs):
    # 向前传播
    z1, a1, z2, a2 = forward(X)

    # 计算损失
    loss = binary_cross_entropy(y, a2)

    if i % (epochs/20) == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")

    # 反向传播
    backward(X, y, z1, a1, z2, a2, learning_rate=0.1)


INPUTT = np.array([[1, 1], [0, 1], [1, 1], [0, 0]], dtype=float)


# 训练结束后，查看最终的预测结果
_, _, _, predictions = forward(INPUTT)
print("\nFinal Predictions:")
print(predictions.round(4))  # 四舍五入到小数点后 4 位，看概率
print("Rounded Predictions (0 or 1):")
print(predictions.round())  # 直接四舍五入得到 0 或 1
