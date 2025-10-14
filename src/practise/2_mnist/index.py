import numpy as np
import kagglehub
import os
import src.utils.activation_func.softmax as softmax
import src.utils.activation_func.relu as relu
import src.utils.vision.handwriting_web.app as webapp

# 用kagglehub下载mnist数据集，不需要每次执行都重复下载，也不需要手动处理数据集，比较方便
_datasetPath = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", _datasetPath)

# 神经网络定义
# 为了简化问题，我们不实现根据数据集自动调整网络结构的功能
# 输入固定位MNIST，输入28*28 + 一个颜色通道，输出固定是0-9这10个类别
input_size = 28 * 28  # 输入层节点数
hidden_size = 128  # 隐藏层节点数，可以调整
output_size = 10  # 输出层节点数

# 超参数
learning_rate = 0.1  # 学习率，可以调整
num_epochs = 10  # 训练轮数，可以调整
batch_size = 32  # mini-batch的批处理大小，可以调整
momentum = 0.9  # 动量，可以调整


def load_mnist_images(filename):
    with open(os.path.join(_datasetPath, filename, filename), "rb") as f:
        # 读取文件头信息
        # magic number用于标记文件类型，2051表示图片文件，2049表示标签文件
        # num_images表示图片数量
        # rows和cols表示图片的行数和列数（像素）
        magic, num_images, rows, cols = np.frombuffer(
            f.read(16), dtype=np.uint32
        ).byteswap()

        # 读取图像数据, 现在是一个一维数组，元素个数=num_images*rows*cols=60000*28*28
        images = np.frombuffer(f.read(), dtype=np.uint8)

        # 将这个一维数组按照图片切分
        images = images.reshape(num_images, rows * cols)

        # 将像素值归一化到0-1之间
        images = images.astype(np.float32) / 255.0

        # print("Filename \"{}\", Images {}, Size {}x{}, Shape: {}".format(filename, num_images, rows, cols, images.shape))
        return images


def load_mnist_labels(filename):
    with open(os.path.join(_datasetPath, filename, filename), "rb") as f:
        # 读取文件头信息
        # magic number用于标记文件类型，2051表示图片文件，2049表示标签文件
        # num_labels表示标签数量
        magic, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()

        # 读取标签数据, 现在是一个一维数组，元素个数=num_labels=60000
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        # print("Filename \"{}\", Labels {}, Shape {}".format(filename, num_labels, labels.shape))
        return labels


def loaddataset():
    train_images = load_mnist_images("train-images-idx3-ubyte")
    train_labels = load_mnist_labels("train-labels-idx1-ubyte")
    test_images = load_mnist_images("t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("t10k-labels-idx1-ubyte")
    return (train_images, train_labels), (test_images, test_labels)


# 创建one hot标签, 把image和label关联起来
def one_hot_encode(labels, num_classes=10):
    # 创建一个全零矩阵，行数为标签数量，列数为类别数量
    one_hot = np.zeros((labels.shape[0], num_classes))
    # 将对应类别的位置设为1
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


# 我们设计一个三层的神经网络，输入层-隐藏层-输出层

# 初始化权重和偏置，使用了He初始化
np.random.seed(42)  # 为了结果可复现，设置随机种子
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))  # 隐藏层偏置
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))  # 输出层偏置
velocity_W1 = np.zeros_like(W1)
velocity_b1 = np.zeros_like(b1)
velocity_W2 = np.zeros_like(W2)
velocity_b2 = np.zeros_like(b2)


# 损失函数用交叉熵损失
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]

    # 避免log(0)的情况
    y_pred_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # 选出每一个样本，对应真实类别的预测概率
    loss = -np.log(y_pred_clip[range(m), np.argmax(y_true, axis=1)])

    # 求平均值
    loss = np.sum(loss) / m
    return loss


# 向前传播
def forward(x):
    z1 = np.dot(x, W1) + b1
    a1 = relu.activation(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax.activation(z2)
    return z1, a1, z2, a2


# 向后传播
def backward(x, y, z1, a1, z2, a2):
    m = y.shape[0]

    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu.derivative(z1)
    dW1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def update_parameters(dW1, db1, dW2, db2):
    global W1, b1, W2, b2, velocity_W1, velocity_b1, velocity_W2, velocity_b2
    # 使用动量法更新参数
    velocity_W1 = momentum * velocity_W1 + learning_rate * dW1
    velocity_b1 = momentum * velocity_b1 + learning_rate * db1
    velocity_W2 = momentum * velocity_W2 + learning_rate * dW2
    velocity_b2 = momentum * velocity_b2 + learning_rate * db2
    W1 -= velocity_W1
    b1 -= velocity_b1
    W2 -= velocity_W2
    b2 -= velocity_b2


# 训练模型

count = 0


def train_model():
    (train_images, train_labels), (test_images, test_labels) = loaddataset()
    train_labels_one_hot = one_hot_encode(train_labels)
    test_labels_one_hot = one_hot_encode(test_labels)

    for epoch in range(num_epochs):

        for i in range(0, train_images.shape[0], batch_size):
            global count
            image_batch = train_images[i : i + batch_size]
            label_batch = train_labels_one_hot[i : i + batch_size]

            # 向前传播
            z1, a1, z2, a2 = forward(image_batch)

            # 计算损失
            loss = compute_loss(label_batch, a2)

            # 向后传播
            dW1, db1, dW2, db2 = backward(image_batch, label_batch, z1, a1, z2, a2)

            # 更新参数
            update_parameters(dW1, db1, dW2, db2)
            count += 1
            if count % 100 == 0:
                print(
                    f"Count {count} Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size+1}, Loss: {loss:.4f}"
                )

        # # 向前传播
        # z1, a1, z2, a2 = forward(train_images)

        # # 计算损失
        # loss = compute_loss(train_labels_one_hot, a2)
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        # # 向后传播
        # dW1, db1, dW2, db2 = backward(
        #     train_images, train_labels_one_hot, z1, a1, z2, a2
        # )

        # # 更新参数
        # update_parameters(dW1, db1, dW2, db2)

    # 在测试集上评估模型
    _, _, _, test_a2 = forward(test_images)
    test_predictions = np.argmax(test_a2, axis=1)
    accuracy = np.mean(test_predictions == test_labels)
    print(f"Test Accuracy For Test Suite: {accuracy * 100:.2f}%")

    _, _, _, train_a2 = forward(train_images)
    train_predictions = np.argmax(train_a2, axis=1)
    accuracy = np.mean(train_predictions == train_labels)
    print(f"Test Accuracy For Train Suite: {accuracy * 100:.2f}%")


def predict(image):
    # image是一个一维数组，长度是28*28=784，值是0-255
    # 需要把它转换成和训练时一样的格式
    image = np.array(image).astype(np.float32) / 255.0  # 归一化到0-1
    image = image.reshape(1, -1)  # 转换成二维数组，行数是1，列数是784

    # 向前传播
    _, _, _, a2 = forward(image)

    # 取最大值的索引作为预测结果
    prediction = np.argmax(a2, axis=1)[0]
    return prediction


if __name__ == "__main1__":
    now = os.times()
    train_model()
    print("Time used:", os.times()[0] - now[0], "seconds")
    webapp.Launch(predict)

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = loaddataset()
    first_image = train_images[0]
    print("First image shape:", first_image.shape)
    print("First image label:", first_image)
    import matplotlib.pyplot as plt

# 跟上一章节的区别：
# 1. 增加了mini-batch的支持
# 2. 增加了He初始化
# 3. 增加了动量法
