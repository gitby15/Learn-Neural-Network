
import numpy as np
import kagglehub
import os
import src.utils.activation_func.softmax as softmax
import src.utils.activation_func.relu as relu

# 用kagglehub下载mnist数据集，不需要每次执行都重复下载，也不需要手动处理数据集，比较方便
_datasetPath = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", _datasetPath)


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
        print(images.shape)

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


def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)  # He初始化
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢出
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs

    def backward(self, X, y, learning_rate):
        m = X.shape[0]  # 样本数

        # 输出层误差
        dz2 = self.probs - y  # 对于 softmax + 交叉熵 的梯度
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        # 隐藏层误差
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # 更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def compute_loss(self, probs, y):
        # 交叉熵损失
        m = y.shape[0]
        
        log_probs = -np.log(probs[range(m), np.argmax(y, axis=1)])
        loss = np.sum(log_probs) / m
        return loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        true = np.argmax(y, axis=1)
        return np.mean(preds == true)


def main():
    # 加载数据
    (train_images, train_labels), (test_images, test_labels) = loaddataset()

    # 数据预处理
    X_train = train_images
    Y_train = one_hot(train_labels)
    X_test = test_images
    Y_test = one_hot(test_labels)

    # 创建网络：784 -> 128 -> 10
    input_size = 784
    hidden_size = 128
    output_size = 10
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    # 超参数
    epochs = 20
    batch_size = 64
    learning_rate = 0.01
    n_samples = X_train.shape[0]

    # 记录训练过程
    losses = []
    accuracies = []

    for epoch in range(epochs):
        # 小批量训练
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            # 前向 + 反向 + 更新
            probs = nn.forward(X_batch)
            nn.backward(X_batch, Y_batch, learning_rate)

        # 每个 epoch 计算训练集 loss 和准确率
        train_probs = nn.forward(X_train)
        loss = nn.compute_loss(train_probs, Y_train)
        acc = nn.accuracy(X_train, Y_train)
        test_acc = nn.accuracy(X_test, Y_test)

        losses.append(loss)
        accuracies.append(test_acc)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()