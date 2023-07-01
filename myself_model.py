from myself_tensor import myTensor
import numpy as np

'''定义神经网络类'''
class NeuralNetwork:
    def __init__(self):
        # 定义神经网络的构造函数
        # 初始化四个张量对象，作为神经网络的权重参数
        # 使用随机数填充张量数据，并乘以一个缩放因子，以避免梯度消失或爆炸
        self.w1 = myTensor(np.random.randn(4, 10) * np.sqrt(2 / 4))
        self.w2 = myTensor(np.random.randn(10, 20) * np.sqrt(2 / 10))
        self.w3 = myTensor(np.random.randn(20, 10) * np.sqrt(2 / 20))
        self.w4 = myTensor(np.random.randn(10, 3) * np.sqrt(2 / 10))
        # 初始化四个张量对象，作为神经网络的偏置参数
        # 使用零填充张量数据
        self.b1 = myTensor(np.zeros((1, 10)))
        self.b2 = myTensor(np.zeros((1, 20)))
        self.b3 = myTensor(np.zeros((1, 10)))
        self.b4 = myTensor(np.zeros((1, 3)))
        # 将所有的参数放入一个列表中，方便后续的优化器使用
        self.params = [self.w1, self.w2, self.w3, self.w4, self.b1, self.b2, self.b3, self.b4]

    def forward(self, x):
        # 定义神经网络的前向传播方法
        # 如果x是张量对象，则将其数据作为输入，否则将x本身作为输入
        if isinstance(x, myTensor):
            x = x.data
        # 计算第一层的输出，使用线性变换,无激活函数
        z1 = x.dot(self.w1) + self.b1
        a1 = z1
        # 计算第二层的输出，使用线性变换和Tanh激活函数
        z2 = a1.dot(self.w2) + self.b2
        a2 = tanh(z2)
        # 计算第三层的输出，使用线性变换和Tanh激活函数
        z3 = a2.dot(self.w3) + self.b3
        a3 = tanh(z3)
        # 计算第四层的输出，使用线性变换和LogSoftmax激活函数
        z4 = a3.dot(self.w4) + self.b4
        a4 = logsoftmax(z4) # 使用LogSoftmax激活函数，将线性变换的结果转换为对数概率，并使得它们的和为0
        # 同时返回每一层的输出，并保存在一个列表中
        return logsoftmax(a4), [x, a1, a2, a3, a4]

    def backward(self, y_pred, y_true, inputs):
        # 定义神经网络的反向传播方法
        # 如果y_pred和y_true都是张量对象，则将其数据作为输入，否则将它们本身作为输入
        if isinstance(y_pred, myTensor) and isinstance(y_true, myTensor):
            y_pred = y_pred.data
            y_true = y_true.data
        # 计算损失函数的梯度，这里使用交叉熵损失函数
        dL = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
        # 从输入列表中获取每一层的输出，并赋值给对应的变量
        x, a1, a2, a3, a4 = inputs
        # 计算第四层参数的梯度，使用链式法则和LogSoftmax函数的导数
        dZ4 = dL # LogSoftmax函数的导数就是损失函数的梯度
        dW4 = a3.transpose().dot(dZ4)
        dB4 = dZ4.sum(axis=0)
        # 计算第三层参数的梯度，使用链式法则和Tanh函数的导数
        dA3 = dZ4.dot(self.w4.transpose())
        dZ3 = dA3 * (1 - a3 ** 2) # Tanh函数的导数是1减去平方
        dW3 = a2.transpose().dot(dZ3)
        dB3 = dZ3.sum(axis=0)
        # 计算第二层参数的梯度，使用链式法则和Tanh函数的导数
        dA2 = dZ3.dot(self.w3.transpose())
        dZ2 = dA2 * (1 - a2 ** 2) # Tanh函数的导数是1减去平方
        dW2 = a1.transpose().dot(dZ2)
        dB2 = dZ2.sum(axis=0)
        # 计算第一层参数的梯度，使用链式法则和无激活函数导数
        dA1 = dZ2.dot(self.w2.transpose())
        dZ1 = dA1 # 无激活函数，导数为1
        # 使用神经网络的输入计算第一层权重矩阵的梯度
        dW1 = x.transpose().dot(dZ1)
        dB1 = dZ1.sum(axis=0)
        # 将计算得到的梯度赋值给对应参数的梯度属性
        self.w4.grad = dW4
        self.b4.grad = dB4
        self.w3.grad = dW3
        self.b3.grad = dB3
        self.w2.grad = dW2
        self.b2.grad = dB2
        self.w1.grad = dW1
        self.b1.grad = dB1
    

'''定义一些激活函数，用于给神经网络添加非线性函数'''
def relu(x):
    # 定义ReLU激活函数
    # 如果x是张量对象，则返回一个新的张量对象，其数据为x的数据经过ReLU函数变换后的值
    if isinstance(x, myTensor):
        return myTensor(np.maximum(0, x.data))
    # 如果x是数字，则返回一个数字，其值为x经过ReLU函数变换后的值
    elif isinstance(x, (int, float)):
        return max(0, x)
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for relu")

def sigmoid(x):
    # 定义Sigmoid激活函数
    # 如果x是张量对象，则返回一个新的张量对象，其数据为x的数据经过Sigmoid函数变换后的值
    if isinstance(x, myTensor):
        return myTensor(1 / (1 + np.exp(-x.data)))
    # 如果x是数字，则返回一个数字，其值为x经过Sigmoid函数变换后的值
    elif isinstance(x, (int, float)):
        return 1 / (1 + np.exp(-x))
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for sigmoid")

def tanh(x):
    # 定义Tanh激活函数
    # 如果x是张量对象，则返回一个新的张量对象，其数据为x的数据经过Tanh函数变换后的值
    if isinstance(x, myTensor):
        return myTensor(np.tanh(x.data))
    # 如果x是数字，则返回一个数字，其值为x经过Tanh函数变换后的值
    elif isinstance(x, (int, float)):
        return np.tanh(x)
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for tanh")
    
def logsoftmax(x):
    # 定义LogSoftmax激活函数
    # 如果x是张量对象，则返回一个新的张量对象，其数据为x的数据经过LogSoftmax函数变换后的值
    if isinstance(x, myTensor):
        # 使用numpy库计算输入张量数据的指数
        exp_x = np.exp(x.data)
        # 使用numpy库计算输入张量数据的指数之和
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        # 使用numpy库计算输入张量数据的对数概率
        log_prob_x = np.log(exp_x / sum_exp_x)
        # 返回一个新的张量对象，其数据为输入张量数据的对数概率
        return myTensor(log_prob_x)
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for logsoftmax")


'''定义损失函数'''
def mse(y_pred, y_true):
    # 定义均方误差损失函数
    # 如果y_pred和y_true都是张量对象，则返回一个新的张量对象，其数据为y_pred和y_true之间的均方误差
    if isinstance(y_pred, myTensor) and isinstance(y_true, myTensor):
        return ((y_pred - y_true) ** 2).mean()
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for mse")

def cross_entropy(y_pred, y_true):
    # 定义交叉熵损失函数
    # 如果y_pred和y_true都是张量对象，则返回一个新的张量对象，其数据为y_pred和y_true之间的交叉熵损失
    if isinstance(y_pred, myTensor) and isinstance(y_true, myTensor):
        # 使用numpy库将真实值转换为one-hot编码形式
        y_true_onehot = np.eye(3)[y_true.data]
        # 使用numpy库计算每个样本的交叉熵损失，并求均值
        return myTensor(-np.mean(y_true_onehot * np.log(y_pred.data)))
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for cross_entropy")
    
def nll(y_pred, y_true):
    # 定义负对数似然损失函数
    # 如果y_pred和y_true都是张量对象，则返回一个新的张量对象，其数据为y_pred和y_true之间的负对数似然损失
    if isinstance(y_pred, myTensor) and isinstance(y_true, myTensor):
        # 使用numpy库获取每个样本的预测类别对应的对数概率
        log_prob = y_pred.data[np.arange(len(y_pred)), y_true.data]
        # 使用numpy库计算每个样本的负对数似然损失，并求均值
        return myTensor(-np.mean(log_prob))
    # 否则抛出类型错误异常
    else:
        raise TypeError("Unsupported type for nll")



'''定义若干优化器，用于根据损失函数的梯度更新神经网络的参数'''
class Optimizer:
    def __init__(self, params, lr):
        # 定义优化器的基类
        # params是一个张量对象的列表，表示神经网络的参数
        # lr是一个数字，表示学习率
        self.params = params
        self.lr = lr

    def step(self):
        # 定义优化器的更新步骤
        # 这是一个抽象方法，需要在子类中实现
        raise NotImplementedError

    def zero_grad(self):
        # 定义优化器的梯度清零方法
        # 遍历参数列表中的每个张量对象，将其梯度属性设为零
        for param in self.params:
            param.grad = np.zeros_like(param.data)

class SGD(Optimizer):
    def __init__(self, params, lr):
        # 定义随机梯度下降优化器
        # 调用父类的构造函数
        super().__init__(params, lr)

    def step(self):
        # 定义随机梯度下降优化器的更新步骤
        # 遍历参数列表中的每个张量对象，根据其梯度和学习率更新其数据
        for param in self.params:
            param.data -= self.lr * param.grad

class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # 定义Adam优化器
        # 调用父类的构造函数
        super().__init__(params, lr)
        # beta1和beta2是两个数字，表示一阶和二阶动量的衰减系数
        self.beta1 = beta1
        self.beta2 = beta2
        # eps是一个数字，表示数值稳定性的常数
        self.eps = eps
        # 初始化一个和参数列表长度相同的列表，用于存储一阶和二阶动量
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]
        # 初始化一个计数器t为0，用于记录更新步骤的次数
        self.t = 0

    def step(self):
        # 定义Adam优化器的更新步骤
        # 将计数器t加一
        self.t += 1
        # 遍历参数列表中的每个张量对象及其对应的一阶和二阶动量
        for i, (param, m, v) in enumerate(zip(self.params, self.m, self.v)):
            # 根据公式更新一阶和二阶动量
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            v = self.beta2 * v + (1 - self.beta2) * (param.grad ** 2)
            # 根据公式计算偏差修正后的一阶和二阶动量
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            # 根据公式更新参数数据
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            # 将更新后的一阶和二阶动量保存到列表中
            self.m[i] = m
            self.v[i] = v
