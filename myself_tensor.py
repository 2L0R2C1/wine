import numpy as np

'''自定义张量对象，实现张量的加减乘除基本运算'''
class myTensor:
    def __init__(self, data):
        # 将数据转换为numpy数组，并保存为self.data属性
        self.data = np.array(data)
        # 初始化一个和数据形状相同的零数组，并保存为self.grad属性
        self.grad = np.zeros_like(self.data)

    def __getitem__(self, index):
        # 返回指定索引位置的数据和梯度
        return self.data[index], self.grad[index]

    def __setitem__(self, index, value):
        # 修改指定索引位置的数据和梯度
        # 如果value是一个元组或列表，则分别赋值给数据和梯度
        # 否则，只赋值给数据
        if isinstance(value, (tuple, list)):
            self.data[index] = value[0]
            self.grad[index] = value[1]
        else:
            self.data[index] = value
        
    def __add__(self, other):
        # 返回两个张量对象相加的结果
        # 如果other是一个数值，则将其广播到与self.data相同的形状
        # 如果other是一个张量对象，则将其数据与self.data相加
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = myTensor(self.data + other)
        return result

    def __radd__(self, other):
        # 返回两个张量对象相加的结果
        # 这里直接调用__add__方法即可，因为加法运算是可交换的
        return self.__add__(other)

    def __sub__(self, other):
        # 返回两个张量对象相减的结果
        # 如果other是一个数值，则将其广播到与self.data相同的形状
        # 如果other是一个张量对象，则将其数据与self.data相减
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = myTensor(self.data - other)
        return result

    def __rsub__(self, other):
        # 返回两个张量对象相减的结果
        # 这里需要注意，反向减法运算不是可交换的，所以需要反转操作数的顺序
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = myTensor(other - self.data)
        return result

    def __mul__(self, other):
        # 返回两个张量对象相乘的结果
        # 如果other是一个数值，则将其广播到与self.data相同的形状
        # 如果other是一个张量对象，则将其数据与self.data进行逐元素乘法运算
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = myTensor(self.data * other)
        return result

    def __rmul__(self, other):
        # 返回两个张量对象相乘的结果
        # 这里直接调用__mul__方法即可，因为乘法运算是可交换的
        return self.__mul__(other)

    def __truediv__(self, other):
        # 返回两个张量对象相除的结果
        # 如果other是一个数值，则将其广播到与self.data相同的形状
        # 如果other是一个张量对象，则将其数据与self.data进行逐元素除法运算
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = myTensor(self.data / other)
        return result

    def __rtruediv__(self, other):
        # 返回两个张量对象相除的结果
        # 这里需要注意，反向除法运算不是可交换的，所以需要反转操作数的顺序
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = myTensor(other / self.data)
        return result

    def dot(self, other):
        # 返回两个张量对象的点积
        # 如果other是一个数值，则将其广播到与self.data相同的形状
        # 如果other是一个张量对象，则将其数据与self.data进行点积运算
        if isinstance(other, (int, float)):
            other = np.full_like(self.data, other)
        elif isinstance(other, myTensor):
            other = other.data
        result = np.dot(self.data, other)
        return result

    def sum(self):
        # 定义张量的求和运算
        # 返回一个新的张量对象，其数据为原始张量的数据之和
        return myTensor(np.sum(self.data))

    def mean(self):
        # 定义张量的求均值运算
        # 返回一个新的张量对象，其数据为原始张量的数据之均值
        return myTensor(np.mean(self.data))

    def max(self):
        # 定义张量的求最大值运算
        # 返回一个新的张量对象，其数据为原始张量的数据之最大值
        return myTensor(np.max(self.data))

    def argmax(self):
        # 定义张量的求最大值索引运算
        # 返回一个新的张量对象，其数据为原始张量的数据之最大值索引
        return myTensor(np.argmax(self.data))

    def exp(self):
        # 定义张量的指数运算
        # 返回一个新的张量对象，其数据为原始张量的数据之指数
        return myTensor(np.exp(self.data))

    def log(self):
        # 定义张量的对数运算
        # 返回一个新的张量对象，其数据为原始张量的数据之对数
        return myTensor(np.log(self.data))

    def reshape(self, shape):
        # 定义张量的重塑运算
        # 返回一个新的张量对象，其数据为原始张量的数据之重塑
        return myTensor(np.reshape(self.data, shape))

    def transpose(self):
        # 定义张量的转置运算
        # 返回一个新的张量对象，其数据为原始张量的数据之转置
        return myTensor(np.transpose(self.data))

    def __repr__(self):
        # 定义张量的字符串表示方法
        # 返回一个字符串，显示张量的数据和梯度
        return f"myTensor(data={self.data}, grad={self.grad})"