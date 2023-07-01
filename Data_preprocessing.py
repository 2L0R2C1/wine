import numpy as np                      # 导入numpy库，用于处理多维数组
import pandas as pd                     # 导入pandas库，用于处理数据表格
import torch                            # 导入torch库，用于深度学习
from myself_model import myTensor
from torch.utils.data import Dataset    # 导入Dataset类，用于创建自定义数据集
import torch.nn.functional as tnf       # 导入torch.nn.functional模块，用于提供一些神经网络的函数

class CSV_dataset(Dataset): # 定义一个CSV_dataset类，继承自Dataset类

    # 定义初始化方法，接受两个参数：dir是csv文件的路径，if_normalize是是否对数据进行归一化
    def __init__(self, dir, if_normalize=True): 
        super().__init__() # 调用父类的初始化方法

        data = pd.read_csv(dir) # 使用pandas读取csv文件，默认以逗号分隔

        nplist = data.T.to_numpy() # 将数据表格转置后转换为numpy数组
        self.data = nplist[:-1].T.astype(np.float64) # 取出除了第一行和最后一行的数据，并转置回来，并转换为64位浮点数，并赋值给self.data属性
        self.target = nplist[-1] # 取出最后一行的数据，并赋值给self.target属性

        # # Tensor化
        self.data = torch.FloatTensor(self.data) # 将数据转换为torch的浮点张量
        self.target_num = torch.LongTensor(self.target.astype(float)) # 将目标转换为浮点数后再转换为torch的长整数张量

        # myTensor化
        # self.data = myTensor(self.data) # 将数据转换为自定义的张量对象
        # self.target_num = myTensor(self.target.astype(float)) # 将目标转换为浮点数后再转换为自定义的张量对象

        if if_normalize: # 如果需要归一化
            self.data = tnf.normalize(self.data) 

    # 获取某个索引位置的数据和目标
    def __getitem__(self, index): 
        return self.data[index], self.target_num[index] 

    # 获取数据集长度
    def __len__(self): 
        return len(self.target) 

# 将数据集按照比例划分为训练集和测试集，data是数据集，rate是训练集占比
def data_split(data, rate): 
    train_l = int(len(data) * rate) 
    test_l = len(data) - train_l 
    # 打乱数据集，进行随机划分
    return torch.utils.data.random_split(data, [train_l, test_l]) 

    '''myTensor化后'''
    # train_l = int(len(data) * rate) 
    # test_l = len(data) - train_l 
    # # 打乱数据集，进行随机划分
    # # 使用numpy.random.permutation函数生成一个随机排列数组作为索引
    # indices = np.random.permutation(len(data))
    # # 使用切片操作获取训练集和测试集的索引子数组
    # train_indices = indices[:train_l]
    # test_indices = indices[train_l:]
    # # 使用列表推导式获取训练集和测试集的数据和目标子列表
    # train_data = [data[i] for i in train_indices]
    # test_data = [data[i] for i in test_indices]
    # # 返回训练集和测试集
    # return train_data, test_data

