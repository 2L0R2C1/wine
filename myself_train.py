import time                             # 导入time模块，用于计算训练时间
from torch.utils.data import DataLoader # 导入数据加载器类DataLoader
from myself_model import NeuralNetwork        # 导入自定义的神经网络类Net
from myself_model import myTensor,Adam,cross_entropy        # 导入自定义模块
from Data_preprocessing import data_split, CSV_dataset # 导入自定义的数据预处理函数data_split和CSV_dataset
import matplotlib.pyplot as plt         # 导入matplotlib库

if __name__ == "__main__":

    # 读入数据并进行预处理
    data = CSV_dataset('Iris.csv', if_normalize=False)  # 从csv文件中读取数据和目标，并将它们转换为torch张量
    data_train, data_test = data_split(data, 0.9)       # 将数据集按照0.9的比例划分为训练集和测试集
    batchSize = 50      # 设置每个批次的数据量为50
    studyRate = 0.001   # 设置学习率为0.001

    # 创建一个数据加载器实例data_loader，从训练集中按照批次大小和是否打乱顺序的设置获取数据
    data_loader = DataLoader(data_train, batch_size=batchSize, shuffle=True) 

    # 创建一个神经网络对象
    model = NeuralNetwork()

    # 创建一个优化器对象，使用Adam优化器
    optimizer = Adam(model.params, lr=0.01)

    # 创建一个损失函数对象，使用交叉熵损失函数
    loss_func = cross_entropy

    train_erres = [] # 创建一个空列表，用于记录每个迭代次数的训练损失
    test_accs = [] # 创建一个空列表，用于记录每个迭代次数的测试准确率

    # 获取开始训练时间
    start = time.time() 

    for epoch in range(500): # 迭代500次
        # 训练BP神经网络
        for x_batch, y_batch in data_loader: 
            x_batch = myTensor(x_batch)
            y_batch = myTensor(y_batch)
            # 计算神经网络的输出和损失值
            y_pred = model.forward(x_batch)
            loss = loss_func(y_pred, y_batch)
            # 打印周期数和损失值
            print(f"Epoch {epoch}, Loss: {loss.data}")
            # 清空优化器的梯度缓存
            optimizer.zero_grad()
            # 计算神经网络参数的梯度
            model.backward(y_pred, y_batch)
            # 更新神经网络参数
            optimizer.step()

        # 记录训练损失（即误差）
        train_erres.append(err.item()) 

        # 每迭代5次进行一次测试，记录模型分类准确率随训练次数的关系
        if epoch % 5 == 0:  
            sum = 0     # 初始化一个计数器sum为0，用于记录预测正确的样本数
            total = len(data_test) # 获取测试集的总数
            test_dataloader = DataLoader(data_test, batch_size=1, shuffle=True) 
            
            for i, (inputs, target) in test_dataloader: 
                # 计算网络每一层的输出（前向传播一次）
                y_pred = model.forward(inputs) 
                # 使用axis=1参数，表示在第二个维度上求最大值索引，即每个样本的预测类别
                y_pred = y_pred.argmax(axis=1)
                # 使用axis=0参数，表示在第一个维度上求均值，即所有样本的准确率
                acc = (y_pred == target).mean(axis=0)
                print(f"Test accuracy: {acc.data}")
            
            # 记录准确率
            test_accs.append(acc) 

            err = err.item() # 将损失值从张量转换为标量

            # 打印迭代次数，训练损失和测试准确率
            print("训练次数：", epoch , "   训练损失: ", err,"    精确度: ",float(acc * 100),"%") 

    print("训练时间：",time.time() - start) 

    # 绘制训练损失和测试准确率的曲线
    plt.figure(figsize=(10, 5)) # 设置画布大小
    plt.subplot(1, 2, 1) # 设置第一个子图
    plt.plot(train_erres, label="Train err") # 绘制训练损失曲线
    plt.xlabel("Epoch") # 设置x轴标签
    plt.ylabel("err") # 设置y轴标签
    plt.legend() # 显示图例
    plt.subplot(1, 2, 2) # 设置第二个子图
    plt.plot(test_accs, label="Test Accuracy") # 绘制测试准确率曲线
    plt.xlabel("Epoch") # 设置x轴标签
    plt.ylabel("Accuracy") # 设置y轴标签
    plt.legend() # 显示图例
    plt.show() # 显示图像

