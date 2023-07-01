import time                             # 导入time模块，用于计算训练时间
import torch                            # 导入torch模块，用于构建和训练神经网络模型
from model import Neural_network        # 导入自定义的神经网络类Net
from torch.utils.data import DataLoader # 导入数据加载器类DataLoader
from Data_preprocessing import data_split, CSV_dataset # 导入自定义的数据预处理函数data_split和CSV_dataset
import matplotlib.pyplot as plt         # 导入matplotlib库

if __name__ == "__main__": 

    # 读入数据并进行预处理
    data = CSV_dataset('wine_data.csv', if_normalize=False)  # 从csv文件中读取数据和目标，并将它们转换为torch张量
    data_train, data_test = data_split(data, 0.9)       # 将数据集按照0.9的比例划分为训练集和测试集
    batchSize = 30      # 设置每个批次的数据量为30
    studyRate = 0.0002   # 设置学习率为0.001

    # 创建一个数据加载器实例data_loader，从训练集中按照批次大小和是否打乱顺序的设置获取数据
    data_loader = DataLoader(data_train, batch_size=batchSize, shuffle=True) 

    # 设置训练设备为GPU或CPU，根据是否有可用的GPU自动选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    print(device) 
    
    # 创建神经网络模型
    net = Neural_network() 

    # 创建一个优化器实例optimizer，使用Adam算法对模型参数进行更新
    # Adam算法是一种基于“momentum”思想的随机梯度下降优化方法，通过迭代更新之前每次计算梯度的一阶moment和二阶moment，并计算滑动平均值，用来更新当前的参数。
    optimizer = torch.optim.Adam(net.parameters(), studyRate) 

    # 创建一个损失函数实例err_func，使用交叉熵损失函数计算模型输出和真实标签之间的差异
    err_func = torch.nn.CrossEntropyLoss() 

    # 创建一个优化器实例optimizer，使用普通的梯度下降算法对模型参数进行更新
    # optimizer = torch.optim.SGD(net.parameters(), studyRate)

    # # # 创建一个损失函数实例err_func，使用均方误差损失函数计算模型输出和真实标签之间的差异
    # err_func = torch.nn.MSELoss()

    train_erres = [] # 创建一个空列表，用于记录每个迭代次数的训练损失
    test_accs = [] # 创建一个空列表，用于记录每个迭代次数的测试准确率

    # 获取开始训练时间
    start = time.time() 

    for epoch in range(500): # 迭代500次
        # 训练BP神经网络
        for step, data in enumerate(data_loader): 
            # 将模型设置为训练模式
            net.train() 
            inputs, target = data 
            # 前向传播
            out = net(inputs) 

            # 使用均方误差损失函数时nn.MSELoss()
            # 将目标转换为独热编码向量
            # target = torch.nn.functional.one_hot(target)
            # # 将神经网络的输出和目标都转换为浮点型张量
            # out = out.float()
            # target = target.float()

            # 计算误差
            err = err_func(out, target) 
            # 清空上一轮的梯度缓存
            optimizer.zero_grad() 
            # 误差反向传播
            err.backward() 
            # 参数更新
            optimizer.step() 

        # 记录训练损失（即误差）
        train_erres.append(err.item()) 

        # 每迭代5次进行一次测试，记录模型分类准确率随训练次数的关系
        if epoch % 5 == 0:  
            net.eval()      # 将模型设置为评估模式
            with torch.no_grad(): # 使用上下文管理器禁用梯度计算，节省内存

                sum = 0     # 初始化一个计数器sum为0，用于记录预测正确的样本数
                total = len(data_test) # 获取测试集的总数
                test_dataloader = DataLoader(data_test, batch_size=1, shuffle=True) 
                
                for i, (inputs, target) in enumerate(test_dataloader): 
                    # 计算网络每一层的输出（前向传播一次）
                    outputs = net(inputs) 

                    # 提取输出量的最大值
                    prediction = torch.max(outputs, 1)[1]  
                    pred_y = prediction.cpu().detach().numpy()  # 将预测张量从GPU取到CPU中，并转换为numpy数组

                    # 计算测试集的准确率
                    if pred_y[0] == target.data.numpy()[0]: 
                        sum += 1 
                    acc = sum / total 
                
                # 记录准确率
                test_accs.append(acc) 

                err = err.item() # 将损失值从张量转换为标量

                # 打印迭代次数，训练损失和测试准确率
                print("训练次数：", epoch , "   训练损失: ", err,"    精确度: ",float(acc * 100),"%") 

    print("训练时间：",time.time() - start) 

    save_path = './Model.pth' 
    torch.save(net.state_dict(), save_path)

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