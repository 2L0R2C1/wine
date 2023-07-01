import torch
import time
import numpy as np 

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler    # 导入sklearn.preprocessing模块
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader # 导入数据加载器类DataLoader
from Data_preprocessing import data_split, CSV_dataset # 导入自定义的数据预处理函数data_split和CSV_dataset
import matplotlib.pyplot as plt         # 导入matplotlib库

def main():
    # 加载数据集
    wine = load_wine()
    X = wine.data
    y = wine.target
    
    # 创建一个StandardScaler对象
    scaler = StandardScaler()

    # 使用fit()方法对训练集进行拟合，计算训练集的均值和标准差
    scaler.fit(X)

    # 使用transform()方法对训练集和测试集进行标准化转换，得到新的数据集
    X_train_scaled = scaler.transform(X)
    # Y_test_scaled = scaler.transform(y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y, test_size=0.3)

    # 选择一个合适的k值和距离度量方式，创建并拟合knn分类器
    k = 1 # 可以通过交叉验证来选择最优的k值
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)

    # 评估knn分类器
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 打印评估结果
    print("Accuracy:", accuracy)
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    # 绘制准确率随k值的变化曲线
    k_range = range(1, 30)
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracies.append(knn.score(X_test, y_test))

    plt.figure(figsize=(15, 5)) # 设置画布大小
    plt.subplot(1, 2, 1) # 设置第一个子图
    plt.plot(k_range, accuracies)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Accuracy')

    # 绘制混淆矩阵
    plt.subplot(1, 2, 2) # 设置第二个子图
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = range(len(wine.target_names))
    plt.xticks(tick_marks, wine.target_names, rotation=45)
    plt.yticks(tick_marks, wine.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # # 读入数据并进行预处理
    # data = CSV_dataset('wine_data.csv', if_normalize=True)  # 从csv文件中读取数据和目标，并将它们转换为torch张量
    # data_train, data_test = data_split(data, 0.9)       # 将数据集按照0.9的比例划分为训练集和测试集

    # batchSize = 30      # 设置每个批次的数据量为30

    # # 创建一个数据加载器实例data_loader，从训练集中按照批次大小和是否打乱顺序的设置获取数据
    # data_loader = DataLoader(data_train, batch_size=batchSize, shuffle=True) 

    # # 训练决策树模型
    # clf = DecisionTreeClassifier()

    # train_erres = [] # 创建一个空列表，用于记录每个迭代次数的训练损失
    # test_accs = [] # 创建一个空列表，用于记录每个迭代次数的测试准确率

    # # 获取开始训练时间
    # start = time.time() 

    # for epoch in range(500): 
    #     train_loss =0
    #     for step, (x,y) in enumerate(data_loader): 
    #         x = x.numpy()
    #         y = y.numpy()
    #         clf.fit(x,y)
    #         pred = clf.predict(x)
    #         # 计算本批次的损失值，这里使用平方差损失函数
    #         loss = ((pred - y) ** 2).mean()
    #         # 累加本批次的损失值到本轮训练的损失值中
    #         train_loss += loss

    #     # 记录训练损失（即误差）
    #     train_loss /= len(data_loader)
    #     train_erres.append(train_loss)
    #     print("time:",time.time() - start, 'epoch:', epoch + 1, 'loss', train_loss) 

    #     if epoch % 5 == 0:  
    #         test_acc = 0.0
    #         # 遍历测试集的数据加载器，对分类器进行预测
    #         test_dataloader = DataLoader(data_test, batch_size=1, shuffle=True) 
    #         for i, (x, y) in enumerate(test_dataloader): 
    #             x = x.numpy()
    #             y = y.numpy()
    #             test_acc += clf.score(x,y).mean()
            
    #         test_acc /= len(test_dataloader)
    #         test_accs.append(test_acc)    
    #         print("time:",time.time() - start, 'epoch:', epoch + 1, 'acc', test_acc)
             

    # # 绘制训练损失和测试准确率的曲线
    # plt.figure(figsize=(10, 5)) # 设置画布大小
    # plt.subplot(1, 2, 1) # 设置第一个子图
    # plt.plot(train_erres, label="Train err") # 绘制训练损失曲线
    # plt.xlabel("Epoch") # 设置x轴标签
    # plt.ylabel("err") # 设置y轴标签
    # plt.legend() # 显示图例
    # plt.subplot(1, 2, 2) # 设置第二个子图
    # plt.plot(test_accs, label="Test Accuracy") # 绘制测试准确率曲线
    # plt.xlabel("Epoch") # 设置x轴标签
    # plt.ylabel("Accuracy") # 设置y轴标签
    # plt.legend() # 显示图例
    # plt.show() # 显示图像


if __name__ == "__main__": 
    main()
