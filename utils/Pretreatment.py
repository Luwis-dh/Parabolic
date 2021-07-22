import os
import numpy as np
import pandas as pd
import re
import cv2 as cv
import pandas as pd
pd.plotting.register_matplotlib_converters()
def SplitData(X,Y):
    x=np.array(X)
    y=np.array(Y)
    # 转变为适合opencv的数组格式
    # 为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
    return x, y
def Format(Data):
    #小于115长度的加入空
    if(len(Data)<115):
        FOData = []
        for i in range(len(Data)):
            FOData.append(Data[i])
        for i in range(len(Data),115):
            FOData.append(Data[np.random.randint(0,len(Data))])
    #大于115长度的随机顺序提取
    elif(len(Data)>=115):
        FOData = []
        temp = np.zeros([115])
        # 从此数据中随机提取115个函数
        for i in range(115):
            temp[i]=np.random.randint(0,len(Data))
        temp.sort()
        for i in range(115):
            FOData.append(Data[int(temp[i])])
    return FOData
def Preparing(Truepath,Falsepath):
    #数据集预处理
    # Delet(Truepath)
    # Delet(Falsepath)
    #True数据集收集
    Tdatas=ReadData(Truepath)
    # Min = []
    # for i in range(len(Tdatas)):
    #     Min.append(len(Tdatas[i]))
    # print(Min)

    #False数据集收集
    Fdatas=ReadData(Falsepath)

    #数据格式处理
    X=[]
    Y=[]
    conut=0
    for Tdata in Tdatas:
        X.append(Tdata)
        Y.append(1)
        #conut+=1
    for Fdata in Fdatas:
        X.append(Fdata)
        Y.append(0)
    x,y=SplitData(X,Y)

    #x= np.reshape(x, (x.shape[0], -1))
    return x,y
def ReadData(path):
    # 读取文件夹下面的文档
    txts = os.listdir(path)
    datas=[]
    # 1.读取数据集
    for txt in txts:
        try:
            data = np.loadtxt(path+'//'+txt,np.float32)
            #datas.append(txt)
            datas.append(data)
        except:
            print(path+'//'+txt)
    # converters：将数据列与转换函数进行映射的字典 converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
    #print(data.shape)
    return datas
def GetMax(X):
    Max = 0
    for X_ in X:
        if Max<X_.shape[0]:
            Max = X_.shape[0]
    return Max
def Format(Data,Max):
    if(len(Data)<=Max):
        FOData = []
        for i in range(Data.shape[0]):
            FOData.append(Data[i])
        for i in range(Data.shape[0],Max):
            FOData.append(np.zeros(2))
        return FOData
def SplitData(X,Y):
    x_new = []
    for X_ in X:
        x_new.append(X_[:, 0:2])
    x = np.array(x_new)
    # x=np.array(X)
    y = np.array(Y)
    # 转变为适合opencv的数组格式
    # 为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
    return x, y
def Unify(X):
    Max = GetMax(X)
    x = []
    for X_ in X:
        x.append(Format(X_,Max))
    x=np.array(x)
    x = np.reshape(x, (x.shape[0], -1))
    return x
def Dbscan_Denoise(x,model):
    x_new = []
    for num in range(int(len(x))):
        # 获取DB聚类结果
        X = []
        label = model.fit_predict(x[num][:, 0:2])
        # 获取数量最多的元素
        Maxnum = np.argmax(np.bincount(label))
        # 最多的元素
        for i in range(len(x[num])):
            if label[i] == Maxnum:
                X.append(x[num][i])
        x_new.append(np.array(X))
    X = np.array(x_new)
    return X