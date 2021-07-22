from sklearn.cluster import DBSCAN
from .Visual import *
import sklearn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
import numpy as np

#Dbscan-聚类
def Dbscan(num=1,x=None):
    model = DBSCAN(eps=100, min_samples=1)
    return model
#SVM-支持向量机
def SVM(train_data, test_data, train_label, test_label):
    svm = sklearn.svm.LinearSVC()
    svm.fit(train_data,train_label)
    print('SVM accuarcy is :%.2f %%'%(svm.score(test_data,test_label)*100))
    return svm
#RandomForest-随机森林
def RandomForest(train_data, test_data, train_label, test_label):
    forest_model = RandomForestClassifier(random_state=1)
    forest_model.fit(train_data, train_label)
    result = forest_model.predict(test_data)
    mask = result == test_label
    correct = np.count_nonzero(mask)
    print("RandomForest accuarcy is: %.2f %%"%(correct * 100.0 / len(result)))
    return forest_model

#GBDT-梯度下降树
def GBDT(train_data, test_data, train_label, test_label):
    #训练模型
    gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=5, subsample=1
                                      , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                      , init=None, random_state=None, max_features=None
                                      , verbose=0, max_leaf_nodes=None, warm_start=False
                                      )
    gbdt.fit(train_data,train_label.ravel())
    pred=gbdt.predict(test_data)

    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(test_label)):
        if pred[i] == test_label[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    print("GBDT Accuracy is: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
    return gbdt

# Xgboost
def xgboost(X_train, X_test, y_train, y_test):
    #训练模型
    model=XGBClassifier()
    #数据转换警告：当需要一维数组时，传递了列向量y。请将Y的形状更改为（n_samples，），例如使用ravel（）。
    model.fit(X_train,y_train)

    #对测试集进行预测
    ans=model.predict(X_test)

    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(y_test)):
        if ans[i] == y_test[i]:
            cnt1 += 1
        else:
            cnt2 += 1

    print("xgboost Accuracy is: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
    return model

def Measure(test_data,test_label,model):
    M = sklearn.metrics.confusion_matrix(test_label, model.predict(test_data))
    Missing_Alarm = (M[0][1] / (M[0][0] + M[1][0])) * 100
    False_Alarm = (M[1][0] / (M[0][0] + M[1][0])) * 100
    print('混淆矩阵:\n', M)
    print('漏警概率:%.2f %%' % (Missing_Alarm))
    print('虚警概率:%.2f %%' % (False_Alarm))
