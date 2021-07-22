import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn

def Draw(num=1,x=None):
    sns.scatterplot(x=x[num][:, 0], y=x[num][:, 1])
    plt.show()

def Draw_model(num=1,x=None,model=None):
    sns.scatterplot(x=x[num][:, 0], y=x[num][:, 1], hue=model.fit_predict(x[num][:, 0:2]))
    plt.show()

def Draw_Compare(num=1,x=None,X=None,model=None):
    f = plt.figure()
    # 去噪前
    f.add_subplot(3, 1, 1)
    plt.title('Before Benoising')
    plt.xlim((-100, 2000))
    plt.ylim((-100, 1200))
    sns.scatterplot(x=x[num][:, 0], y=x[num][:, 1])
    # 去噪后
    f.add_subplot(3, 1, 3)
    plt.xlim((-100, 2000))
    plt.ylim((-100, 1200))
    plt.title('After Benoising')
    sns.scatterplot(x=X[num][:, 0], y=X[num][:, 1])
    plt.show()
    return X

def Draw_Roc(model,test_data, test_label):
    sklearn.metrics.plot_roc_curve(model, test_data, test_label)
    plt.show()