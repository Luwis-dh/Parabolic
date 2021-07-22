from utils.Pretreatment import *
from utils.Visual import *
from utils.Model import *
#数据路径
Truepath=r'./docs/true'
Falsepath=r'./docs/false'

if __name__=="__main__":
    #数据预处理
    x,y=Preparing(Truepath, Falsepath)
    #数据可视化，num为样本编号，num（0～425）
    num = 1
    #Draw(num=num,x=x)
    #聚类并可视化
    model = Dbscan(num=num,x=x)
    #Draw_model(num=1, x=x, model=model)
    #去噪并比较
    X=Dbscan_Denoise(x,model)
    #x=Draw_Compare(num=num,x=x,X=X,model=model)
    #数据统一
    x = Unify(x)
    #数据集随机分类
    train_data, test_data, train_label, test_label = \
        sklearn.model_selection.train_test_split(x[:], y,random_state=1,train_size=0.6,test_size=0.4)
    #选择模型进行训练，SVM、RandomForest、XGboost、GBDT

    #SVM
    #model = SVM(train_data, test_data, train_label, test_label)

    #RandomForest
    #model = RandomForest(train_data, test_data, train_label, test_label)

    #XGboost
    model = xgboost(train_data, test_data, train_label, test_label)

    #性能度量

    ##Roc曲线
    #Draw_Roc(model,test_data, test_label)
    ##漏警概率,虚警概率,混淆矩阵
    Measure(test_data,test_label,model)