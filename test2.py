import util
from __init__ import SS3
from util import Dataset, Evaluation, span
#from server import Live_Test
import selfadaptionbagging
from selfadaptionbagging import bagging
from selfadaptionbagging import self_adaption_bagging
from random import randint as ra

from sklearn.metrics import accuracy_score
from sklearn import metrics
import math
import Gmean_search


util.Evaluation.clear_cache()#先清除内存
clf = SS3()

s, l, p, _ = clf.get_hyperparameters()

print("Smoothness(s):", s)
print("Significance(l):", l)
print("Sanction(p):", p)

x , y = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data)/kflod")

def builddata(x,y,n):
    len_data = len(y)
    len_po = 0
    len_ne = 0
    for i in range(0, int(len_data)):
        if y[i] == 'po':
            len_po = len_po+1
        if y[i] == 'ne':
            len_ne = len_ne+1

    x_test=[]
    y_test=[]
    x_train=[]
    y_train=[]

    '''ne'''
    for i in range(0, int(len_ne/n)):
        random = ra(1, len_ne)
        x_test.append(x[random])
        y_test.append(y[random])

    for i in range(0, int((n-1)*len_ne/n)):
        random = ra(1, len_ne)
        x_train.append(x[random])
        y_train.append(y[random])

    '''po'''
    for j in range(0, int(len_po/n)):
        random = ra(len_ne, len_ne+len_po)
        x_test.append(x[random])
        y_test.append(y[random])

    for j in range(0, int((n-1)*len_po/n)):
        random = ra(len_ne, len_ne+len_po-1)
        x_train.append(x[random])
        y_train.append(y[random])

    return x_train,y_train,x_test,y_test

'''n折交叉验证'''
def kflod(x,y,n):
    len_data = len(y)
    len_po = 0
    len_ne = 0
    for i in range(0, int(len_data)):
        if y[i] == 'po':
            len_po = len_po+1
        if y[i] == 'ne':
            len_ne = len_ne+1

    flod_ne = len_ne/n
    flod_po= len_po/n

    for j in range(0,n):

        x_train_ne = []
        x_train_po = []

        y_train_ne = []
        y_train_po = []

        x_test_ne = []
        x_test_po = []

        y_test_ne = []
        y_test_po = []

        util.Evaluation.clear_cache()  # 先清除内存

        '''构造测试集'''
        '''
        for z in range(int(j*flod_ne),int((j+1)*flod_ne)):
            x_test_ne.append(x[z])
            y_test_ne.append(y[z])

        for z in range(int(len_ne+j*flod_po),int(len_ne+(j+1)*flod_po)):
            x_test_po.append(x[z])
            y_test_po.append(y[z])

        x_test=x_test_ne+x_test_po
        y_test=y_test_ne+y_test_po
        '''
        '''构造训练集'''
        '''
        for z in range(0,int(j*flod_ne)):
            x_train_ne.append(x[z])
            y_train_ne.append(y[z])

        for z in range(int((j+1)*flod_ne),int(len_ne-1)):
            x_train_ne.append(x[z])
            y_train_ne.append(y[z])

        for z in range(int(len_ne),int(len_ne+j*flod_po)):
            x_train_po.append(x[z])
            y_train_po.append(y[z])


        for z in range(int(len_ne+(j+1)*flod_po),int(len_po+len_ne-1)):
            x_train_po.append(x[z])
            y_train_po.append(y[z])

        x_train=x_train_ne+x_train_po
        y_train=y_train_ne+y_train_po
        '''
        '''开始训练'''
        """
        clf.train(x_train, y_train, n_grams=3)
        clf.set_hyperparameters(s=0.32, l=0.48, p=0.5)
        y_pred = clf.predict(x_test)

        print("Accuracy:", accuracy_score(y_pred, y_test))

        print("f1_score", metrics.f1_score(y_test, y_pred, labels=None, pos_label='po'))
        print("recall", metrics.recall_score(y_test, y_pred, labels=None, pos_label='po'))

        matrixes = metrics.confusion_matrix(y_test, y_pred, labels=['ne', 'po'])
        print(matrixes)

        '''计算G-Mean评分'''
        m = matrixes
        Recall = m[1][1] / (m[1][1] + m[1][0])
        Specificity = m[0][0] / (m[0][1] + m[0][0])
        GMean_Score = math.sqrt(Recall * Specificity)
        print("G-Mean Score: " + str(GMean_Score))

        '''计算特异度'''
        print("Specificity is : ",Specificity)

        '''计算未加权平均召回率'''
        UAR=((m[0][0]/(m[0][0]+m[0][1]))+(m[1][1]/(m[1][0]+m[1][1])))/2
        print("UAR is : ",UAR)

        '''生成词云'''
        clf.save_wordcloud("po", plot=True)  # 正类
        clf.save_wordcloud("ne", color="tomato", plot=True)  # 负类
        clf.save_wordcloud("po", n_grams=3, plot=True)  # 3-gram 正类 
        clf.save_wordcloud("ne", n_grams=3, color="tomato", plot=True)  # 3-gram 负类

        '''绘制混淆矩阵'''
        Evaluation.test(clf, x_test, y_test)  # 绘制混淆矩阵
        """
        x_train,y_train,x_test,y_test=builddata(x,y,5)
        Y_pred=bagging(x_train, y_train, x_test, y_test,5000, self_adaption_bagging(x_train, y_train, x_test, y_test, 5000, 1, 2))

        "记录列表重新启动"
        selfadaptionbagging.GMEAN.clear()
        selfadaptionbagging.GMEAN.append(0)

        '''输出评价指标'''
        '''评价结果'''
        print("f1_score", metrics.f1_score(y_test, Y_pred, labels=None, pos_label='po'))
        print("recall", metrics.recall_score(y_test, Y_pred, labels=None, pos_label='po'))
        matrixes = metrics.confusion_matrix(y_test, Y_pred, labels=['ne', 'po'])
        print(matrixes)
        '''计算G-Mean评分'''
        m = matrixes
        Recall = m[1][1] / (m[1][1] + m[1][0])
        Specificity = m[0][0] / (m[0][1] + m[0][0])
        GMean_Score = math.sqrt(Recall * Specificity)
        print("G-Mean Score: " + str(GMean_Score))
        '''计算特异度'''
        print("Specificity is : ", Specificity)
        '''计算未加权平均召回率'''
        UAR = ((m[0][0] / (m[0][0] + m[0][1])) + (m[1][1] / (m[1][0] + m[1][1]))) / 2
        print("UAR is : ", UAR)


clf=SS3()
kflod(x,y,5)

'''
网格调参
s_vals=span(0.2, 0.8, 6)  # [0.2 , 0.32, 0.44, 0.56, 0.68, 0.8]
l_vals=span(0.1, 2, 6)    # [0.1 , 0.48, 0.86, 1.24, 1.62, 2]
p_vals=span(0.5, 2, 6)    # [0.5, 0.8, 1.1, 1.4, 1.7, 2]
s,l,p=Gmean_search.GMean_Search(clf,x_test,y_test,s_vals,l_vals,p_vals)
print("best_s:",s)
print("best_l:",l)
print("best_p:",p)
'''

