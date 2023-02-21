# coding=utf-8
from __init__ import SS3
from util import Dataset
from random import randint as ra
from sklearn.metrics import accuracy_score
from sklearn import metrics
import math

clf = SS3()

#X_train, Y_train = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data) - 副本 - 副本/train")
#X_test, Y_test = Dataset.load_from_files("C:/Users/GYM/Desktop/大创与互联网+：基于集成学习的心理状态分析与评测系统/代码/ss3 and t-ss3/erisk2018/eRisk2018/2018/task 1 - depression (test split, train split is 2017 data) - 副本 - 副本/test")


def gmean(matrixes):
    '''gmean函数'''
    m=matrixes
    Recall = m[1][1] / (m[1][1] + m[1][0])
    Specificity = m[0][0] / (m[0][1] + m[0][0])
    GMean_Score=math.sqrt(Recall*Specificity)
    #print("G-Mean Score: "+str(GMean_Score))
    return GMean_Score

def random_sample(samplex,sampley, size: int,strength):
    """分层增强随机抽样:(保证少量数据集先进行数据载入）
    :param sample: 待采样样本
    :param size: 采样个数
    :param strength :采样系数
    """
    '''计算分层系数'''
    a = 0
    b = 0
    for i in range(0, len(sampley)):
        if sampley[i] == 'po': #分层依据可以随着标签不同而更改
            a = a + 1
        else:
            b = b + 1
    if a>b:
        floor=b #分层隔板
        floor_co=floor/(a+b) #分层系数
    else:
        floor=a
        floor_co=floor/(a+b)

    sample_x=[]
    sample_y=[]
    lenx=len(samplex)

    '''第一层采样'''
    for i in range(0,int(size*floor_co*strength)):#设置采样次数
        random=ra(1,floor-1)
        sample_x.append(samplex[random])
        sample_y.append(sampley[random])

    '''第二层采样'''
    for j in range(0,size-int(size*floor_co*strength)):
        random2=ra(floor,lenx-1)
        sample_x.append(samplex[random2])
        sample_y.append(sampley[random2])


    return sample_x,sample_y

GMEAN=[0]

def bagging(X_train,Y_train,X_test,Y_test,smaple_size:int,strength):
    '''训练第i个模型，并进行网格调参'''
    for i in range(0,5):
        '''为每个模型随机采样'''
        x_train,y_train=random_sample(X_train,Y_train,smaple_size,strength)
        '''4个模型集成'''
        clf.train(x_train, y_train, n_grams=i+2)
        clf.set_hyperparameters(s=0.32, l=2.0, p=0.8)
        y_pred = clf.predict(X_test)
        '''保存预测结果'''
        if i==0 :
            y1=y_pred
        if i==1 :
            y2=y_pred
        if i==2 :
            y3=y_pred
        if i==3 :
            y4=y_pred
        if i==4 :
            y5=y_pred

    '''对弱学习器进行评估，效果过差的会丢弃'''
    matrixes1=metrics.confusion_matrix(Y_test,y1,labels=['ne','po'])
    #print(matrixes1)
    matrixes2=metrics.confusion_matrix(Y_test,y2,labels=['ne','po'])
    #print(matrixes2)
    matrixes3=metrics.confusion_matrix(Y_test,y3,labels=['ne','po'])
    #print(matrixes3)
    matrixes4=metrics.confusion_matrix(Y_test,y4,labels=['ne','po'])
    #print(matrixes4)
    matrixes5=metrics.confusion_matrix(Y_test,y5,labels=['ne','po'])
    #print(matrixes5)
    y1_score=gmean(matrixes1)
    y2_score=gmean(matrixes2)
    y3_score=gmean(matrixes3)
    y4_score=gmean(matrixes4)
    y5_score=gmean(matrixes5)
    '''投票得出结果'''
    Y_pred=[]
    test_len=len(y_pred)
    for i in range(0,test_len):
        votes_number = 1
        live_learner = 5
        if y1_score > 0.6:
            if y3[i] == y1[i]:
                votes_number = votes_number + 1
            else:
                different = y1[i]
        else:
            live_learner = live_learner - 1
        if y2_score > 0.6:
            if y3[i] == y2[i]:
                votes_number = votes_number + 1
            else:
                different = y2[i]
        else:
            live_learner = live_learner - 1
        if y4_score > 0.6:
            if y3[i] == y4[i]:
                votes_number = votes_number + 1
            else:
                different = y4[i]
        else:
            live_learner = live_learner - 1
        if y5_score > 0.6:
            if y3[i] == y5[i]:
                votes_number = votes_number + 1
            else:
                different = y5[i]
        else:
            live_learner = live_learner - 1

        if votes_number > (live_learner / 2):  # 投票数大于存活学习器的一半
            Y_pred.append(y3[i])
        else:
            Y_pred.append(different)
    '''计算G-Mean评分'''
    matrixes = metrics.confusion_matrix(Y_test, Y_pred, labels=['ne', 'po'])
    m = matrixes
    Recall = m[1][1] / (m[1][1] + m[1][0])
    Specificity = m[0][0] / (m[0][1] + m[0][0])
    GMean_Score = math.sqrt(Recall * Specificity)
    GMEAN.append(GMean_Score)

    return Y_pred
    #print(Y_pred)



def self_adaption_bagging(X_train,Y_train,X_test,Y_test,sample_size:int,down,up):
    """分层增强随机抽样:(保证少量数据集先进行数据载入）
        :param X_train,Y_train: 待采样样本
        :param sample_size: 样本规模
        down:搜索下界
        up:搜索上界

        return 最佳采样系数
        """
    for i in range(int(down*10),int(up*10)+1):
        bagging(X_train,Y_train,X_test,Y_test,sample_size,i/10)
    best=(GMEAN.index(max(GMEAN))/10)+down-0.1

    print("最优采样系数: ",best)
    return best


#Y_pred=bagging(X_train,Y_train,X_test,Y_test,5000,self_adaption_bagging(X_train,Y_train,X_test,Y_test,5000,7,8))



"""
#print("f1_score",metrics.f1_score(Y_test,Y_pred,labels=None,pos_label='po'))
#print("recall",metrics.recall_score(Y_test,Y_pred,labels=None,pos_label='po'))
#matrixes=metrics.confusion_matrix(Y_test,Y_pred,labels=['ne','po'])
#print(matrixes)

m = matrixes
Recall = m[1][1] / (m[1][1] + m[1][0])
Specificity = m[0][0] / (m[0][1] + m[0][0])
GMean_Score = math.sqrt(Recall * Specificity)
#print("G-Mean Score: " + str(GMean_Score))

#print("Specificity is : ", Specificity)

UAR = ((m[0][0] / (m[0][0] + m[0][1])) + (m[1][1] / (m[1][0] + m[1][1]))) / 2
#print("UAR is : ", UAR)
"""

